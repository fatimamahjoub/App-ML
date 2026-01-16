from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target


@dataclass(frozen=True)
class TrainConfig:
    target_col: str
    test_size: float
    random_state: int
    model_id: str  # key from MODEL_SPECS


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    build: callable


MODEL_SPECS: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        id="logreg",
        label="Logistic Regression",
        build=lambda: LogisticRegression(max_iter=3000),
    ),
    "rf": ModelSpec(
        id="rf",
        label="Random Forest",
        build=lambda: RandomForestClassifier(n_estimators=400, random_state=42),
    ),
    "gb": ModelSpec(
        id="gb",
        label="Gradient Boosting",
        build=lambda: GradientBoostingClassifier(random_state=42),
    ),
    "hgb": ModelSpec(
        id="hgb",
        label="Hist Gradient Boosting",
        build=lambda: HistGradientBoostingClassifier(random_state=42),
    ),
}


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        # Some estimators (e.g. HistGradientBoostingClassifier) require dense input.
        # Setting this forces the combined output to be dense.
        sparse_threshold=0.0,
    )


def get_model(model_id: str):
    spec = MODEL_SPECS.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown model: {model_id}. Expected one of: {sorted(MODEL_SPECS)}")
    return spec.build()


def train_and_evaluate(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cfg: TrainConfig,
) -> tuple[Pipeline, dict[str, Any]]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found.")

    X = df.drop(columns=[cfg.target_col])
    y_raw = df[cfg.target_col]

    # map to binary like the notebook (but keep original names for display)
    if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
        y = y_raw.map({"Risque Elevé": 0, "Risque Faible": 1})
        if y.isna().any():
            # fallback: factorize
            y, uniques = pd.factorize(y_raw)
            target_names = [str(u) for u in uniques]
        else:
            target_names = ["Risque Elevé", "Risque Faible"]
    else:
        y = y_raw
        target_names = sorted(pd.Series(y_raw).dropna().unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if len(pd.unique(y)) > 1 else None
    )

    pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    clf = get_model(cfg.model_id)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_kind = type_of_target(y_test)
    is_binary = y_kind == "binary" and len(pd.unique(y_test)) <= 2

    metric_kwargs: dict[str, Any]
    if is_binary:
        metric_kwargs = {"average": "binary", "zero_division": 0}
    else:
        metric_kwargs = {"average": "weighted", "zero_division": 0}

    metrics: dict[str, Any] = {
        "model_id": cfg.model_id,
        "model_label": MODEL_SPECS[cfg.model_id].label,
        "test_size": cfg.test_size,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, **metric_kwargs)),
        "recall": float(recall_score(y_test, y_pred, **metric_kwargs)),
        "f1": float(f1_score(y_test, y_pred, **metric_kwargs)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "target_names": target_names,
    }

    # AUC if probabilities exist and binary target
    if hasattr(pipe, "predict_proba"):
        try:
            if is_binary:
                proba = pipe.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            pass

    # training ranges for sanity checks on user inputs
    ranges = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        ranges[c] = {"min": float(np.nanmin(s)), "max": float(np.nanmax(s))}
    metrics["numeric_ranges"] = ranges

    return pipe, metrics


def save_artifact(path: Path, *, pipeline: Pipeline, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, path)


def load_artifact(path: Path) -> dict[str, Any]:
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj:
        raise ValueError("Invalid artifact file.")
    return obj


