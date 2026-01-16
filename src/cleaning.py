from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CleaningConfig:
    target_col: str
    # Missing values
    numeric_missing: str  # "drop" | "mean" | "median"
    categorical_missing: str  # "drop" | "mode"
    drop_duplicates: bool
    # Outliers
    outlier_method: str  # "none" | "zscore" | "mean_std"
    outlier_cols: list[str]
    zscore_threshold: float
    mean_std_k: float


def infer_column_types(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def apply_missing_values_strategy(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_strategy: str,
    categorical_strategy: str,
) -> pd.DataFrame:
    out = df.copy()

    # Numeric
    if numeric_strategy == "drop":
        out = out.dropna(subset=numeric_cols)
    elif numeric_strategy in {"mean", "median"}:
        for c in numeric_cols:
            if numeric_strategy == "mean":
                fill = out[c].mean()
            else:
                fill = out[c].median()
            out[c] = out[c].fillna(fill)
    else:
        raise ValueError(f"Unknown numeric missing strategy: {numeric_strategy}")

    # Categorical
    if categorical_strategy == "drop":
        out = out.dropna(subset=categorical_cols)
    elif categorical_strategy == "mode":
        for c in categorical_cols:
            if out[c].isna().any():
                mode = out[c].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else "UNKNOWN"
                out[c] = out[c].fillna(fill)
    else:
        raise ValueError(f"Unknown categorical missing strategy: {categorical_strategy}")

    return out


def _zscore_mask(series: pd.Series, threshold: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sigma = s.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series([True] * len(series), index=series.index)
    z = (s - mu) / sigma
    return z.abs() <= threshold


def _mean_std_mask(series: pd.Series, k: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sigma = s.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series([True] * len(series), index=series.index)
    low = mu - k * sigma
    high = mu + k * sigma
    return (s >= low) & (s <= high)


def apply_outlier_filter(
    df: pd.DataFrame,
    *,
    method: str,
    cols: list[str],
    zscore_threshold: float,
    mean_std_k: float,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns: (filtered_df, report)
    report: per-column kept/removed counts.
    """
    if method == "none" or not cols:
        return df.copy(), {"method": method, "cols": cols, "removed_rows": 0}

    out = df.copy()
    initial_n = len(out)

    mask = pd.Series([True] * len(out), index=out.index)
    per_col: dict[str, dict[str, int]] = {}

    for c in cols:
        if c not in out.columns:
            continue
        if method == "zscore":
            m = _zscore_mask(out[c], zscore_threshold)
        elif method == "mean_std":
            m = _mean_std_mask(out[c], mean_std_k)
        else:
            raise ValueError(f"Unknown outlier method: {method}")

        per_col[c] = {"kept": int(m.sum()), "removed": int((~m).sum())}
        mask &= m

    out = out.loc[mask].copy()
    return out, {
        "method": method,
        "cols": cols,
        "initial_rows": int(initial_n),
        "final_rows": int(len(out)),
        "removed_rows": int(initial_n - len(out)),
        "per_col": per_col,
    }


def clean_dataframe(df: pd.DataFrame, cfg: CleaningConfig) -> tuple[pd.DataFrame, dict]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in dataset.")

    out = df.copy()

    numeric_cols, categorical_cols = infer_column_types(out, cfg.target_col)

    # duplicates
    dup_before = int(out.duplicated().sum())
    if cfg.drop_duplicates:
        out = out.drop_duplicates()
    dup_after = int(out.duplicated().sum())

    # missing values
    nulls_before = int(out.isna().sum().sum())
    out = apply_missing_values_strategy(
        out,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_strategy=cfg.numeric_missing,
        categorical_strategy=cfg.categorical_missing,
    )
    nulls_after = int(out.isna().sum().sum())

    # outliers
    outlier_df, outlier_report = apply_outlier_filter(
        out,
        method=cfg.outlier_method,
        cols=cfg.outlier_cols,
        zscore_threshold=cfg.zscore_threshold,
        mean_std_k=cfg.mean_std_k,
    )

    report = {
        "rows_before": int(df.shape[0]),
        "rows_after": int(outlier_df.shape[0]),
        "duplicates_before": dup_before,
        "duplicates_after": dup_after,
        "nulls_before": nulls_before,
        "nulls_after": nulls_after,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "outliers": outlier_report,
    }

    return outlier_df, report


