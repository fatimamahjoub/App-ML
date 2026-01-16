from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import (
    ARTIFACT_PATH,
    SessionKeys,
    compute_cleaned,
    get_feature_columns,
    get_raw_df,
    require_target_column,
)
from src.constants import TARGET_COL
from src.modeling import MODEL_SPECS, TrainConfig, save_artifact, train_and_evaluate


st.set_page_config(page_title="Modeling • Credit Risk", layout="wide")

st.title("Modeling")
st.caption("Train and evaluate models on the cleaned dataset, then optionally save one to `artifacts/`.")

raw_df = get_raw_df()
require_target_column(raw_df)

st.subheader("Training setup")

t1, t2 = st.columns(2)
with t1:
    test_size = st.slider("Test size", 0.10, 0.50, 0.30, 0.05)
with t2:
    seed = st.number_input("Random seed", value=42, step=1)

st.divider()

clean_df, _report = compute_cleaned(raw_df)
num_cols, cat_cols = get_feature_columns(clean_df)

model_ids = list(MODEL_SPECS.keys())

if st.button(f"Train & evaluate ({len(model_ids)} models)", type="primary"):
    results: list[dict[str, object]] = []
    pipes: dict[str, dict[str, object]] = {}

    with st.spinner("Training models..."):
        for model_id in model_ids:
            pipe, metrics = train_and_evaluate(
                clean_df,
                numeric_cols=num_cols,
                categorical_cols=cat_cols,
                cfg=TrainConfig(
                    target_col=TARGET_COL,
                    test_size=float(test_size),
                    random_state=int(seed),
                    model_id=model_id,
                ),
            )
            results.append(
                {
                    k: v
                    for k, v in metrics.items()
                    if k in {"model_id", "model_label", "accuracy", "precision", "recall", "f1", "roc_auc"}
                }
            )
            pipes[model_id] = {"pipeline": pipe, "metrics": metrics}

    st.session_state[SessionKeys.MODEL_RESULTS] = results
    st.session_state[SessionKeys.MODEL_PIPES] = pipes
    st.success("Done.")


if SessionKeys.MODEL_RESULTS not in st.session_state:
    st.info("Train models to compare results.")
    st.stop()

res_df = pd.DataFrame(st.session_state[SessionKeys.MODEL_RESULTS]).sort_values("f1", ascending=False)
st.subheader("Results")
st.dataframe(res_df, use_container_width=True)

best_id = str(res_df.iloc[0]["model_id"])
best_label = str(res_df.iloc[0].get("model_label", best_id))
st.caption(f"Best by F1: **{best_label}**")

chosen_id = st.selectbox(
    "Model to save / use",
    options=model_ids,
    index=model_ids.index(best_id) if best_id in model_ids else 0,
    format_func=lambda mid: MODEL_SPECS[mid].label,
)

col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("Save selected model", type="secondary"):
        bundle = st.session_state[SessionKeys.MODEL_PIPES][chosen_id]
        meta = {
            "model_id": chosen_id,
            "model_label": MODEL_SPECS[chosen_id].label,
            "metrics": bundle["metrics"],
            "cleaning_config": st.session_state.get(SessionKeys.CLEANING_CFG).__dict__
            if st.session_state.get(SessionKeys.CLEANING_CFG)
            else None,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
        }
        save_artifact(ARTIFACT_PATH, pipeline=bundle["pipeline"], metadata=meta)
        st.success(f"Saved: {ARTIFACT_PATH}")

with col_b:
    with st.expander("Best model classification report", expanded=False):
        st.text(st.session_state[SessionKeys.MODEL_PIPES][best_id]["metrics"]["classification_report"])


