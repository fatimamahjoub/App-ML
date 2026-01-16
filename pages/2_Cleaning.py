from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import (
    compute_cleaned,
    get_feature_columns,
    get_raw_df,
    require_target_column,
    set_cleaning_cfg,
)
from src.cleaning import CleaningConfig
from src.constants import TARGET_COL


st.set_page_config(page_title="Cleaning • Credit Risk", layout="wide")

st.title("Cleaning")
st.caption("Configure missing values, duplicates, and outlier filtering. Then generate a cleaned dataset.")

raw_df = get_raw_df()
require_target_column(raw_df)
numeric_cols, _cat_cols = get_feature_columns(raw_df)

st.subheader("Configuration")

with st.form("cleaning_form", border=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        numeric_missing = st.selectbox(
            "Numeric missing values",
            options=["drop", "mean", "median"],
            index=1,
        )
        categorical_missing = st.selectbox(
            "Categorical missing values",
            options=["drop", "mode"],
            index=1,
        )
        drop_dups = st.checkbox("Drop duplicate rows", value=True)

    with c2:
        outlier_method = st.selectbox(
            "Outlier removal method",
            options=["none", "zscore", "mean_std"],
            index=0,
        )
        outlier_cols = st.multiselect(
            "Columns for outlier filtering",
            options=numeric_cols,
            default=numeric_cols[:1] if numeric_cols else [],
        )

    with c3:
        z_thr = st.slider("Z-score threshold (zscore)", 1.0, 5.0, 3.0, 0.1)
        k_std = st.slider("k * std (mean±k·std)", 1.0, 5.0, 3.0, 0.1)

    submitted = st.form_submit_button("Apply cleaning", type="primary")

if submitted:
    cfg = CleaningConfig(
        target_col=TARGET_COL,
        numeric_missing=numeric_missing,
        categorical_missing=categorical_missing,
        drop_duplicates=drop_dups,
        outlier_method=outlier_method,
        outlier_cols=outlier_cols,
        zscore_threshold=float(z_thr),
        mean_std_k=float(k_std),
    )
    set_cleaning_cfg(cfg)
    st.success("Cleaning configuration saved. Scroll down to see results.")

st.divider()
st.subheader("Results")

clean_df, report = compute_cleaned(raw_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows (before)", report["rows_before"])
m2.metric("Rows (after)", report["rows_after"])
m3.metric("Nulls (after)", report["nulls_after"])
m4.metric("Duplicates (before)", report["duplicates_before"])

with st.expander("Cleaning report (full)", expanded=False):
    st.json(report)

st.subheader("Preview")
st.dataframe(clean_df.head(25), use_container_width=True)

# Extra feature (only one): allow download of cleaned dataset
st.download_button(
    "Download cleaned dataset (CSV)",
    data=clean_df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)


