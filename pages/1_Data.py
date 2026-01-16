from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app_state import clear_uploaded_dataset, get_raw_df, require_target_column, set_uploaded_dataset
from src.constants import DEFAULT_DATASET_PATH, TARGET_COL


st.set_page_config(page_title="Dataset • Credit Risk", layout="wide")

st.title("Dataset")
st.caption("Use the bundled dataset or upload your own (.xlsx/.csv). All pages will use the selected dataset.")

with st.expander("Dataset source", expanded=True):
    uploaded = st.file_uploader("Upload dataset", type=["xlsx", "xls", "csv"])
    c_a, c_b = st.columns([1, 2])
    with c_a:
        if st.button("Use bundled dataset", type="secondary"):
            clear_uploaded_dataset()
            st.success("Switched back to the bundled dataset.")
    with c_b:
        st.caption(f"Bundled dataset path: `{DEFAULT_DATASET_PATH}`")

    if uploaded is not None:
        set_uploaded_dataset(data=uploaded.getvalue(), filename=uploaded.name)
        st.success(f"Uploaded: `{uploaded.name}`")

df = get_raw_df()
require_target_column(df)

top = st.slider("Rows to preview", min_value=10, max_value=200, value=50, step=10)
st.subheader("Preview")
st.dataframe(df.head(int(top)), use_container_width=True)

st.divider()
st.subheader("Quick summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", int(df.shape[0]))
c2.metric("Columns", int(df.shape[1]))
c3.metric("Nulls", int(df.isna().sum().sum()))
c4.metric("Duplicates", int(df.duplicated().sum()))

with st.expander("Dtypes", expanded=False):
    st.json(df.dtypes.astype(str).to_dict())

with st.expander(f"Target distribution: `{TARGET_COL}`", expanded=False):
    vc = df[TARGET_COL].value_counts(dropna=False).rename_axis(TARGET_COL).reset_index(name="count")
    st.dataframe(vc, use_container_width=True)

st.divider()
st.subheader("Pandas describe")
st.dataframe(df.describe(include="all"), use_container_width=True)


