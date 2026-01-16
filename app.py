from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Credit Risk", layout="wide")

try:
    st.switch_page("pages/1_Data.py")
except Exception:
    # Fallback for environments where switching isn't available for some reason.
    st.title("Credit Risk")
    st.caption("Open the pages menu (top-left) and select **Data** to start.")

