from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.cleaning import CleaningConfig, clean_dataframe, infer_column_types
from src.constants import ARTIFACTS_DIR, DEFAULT_DATASET_PATH, TARGET_COL
from src.data_io import read_dataset_from_bytes, read_dataset_from_path


ARTIFACT_PATH = ARTIFACTS_DIR / "credit_risk_model.joblib"


class SessionKeys:
    DATASET_BYTES = "dataset_bytes"
    DATASET_FILENAME = "dataset_filename"
    CLEANING_CFG = "cleaning_cfg"
    CLEAN_DF = "clean_df"
    CLEAN_REPORT = "clean_report"
    MODEL_RESULTS = "model_results"
    MODEL_PIPES = "model_pipes"


@st.cache_data(show_spinner=False)
def load_default_dataset(path_str: str) -> pd.DataFrame:
    return read_dataset_from_path(Path(path_str))


@st.cache_data(show_spinner=False)
def load_uploaded_dataset(data: bytes, *, filename: str) -> pd.DataFrame:
    return read_dataset_from_bytes(data, filename=filename)


def set_uploaded_dataset(*, data: bytes, filename: str) -> None:
    st.session_state[SessionKeys.DATASET_BYTES] = data
    st.session_state[SessionKeys.DATASET_FILENAME] = filename
    # invalidate derived state
    st.session_state.pop(SessionKeys.CLEANING_CFG, None)
    st.session_state.pop(SessionKeys.CLEAN_DF, None)
    st.session_state.pop(SessionKeys.CLEAN_REPORT, None)
    st.session_state.pop(SessionKeys.MODEL_RESULTS, None)
    st.session_state.pop(SessionKeys.MODEL_PIPES, None)


def clear_uploaded_dataset() -> None:
    st.session_state.pop(SessionKeys.DATASET_BYTES, None)
    st.session_state.pop(SessionKeys.DATASET_FILENAME, None)
    # invalidate derived state
    st.session_state.pop(SessionKeys.CLEANING_CFG, None)
    st.session_state.pop(SessionKeys.CLEAN_DF, None)
    st.session_state.pop(SessionKeys.CLEAN_REPORT, None)
    st.session_state.pop(SessionKeys.MODEL_RESULTS, None)
    st.session_state.pop(SessionKeys.MODEL_PIPES, None)


def get_raw_df() -> pd.DataFrame:
    try:
        if SessionKeys.DATASET_BYTES in st.session_state and SessionKeys.DATASET_FILENAME in st.session_state:
            df = load_uploaded_dataset(
                st.session_state[SessionKeys.DATASET_BYTES],
                filename=st.session_state[SessionKeys.DATASET_FILENAME],
            )
        else:
            df = load_default_dataset(str(DEFAULT_DATASET_PATH))
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()
    return df


def require_target_column(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        st.error(
            f"Expected target column `{TARGET_COL}` not found.\n\n"
            f"Available columns: {df.columns.tolist()}"
        )
        st.stop()


def set_cleaning_cfg(cfg: CleaningConfig) -> None:
    st.session_state[SessionKeys.CLEANING_CFG] = cfg
    # invalidate prior results
    st.session_state.pop(SessionKeys.CLEAN_DF, None)
    st.session_state.pop(SessionKeys.CLEAN_REPORT, None)


def has_cleaning_cfg() -> bool:
    return SessionKeys.CLEANING_CFG in st.session_state


def get_cleaning_cfg() -> CleaningConfig | None:
    return st.session_state.get(SessionKeys.CLEANING_CFG)


def compute_cleaned(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = get_cleaning_cfg()
    if cfg is None:
        st.info("Go to **Cleaning** and click **Apply cleaning** first.")
        st.stop()

    if SessionKeys.CLEAN_DF in st.session_state and SessionKeys.CLEAN_REPORT in st.session_state:
        return st.session_state[SessionKeys.CLEAN_DF], st.session_state[SessionKeys.CLEAN_REPORT]

    clean_df, report = clean_dataframe(df, cfg)
    st.session_state[SessionKeys.CLEAN_DF] = clean_df
    st.session_state[SessionKeys.CLEAN_REPORT] = report
    return clean_df, report


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    require_target_column(df)
    return infer_column_types(df, TARGET_COL)

