from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


def read_dataset_from_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported dataset format: {suffix}. Use .xlsx/.xls/.csv")


def read_dataset_from_bytes(data: bytes, *, filename: str) -> pd.DataFrame:
    """
    Read a dataset from uploaded bytes.
    filename is used only to infer the format (.xlsx/.xls/.csv).
    """
    suffix = Path(filename).suffix.lower()
    bio = BytesIO(data)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(bio)
    if suffix == ".csv":
        return pd.read_csv(bio)

    raise ValueError(f"Unsupported dataset format: {suffix}. Use .xlsx/.xls/.csv")
