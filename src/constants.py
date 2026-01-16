from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = APP_DIR / "artifacts"

# Default dataset path (relative to this app folder)
DEFAULT_DATASET_PATH = (APP_DIR / "data" / "Risque_data.xlsx").resolve()

# Target column (as specified in your notebook)
TARGET_COL = "Risque"


