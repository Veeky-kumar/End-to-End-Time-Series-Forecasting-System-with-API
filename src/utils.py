from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Forecasting Case.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
MODEL_BUNDLE_PATH = SAVED_MODELS_DIR / "forecast_bundle.joblib"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
FORECASTS_PATH = OUTPUTS_DIR / "forecasts.csv"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
FORECAST_HORIZON = 8
RESAMPLE_FREQUENCY = "W-SUN"
VALIDATION_WEEKS = 8
TEST_WEEKS = 8
MIN_HISTORY_POINTS = 70
LAG_STEPS = [1, 7, 30]
ROLLING_WINDOWS = [7, 30]
SEASONAL_PERIOD = 52
LSTM_SEQUENCE_LENGTH = 12
LSTM_EPOCHS = 25
LSTM_BATCH_SIZE = 16
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "reg:squarederror",
    "random_state": 42,
}


def setup_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_project_structure() -> None:
    """Create the folders used by the pipeline so the flow works from a clean clone."""

    ensure_directories(
        RAW_DATA_PATH.parent,
        PROCESSED_DATA_PATH.parent,
        SAVED_MODELS_DIR,
        OUTPUTS_DIR,
        PLOTS_DIR,
        PROJECT_ROOT / "notebooks",
    )
