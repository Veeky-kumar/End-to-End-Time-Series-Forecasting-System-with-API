from __future__ import annotations

from src.feature_engineering import build_supervised_features
from src.predict import forecast_all_states
from src.preprocessing import preprocess_data
from src.train import train_all_states
from src.utils import ensure_project_structure, setup_logging


def run_pipeline() -> None:
    """Run the full forecasting flow in the same order a beginner would learn it."""

    setup_logging()
    ensure_project_structure()

    weekly_data = preprocess_data()
    _ = build_supervised_features(weekly_data)
    bundle = train_all_states(weekly_data)
    forecast_all_states(bundle)


if __name__ == "__main__":
    run_pipeline()
