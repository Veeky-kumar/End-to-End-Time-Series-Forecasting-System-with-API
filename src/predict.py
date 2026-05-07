from __future__ import annotations

from typing import Dict, List

import joblib
import pandas as pd

from src.utils import FORECAST_HORIZON, FORECASTS_PATH, MODEL_BUNDLE_PATH


def load_model_bundle(path=MODEL_BUNDLE_PATH) -> Dict[str, object]:
    """Load the saved models and metrics used by the API and batch forecasts."""

    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}. Run `python -m src.run_pipeline` first.")
    return joblib.load(path)


def available_states(bundle: Dict[str, object] | None = None) -> List[str]:
    bundle = bundle or load_model_bundle()
    return sorted(bundle["models"].keys())


def predict_state(state: str, bundle: Dict[str, object] | None = None) -> pd.DataFrame:
    """Generate the next 8 weekly forecasts for one state."""

    bundle = bundle or load_model_bundle()
    models = bundle["models"]
    if state not in models:
        raise KeyError(f"State `{state}` not found. Available states: {', '.join(available_states(bundle))}")

    model = models[state]
    forecast = model.predict(FORECAST_HORIZON).copy()
    forecast["State"] = state
    return forecast[["State", "Date", "prediction"]]


def forecast_all_states(bundle: Dict[str, object] | None = None, output_path=FORECASTS_PATH) -> pd.DataFrame:
    """Create one combined forecast file for every trained state."""

    bundle = bundle or load_model_bundle()
    frames = [predict_state(state, bundle) for state in available_states(bundle)]
    forecasts = pd.concat(frames, ignore_index=True).rename(columns={"prediction": "forecast"})
    forecasts.to_csv(output_path, index=False)
    return forecasts


def load_metrics(bundle: Dict[str, object] | None = None) -> Dict[str, object]:
    bundle = bundle or load_model_bundle()
    return bundle["metrics"]
