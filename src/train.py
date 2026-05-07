from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import joblib
import pandas as pd

from src.evaluate import (
    evaluate_forecast,
    save_actual_vs_predicted_plot,
    save_feature_importance_plot,
    save_trend_plot,
    summarise_metrics,
)
from src.models.arima_model import ArimaForecaster
from src.models.lstm_model import LSTMForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.preprocessing import iter_state_splits, preprocess_data, quality_report
from src.utils import (
    FORECAST_HORIZON,
    LAG_STEPS,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_SEQUENCE_LENGTH,
    METRICS_PATH,
    MIN_HISTORY_POINTS,
    MODEL_BUNDLE_PATH,
    PLOTS_DIR,
    RESAMPLE_FREQUENCY,
    ROLLING_WINDOWS,
    SAVED_MODELS_DIR,
    SEASONAL_PERIOD,
    TEST_WEEKS,
    VALIDATION_WEEKS,
    XGBOOST_PARAMS,
    save_json,
)

LOGGER = logging.getLogger(__name__)


def build_models() -> Dict[str, object]:
    """Create the candidate models used in the comparison step."""

    return {
        "ARIMA_SARIMA": ArimaForecaster(seasonal_period=SEASONAL_PERIOD),
        "Prophet": ProphetForecaster(),
        "XGBoost": XGBoostForecaster(
            params=XGBOOST_PARAMS,
            lags=LAG_STEPS,
            rolling_windows=ROLLING_WINDOWS,
            freq=RESAMPLE_FREQUENCY,
        ),
        "LSTM": LSTMForecaster(
            sequence_length=LSTM_SEQUENCE_LENGTH,
            freq=RESAMPLE_FREQUENCY,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
        ),
    }


def fit_and_score_model(model: object, train_frame: pd.DataFrame, actual_frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Train one model and score it on the next time block."""

    model.fit(train_frame)
    predictions = model.predict(len(actual_frame))
    metrics = evaluate_forecast(actual_frame["sales"], predictions["prediction"])
    return predictions, metrics


def choose_best_model_for_state(train_frame: pd.DataFrame, validation_frame: pd.DataFrame) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """Pick the best validation performer for one state."""

    state_metrics: Dict[str, Dict[str, float]] = {}

    for model_name, model in build_models().items():
        try:
            _, metrics = fit_and_score_model(model, train_frame, validation_frame)
            state_metrics[model_name] = metrics
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping %s for this state: %s", model_name, exc)

    if not state_metrics:
        raise RuntimeError("No models completed successfully for this state.")

    best_model_name = min(state_metrics.items(), key=lambda item: item[1]["RMSE"])[0]
    return best_model_name, state_metrics


def train_all_states(data: pd.DataFrame | None = None) -> Dict[str, object]:
    """Train models for every state, save plots, and persist the winning models."""

    modeling_data = data if data is not None else preprocess_data()
    model_metrics: Dict[str, List[Dict[str, float]]] = {
        "ARIMA_SARIMA": [],
        "Prophet": [],
        "XGBoost": [],
        "LSTM": [],
    }
    models_by_state: Dict[str, object] = {}
    state_level_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for state, train_frame, validation_frame, test_frame in iter_state_splits(
        modeling_data,
        validation_size=VALIDATION_WEEKS,
        test_size=TEST_WEEKS,
    ):
        if len(train_frame) < MIN_HISTORY_POINTS:
            LOGGER.warning("Skipping %s because there is not enough history.", state)
            continue

        best_model_name, validation_metrics = choose_best_model_for_state(train_frame, validation_frame)
        for model_name, metrics in validation_metrics.items():
            model_metrics[model_name].append(metrics)

        final_model = build_models()[best_model_name]
        final_train = pd.concat([train_frame, validation_frame], ignore_index=True)
        test_predictions, test_metrics = fit_and_score_model(final_model, final_train, test_frame)

        models_by_state[state] = final_model
        state_level_metrics[state] = validation_metrics
        state_level_metrics[state][f"{best_model_name}_test"] = test_metrics

        full_state_history = pd.concat([train_frame, validation_frame, test_frame], ignore_index=True)
        save_trend_plot(full_state_history, PLOTS_DIR / f"{state}_trend.png", state)
        save_actual_vs_predicted_plot(
            test_frame,
            test_predictions,
            PLOTS_DIR / f"{state}_{best_model_name}_actual_vs_predicted.png",
            state,
            best_model_name,
        )
        if hasattr(final_model, "feature_importance_"):
            save_feature_importance_plot(
                getattr(final_model, "feature_importance_"),
                PLOTS_DIR / f"{state}_{best_model_name}_feature_importance.png",
                state,
            )

    comparison_table = summarise_metrics(model_metrics)
    quality = quality_report(modeling_data)
    metrics_payload = {
        "best_global_model": comparison_table.iloc[0]["Model"] if not comparison_table.empty else None,
        "comparison_table": comparison_table.to_dict(orient="records"),
        "state_level_metrics": state_level_metrics,
        "data_quality_report": quality,
    }
    bundle = {
        "forecast_horizon": FORECAST_HORIZON,
        "metrics": metrics_payload,
        "models": models_by_state,
    }

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    save_json(metrics_payload, METRICS_PATH)
    LOGGER.info("Saved trained models for %s states", len(models_by_state))
    return bundle
