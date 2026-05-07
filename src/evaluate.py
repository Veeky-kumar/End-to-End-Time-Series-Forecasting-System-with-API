from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean_absolute_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def mean_absolute_percentage_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    safe_actual = np.where(actual == 0, 1e-8, actual)
    return float(np.mean(np.abs((actual - predicted) / safe_actual)) * 100)


def evaluate_forecast(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Calculate the three core metrics used to compare models fairly."""

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def summarise_metrics(model_metrics: Dict[str, List[Dict[str, float]]]) -> pd.DataFrame:
    """Average validation metrics across states so we can rank models."""

    rows = []
    for model_name, bundles in model_metrics.items():
        if not bundles:
            continue
        rows.append(
            {
                "Model": model_name,
                "MAE": float(np.mean([bundle["MAE"] for bundle in bundles])),
                "RMSE": float(np.mean([bundle["RMSE"] for bundle in bundles])),
                "MAPE": float(np.mean([bundle["MAPE"] for bundle in bundles])),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def save_trend_plot(frame: pd.DataFrame, output_path: Path, state: str) -> None:
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(frame["Date"], frame["sales"], label="Sales", color="#1f77b4")
    axis.set_title(f"Sales Trend - {state}")
    axis.set_xlabel("Date")
    axis.set_ylabel("Sales")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_actual_vs_predicted_plot(
    actual_frame: pd.DataFrame,
    predicted_frame: pd.DataFrame,
    output_path: Path,
    state: str,
    model_name: str,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(actual_frame["Date"], actual_frame["sales"], label="Actual", color="#1f77b4")
    axis.plot(predicted_frame["Date"], predicted_frame["prediction"], label="Predicted", color="#ff7f0e")
    axis.set_title(f"Actual vs Predicted - {state} - {model_name}")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_feature_importance_plot(feature_importance: dict[str, float], output_path: Path, state: str) -> None:
    if not feature_importance:
        return

    top_items = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:10]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.barh([key for key, _ in top_items][::-1], [value for _, value in top_items][::-1], color="#2ca02c")
    axis.set_title(f"XGBoost Feature Importance - {state}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
