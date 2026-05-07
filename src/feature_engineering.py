from __future__ import annotations

import logging
from typing import List

import holidays
import pandas as pd

from src.utils import LAG_STEPS, ROLLING_WINDOWS

LOGGER = logging.getLogger(__name__)


def add_temporal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar fields help models learn weekly and monthly demand patterns."""

    enriched = frame.copy()
    enriched["day_of_week"] = enriched["Date"].dt.dayofweek
    enriched["week_of_year"] = enriched["Date"].dt.isocalendar().week.astype(int)
    enriched["month"] = enriched["Date"].dt.month
    enriched["quarter"] = enriched["Date"].dt.quarter
    enriched["is_weekend"] = (enriched["day_of_week"] >= 5).astype(int)
    return enriched


def add_holiday_feature(frame: pd.DataFrame) -> pd.DataFrame:
    """Holiday flags give the model a simple signal for special demand periods."""

    enriched = frame.copy()
    years = sorted(enriched["Date"].dt.year.unique().tolist())
    india_holidays = holidays.India(years=years)
    enriched["is_holiday"] = enriched["Date"].dt.date.astype("object").isin(india_holidays).astype(int)
    return enriched


def add_lag_and_rolling_features(
    frame: pd.DataFrame,
    lags: List[int],
    rolling_windows: List[int],
) -> pd.DataFrame:
    """Lags and rolling windows give the model memory about recent sales behaviour."""

    enriched = frame.copy().sort_values(["State", "Date"])

    for lag in lags:
        # Lag features teach the model that "what happened recently" often predicts "what happens next".
        enriched[f"sales_lag_{lag}"] = enriched.groupby("State")["sales"].shift(lag)

    for window in rolling_windows:
        grouped = enriched.groupby("State")["sales"]
        enriched[f"rolling_mean_{window}"] = grouped.shift(1).rolling(window=window).mean()
        enriched[f"rolling_std_{window}"] = grouped.shift(1).rolling(window=window).std()

    return enriched


def build_supervised_features(
    frame: pd.DataFrame,
    lags: List[int] | None = None,
    rolling_windows: List[int] | None = None,
) -> pd.DataFrame:
    """Build the full feature table used by XGBoost and useful for inspection."""

    lags = lags or LAG_STEPS
    rolling_windows = rolling_windows or ROLLING_WINDOWS
    featured = add_temporal_features(frame)
    featured = add_holiday_feature(featured)
    featured = add_lag_and_rolling_features(featured, lags, rolling_windows)
    return featured


def feature_columns(frame: pd.DataFrame) -> List[str]:
    """Return the columns a supervised model should train on."""

    excluded = {"State", "Date", "Category", "sales"}
    return [column for column in frame.columns if column not in excluded]
