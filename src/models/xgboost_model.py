from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

from src.feature_engineering import build_supervised_features, feature_columns

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class XGBoostMetadata:
    lags: List[int]
    rolling_windows: List[int]
    freq: str


class XGBoostForecaster:
    name = "XGBoost"

    def __init__(self, params: dict, lags: List[int], rolling_windows: List[int], freq: str) -> None:
        self.params = params
        self.metadata = XGBoostMetadata(lags=lags, rolling_windows=rolling_windows, freq=freq)
        self.model = None
        self.history_frame: pd.DataFrame | None = None
        self.columns: List[str] = []
        self.feature_importance_: dict[str, float] = {}

    def fit(self, train_frame: pd.DataFrame) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost is required. Install the `xgboost` package.") from exc

        featured = build_supervised_features(
            train_frame,
            lags=self.metadata.lags,
            rolling_windows=self.metadata.rolling_windows,
        ).dropna()

        self.columns = feature_columns(featured)
        model = XGBRegressor(**self.params)
        model.fit(featured[self.columns], featured["sales"])

        self.model = model
        self.history_frame = train_frame.copy()
        self.feature_importance_ = dict(zip(self.columns, model.feature_importances_.tolist()))
        LOGGER.info("Trained XGBoost model with %s features", len(self.columns))

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.model is None or self.history_frame is None:
            raise RuntimeError("XGBoost model must be fit before predict is called.")

        history = self.history_frame.copy()
        state_name = history["State"].iloc[0]
        category = history["Category"].iloc[0]
        forecasts = []

        for _ in range(horizon):
            next_date = history["Date"].max() + pd.tseries.frequencies.to_offset(self.metadata.freq)
            next_row = pd.DataFrame(
                [{"State": state_name, "Date": next_date, "Category": category, "sales": float("nan")}]
            )
            history = pd.concat([history, next_row], ignore_index=True)
            featured = build_supervised_features(
                history,
                lags=self.metadata.lags,
                rolling_windows=self.metadata.rolling_windows,
            )
            current = featured.tail(1)[self.columns]
            prediction = float(self.model.predict(current)[0])
            history.loc[history.index[-1], "sales"] = prediction
            forecasts.append({"Date": next_date, "prediction": prediction})

        return pd.DataFrame(forecasts)
