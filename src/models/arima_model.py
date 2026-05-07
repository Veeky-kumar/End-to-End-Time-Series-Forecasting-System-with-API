from __future__ import annotations

import logging

import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

LOGGER = logging.getLogger(__name__)


class ArimaForecaster:
    """SARIMA wrapper with a simple, explainable parameter strategy."""

    name = "ARIMA_SARIMA"

    def __init__(self, seasonal_period: int = 52) -> None:
        self.seasonal_period = seasonal_period
        self.model_fit = None
        self.training_dates: pd.Series | None = None

    @staticmethod
    def adf_stationarity_test(series: pd.Series) -> dict[str, float]:
        statistic, pvalue, _, _, critical_values, _ = adfuller(series.dropna())
        return {
            "adf_statistic": float(statistic),
            "pvalue": float(pvalue),
            "critical_1pct": float(critical_values["1%"]),
            "critical_5pct": float(critical_values["5%"]),
        }

    def fit(self, train_frame: pd.DataFrame) -> None:
        series = train_frame["sales"].astype(float)
        self.training_dates = train_frame["Date"]
        stationarity = self.adf_stationarity_test(series)
        differencing = 0 if stationarity["pvalue"] < 0.05 else 1

        # In ARIMA(p, d, q), p captures autoregressive memory, d handles differencing,
        # and q captures moving-average error correction. We keep them modest for stability.
        order = (1, differencing, 1)
        seasonal_order = (1, 1, 1, self.seasonal_period) if len(series) > self.seasonal_period * 2 else (0, 0, 0, 0)

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.model_fit = model.fit(disp=False)
        LOGGER.info("Trained SARIMA model with order=%s seasonal_order=%s", order, seasonal_order)

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.model_fit is None or self.training_dates is None:
            raise RuntimeError("ARIMA model must be fit before predict is called.")

        forecast = self.model_fit.forecast(steps=horizon)
        last_date = self.training_dates.iloc[-1]
        future_dates = pd.date_range(last_date, periods=horizon + 1, freq="W-SUN")[1:]
        return pd.DataFrame({"Date": future_dates, "prediction": forecast.values})
