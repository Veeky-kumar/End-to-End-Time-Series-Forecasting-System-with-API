from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


class ProphetForecaster:
    name = "Prophet"

    def __init__(self) -> None:
        self.model = None
        self.training_frame: pd.DataFrame | None = None

    def fit(self, train_frame: pd.DataFrame) -> None:
        try:
            from prophet import Prophet
        except ImportError as exc:
            raise ImportError("Prophet is required. Install the `prophet` package.") from exc

        prophet_frame = train_frame.rename(columns={"Date": "ds", "sales": "y"})[["ds", "y"]]
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            stan_backend="CMDSTANPY",
        )
        model.add_country_holidays(country_name="IN")
        model.fit(prophet_frame)
        self.model = model
        self.training_frame = prophet_frame
        LOGGER.info("Trained Prophet model")

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Prophet model must be fit before predict is called.")

        future = self.model.make_future_dataframe(periods=horizon, freq="W")
        forecast = self.model.predict(future)
        result = forecast[["ds", "yhat"]].tail(horizon).rename(columns={"ds": "Date", "yhat": "prediction"})
        return result.reset_index(drop=True)
