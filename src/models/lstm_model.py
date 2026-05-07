from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LSTMMetadata:
    sequence_length: int
    freq: str
    epochs: int
    batch_size: int


class LSTMForecaster:
    name = "LSTM"

    def __init__(self, sequence_length: int, freq: str, epochs: int, batch_size: int) -> None:
        self.metadata = LSTMMetadata(
            sequence_length=sequence_length,
            freq=freq,
            epochs=epochs,
            batch_size=batch_size,
        )
        self.model = None
        self.scaler = MinMaxScaler()
        self.history_values: np.ndarray | None = None
        self.training_dates: pd.Series | None = None

    def _create_sequences(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_values, y_values = [], []
        for index in range(self.metadata.sequence_length, len(values)):
            x_values.append(values[index - self.metadata.sequence_length : index, 0])
            y_values.append(values[index, 0])
        return np.array(x_values), np.array(y_values)

    def fit(self, train_frame: pd.DataFrame) -> None:
        try:
            from tensorflow.keras import Input
            from tensorflow.keras.layers import LSTM, Dense
            from tensorflow.keras.models import Sequential
        except ImportError as exc:
            raise ImportError("TensorFlow is required. Install `tensorflow` or `tensorflow-cpu`.") from exc

        values = train_frame[["sales"]].astype(float).values
        scaled = self.scaler.fit_transform(values)
        x_values, y_values = self._create_sequences(scaled)
        if len(x_values) == 0:
            raise ValueError("Not enough history to create LSTM sequences.")

        x_values = x_values.reshape((x_values.shape[0], x_values.shape[1], 1))

        model = Sequential(
            [
                Input(shape=(x_values.shape[1], 1)),
                LSTM(64, activation="tanh"),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(
            x_values,
            y_values,
            epochs=self.metadata.epochs,
            batch_size=self.metadata.batch_size,
            verbose=0,
        )

        self.model = model
        self.history_values = scaled
        self.training_dates = train_frame["Date"]
        LOGGER.info("Trained LSTM model")

    def predict(self, horizon: int) -> pd.DataFrame:
        if self.model is None or self.history_values is None or self.training_dates is None:
            raise RuntimeError("LSTM model must be fit before predict is called.")

        window = self.history_values[-self.metadata.sequence_length :].copy()
        predictions = []

        for _ in range(horizon):
            batch = window.reshape((1, window.shape[0], 1))
            scaled_prediction = self.model.predict(batch, verbose=0)[0][0]
            predictions.append(scaled_prediction)
            window = np.vstack([window[1:], [[scaled_prediction]]])

        forecast = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(
            self.training_dates.iloc[-1],
            periods=horizon + 1,
            freq=self.metadata.freq,
        )[1:]
        return pd.DataFrame({"Date": future_dates, "prediction": forecast})
