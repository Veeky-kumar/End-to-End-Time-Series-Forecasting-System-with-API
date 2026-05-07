from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_metrics, predict_state
from src.utils import setup_logging

setup_logging()
LOGGER = logging.getLogger(__name__)
app = FastAPI(title="Time Series Forecasting Service", version="1.0.0")


class PredictionRequest(BaseModel):
    state: str = Field(..., description="State name to forecast, for example Punjab")


class PredictionResponse(BaseModel):
    state: str
    forecast: list[float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    try:
        return load_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        forecast_frame = predict_state(request.state)
        forecast = [float(value) for value in forecast_frame["prediction"].tolist()]
        return PredictionResponse(state=request.state, forecast=forecast)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Prediction failed for state %s", request.state)
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
