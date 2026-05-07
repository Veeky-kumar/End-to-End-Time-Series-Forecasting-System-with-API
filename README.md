# Time Series Forecasting System

This project forecasts the next 8 weeks of total sales for each state using the dataset columns `State`, `Date`, `Total`, and `Category`.

The refactor goal is simple: keep the project modular like a real backend or ML system, but make the execution flow easy to read from top to bottom.

## Project structure

```text
time-series-forecasting/
|
|-- data/
|   |-- raw/
|   |   `-- Forecasting Case.csv
|   `-- processed/
|       `-- cleaned_data.csv
|
|-- notebooks/
|   `-- eda.ipynb
|
|-- src/
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- utils.py
|   |-- run_pipeline.py
|   `-- models/
|       |-- arima_model.py
|       |-- prophet_model.py
|       |-- xgboost_model.py
|       `-- lstm_model.py
|
|-- api/
|   `-- main.py
|
|-- saved_models/
|
|-- outputs/
|   |-- forecasts.csv
|   |-- metrics.json
|   `-- plots/
|
|-- requirements.txt
|-- README.md
|-- Dockerfile
`-- .gitignore
```

## Beginner-friendly flow

If you want to understand the project in the simplest order, read the files like this:

1. `src/run_pipeline.py`
2. `src/preprocessing.py`
3. `src/feature_engineering.py`
4. `src/train.py`
5. `src/evaluate.py`
6. `src/predict.py`
7. `api/main.py`

`run_pipeline.py` is the best starting point because it shows the full forecasting workflow in one place:

1. preprocess the raw data
2. build features
3. train and compare models
4. save the winning models
5. generate 8-week forecasts

## What each module does

- `preprocessing.py` loads the raw dataset, validates the schema, parses dates, renames `Total` to `sales`, and resamples state sales into weekly series.
- `feature_engineering.py` creates temporal, lag, rolling, and holiday features.
- `train.py` trains ARIMA, Prophet, XGBoost, and LSTM state by state, compares them using validation RMSE, and saves the best model per state.
- `evaluate.py` contains metrics and plotting helpers.
- `predict.py` loads saved models and creates future forecasts.
- `api/main.py` exposes a simple FastAPI interface for prediction.

## Why these features matter

- Lag features help the model use recent history, which is one of the strongest signals in forecasting.
- Rolling statistics help capture short-term trend and volatility.
- Temporal features help the model recognize weekly, monthly, and quarterly patterns.
- Holiday flags help the model react to unusual demand around special dates.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python -m src.run_pipeline
```

This will create:

- `data/processed/cleaned_data.csv`
- `saved_models/forecast_bundle.joblib`
- `outputs/metrics.json`
- `outputs/forecasts.csv`
- `outputs/plots/*.png`

## Explore the data first

Open `notebooks/eda.ipynb` to inspect:

- trend visualization
- seasonality analysis
- missing value analysis
- state-wise sales patterns

## Run the API

```bash
uvicorn api.main:app --reload
```

Swagger docs will be available at `http://127.0.0.1:8000/docs`.

## API examples

### `GET /health`

```json
{
  "status": "ok"
}
```

### `GET /metrics`

Returns the saved model comparison and state-level metrics.

### `POST /predict`

Request:

```json
{
  "state": "Alabama"
}
```

Response:

```json
{
  "state": "Alabama",
  "forecast": [236709479.5, 233332492.6, 231813792.7, 346693411.1, 355352953.1, 233780694.8, 223035049.7, 276511952.5]
}
```

## Notes

- The raw dataset contains mixed date formats, so preprocessing handles that explicitly.
- Weekly resampling is used so the "next 8 weeks" forecast has a consistent time step.
- Prophet is included in the codebase, but on some Windows environments it may require a working CmdStan installation.
