from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import pandas as pd

from src.utils import PROCESSED_DATA_PATH, RAW_DATA_PATH, RESAMPLE_FREQUENCY

LOGGER = logging.getLogger(__name__)


def load_sales_data(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from CSV or Excel."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(file_path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    LOGGER.info("Loaded %s rows from %s", len(frame), file_path.name)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    """Check that the raw file matches the expected assignment schema."""

    expected = {"State", "Date", "Total", "Category"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    missing_mask = parsed.isna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            series.loc[missing_mask].astype(str).str.strip(),
            errors="coerce",
            dayfirst=True,
        )

    if parsed.isna().any():
        bad_values = series.loc[parsed.isna()].astype(str).head(5).tolist()
        raise ValueError(f"Failed to parse some dates. Sample values: {bad_values}")

    return parsed


def clean_sales_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and fix the raw data so every later step sees a clean time series."""

    validate_schema(frame)

    clean = frame.copy()
    clean["State"] = clean["State"].astype(str).str.strip()
    clean["Category"] = clean["Category"].astype(str).str.strip()
    clean["Date"] = _parse_mixed_dates(clean["Date"])
    clean["sales"] = (
        clean["Total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", pd.NA)
        .astype(float)
    )

    clean = clean.drop(columns=["Total"])
    clean = clean.dropna(subset=["State", "Date", "sales"])
    clean = clean.sort_values(["State", "Date"]).reset_index(drop=True)

    duplicates = clean.duplicated(subset=["State", "Date"], keep=False)
    if duplicates.any():
        LOGGER.warning("Found duplicate state/date rows. Aggregating them by sum.")
        clean = (
            clean.groupby(["State", "Date", "Category"], as_index=False)["sales"]
            .sum()
            .sort_values(["State", "Date"])
        )

    return clean


def resample_weekly_sales(frame: pd.DataFrame, freq: str = RESAMPLE_FREQUENCY) -> pd.DataFrame:
    """Convert each state into a weekly series so the forecast horizon is consistent."""

    outputs: List[pd.DataFrame] = []
    for state, group in frame.groupby("State"):
        state_frame = group.sort_values("Date").set_index("Date")
        resampled = state_frame[["sales"]].resample(freq).sum(min_count=1)
        resampled["State"] = state
        resampled["Category"] = group["Category"].mode().iat[0]
        resampled["sales"] = resampled["sales"].interpolate(method="linear")
        resampled["sales"] = resampled["sales"].ffill().bfill()

        outputs.append(resampled.reset_index())

    combined = pd.concat(outputs, ignore_index=True)
    combined = combined.sort_values(["State", "Date"]).reset_index(drop=True)
    return combined


def save_processed_data(frame: pd.DataFrame, path: Path | str = PROCESSED_DATA_PATH) -> Path:
    """Persist the cleaned weekly data so EDA and training use the same source."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    LOGGER.info("Saved processed data to %s", output_path)
    return output_path


def quality_report(frame: pd.DataFrame) -> Dict[str, object]:
    counts = frame.groupby("State").size()
    return {
        "rows": int(len(frame)),
        "states": int(frame["State"].nunique()),
        "categories": sorted(frame["Category"].dropna().unique().tolist()),
        "date_min": frame["Date"].min(),
        "date_max": frame["Date"].max(),
        "per_state_min_rows": int(counts.min()),
        "per_state_max_rows": int(counts.max()),
        "missing_sales": int(frame["sales"].isna().sum()),
    }


def split_state_frame(frame: pd.DataFrame, validation_size: int, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one state into train, validation, and test blocks without shuffling."""

    frame = frame.sort_values("Date").reset_index(drop=True)
    if len(frame) <= validation_size + test_size:
        raise ValueError("Series is too short for validation and test windows.")

    train_end = len(frame) - (validation_size + test_size)
    validation_end = len(frame) - test_size
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def iter_state_splits(
    frame: pd.DataFrame,
    validation_size: int,
    test_size: int,
) -> Generator[Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
    """Yield one state at a time to keep the training loop easy to follow."""

    for state, state_frame in frame.groupby("State"):
        train_frame, validation_frame, test_frame = split_state_frame(state_frame, validation_size, test_size)
        yield state, train_frame, validation_frame, test_frame


def preprocess_data(input_path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Full preprocessing entrypoint used by the pipeline."""

    raw_data = load_sales_data(input_path)
    cleaned_data = clean_sales_data(raw_data)
    weekly_data = resample_weekly_sales(cleaned_data)
    save_processed_data(weekly_data)
    return weekly_data
