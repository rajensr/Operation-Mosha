import numpy as np
import pandas as pd
from typing import List

FEATURE_COLUMNS: List[str] = [
    # Static & Vulnerability Indicators
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    # Current Observation Weather
    "avg_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "temp_diurnal_range",
    # Historical Rainfall Lags (strictly per ward)
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_4",
    # Rolling Rainfall Summaries
    "rainfall_sum_2w",
    "rainfall_sum_4w",
    # Lag-1 Weather Indicators
    "avg_temp_lag_1",
    "humidity_lag_1",
    # Seasonality
    "sin_week",
    "cos_week",
]


def create_features(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """
    Constructs deterministic temporal lags, rolling aggregations, and calendar
    features grouped strictly by ward_id to prevent cross-ward contamination.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing required meteorological and ward identifiers.
    is_training : bool, default=False
        If True, drops the initial rows with undefined 4-week lag history
        (4 weeks per ward, N=240) to guarantee clean, non-null training observations.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame sorted by ward_id and date.
    """
    required_cols = {
        "date",
        "ward_id",
        "rainfall_mm",
        "avg_temp_c",
        "max_temp_c",
        "min_temp_c",
        "relative_humidity_pct",
        "population_density_sqkm",
        "water_logging_risk_index",
        "sanitation_score",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {sorted(missing)}")

    # Ensure clean copy and enforce strict chronological sorting
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["ward_id", "date"]).reset_index(drop=True)

    # 1. Diurnal temperature range
    data["temp_diurnal_range"] = data["max_temp_c"] - data["min_temp_c"]

    # 2. Rainfall lags (1, 2, 3, 4 weeks) partitioned strictly by ward_id
    grouped_ward = data.groupby("ward_id")
    data["rainfall_lag_1"] = grouped_ward["rainfall_mm"].shift(1)
    data["rainfall_lag_2"] = grouped_ward["rainfall_mm"].shift(2)
    data["rainfall_lag_3"] = grouped_ward["rainfall_mm"].shift(3)
    data["rainfall_lag_4"] = grouped_ward["rainfall_mm"].shift(4)

    # 3. Rolling rainfall summaries based strictly on past observed weeks
    data["rainfall_sum_2w"] = data["rainfall_lag_1"] + data["rainfall_lag_2"]
    data["rainfall_sum_4w"] = (
        data["rainfall_lag_1"]
        + data["rainfall_lag_2"]
        + data["rainfall_lag_3"]
        + data["rainfall_lag_4"]
    )

    # 4. Lag-1 Weather indicators
    data["avg_temp_lag_1"] = grouped_ward["avg_temp_c"].shift(1)
    data["humidity_lag_1"] = grouped_ward["relative_humidity_pct"].shift(1)

    # 5. Cyclical Epidemiological Week Encoding
    week_of_year = data["date"].dt.isocalendar().week
    data["sin_week"] = np.sin(2.0 * np.pi * week_of_year / 52.0)
    data["cos_week"] = np.cos(2.0 * np.pi * week_of_year / 52.0)

    if is_training:
        # Truncate undefined initial 4 observation weeks per ward to eliminate nulls without forward filling
        data = data.dropna(subset=["rainfall_lag_4"]).reset_index(drop=True)
    else:
        # In inference mode, if individual records without sufficient history are scored,
        # backfill missing lags with available current-week observation to avoid fatal NaNs.
        data["rainfall_lag_1"] = data["rainfall_lag_1"].fillna(data["rainfall_mm"])
        data["rainfall_lag_2"] = data["rainfall_lag_2"].fillna(data["rainfall_lag_1"])
        data["rainfall_lag_3"] = data["rainfall_lag_3"].fillna(data["rainfall_lag_2"])
        data["rainfall_lag_4"] = data["rainfall_lag_4"].fillna(data["rainfall_lag_3"])
        data["rainfall_sum_2w"] = data["rainfall_lag_1"] + data["rainfall_lag_2"]
        data["rainfall_sum_4w"] = (
            data["rainfall_lag_1"]
            + data["rainfall_lag_2"]
            + data["rainfall_lag_3"]
            + data["rainfall_lag_4"]
        )
        data["avg_temp_lag_1"] = data["avg_temp_lag_1"].fillna(data["avg_temp_c"])
        data["humidity_lag_1"] = data["humidity_lag_1"].fillna(data["relative_humidity_pct"])

    return data
