"""Temporal feature engineering pipeline with strict ward isolation and leakage prevention.

Implements ward-grouped meteorological lags, rolling cumulative sums,
environmental indices, and cyclical epidemiological calendar features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "diurnal_temp_range",
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_4",
    "rainfall_sum_2w",
    "rainfall_sum_4w",
    "avg_temp_lag_1",
    "humidity_lag_1",
    "sin_week",
    "cos_week",
]

LAG_COLUMNS: list[str] = [
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "rainfall_lag_4",
    "rainfall_sum_2w",
    "rainfall_sum_4w",
    "avg_temp_lag_1",
    "humidity_lag_1",
]


def create_features(
    df: pd.DataFrame,
    is_training: bool = False,
) -> pd.DataFrame:
    """Generate lag features, rolling summaries, and calendar encodings.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing weekly observations for one or more wards.
    is_training : bool, default False
        When True, rows with undefined lag history (the initial 4 observation
        weeks per ward) are cleanly dropped rather than imputed. When False,
        missing lags are imputed using the earliest available lag values within
        each ward to ensure unbroken inference capability.

    Returns
    -------
    pd.DataFrame
        DataFrame with computed feature columns, sorted by [ward_id, date].
    """
    if df.empty:
        return df.copy()

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    # Enforce deterministic chronological order strictly per ward
    data = data.sort_values(by=["ward_id", "date"], ascending=[True, True], kind="mergesort").reset_index(
        drop=True
    )

    # 1. Ward-grouped meteorological lags (strictly never crossing ward boundaries)
    grouped = data.groupby("ward_id", sort=False)

    data["rainfall_lag_1"] = grouped["rainfall_mm"].shift(1)
    data["rainfall_lag_2"] = grouped["rainfall_mm"].shift(2)
    data["rainfall_lag_3"] = grouped["rainfall_mm"].shift(3)
    data["rainfall_lag_4"] = grouped["rainfall_mm"].shift(4)

    # Cumulative past rainfall sums (past 2 weeks: t-1 + t-2; past 4 weeks: t-1..t-4)
    data["rainfall_sum_2w"] = data["rainfall_lag_1"] + data["rainfall_lag_2"]
    data["rainfall_sum_4w"] = (
        data["rainfall_lag_1"]
        + data["rainfall_lag_2"]
        + data["rainfall_lag_3"]
        + data["rainfall_lag_4"]
    )

    # Lag-1 thermal and humidity context
    data["avg_temp_lag_1"] = grouped["avg_temp_c"].shift(1)
    data["humidity_lag_1"] = grouped["relative_humidity_pct"].shift(1)

    # 2. Contemporary meteorological & diurnal indicators
    data["diurnal_temp_range"] = data["max_temp_c"] - data["min_temp_c"]

    # 3. Cyclical epidemiological calendar encoding (week 1..52)
    iso_week = data["date"].dt.isocalendar().week.astype(float)
    data["sin_week"] = np.sin(2.0 * np.pi * iso_week / 52.0)
    data["cos_week"] = np.cos(2.0 * np.pi * iso_week / 52.0)

    # 4. Handle initial observation weeks per ward
    if is_training:
        # Drop initial 4 observation weeks where 4-week lags are undefined
        data = data.dropna(subset=LAG_COLUMNS).reset_index(drop=True)
    else:
        # In inference mode, impute missing lags if input has fewer than 5 weeks
        for col in LAG_COLUMNS:
            if col.startswith("rainfall_lag_"):
                data[col] = data[col].fillna(data["rainfall_mm"])
            elif col == "rainfall_sum_2w":
                data[col] = data[col].fillna(data["rainfall_mm"] * 2.0)
            elif col == "rainfall_sum_4w":
                data[col] = data[col].fillna(data["rainfall_mm"] * 4.0)
            elif col == "avg_temp_lag_1":
                data[col] = data[col].fillna(data["avg_temp_c"])
            elif col == "humidity_lag_1":
                data[col] = data[col].fillna(data["relative_humidity_pct"])

    return data
