import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.features import create_features, FEATURE_COLUMNS
from src.predict import score_rows


@pytest.fixture
def sample_ward_data():
    """
    Creates a deterministic synthetic multi-ward dataset spanning 8 weeks.
    Ward_A has rainfall 10, 20, 30, ...
    Ward_B has rainfall 100, 200, 300, ...
    """
    dates = pd.date_range("2024-01-01", periods=8, freq="W-MON").strftime("%Y-%m-%d")
    rows = []
    for d_idx, d in enumerate(dates):
        # Ward A
        rows.append(
            {
                "date": d,
                "city": "Dhaka North",
                "ward_id": "Ward_A",
                "population_density_sqkm": 50000,
                "water_logging_risk_index": 0.8,
                "sanitation_score": 0.5,
                "avg_temp_c": 25.0 + d_idx,
                "max_temp_c": 30.0 + d_idx,
                "min_temp_c": 20.0 + d_idx,
                "rainfall_mm": 10.0 * (d_idx + 1),
                "relative_humidity_pct": 70.0,
                "dengue_cases": d_idx % 3,
            }
        )
        # Ward B
        rows.append(
            {
                "date": d,
                "city": "Dhaka South",
                "ward_id": "Ward_B",
                "population_density_sqkm": 60000,
                "water_logging_risk_index": 0.9,
                "sanitation_score": 0.4,
                "avg_temp_c": 26.0 + d_idx,
                "max_temp_c": 31.0 + d_idx,
                "min_temp_c": 21.0 + d_idx,
                "rainfall_mm": 100.0 * (d_idx + 1),
                "relative_humidity_pct": 80.0,
                "dengue_cases": (d_idx + 1) % 4,
            }
        )
    return pd.DataFrame(rows)


def test_lags_do_not_bleed_across_wards(sample_ward_data):
    """
    Asserts that rainfall lags for Ward_B never contain values from Ward_A,
    and vice-versa, proving strict spatial isolation.
    """
    feat_df = create_features(sample_ward_data, is_training=False)

    ward_a = feat_df[feat_df["ward_id"] == "Ward_A"].sort_values("date").reset_index(drop=True)
    ward_b = feat_df[feat_df["ward_id"] == "Ward_B"].sort_values("date").reset_index(drop=True)

    # Week 0 (first week) should not have lags from other wards
    # For Ward B, rainfall is 100, 200, 300, 400, ...
    # At index 1 (second week), lag_1 must be 100.0, not Ward A's 10.0
    assert ward_b.loc[1, "rainfall_lag_1"] == 100.0
    assert ward_a.loc[1, "rainfall_lag_1"] == 10.0

    # At index 2, lag_2 for Ward B must be 100.0, not Ward A's 10.0
    assert ward_b.loc[2, "rainfall_lag_2"] == 100.0
    assert ward_a.loc[2, "rainfall_lag_2"] == 10.0

    # 2-week rolling sum for Ward B at index 2 must be 200 + 100 = 300
    assert ward_b.loc[2, "rainfall_sum_2w"] == 300.0
    assert ward_a.loc[2, "rainfall_sum_2w"] == 30.0


def test_temporal_isolation(sample_ward_data):
    """
    Asserts that mutating future observations does NOT alter features computed
    for past observation dates.
    """
    feat_df_original = create_features(sample_ward_data, is_training=False)

    # Mutate the last 2 weeks (weeks 6 and 7) with extreme storm values
    mutated_data = sample_ward_data.copy()
    future_mask = mutated_data["date"] >= mutated_data["date"].unique()[-2]
    mutated_data.loc[future_mask, "rainfall_mm"] = 9999.0
    mutated_data.loc[future_mask, "avg_temp_c"] = 45.0

    feat_df_mutated = create_features(mutated_data, is_training=False)

    # Features for weeks 0 through 4 must be 100% identical
    historical_dates = sample_ward_data["date"].unique()[:5]
    orig_hist = feat_df_original[feat_df_original["date"].dt.strftime("%Y-%m-%d").isin(historical_dates)]
    mut_hist = feat_df_mutated[feat_df_mutated["date"].dt.strftime("%Y-%m-%d").isin(historical_dates)]

    for col in FEATURE_COLUMNS:
        np.testing.assert_array_almost_equal(
            orig_hist[col].values,
            mut_hist[col].values,
            err_msg=f"Column {col} leaked future values into past records!",
        )


def test_feature_columns_and_nonnull_on_training(sample_ward_data):
    """
    Asserts that when is_training=True, all rows returned have complete lag history
    and zero null values in FEATURE_COLUMNS.
    """
    feat_train = create_features(sample_ward_data, is_training=True)

    # In 8 weeks, dropping first 4 weeks leaves 4 weeks per ward (8 rows total)
    assert len(feat_train) == 8
    assert feat_train[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_score_rows_contract():
    """
    Tests that score_rows complies with the shared integration agreement:
    returns [ward_id, date, p_nonzero, expected_cases, risk_tier].
    """
    test_df = pd.DataFrame(
        [
            {
                "date": "2025-06-02",
                "city": "Dhaka North",
                "ward_id": "DNCC_Ward_01",
                "population_density_sqkm": 55000,
                "water_logging_risk_index": 0.75,
                "sanitation_score": 0.6,
                "avg_temp_c": 29.5,
                "max_temp_c": 34.0,
                "min_temp_c": 25.0,
                "rainfall_mm": 45.2,
                "relative_humidity_pct": 82.0,
            }
        ]
    )

    scored = score_rows(test_df)
    assert not scored.empty
    expected_cols = {"ward_id", "date", "p_nonzero", "expected_cases", "risk_tier"}
    assert expected_cols.issubset(scored.columns)
    assert scored.loc[0, "ward_id"] == "DNCC_Ward_01"
    assert 0.0 <= scored.loc[0, "p_nonzero"] <= 1.0
    assert scored.loc[0, "expected_cases"] >= 0.0
    assert scored.loc[0, "risk_tier"] in {"Low", "Medium", "High", "Critical"}
