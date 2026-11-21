"""Unit tests for feature engineering, lag isolation, and data leakage prevention."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, LAG_COLUMNS, create_features


def _sample_ward_series(ward_id: str, n_weeks: int = 10, base_rainfall: float = 10.0) -> list[dict]:
    dates = pd.date_range("2025-01-05", periods=n_weeks, freq="7D")
    records = []
    for i, dt in enumerate(dates):
        records.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "city": "Chattogram",
                "ward_id": ward_id,
                "population_density_sqkm": 25000.0,
                "water_logging_risk_index": 0.5,
                "sanitation_score": 0.6,
                "avg_temp_c": 28.0 + (i * 0.2),
                "max_temp_c": 32.0 + (i * 0.2),
                "min_temp_c": 24.0 + (i * 0.2),
                "rainfall_mm": base_rainfall + (i * 5.0),
                "relative_humidity_pct": 75.0,
                "dengue_cases": i % 3,
            }
        )
    return records


def test_rainfall_lags_never_cross_ward_boundaries():
    """Prove that lag values from Ward A never bleed into the start of Ward B."""
    ward_a = _sample_ward_series("Ward_A", n_weeks=6, base_rainfall=100.0)
    ward_b = _sample_ward_series("Ward_B", n_weeks=6, base_rainfall=10.0)

    # Scramble initial order to prove sorting by [ward_id, date] is deterministic
    df = pd.DataFrame(ward_b + ward_a).sample(frac=1.0, random_state=42)

    features = create_features(df, is_training=False)

    # Ward B row 0 should NEVER have lag values from Ward A (which had 100+ mm rainfall)
    ward_b_rows = features[features["ward_id"] == "Ward_B"].sort_values("date").reset_index(drop=True)

    # On week 1 (index 1), lag 1 of Ward B must be Ward B's week 0 rainfall (10.0), NOT Ward A
    assert ward_b_rows.loc[1, "rainfall_lag_1"] == 10.0
    assert ward_b_rows.loc[2, "rainfall_lag_2"] == 10.0

    # Week 0 imputed value must be Ward B's own current rainfall (10.0), never 100.0
    assert ward_b_rows.loc[0, "rainfall_lag_1"] == 10.0


def test_future_observations_cannot_change_features_for_earlier_weeks():
    """Prove that adding future rows does not alter feature vectors of historical weeks."""
    base_rows = _sample_ward_series("Ward_A", n_weeks=5, base_rainfall=20.0)
    df_base = pd.DataFrame(base_rows)

    feat_base = create_features(df_base, is_training=False)
    vec_week_3_before = feat_base.loc[3, FEATURE_COLUMNS].to_dict()

    # Append 5 future weeks with drastically higher rainfall & temperatures
    future_rows = _sample_ward_series("Ward_A", n_weeks=5, base_rainfall=200.0)
    # Adjust future dates
    future_dates = pd.date_range("2025-02-09", periods=5, freq="7D")
    for r, dt in zip(future_rows, future_dates):
        r["date"] = dt.strftime("%Y-%m-%d")

    df_extended = pd.DataFrame(base_rows + future_rows)
    feat_extended = create_features(df_extended, is_training=False)
    vec_week_3_after = feat_extended.loc[3, FEATURE_COLUMNS].to_dict()

    for col in FEATURE_COLUMNS:
        assert np.isclose(vec_week_3_before[col], vec_week_3_after[col]), f"Feature leakage detected in {col}!"


def test_cumulative_rainfall_sums_are_exact():
    """Verify that 2-week and 4-week rainfall sums strictly equal the sum of lags."""
    rows = _sample_ward_series("Ward_A", n_weeks=8, base_rainfall=15.0)
    df = pd.DataFrame(rows)
    features = create_features(df, is_training=True)

    for _, row in features.iterrows():
        sum_2w = row["rainfall_lag_1"] + row["rainfall_lag_2"]
        sum_4w = (
            row["rainfall_lag_1"]
            + row["rainfall_lag_2"]
            + row["rainfall_lag_3"]
            + row["rainfall_lag_4"]
        )
        assert np.isclose(row["rainfall_sum_2w"], sum_2w)
        assert np.isclose(row["rainfall_sum_4w"], sum_4w)


def test_training_mode_drops_initial_four_weeks():
    """Verify that is_training=True cleanly drops the first 4 weeks per ward where 4-week lags are undefined."""
    rows_a = _sample_ward_series("Ward_A", n_weeks=10)
    rows_b = _sample_ward_series("Ward_B", n_weeks=10)
    df = pd.DataFrame(rows_a + rows_b)

    feat_train = create_features(df, is_training=True)

    # 10 weeks * 2 wards = 20 total. Dropping 4 weeks per ward = (10-4) * 2 = 12 rows
    assert len(feat_train) == 12
    assert not feat_train[LAG_COLUMNS].isna().any().any()
    assert (feat_train.groupby("ward_id").size() == 6).all()


def test_score_rows_contract_and_valid_values():
    """Verify that score_rows complies with the hackathon interface contract."""
    from src.predict import score_rows

    rows = _sample_ward_series("Ward_Test", n_weeks=6, base_rainfall=45.0)
    df = pd.DataFrame(rows)

    res = score_rows(df)

    expected_cols = {"ward_id", "date", "p_nonzero", "expected_cases", "risk_tier"}
    assert expected_cols.issubset(res.columns)
    assert len(res) == 6
    assert (res["p_nonzero"] >= 0.0).all() and (res["p_nonzero"] <= 1.0).all()
    assert (res["expected_cases"] >= 0.0).all()
    assert set(res["risk_tier"]).issubset({"Low", "Medium", "High", "Critical"})


def test_score_rows_missing_bundle_raises_clear_error(tmp_path: Path):
    """Verify that score_rows raises a descriptive FileNotFoundError if the artifact is missing."""
    from src.predict import score_rows

    df = pd.DataFrame(_sample_ward_series("Ward_Test", n_weeks=2))
    non_existent_bundle = tmp_path / "missing_model.joblib"

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        score_rows(df, model_path=non_existent_bundle)

