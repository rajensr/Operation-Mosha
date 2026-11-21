"""Unit tests for src/clean_data.py using tiny synthetic datasets."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path for test discovery
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import pytest

from src.clean_data import DataValidationError, clean_and_validate, run_pipeline


def _base_row(**overrides) -> dict:
    row = {
        "date": "2025-01-05",
        "city": "Chattogram",
        "ward_id": "W01",
        "population_density_sqkm": 12000.0,
        "water_logging_risk_index": 0.4,
        "sanitation_score": 0.7,
        "avg_temp_c": 26.5,
        "max_temp_c": 30.0,
        "min_temp_c": 22.0,
        "rainfall_mm": 12.5,
        "relative_humidity_pct": 75.0,
        "dengue_cases": 0,
    }
    row.update(overrides)
    return row


def test_preserves_zero_dengue_cases_and_strips_whitespace():
    df = pd.DataFrame(
        [
            _base_row(date="2025-01-05", city="  Chattogram ", ward_id=" W01 ", dengue_cases=0),
            _base_row(date="2025-01-12", city="Chattogram", ward_id="W01", dengue_cases=0),
            _base_row(date="2025-01-19", city="Chattogram", ward_id="W01", dengue_cases=3),
        ]
    )
    cleaned, audit = clean_and_validate(df)

    assert len(cleaned) == 3
    assert cleaned["city"].iloc[0] == "Chattogram"
    assert cleaned["ward_id"].iloc[0] == "W01"
    assert audit.zero_dengue_count == 2
    assert audit.zero_dengue_pct == pytest.approx(66.67, abs=0.01)


def test_sorts_by_ward_id_then_date():
    df = pd.DataFrame(
        [
            _base_row(ward_id="W02", date="2025-01-12"),
            _base_row(ward_id="W01", date="2025-01-19"),
            _base_row(ward_id="W01", date="2025-01-05"),
            _base_row(ward_id="W02", date="2025-01-05"),
        ]
    )
    cleaned, _ = clean_and_validate(df)

    pairs = list(zip(cleaned["ward_id"], cleaned["date"].dt.strftime("%Y-%m-%d")))
    assert pairs == [
        ("W01", "2025-01-05"),
        ("W01", "2025-01-19"),
        ("W02", "2025-01-05"),
        ("W02", "2025-01-12"),
    ]


def test_exact_duplicate_keys_are_removed_and_reported():
    row = _base_row(ward_id="W01", date="2025-01-05", dengue_cases=2)
    df = pd.DataFrame([row, row])

    cleaned, audit = clean_and_validate(df)

    assert len(cleaned) == 1
    assert audit.exact_duplicates_removed == 1
    assert audit.conflicting_duplicate_keys == []


def test_conflicting_duplicate_keys_raise_error():
    df = pd.DataFrame(
        [
            _base_row(ward_id="W01", date="2025-01-05", dengue_cases=1),
            _base_row(ward_id="W01", date="2025-01-05", dengue_cases=5),
        ]
    )
    with pytest.raises(DataValidationError, match="conflicting duplicate"):
        clean_and_validate(df)


def test_invalid_humidity_raises_error():
    df = pd.DataFrame([_base_row(relative_humidity_pct=105.0)])
    with pytest.raises(DataValidationError, match="relative_humidity_pct"):
        clean_and_validate(df)


def test_negative_rainfall_raises_error():
    df = pd.DataFrame([_base_row(rainfall_mm=-2.5)])
    with pytest.raises(DataValidationError, match="rainfall_mm"):
        clean_and_validate(df)


def test_negative_or_non_integer_dengue_cases_raises_error():
    df_neg = pd.DataFrame([_base_row(dengue_cases=-1)])
    with pytest.raises(DataValidationError, match="negative dengue_cases"):
        clean_and_validate(df_neg)

    df_float = pd.DataFrame([_base_row(dengue_cases=2.5)])
    with pytest.raises(DataValidationError, match="non-integer dengue_cases"):
        clean_and_validate(df_float)


def test_out_of_scale_index_flagged_not_silently_rescaled():
    df = pd.DataFrame([_base_row(water_logging_risk_index=7.5)])
    with pytest.raises(DataValidationError, match="Flagged without rescaling"):
        clean_and_validate(df)


def test_missing_values_reported_without_fabrication():
    df = pd.DataFrame(
        [
            _base_row(date="2025-01-05", rainfall_mm=None),
            _base_row(date="2025-01-12", rainfall_mm=20.0),
        ]
    )
    cleaned, audit = clean_and_validate(df)

    assert len(cleaned) == 2
    assert pd.isna(cleaned["rainfall_mm"].iloc[0])
    assert audit.missing_values["rainfall_mm"] == 1


def test_detects_weekly_gaps_per_ward_without_fabricating_rows():
    df = pd.DataFrame(
        [
            _base_row(ward_id="W01", date="2025-01-05"),
            _base_row(ward_id="W01", date="2025-01-26"),  # 21-day gap -> 2 missing weeks
            _base_row(ward_id="W02", date="2025-01-05"),
            _base_row(ward_id="W02", date="2025-01-12"),  # 7-day gap -> no missing weeks
        ]
    )
    cleaned, audit = clean_and_validate(df)

    assert len(cleaned) == 4  # No fabricated rows
    assert "W01" in audit.weekly_gaps
    assert "W02" not in audit.weekly_gaps
    assert audit.total_missing_weeks == 2


def test_run_pipeline_writes_outputs_and_preserves_raw_file(tmp_path: Path):
    raw_csv = tmp_path / "dengue.csv"
    out_csv = tmp_path / "cleaned.csv"
    report_md = tmp_path / "data_quality.md"

    df = pd.DataFrame(
        [
            _base_row(ward_id="W02", date="2025-01-12", dengue_cases=0),
            _base_row(ward_id="W01", date="2025-01-05", dengue_cases=4),
        ]
    )
    df.to_csv(raw_csv, index=False)
    raw_bytes_before = raw_csv.read_bytes()

    audit = run_pipeline(input_path=raw_csv, output_path=out_csv, report_path=report_md)

    assert raw_csv.read_bytes() == raw_bytes_before
    assert out_csv.exists()
    assert report_md.exists()
    assert audit.cleaned_row_count == 2
    assert "Data Quality Audit Report" in report_md.read_text(encoding="utf-8")
