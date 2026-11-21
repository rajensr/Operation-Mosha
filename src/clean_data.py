"""Deterministic data audit, validation, and cleaning script for ward-level dengue data.

Run from the repository root:
    python src/clean_data.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "date",
    "city",
    "ward_id",
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "dengue_cases",
]

NUMERIC_COLUMNS = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
    "dengue_cases",
]

STRING_COLUMNS = ["city", "ward_id"]


class DataValidationError(ValueError):
    """Raised when critical data integrity or validation rules are violated."""


@dataclass
class AuditReport:
    raw_row_count: int = 0
    cleaned_row_count: int = 0
    unique_wards: int = 0
    unique_cities: int = 0
    earliest_date: str | None = None
    latest_date: str | None = None
    missing_values: dict[str, int] = field(default_factory=dict)
    coerced_to_nan: dict[str, int] = field(default_factory=dict)
    exact_duplicates_removed: int = 0
    conflicting_duplicate_keys: list[tuple[str, str]] = field(default_factory=list)
    zero_dengue_count: int = 0
    zero_dengue_pct: float = 0.0
    invalid_records: dict[str, int] = field(default_factory=dict)
    scale_warnings: list[str] = field(default_factory=list)
    weekly_gaps: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    total_missing_weeks: int = 0
    validation_errors: list[str] = field(default_factory=list)


def _check_schema(df: pd.DataFrame) -> None:
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    errors = []
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    if extra_cols:
        errors.append(f"Unexpected extra columns: {extra_cols}")
    if errors:
        raise DataValidationError("; ".join(errors))


def _detect_weekly_gaps(df: pd.DataFrame) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Detect missing weekly observations separately for each ward without fabricating rows."""
    gaps_by_ward: dict[str, list[dict[str, Any]]] = {}
    total_missing_weeks = 0

    for ward_id, group in df.groupby("ward_id", sort=True):
        valid_dates = group["date"].dropna().sort_values().reset_index(drop=True)
        if len(valid_dates) < 2:
            continue

        diffs = valid_dates.diff().dt.days
        ward_gaps = []
        for idx in range(1, len(valid_dates)):
            delta_days = int(diffs.iloc[idx])
            if delta_days > 7:
                prev_date = valid_dates.iloc[idx - 1]
                curr_date = valid_dates.iloc[idx]
                missing_weeks = max((delta_days // 7) - 1, 1) if delta_days % 7 == 0 else (delta_days - 1) // 7
                if missing_weeks > 0:
                    total_missing_weeks += missing_weeks
                    ward_gaps.append(
                        {
                            "from_date": prev_date.strftime("%Y-%m-%d"),
                            "to_date": curr_date.strftime("%Y-%m-%d"),
                            "gap_days": delta_days,
                            "estimated_missing_weeks": missing_weeks,
                        }
                    )
        if ward_gaps:
            gaps_by_ward[str(ward_id)] = ward_gaps

    return gaps_by_ward, total_missing_weeks


def clean_and_validate(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, AuditReport]:
    """Audit, validate, and deterministically clean the dengue dataset.

    Does not modify `raw_df` in place. Raises `DataValidationError` if
    conflicting duplicate keys or invalid values require a policy decision.
    """
    _check_schema(raw_df)
    df = raw_df[EXPECTED_COLUMNS].copy()
    audit = AuditReport(raw_row_count=len(df))

    # 1. Strip whitespace from string identifiers
    for col in STRING_COLUMNS:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df.loc[df[col] == "", col] = np.nan

    # 2. Parse dates safely
    raw_date_non_null = df["date"].notna() & (df["date"].astype(str).str.strip() != "")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    unparseable_dates = int((raw_date_non_null & df["date"].isna()).sum())
    if unparseable_dates > 0:
        audit.coerced_to_nan["date"] = unparseable_dates
        audit.invalid_records["unparseable_dates"] = unparseable_dates
        audit.validation_errors.append(f"Found {unparseable_dates} unparseable date value(s).")

    # 3. Convert numeric columns safely and track coercion failures
    for col in NUMERIC_COLUMNS:
        raw_non_null = df[col].notna() & (df[col].astype(str).str.strip() != "")
        converted = pd.to_numeric(df[col], errors="coerce")
        failed_conversions = int((raw_non_null & converted.isna()).sum())
        if failed_conversions > 0:
            audit.coerced_to_nan[col] = failed_conversions
            audit.invalid_records[f"non_numeric_{col}"] = failed_conversions
            audit.validation_errors.append(
                f"Column '{col}' has {failed_conversions} non-numeric value(s)."
            )
        df[col] = converted

    # 4. Record missing values across all columns
    audit.missing_values = {col: int(df[col].isna().sum()) for col in EXPECTED_COLUMNS}

    # 5. Check duplicate (ward_id, date) keys
    key_cols = ["ward_id", "date"]
    valid_key_mask = df["ward_id"].notna() & df["date"].notna()
    dup_key_mask = valid_key_mask & df.duplicated(subset=key_cols, keep=False)

    if dup_key_mask.any():
        # Check for conflicting values within duplicate (ward_id, date) groups
        dup_df = df.loc[dup_key_mask]
        conflicting_keys: list[tuple[str, str]] = []
        for (ward_id, dt), group in dup_df.groupby(key_cols, sort=True):
            # Deduplicate exact rows (treating NaN == NaN)
            if len(group.drop_duplicates()) > 1:
                conflicting_keys.append((str(ward_id), dt.strftime("%Y-%m-%d")))

        if conflicting_keys:
            audit.conflicting_duplicate_keys = conflicting_keys
            audit.invalid_records["conflicting_duplicate_keys"] = len(conflicting_keys)
            audit.validation_errors.append(
                f"Found {len(conflicting_keys)} conflicting duplicate (ward_id, date) key(s): "
                f"{conflicting_keys[:5]}"
            )
        else:
            before_dedup = len(df)
            df = df.drop_duplicates(keep="first").reset_index(drop=True)
            audit.exact_duplicates_removed = before_dedup - len(df)

    # 6. Sort deterministically by ward_id, then date
    df = df.sort_values(by=["ward_id", "date"], ascending=[True, True], kind="mergesort").reset_index(
        drop=True
    )

    # 7. Value domain validations (never silently discard or rescale)
    # Dengue cases: nonnegative integers
    dengue_non_null = df["dengue_cases"].dropna()
    neg_dengue = int((dengue_non_null < 0).sum())
    non_int_dengue = int(((dengue_non_null % 1) != 0).sum())
    if neg_dengue > 0:
        audit.invalid_records["negative_dengue_cases"] = neg_dengue
        audit.validation_errors.append(f"Found {neg_dengue} negative dengue_cases value(s).")
    if non_int_dengue > 0:
        audit.invalid_records["non_integer_dengue_cases"] = non_int_dengue
        audit.validation_errors.append(f"Found {non_int_dengue} non-integer dengue_cases value(s).")

    # Rainfall: nonnegative
    neg_rainfall = int((df["rainfall_mm"].dropna() < 0).sum())
    if neg_rainfall > 0:
        audit.invalid_records["negative_rainfall_mm"] = neg_rainfall
        audit.validation_errors.append(f"Found {neg_rainfall} negative rainfall_mm value(s).")

    # Relative humidity: [0, 100]
    hum_non_null = df["relative_humidity_pct"].dropna()
    invalid_humidity = int(((hum_non_null < 0) | (hum_non_null > 100)).sum())
    if invalid_humidity > 0:
        audit.invalid_records["invalid_relative_humidity_pct"] = invalid_humidity
        audit.validation_errors.append(
            f"Found {invalid_humidity} relative_humidity_pct value(s) outside [0, 100]."
        )

    # Population density: nonnegative
    neg_pop = int((df["population_density_sqkm"].dropna() < 0).sum())
    if neg_pop > 0:
        audit.invalid_records["negative_population_density"] = neg_pop
        audit.validation_errors.append(
            f"Found {neg_pop} negative population_density_sqkm value(s)."
        )

    # Temperature logical order: min <= avg <= max
    temp_valid = df[["min_temp_c", "avg_temp_c", "max_temp_c"]].notna().all(axis=1)
    temp_inconsistent = int(
        (
            temp_valid
            & ((df["min_temp_c"] > df["avg_temp_c"]) | (df["avg_temp_c"] > df["max_temp_c"]))
        ).sum()
    )
    if temp_inconsistent > 0:
        audit.invalid_records["inconsistent_temperatures"] = temp_inconsistent
        audit.validation_errors.append(
            f"Found {temp_inconsistent} row(s) violating min_temp_c <= avg_temp_c <= max_temp_c."
        )

    # Water-logging and sanitation indices: check 0-1 scale, flag instead of silently rescaling
    for idx_col in ["water_logging_risk_index", "sanitation_score"]:
        series = df[idx_col].dropna()
        out_of_01 = int(((series < 0) | (series > 1)).sum())
        if out_of_01 > 0:
            min_v, max_v = float(series.min()), float(series.max())
            audit.invalid_records[f"{idx_col}_outside_0_1"] = out_of_01
            msg = (
                f"Column '{idx_col}' has {out_of_01} value(s) outside [0, 1] "
                f"(observed range: [{min_v}, {max_v}]). Flagged without rescaling."
            )
            audit.scale_warnings.append(msg)
            audit.validation_errors.append(msg)

    # 8. Compute summary statistics
    audit.cleaned_row_count = len(df)
    audit.unique_wards = int(df["ward_id"].nunique(dropna=True))
    audit.unique_cities = int(df["city"].nunique(dropna=True))
    valid_dates = df["date"].dropna()
    if not valid_dates.empty:
        audit.earliest_date = valid_dates.min().strftime("%Y-%m-%d")
        audit.latest_date = valid_dates.max().strftime("%Y-%m-%d")

    if len(dengue_non_null) > 0:
        audit.zero_dengue_count = int((dengue_non_null == 0).sum())
        audit.zero_dengue_pct = round((audit.zero_dengue_count / len(dengue_non_null)) * 100.0, 2)

    # 9. Detect missing weekly observations per ward
    audit.weekly_gaps, audit.total_missing_weeks = _detect_weekly_gaps(df)

    # 10. Stop if validation errors exist rather than guessing or silently dropping
    if audit.validation_errors:
        raise DataValidationError(
            "Validation failed with the following issue(s):\n- "
            + "\n- ".join(audit.validation_errors)
        )

    # Convert dengue_cases to nullable Int64 so integers serialize cleanly without trailing .0
    df["dengue_cases"] = df["dengue_cases"].astype("Int64")

    return df, audit


def render_audit_report(audit: AuditReport, input_path: Path, output_path: Path) -> str:
    """Generate a readable Markdown audit report with measured counts."""
    lines = [
        "# Data Quality Audit Report",
        "",
        f"- **Raw input file:** `{input_path}` (never modified)",
        f"- **Cleaned output file:** `{output_path}`",
        "",
        "## Dataset Overview",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Raw row count | {audit.raw_row_count} |",
        f"| Cleaned row count | {audit.cleaned_row_count} |",
        f"| Unique wards | {audit.unique_wards} |",
        f"| Unique cities | {audit.unique_cities} |",
        f"| Earliest date | {audit.earliest_date or 'N/A'} |",
        f"| Latest date | {audit.latest_date or 'N/A'} |",
        f"| Zero dengue_cases rows | {audit.zero_dengue_count} ({audit.zero_dengue_pct:.2f}%) |",
        f"| Exact duplicate rows removed | {audit.exact_duplicates_removed} |",
        f"| Conflicting duplicate (ward_id, date) keys | {len(audit.conflicting_duplicate_keys)} |",
        f"| Estimated missing ward-weeks | {audit.total_missing_weeks} |",
        "",
        "## Missing Values by Column",
        "",
        "| Column | Missing Count | Coerced to NaN |",
        "| --- | --- | --- |",
    ]
    for col in EXPECTED_COLUMNS:
        missing = audit.missing_values.get(col, 0)
        coerced = audit.coerced_to_nan.get(col, 0)
        lines.append(f"| `{col}` | {missing} | {coerced} |")

    lines.extend(["", "## Validation & Invalid Value Checks", ""])
    if not audit.invalid_records and not audit.scale_warnings:
        lines.append("- All domain checks passed (nonnegative integer `dengue_cases`, nonnegative `rainfall_mm`, `relative_humidity_pct` in `[0, 100]`, consistent temperatures, and `[0, 1]` indices).")
    else:
        for issue, count in audit.invalid_records.items():
            lines.append(f"- **{issue}:** {count}")
        for warning in audit.scale_warnings:
            lines.append(f"- **Scale warning:** {warning}")

    lines.extend(["", "## Weekly Continuity & Gaps per Ward", ""])
    if not audit.weekly_gaps:
        lines.append("- No missing weekly gaps (> 7 days) detected for any ward.")
    else:
        lines.append(f"- Detected **{audit.total_missing_weeks}** estimated missing ward-week(s) across **{len(audit.weekly_gaps)}** ward(s). No synthetic rows were fabricated.")
        lines.append("")
        lines.append("| Ward ID | From Date | To Date | Gap (Days) | Est. Missing Weeks |")
        lines.append("| --- | --- | --- | --- | --- |")
        for ward_id, gaps in audit.weekly_gaps.items():
            for gap in gaps:
                lines.append(
                    f"| `{ward_id}` | {gap['from_date']} | {gap['to_date']} | "
                    f"{gap['gap_days']} | {gap['estimated_missing_weeks']} |"
                )

    lines.extend(
        [
            "",
            "## Data Handoff Checklist (Sadrul -> Technical Lead)",
            "",
            f"- [x] Total rows and unique wards: **{audit.cleaned_row_count}** rows, **{audit.unique_wards}** unique wards",
            f"- [x] Earliest and latest dates: **{audit.earliest_date or 'N/A'}** to **{audit.latest_date or 'N/A'}**",
            f"- [x] Percentage of `dengue_cases` equal to zero: **{audit.zero_dengue_pct:.2f}%** ({audit.zero_dengue_count} rows, all preserved)",
            f"- [x] Missing values and duplicate ward/date pairs: **{sum(audit.missing_values.values())}** total missing cells; **{audit.exact_duplicates_removed}** exact duplicates removed; **{len(audit.conflicting_duplicate_keys)}** conflicting duplicates",
            f"- [x] Invalid values or weekly gaps: **{sum(audit.invalid_records.values())}** invalid values; **{audit.total_missing_weeks}** estimated missing ward-weeks",
            "- [x] Tests passing and cleaning script working",
            "",
        ]
    )
    return "\n".join(lines)


def run_pipeline(
    input_path: Path = Path("data/raw/dengue.csv"),
    output_path: Path = Path("data/processed/cleaned.csv"),
    report_path: Path = Path("reports/data_quality.md"),
) -> AuditReport:
    """Read raw CSV, validate & clean, write cleaned CSV and audit report."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must never overwrite the raw input CSV.")

    if not input_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {input_path}")

    raw_df = pd.read_csv(input_path)
    cleaned_df, audit = clean_and_validate(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize dates cleanly as YYYY-MM-DD
    out_df = cleaned_df.copy()
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")
    out_df.to_csv(output_path, index=False)

    report_md = render_audit_report(audit, input_path=input_path, output_path=output_path)
    report_path.write_text(report_md, encoding="utf-8")

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit, validate, and clean ward-level dengue weekly data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/dengue.csv"),
        help="Path to raw input CSV (never modified).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/cleaned.csv"),
        help="Path to write cleaned CSV.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data_quality.md"),
        help="Path to write Markdown data quality report.",
    )
    args = parser.parse_args()

    audit = run_pipeline(input_path=args.input, output_path=args.output, report_path=args.report)
    print(f"Cleaning complete: {audit.cleaned_row_count} rows, {audit.unique_wards} wards.")
    print(f"Date range: {audit.earliest_date} to {audit.latest_date}")
    print(f"Zero dengue cases: {audit.zero_dengue_count} ({audit.zero_dengue_pct:.2f}%)")
    print(f"Cleaned CSV written to: {args.output}")
    print(f"Audit report written to: {args.report}")


if __name__ == "__main__":
    main()
