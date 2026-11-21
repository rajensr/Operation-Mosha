# Data Quality Audit Report

- **Raw input file:** `data\raw\dengue.csv` (never modified)
- **Cleaned output file:** `data\processed\cleaned.csv`

## Dataset Overview

| Metric | Value |
| --- | --- |
| Raw row count | 31320 |
| Cleaned row count | 31320 |
| Unique wards | 60 |
| Unique cities | 5 |
| Earliest date | 2016-01-04 |
| Latest date | 2025-12-29 |
| Zero dengue_cases rows | 18176 (58.03%) |
| Exact duplicate rows removed | 0 |
| Conflicting duplicate (ward_id, date) keys | 0 |
| Estimated missing ward-weeks | 0 |

## Missing Values by Column

| Column | Missing Count | Coerced to NaN |
| --- | --- | --- |
| `date` | 0 | 0 |
| `city` | 0 | 0 |
| `ward_id` | 0 | 0 |
| `population_density_sqkm` | 0 | 0 |
| `water_logging_risk_index` | 0 | 0 |
| `sanitation_score` | 0 | 0 |
| `avg_temp_c` | 0 | 0 |
| `max_temp_c` | 0 | 0 |
| `min_temp_c` | 0 | 0 |
| `rainfall_mm` | 0 | 0 |
| `relative_humidity_pct` | 0 | 0 |
| `dengue_cases` | 0 | 0 |

## Validation & Invalid Value Checks

- All domain checks passed (nonnegative integer `dengue_cases`, nonnegative `rainfall_mm`, `relative_humidity_pct` in `[0, 100]`, consistent temperatures, and `[0, 1]` indices).

## Weekly Continuity & Gaps per Ward

- No missing weekly gaps (> 7 days) detected for any ward.

## Data Handoff Checklist (Sadrul -> Technical Lead)

- [x] Total rows and unique wards: **31320** rows, **60** unique wards
- [x] Earliest and latest dates: **2016-01-04** to **2025-12-29**
- [x] Percentage of `dengue_cases` equal to zero: **58.03%** (18176 rows, all preserved)
- [x] Missing values and duplicate ward/date pairs: **0** total missing cells; **0** exact duplicates removed; **0** conflicting duplicates
- [x] Invalid values or weekly gaps: **0** invalid values; **0** estimated missing ward-weeks
- [x] Tests passing and cleaning script working
