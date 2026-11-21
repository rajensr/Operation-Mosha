# Data Quality and Validation Audit Report

**Author:** Sadrul (Data Cleaning & Validation Lead)  
**Status:** Validated & Clean  
**Date Generated:** 2026-11-21 21:09:27  

---

## 1. Executive Summary
The weekly dengue dataset has been audited, validated, and normalized without altering legitimate zero values or fabricating synthetic records. The pipeline strictly enforces temporal continuity, key uniqueness, physical bounds, and type safety.

- **Total Rows Audited:** 31,320
- **Cleaned Rows Exported:** 31,320
- **Unique Cities (5):** Chattogram, Dhaka North, Dhaka South, Gazipur, Narayanganj
- **Unique Wards:** 60
- **Temporal Horizon:** 2016-01-04 to 2025-12-29 (522 consecutive weeks)
- **Wards with Interval Anomalies / Gaps:** 0 (0 gaps detected)
- **Duplicate (ward_id, date) Keys:** 0 conflicts, 0 redundant duplicates removed

---

## 2. Target Variable Distribution (`dengue_cases`)
Dengue occurrence exhibits strong zero-inflation representing non-epidemic weeks. These zero-case records reflect genuine non-outbreak conditions and are 100% preserved.

| Metric | Measured Value | Percentage of Dataset |
|---|---|---|
| Zero-Case Weeks (`dengue_cases == 0`) | 18,176 | 58.03% |
| Positive Outbreak Weeks (`dengue_cases > 0`) | 13,144 | 41.97% |
| Peak Weekly Ward Case Count | 41 cases | N/A |

---

## 3. Data Integrity & Validation Checks
The cleaning pipeline performed and passed the following automated assertions:

1. **Schema Integrity:** All 12 required fields are present with correct dtypes.
2. **Whitespace Trimming:** Leading and trailing whitespace was stripped from all `city` and `ward_id` identifiers.
3. **Temporal Sorting:** Records are strictly sorted by `ward_id` ASC, then `date` ASC.
4. **Physical Boundary Checks:**
   - Rainfall: `[0.0, 268.83] mm` (All values non-negative).
   - Relative Humidity: `[41.84, 99.00] %` (Within valid physical limits [0, 100]%).
   - Temperatures: `min_temp_c <= avg_temp_c <= max_temp_c` verified across all records.
   - Vulnerability Indices: `water_logging_risk_index` and `sanitation_score` confirmed within canonical `[0.0, 1.0]` range.
5. **No Leakage or Fabrication:** No forward filling across wards, no interpolation across missing intervals, and raw data remains immutable.

---

## 4. Downstream Recommendations for Machine Learning
- Target is zero-inflated (58.03% zeros): A two-stage hurdle architecture (classifier for `p_nonzero` + conditional regressor for severity) is strongly recommended over standard MSE regression.
- Lags must be computed strictly per `ward_id` after chronological ordering to prevent spatial and temporal contamination.
