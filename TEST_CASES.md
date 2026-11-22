# Operation-Mosha: Public Health Testing Runbook & Field Test Scenarios

This document provides structured, field-ready test cases for Public Health Administrators, Municipal Commissioners, Epidemiologists, and Evaluation Judges to systematically validate the Operation-Mosha Dengue Outbreak Intelligence Portal.

Portal URL: http://localhost:8000 (or http://127.0.0.1:8000)

---

## Section 1: Interactive Web Portal Test Cases

### Test Case 1: Baseline Historical Verification (Routine Surveillance)
* **Objective:** Validate that the portal correctly fetches and renders historical epidemiological records without data leakage.
* **Navigation:** Open http://localhost:8000/#surveillance
* **Test Steps:**
  1. Set **Target City:** `Chattogram`
  2. Set **Target Ward:** `CCC_Ward_01`
  3. Set **Observation Date:** `2024-03-15` (historical baseline)
  4. Set **Current Week Rain:** `15.0 mm`
  5. Set **Average Temperature:** `26.0 deg C`
  6. Set **Relative Humidity:** `65.0 %`
  7. Click **Compute Outbreak Nowcast** (or wait for live recalculation).
* **Expected Result:**
  - **Risk Tier:** `Low` (Green badge).
  - **Expected Caseload:** Low or baseline (< 5 cases/week).
  - **Operational Protocol:** Displays routine entomological monitoring with focus on routine surveillance.
  - **Attribution Cards:** Prior 4-week rainfall lag values populate accurately from historical records.

---

### Test Case 2: Extreme Monsoon Outbreak Simulation (Future Date 2026)
* **Objective:** Verify that entering a future date past the training dataset correctly invokes the two-stage hurdle model with projected meteorological shocks.
* **Navigation:** http://localhost:8000/#surveillance
* **Test Steps:**
  1. Set **Target City:** `Chattogram`
  2. Set **Target Ward:** `CCC_Ward_08`
  3. Set **Observation Date:** `2026-07-20` (Future peak monsoon)
  4. Set **Current Week Rain:** `120.0 mm`
  5. Set **Average Temperature:** `31.5 deg C`
  6. Set **Relative Humidity:** `92.0 %`
* **Expected Result:**
  - **Risk Tier:** Escalates to `Critical` or `High` (Red/Orange alert badge).
  - **Expected Caseload:** Significant case spike projected.
  - **Outbreak Probability:** Exceeds 70-85%.
  - **Operational Advisory:** Advises emergency biological larviciding (*Bacillus thuringiensis israelensis* / *Bti*) and vehicle-mounted thermal fogging within 48 hours.
  - **Longitudinal Trend Chart:** Re-renders with the simulated projection point plotted at the future date.

---

### Test Case 3: Relative Humidity Physical Boundary Clamping (<= 100%)
* **Objective:** Ensure the portal strictly prevents physically impossible relative humidity values (>100% or <0%).
* **Navigation:** http://localhost:8000/#surveillance
* **Test Steps:**
  1. Click into the **Relative Humidity** input field.
  2. Type `150` or `120`.
* **Expected Result:**
  - The input field value immediately clamps to `100` in the browser.
  - The helper caption displays: `Physical maximum is 100%`.
  - The model calculates using exactly `100.0%`, avoiding mathematical distortion.
  3. Type `-20`.
  - The input immediately bounds up to `0.0%`.

---

### Test Case 4: Real-Time Debounced Reactive Recalculation
* **Objective:** Verify that administrators do not need to click a submit button after adjusting meteorological parameters.
* **Navigation:** http://localhost:8000/#surveillance
* **Test Steps:**
  1. Focus on the **Current Week Rain** input.
  2. Change the value from `20.0` to `80.0`.
  3. Observe the form without clicking any button.
* **Expected Result:**
  - **Dynamic Preview:** The hint below the input immediately updates: `Estimated 4W Accumulation: [Value] mm`.
  - **Debounce:** After 250 milliseconds, the system automatically triggers a forecast update.
  - The Risk Tier, Caseload, and Protocol text refresh automatically.

---

### Test Case 5: 1-Click Executive Scenario Presets
* **Objective:** Verify that municipal executives can instantaneously evaluate standard climate risk archetypes.
* **Navigation:** http://localhost:8000/#surveillance
* **Test Steps:**
  1. Click the **Monsoon Surge** chip:
     - Sets Rainfall: `110.0 mm`, Temperature: `31.5 deg C`, Humidity: `88.0%`.
     - Output: Critical Outbreak Alert with immediate vector suppression protocols.
  2. Click the **Pre-Monsoon** chip:
     - Sets Rainfall: `35.0 mm`, Temperature: `32.0 deg C`, Humidity: `75.0%`.
     - Output: Moderate/High Alert indicating early larval breeding window.
  3. Click the **Winter Baseline** chip:
     - Sets Rainfall: `0.0 mm`, Temperature: `22.0 deg C`, Humidity: `58.0%`.
     - Output: Low Baseline surveillance.

---

### Test Case 6: Ward Vulnerability Matrix Search & CSV Export
* **Objective:** Verify filtering of high-risk municipal wards and exporting field dispatch rosters for municipal fogging teams.
* **Navigation:** Open http://localhost:8000/#vulnerability
* **Test Steps:**
  1. In the **Filter Wards** search box, type `CCC_Ward_03`.
     - Expected: Table filters instantly to display only rows matching `CCC_Ward_03`.
  2. Clear the search box and type `Chattogram`.
     - Expected: Displays all wards in Chattogram.
  3. Click the **Export CSV** button.
* **Expected Result:**
  - Browser downloads a timestamped CSV file (e.g. `dengue_ward_vulnerability_2026-....csv`).
  - The CSV contains columns: `Ward ID`, `City`, `Peak Historical Cases`, `Mean Weekly Cases`, `Current Vulnerability Index`, `Recommended Intervention`.

---

### Test Case 7: Action Protocols Threshold Simulator
* **Objective:** Test the interactive epidemiological threshold simulation engine.
* **Navigation:** Open http://localhost:8000/#protocols
* **Test Steps:**
  1. In the **Interactive Protocol Simulator** form:
     - Set **Cumulative 4-Week Rain:** `300 mm`
     - Set **Average Temperature:** `29 deg C`
     - Set **Relative Humidity:** `85 %`
* **Expected Result:**
  - Banner updates dynamically to **CRITICAL OUTBREAK ALERT**.
  - Text highlights the **Triple-Threshold Trigger**: 4-week rainfall >= 250 mm combined with optimal vector replication conditions.
  - Specifies immediate mobilization of thermal fogging within 48 hours.

---

### Test Case 8: Mobile & Responsive Layout Audit
* **Objective:** Confirm operational accessibility on mobile devices used by field officers.
* **Test Steps:**
  1. Open DevTools (F12) and toggle Mobile Device mode (iPhone 12 / Galaxy S20).
  2. Verify:
     - Navigation tabs remain accessible with horizontal touch-swipe scrolling.
     - Controls column stacks cleanly above metric display cards without horizontal overflow.
     - Form input addons (`mm`, `deg C`, `%`) remain aligned with zero layout break.
     - Touch targets are >= 44px.

---

## Section 2: Automated Backend & API Test Cases

To run the automated test suite locally:

```bash
# Execute full test suite (25 tests)
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### Direct REST API Verification via PowerShell or cURL

#### API-01: Health & Telemetry Verification
```bash
curl http://127.0.0.1:8000/api/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "system": "Operation-Mosha Localhost Engine"
}
```

#### API-02: Prediction API with Clamped Humidity Test
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"city":"Chattogram","ward_id":"CCC_Ward_01","date":"2026-08-10","rainfall_mm":95.0,"avg_temp_c":30.0,"relative_humidity_pct":125.0}'
```
**Expected Verification:**
- Returns HTTP 200.
- JSON response includes `risk_tier`, `expected_cases`, `outbreak_prob`, and four lag components.
- The relative humidity is safely clamped to 100% on the server.

---

## Section 3: Summary of Design & Operational Compliance
- **Typography:** Inter (Weights 400 body, 700 headings).
- **Color Palette:**
  - Text: `rgb(247, 255, 229)` (`#F7FFE5`)
  - Background: `rgb(0, 41, 46)` (`#00292E`)
  - Primary / Accent: `rgb(223, 255, 82)` (`#DFFF52`)
  - Secondary: `rgb(19, 17, 17)` (`#131111`)
- **Zero Emojis:** Strictly 0 unicode emojis across all pages, charts, tables, and documentation.
- **Model Architecture:** Two-Stage Hurdle Model (HistGradientBoostingClassifier + HistGradientBoostingRegressor).
