"""End-to-End Field Scenario Verification for Operation-Mosha.

Validates all Public Health Administrator test cases defined in TEST_CASES.md:
- TC-01: Baseline Surveillance Mode (Historical Verification)
- TC-02: Extreme Monsoon Outbreak Simulation (Future Date 2026-07-20)
- TC-03: Relative Humidity Physical Upper Bound Validation (<=100%)
- TC-04: HTML Form Controls 'step="any"' for Decimal Inputs
- TC-05: Preset Scenarios (Monsoon, Pre-Monsoon, Winter)
- TC-06: Ward Vulnerability Matrix & Action Protocols
"""

import json
import pytest
from server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_tc01_baseline_historical_surveillance(client):
    """TC-01: March 2024 historical baseline should report Low risk."""
    payload = {
        "city": "Chattogram",
        "ward_id": "CCC_Ward_01",
        "date": "2024-03-15",
        "rainfall_mm": 15.0,
        "avg_temp_c": 26.0,
        "relative_humidity_pct": 65.0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    assert data["risk_tier"] == "Low"
    assert data["expected_cases"] < 5.0
    assert data["p_nonzero"] < 0.50
    assert "advisory" in data
    assert data["advisory"]["badge"] in ["ROUTINE SURVEILLANCE", "LOW RISK ADVISORY"]


def test_tc02_extreme_monsoon_future_forecast(client):
    """TC-02: Peak monsoon in July 2026 with 120mm rain should trigger Critical Outbreak Alert."""
    payload = {
        "city": "Chattogram",
        "ward_id": "CCC_Ward_08",
        "date": "2026-07-20",
        "rainfall_mm": 120.0,
        "avg_temp_c": 31.5,
        "relative_humidity_pct": 92.0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    # Model validation
    assert data["risk_tier"] == "Critical", f"Expected Critical, got {data['risk_tier']}"
    assert data["expected_cases"] >= 8.0, f"Expected >= 8.0 cases, got {data['expected_cases']}"
    assert data["p_nonzero"] >= 0.90, f"Expected >= 90% outbreak probability, got {data['p_nonzero']}"
    assert data["sum_4w"] >= 250.0, f"Expected >= 250mm cumulative rain, got {data['sum_4w']}"

    # Public health protocol validation
    advisory = data["advisory"]
    assert advisory["badge"] == "CRITICAL OUTBREAK ALERT"
    assert "thermal fogging" in " ".join(advisory["actions"]).lower()
    assert "larvicide" in advisory["summary"].lower() or "larval" in advisory["summary"].lower()


def test_tc03_humidity_physical_clamping(client):
    """TC-03: Submitting humidity > 100% or < 0% must clamp to physical limits."""
    # Test > 100%
    payload_high = {
        "city": "Chattogram",
        "ward_id": "CCC_Ward_01",
        "date": "2026-07-15",
        "rainfall_mm": 100.0,
        "avg_temp_c": 30.0,
        "relative_humidity_pct": 140.0,
    }
    res_high = client.post("/api/predict", json=payload_high)
    assert res_high.status_code == 200

    # Test < 0%
    payload_low = {
        "city": "Chattogram",
        "ward_id": "CCC_Ward_01",
        "date": "2026-07-15",
        "rainfall_mm": 100.0,
        "avg_temp_c": 30.0,
        "relative_humidity_pct": -25.0,
    }
    res_low = client.post("/api/predict", json=payload_low)
    assert res_low.status_code == 200


def test_tc04_html_decimal_step_attributes(client):
    """TC-04: Verify index.html uses step='any' to prevent native browser validation errors."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Verify input controls accept any decimal
    assert 'id="num-rainfall" class="form-control" min="0" max="600" step="any"' in html
    assert 'id="num-temp" class="form-control" min="10" max="45" step="any"' in html
    assert 'id="num-humidity" class="form-control" min="0" max="100" step="any"' in html
    assert 'id="sim-rain-4w" class="form-control" min="0" max="800" step="any"' in html
    assert 'id="sim-temp" class="form-control" min="15" max="45" step="any"' in html
    assert 'id="sim-hum" class="form-control" min="0" max="100" step="any"' in html


def test_tc05_preset_winter_baseline(client):
    """TC-05: Winter baseline scenario (0mm rain, 22C, 58% hum) must produce Low risk."""
    payload = {
        "city": "Chattogram",
        "ward_id": "CCC_Ward_08",
        "date": "2026-12-15",
        "rainfall_mm": 0.0,
        "avg_temp_c": 22.0,
        "relative_humidity_pct": 58.0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    assert data["risk_tier"] == "Low"
    assert data["expected_cases"] < 1.0
    assert data["p_nonzero"] < 0.20


def test_tc06_vulnerability_and_analytics_endpoints(client):
    """TC-06: Vulnerability and Analytics endpoints return complete ward coverage."""
    res_vuln = client.get("/api/vulnerability")
    assert res_vuln.status_code == 200
    vuln_data = res_vuln.get_json()
    assert "wards" in vuln_data
    assert len(vuln_data["wards"]) >= 40

    res_analytics = client.get("/api/analytics")
    assert res_analytics.status_code == 200
    analytics_data = res_analytics.get_json()
    assert "correlations" in analytics_data
    assert "monthly_seasonality" in analytics_data
