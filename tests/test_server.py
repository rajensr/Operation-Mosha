
"""Unit and integration tests for Operation-Mosha Flask server and REST APIs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_wards_endpoint(client):
    res = client.get("/api/wards")
    assert res.status_code == 200
    data = res.get_json()
    assert "cities" in data
    assert "wards" in data
    assert len(data["wards"]) == 60
    assert data["wards"][0]["ward_id"] == "CCC_Ward_01"


def test_history_endpoint(client):
    res = client.get("/api/history?ward_id=CCC_Ward_01")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ward_id"] == "CCC_Ward_01"
    assert len(data["records"]) == 522
    assert "rainfall_mm" in data["records"][0]
    assert "dengue_cases" in data["records"][0]


def test_predict_endpoint_historical(client):
    payload = {
        "city": "Chittagong",
        "ward_id": "CCC_Ward_01",
        "date": "2025-08-18",
        "rainfall_mm": 80.0,
        "avg_temp_c": 31.5,
        "relative_humidity_pct": 88.0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["ward_id"] == "CCC_Ward_01"
    assert data["risk_tier"] in ["Low", "Medium", "High", "Critical"]
    assert 0.0 <= data["p_nonzero"] <= 1.0
    assert data["expected_cases"] >= 0.0
    assert "advisory" in data
    assert "actions" in data["advisory"]
    assert len(data["advisory"]["actions"]) > 0


def test_predict_endpoint_future_date(client):
    payload = {
        "city": "Chittagong",
        "ward_id": "CCC_Ward_01",
        "date": "2026-08-17",
        "rainfall_mm": 120.0,
        "avg_temp_c": 32.0,
        "relative_humidity_pct": 90.0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["ward_id"] == "CCC_Ward_01"
    assert data["is_future"] is True
    assert data["risk_tier"] in ["Low", "Medium", "High", "Critical"]
    assert data["expected_cases"] >= 0.0


def test_vulnerability_endpoint(client):
    res = client.get("/api/vulnerability")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total_wards"] == 60
    assert len(data["wards"]) == 60
    assert data["wards"][0]["rank"] == 1
    assert data["wards"][0]["vulnerability_score"] >= data["wards"][-1]["vulnerability_score"]


def test_analytics_endpoint(client):
    res = client.get("/api/analytics")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["monthly_seasonality"]) == 12
    assert len(data["correlations"]) > 0
    # Top correlation should be 4-week rainfall sum
    assert data["correlations"][0]["feature"] == "rainfall_sum_4w"


def test_index_html_serving(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Urban Dengue Risk Intelligence" in html
    assert "Operation-Mosha" in html
