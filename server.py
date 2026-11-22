"""Flask server for Operation-Mosha Urban Dengue Risk Intelligence.

Exposes REST APIs for prediction, ward vulnerabilities, historical analytics,
and serves the Single-Page Application (SPA) frontend.
Zero external cloud or paid API dependencies.
"""

from __future__ import annotations

from datetime import datetime, date
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, jsonify, request, send_from_directory
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, create_features
from src.predict import load_model_bundle, score_rows

app = Flask(__name__, static_folder="web", static_url_path="")

DATA_PATH = Path("data/processed/cleaned.csv")
MODEL_PATH = Path("models/model_bundle.joblib")
VALIDATION_PATH = Path("reports/validation.json")

# In-memory cached dataset and model
_cached_df: pd.DataFrame | None = None
_cached_model: dict[str, Any] | None = None


def get_dataset() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {DATA_PATH}")
        _cached_df = pd.read_csv(DATA_PATH)
        _cached_df["date"] = pd.to_datetime(_cached_df["date"])
    return _cached_df


def get_model() -> dict[str, Any] | None:
    global _cached_model
    if _cached_model is None and MODEL_PATH.exists():
        _cached_model = load_model_bundle(MODEL_PATH)
    return _cached_model


# Frontend SPA Routes
@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/<path:path>")
def static_proxy(path: str):
    file_path = Path("web") / path
    if file_path.exists() and file_path.is_file():
        return send_from_directory("web", path)
    return send_from_directory("web", "index.html")


# REST API Endpoints
@app.route("/api/health", methods=["GET"])
def health_check():
    model_loaded = get_model() is not None
    return jsonify({
        "status": "healthy",
        "system": "Operation-Mosha Localhost Engine",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/wards", methods=["GET"])
def list_wards():
    df = get_dataset()
    latest_date = df["date"].max()
    latest_snapshot = df[df["date"] == latest_date]

    wards_list = []
    for _, row in latest_snapshot.sort_values("ward_id").iterrows():
        wards_list.append({
            "ward_id": str(row["ward_id"]),
            "city": str(row["city"]),
            "population_density_sqkm": float(row["population_density_sqkm"]),
            "water_logging_risk_index": round(float(row["water_logging_risk_index"]), 3),
            "sanitation_score": round(float(row["sanitation_score"]), 3),
            "latest_cases": int(row["dengue_cases"]),
            "latest_rainfall_mm": round(float(row["rainfall_mm"]), 1),
            "latest_temp_c": round(float(row["avg_temp_c"]), 1),
            "latest_humidity_pct": round(float(row["relative_humidity_pct"]), 1),
        })

    cities = sorted(df["city"].dropna().unique().tolist())
    earliest_date = df["date"].min().strftime("%Y-%m-%d")
    latest_date_str = latest_date.strftime("%Y-%m-%d")

    return jsonify({
        "cities": cities,
        "earliest_date": earliest_date,
        "latest_date": latest_date_str,
        "wards": wards_list,
    })


@app.route("/api/history", methods=["GET"])
def ward_history():
    ward_id = request.args.get("ward_id", "CCC_Ward_01")
    df = get_dataset()
    sub_df = df[df["ward_id"] == ward_id].sort_values("date").reset_index(drop=True)

    if sub_df.empty:
        return jsonify({"error": f"Ward '{ward_id}' not found"}), 404

    records = []
    for _, r in sub_df.iterrows():
        records.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "rainfall_mm": round(float(r["rainfall_mm"]), 1),
            "avg_temp_c": round(float(r["avg_temp_c"]), 1),
            "relative_humidity_pct": round(float(r["relative_humidity_pct"]), 1),
            "dengue_cases": int(r["dengue_cases"]),
        })

    return jsonify({
        "ward_id": ward_id,
        "city": str(sub_df.iloc[0]["city"]),
        "records": records,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    ward_id = data.get("ward_id", "CCC_Ward_01")
    city = data.get("city", "Chittagong")
    date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))

    try:
        target_dt = pd.to_datetime(date_str)
    except Exception:
        target_dt = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))

    sim_rainfall = max(float(data.get("rainfall_mm", 20.0)), 0.0)
    sim_avg_temp = float(data.get("avg_temp_c", 28.0))
    sim_humidity = min(max(float(data.get("relative_humidity_pct", 75.0)), 0.0), 100.0)

    df = get_dataset()
    ward_df = df[df["ward_id"] == ward_id].sort_values("date").reset_index(drop=True)
    if ward_df.empty:
        return jsonify({"error": f"Ward '{ward_id}' not found"}), 404

    latest_hist_dt = ward_df["date"].max()
    is_future = target_dt > latest_hist_dt

    ref_row = ward_df.iloc[-1]
    if is_future:
        # For future dates beyond historical records, synthesize preceding 4 weeks using
        # ward-specific seasonal climatology scaled by simulated weather anomalies.
        # This preserves biological lag attribution (+0.8066 correlation with 4-week accumulation).
        hist_df = ward_df.copy()
        hist_df["week"] = hist_df["date"].dt.isocalendar().week
        climatology = hist_df.groupby("week")[["rainfall_mm", "avg_temp_c", "relative_humidity_pct"]].mean()

        target_week = target_dt.isocalendar().week
        norm_rain_target = float(climatology.loc[target_week, "rainfall_mm"]) if target_week in climatology.index else 20.0
        rain_scale_ratio = sim_rainfall / max(norm_rain_target, 10.0)

        synth_rows = []
        for k in range(4, 0, -1):
            prev_dt = target_dt - pd.Timedelta(weeks=k)
            pwk = prev_dt.isocalendar().week
            base_rain = float(climatology.loc[pwk, "rainfall_mm"]) if pwk in climatology.index else sim_rainfall
            base_temp = float(climatology.loc[pwk, "avg_temp_c"]) if pwk in climatology.index else sim_avg_temp
            base_hum = float(climatology.loc[pwk, "relative_humidity_pct"]) if pwk in climatology.index else sim_humidity

            synth_row = ref_row.to_dict()
            synth_row["date"] = prev_dt
            synth_row["rainfall_mm"] = max(base_rain * rain_scale_ratio, 0.0)
            synth_row["avg_temp_c"] = base_temp
            synth_row["max_temp_c"] = base_temp + 3.5
            synth_row["min_temp_c"] = base_temp - 3.5
            synth_row["relative_humidity_pct"] = min(max(base_hum, 0.0), 100.0)
            synth_row["dengue_cases"] = 0
            synth_rows.append(synth_row)

        future_row = ref_row.to_dict()
        future_row["date"] = target_dt
        future_row["city"] = city
        future_row["ward_id"] = ward_id
        future_row["avg_temp_c"] = sim_avg_temp
        future_row["max_temp_c"] = sim_avg_temp + 3.5
        future_row["min_temp_c"] = sim_avg_temp - 3.5
        future_row["rainfall_mm"] = sim_rainfall
        future_row["relative_humidity_pct"] = sim_humidity
        future_row["dengue_cases"] = 0
        synth_rows.append(future_row)

        score_df = pd.concat([ward_df, pd.DataFrame(synth_rows)], ignore_index=True)
    else:
        score_df = ward_df[ward_df["date"] <= target_dt].copy()
        if score_df.empty:
            score_df = ward_df.iloc[[0]].copy()
        last_idx = score_df.index[-1]
        orig_rain = float(score_df.loc[last_idx, "rainfall_mm"])
        score_df.loc[last_idx, "rainfall_mm"] = sim_rainfall
        score_df.loc[last_idx, "avg_temp_c"] = sim_avg_temp
        score_df.loc[last_idx, "relative_humidity_pct"] = sim_humidity

        # If user is simulating a custom weather shock on an observation week,
        # propagate the rainfall anomaly into the preceding 4-week window so biological
        # accumulation features (correlation +0.8066) respond dynamically to simulated shocks.
        if abs(sim_rainfall - orig_rain) > 5.0 and len(score_df) > 1:
            scale_factor = sim_rainfall / max(orig_rain, 15.0)
            for k in range(1, min(5, len(score_df))):
                prev_pos = len(score_df) - 1 - k
                prev_val = float(score_df.iloc[prev_pos]["rainfall_mm"])
                score_df.iloc[prev_pos, score_df.columns.get_loc("rainfall_mm")] = max(prev_val * scale_factor, 0.0)

    # Compute multi-week lags
    n_rows = len(score_df)
    lag1 = float(score_df.loc[n_rows - 2, "rainfall_mm"]) if n_rows >= 2 else sim_rainfall
    lag2 = float(score_df.loc[n_rows - 3, "rainfall_mm"]) if n_rows >= 3 else sim_rainfall
    lag3 = float(score_df.loc[n_rows - 4, "rainfall_mm"]) if n_rows >= 4 else sim_rainfall
    lag4 = float(score_df.loc[n_rows - 5, "rainfall_mm"]) if n_rows >= 5 else sim_rainfall
    sum_4w = lag1 + lag2 + lag3 + lag4

    model = get_model()
    if model is not None:
        scored = score_rows(score_df, model_bundle=model)
        active = scored.iloc[-1]
        p_nonzero = float(active["p_nonzero"])
        expected_cases = float(active["expected_cases"])
        risk_tier = str(active["risk_tier"])
    else:
        p_nonzero = 0.0
        expected_cases = 0.0
        risk_tier = "Low"

    # Color map
    color_map = {
        "Low": "#28a745",
        "Medium": "#ffc107",
        "High": "#fd7e14",
        "Critical": "#dc3545",
    }
    risk_color = color_map.get(risk_tier, "#28a745")

    # Public Health Administrator Action Advisory
    if risk_tier == "Critical":
        advisory = {
            "badge": "CRITICAL OUTBREAK ALERT",
            "title": "Imminent Transmission Surge — Emergency Field Mobilization Required",
            "lead_time": "3 to 4 Week Biological Lag Window Active",
            "summary": f"Cumulative 4-week rainfall ({sum_4w:.1f} mm) and thermal suitability ({sim_avg_temp:.1f} deg C) have triggered extensive mosquito breeding. Because the Aedes aquatic cycle and viral extrinsic incubation period require 3 to 4 weeks, hospital admissions will peak rapidly unless larval and adult populations are collapsed immediately.",
            "actions": [
                f"Deploy vehicle-mounted thermal fogging within 48 hours across high-risk sectors of {ward_id}.",
                "Mobilize ward health inspection teams for rooftop container and construction pit clearance.",
                "Issue urgent advisory to municipal hospitals to prepare IV fluid supplies and dengue triage capacity.",
            ],
        }
    elif risk_tier == "High":
        advisory = {
            "badge": "HIGH RISK ADVISORY",
            "title": "Pre-Outbreak Vector Amplification — 2 to 3 Week Larvicide Window Active",
            "lead_time": "2 to 3 Week Larviciding Window Active",
            "summary": f"Sustained moisture ({sum_4w:.1f} mm 4-week rainfall, {sim_humidity:.0f}% humidity) has created extensive aquatic breeding habitats. Public health teams have a critical 2-to-3 week intervention window to eliminate larvae before adult biting vectors mature.",
            "actions": [
                "Treat stormwater drains and low-lying waterlogged depressions with Bacillus thuringiensis israelensis (Bti).",
                f"Coordinate with municipal engineering crews to clear silted culverts in {ward_id} (Drainage index: {ref_row['water_logging_risk_index']:.2f}).",
                "Issue ward SMS alerts urging residents to empty stagnant water in domestic containers.",
            ],
        }
    elif risk_tier == "Medium":
        advisory = {
            "badge": "ELEVATED RISK WATCH",
            "title": "Microclimate Priming Detected — Intensify Habitat Surveillance",
            "lead_time": "Surveillance & Early Source Reduction",
            "summary": f"Moderate rainfall accumulation ({sum_4w:.1f} mm over 4 weeks). Weather indicators are becoming favorable for mosquito reproduction.",
            "actions": [
                "Inspect known microclimate hotspots (tire shops, empty construction plots).",
                "Maintain accelerated solid waste clearance to remove discarded plastic cups and containers.",
                "Ensure community clinics report suspected fever cases within 24 hours.",
            ],
        }
    else:
        advisory = {
            "badge": "ROUTINE SURVEILLANCE",
            "title": "Baseline Vector Suppression — Low Outbreak Potential",
            "lead_time": "Normal Baseline Monitoring",
            "summary": "Current meteorological indicators (low rainfall or non-conducive temperatures) naturally suppress mosquito proliferation and viral replication.",
            "actions": [
                "Maintain standard municipal conservancy schedules and waste management.",
                "Conduct routine ovitrap entomological surveillance in high-density zones.",
                "Continue ongoing public education regarding clean water storage.",
            ],
        }

    return jsonify({
        "ward_id": ward_id,
        "city": city,
        "date": target_dt.strftime("%Y-%m-%d"),
        "is_future": is_future,
        "p_nonzero": round(p_nonzero, 4),
        "expected_cases": round(expected_cases, 2),
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "vulnerability_index": round(float(ref_row["water_logging_risk_index"]), 3),
        "population_density": int(ref_row["population_density_sqkm"]),
        "sanitation_score": round(float(ref_row["sanitation_score"]), 3),
        "lag1": round(lag1, 1),
        "lag2": round(lag2, 1),
        "lag3": round(lag3, 1),
        "lag4": round(lag4, 1),
        "sum_4w": round(sum_4w, 1),
        "advisory": advisory,
    })


@app.route("/api/vulnerability", methods=["GET"])
def vulnerability_matrix():
    df = get_dataset()
    latest_date = df["date"].max()
    snapshot = df[df["date"] == latest_date].copy()

    # Compute composite vulnerability score
    max_pop = snapshot["population_density_sqkm"].max()
    snapshot["vulnerability_score"] = (
        (snapshot["water_logging_risk_index"] * 0.45)
        + ((snapshot["population_density_sqkm"] / max_pop) * 0.35)
        + ((1.0 - snapshot["sanitation_score"]) * 0.20)
    )

    ranked = snapshot.sort_values("vulnerability_score", ascending=False).reset_index(drop=True)

    items = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        items.append({
            "rank": rank,
            "ward_id": str(row["ward_id"]),
            "city": str(row["city"]),
            "vulnerability_score": round(float(row["vulnerability_score"]), 3),
            "water_logging_risk_index": round(float(row["water_logging_risk_index"]), 3),
            "sanitation_score": round(float(row["sanitation_score"]), 3),
            "population_density_sqkm": int(row["population_density_sqkm"]),
            "latest_cases": int(row["dengue_cases"]),
        })

    return jsonify({
        "date": latest_date.strftime("%Y-%m-%d"),
        "total_wards": len(items),
        "high_vulnerability_count": len([w for w in items if w["vulnerability_score"] >= 0.6]),
        "city_avg_waterlog": round(float(snapshot["water_logging_risk_index"].mean()), 3),
        "city_avg_sanitation": round(float(snapshot["sanitation_score"].mean()), 3),
        "wards": items,
    })


@app.route("/api/analytics", methods=["GET"])
def analytics():
    df = get_dataset()
    feat = create_features(df, is_training=True)
    feat["date"] = pd.to_datetime(feat["date"])
    feat["month"] = feat["date"].dt.month

    # 1. Monthly Seasonality
    monthly = feat.groupby("month")[["dengue_cases", "rainfall_mm", "avg_temp_c", "relative_humidity_pct"]].mean().reset_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_data = []
    for _, r in monthly.iterrows():
        m_idx = int(r["month"])
        monthly_data.append({
            "month_num": m_idx,
            "month_name": month_names[m_idx - 1],
            "dengue_cases": round(float(r["dengue_cases"]), 2),
            "rainfall_mm": round(float(r["rainfall_mm"]), 1),
            "avg_temp_c": round(float(r["avg_temp_c"]), 1),
            "relative_humidity_pct": round(float(r["relative_humidity_pct"]), 1),
        })

    # 2. Pearson Correlations
    corr_cols = [
        "rainfall_sum_4w",
        "rainfall_lag_3",
        "rainfall_sum_2w",
        "rainfall_lag_4",
        "rainfall_lag_2",
        "rainfall_lag_1",
        "rainfall_mm",
        "relative_humidity_pct",
        "avg_temp_c",
        "water_logging_risk_index",
        "population_density_sqkm",
        "sanitation_score",
    ]
    corrs = feat[corr_cols + ["dengue_cases"]].corr()["dengue_cases"].drop("dengue_cases")
    corr_data = []
    for feat_name, c_val in corrs.sort_values(ascending=False).items():
        corr_data.append({
            "feature": str(feat_name),
            "correlation": round(float(c_val), 4),
        })

    return jsonify({
        "monthly_seasonality": monthly_data,
        "correlations": corr_data,
    })


@app.route("/api/telemetry", methods=["GET"])
def telemetry():
    if VALIDATION_PATH.exists():
        try:
            return jsonify(json.loads(VALIDATION_PATH.read_text(encoding="utf-8")))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Validation telemetry report not found"}), 404


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Operation-Mosha Flask Engine")
    print("Public Health Decision Support Portal")
    print("Available at: http://localhost:8000")
    print("Zero Emojis Enforced | Inter Typography | Responsive SPA")
    print("=" * 60)
    app.run(host="0.0.0.0", port=8000, debug=False)
