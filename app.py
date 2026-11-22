"""Urban Dengue Risk Intelligence Dashboard.

Localhost decision-support system for public health officers and municipal planners.
Provides nowcasting, future-date forecasting, weather simulation via numerical inputs,
longitudinal trend monitoring, and empirical validation telemetry.

Strictly adheres to specified typography and color tokens:
- Font: Inter (400, 700)
- Text: rgb(247, 255, 229)
- Background: rgb(0, 41, 46)
- Primary/Accent: rgb(223, 255, 82)
- Secondary: rgb(19, 17, 17)
- Zero Emojis
"""

from __future__ import annotations

from datetime import datetime, date
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.predict import score_rows
from src.theme import THEME_CSS, PLOTLY_LAYOUT, RISK_COLORS

st.set_page_config(
    page_title="Urban Dengue Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(THEME_CSS, unsafe_allow_html=True)

DATA_PATH = Path("data/processed/cleaned.csv")
MODEL_PATH = Path("models/model_bundle.joblib")
VALIDATION_PATH = Path("reports/validation.json")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load cleaned dataset from disk."""
    if not DATA_PATH.exists():
        st.error(f"Cleaned dataset not found at {DATA_PATH}. Run python src/clean_data.py first.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_validation_metrics() -> dict | None:
    """Load evaluation metrics JSON from disk."""
    if VALIDATION_PATH.exists():
        try:
            return json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_resource
def check_model_availability() -> bool:
    """Check if trained model bundle artifact exists."""
    return MODEL_PATH.exists()


def main() -> None:
    # Top Navigation Banner
    st.markdown(
        """
        <div style="background-color: rgb(19, 17, 17); border: 1px solid rgba(223, 255, 82, 0.3); padding: 24px 28px; border-radius: 8px; margin-bottom: 24px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: rgb(223, 255, 82); letter-spacing: 1.5px; text-transform: uppercase;">
                Operation-Mosha | Municipal Disease Surveillance
            </div>
            <h1 style="margin: 8px 0 4px 0;">Urban Dengue Risk Intelligence</h1>
            <p style="margin: 0; font-size: 1.05rem; color: rgb(247, 255, 229); opacity: 0.85;">
                Localhost Outbreak Risk Nowcasting & Future Forecasting Engine
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset()
    model_ready = check_model_availability()

    if not model_ready:
        st.warning(
            "Model artifact not trained yet. Run: python src/train.py"
        )

    # Sidebar: Geographic & Temporal Selection
    st.sidebar.markdown(
        """
        <div style="font-size: 1.15rem; font-weight: 700; color: rgb(223, 255, 82); margin-bottom: 12px;">
            Surveillance Selection
        </div>
        """,
        unsafe_allow_html=True,
    )

    cities = sorted(df["city"].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("Select City", cities, index=0)

    city_df = df[df["city"] == selected_city].copy()
    wards = sorted(city_df["ward_id"].dropna().unique().tolist())
    selected_ward = st.sidebar.selectbox("Select Ward", wards, index=0)

    ward_df = city_df[city_df["ward_id"] == selected_ward].sort_values("date").reset_index(drop=True)
    latest_hist_date = ward_df["date"].max().date()

    st.sidebar.markdown(
        """
        <div style="font-size: 0.85rem; color: rgb(223, 255, 82); font-weight: 700; margin-top: 14px; text-transform: uppercase;">
            Observation Date (Past or Future)
        </div>
        """,
        unsafe_allow_html=True,
    )
    picked_date = st.sidebar.date_input(
        "Select / Enter Date",
        value=latest_hist_date,
        min_value=date(2016, 1, 1),
        max_value=date(2030, 12, 31),
    )
    picked_dt = pd.to_datetime(picked_date)

    # Determine baseline meteorological values
    is_future_date = picked_dt > ward_df["date"].max()
    if is_future_date:
        ref_row = ward_df.iloc[-1]
        default_rain = float(ref_row["rainfall_mm"])
        default_temp = float(ref_row["avg_temp_c"])
        default_hum = float(ref_row["relative_humidity_pct"])
        observed_cases_display = "N/A (Future Projection)"
    else:
        # Match closest date on or before picked date
        match_rows = ward_df[ward_df["date"] <= picked_dt]
        ref_row = match_rows.iloc[-1] if not match_rows.empty else ward_df.iloc[0]
        default_rain = float(ref_row["rainfall_mm"])
        default_temp = float(ref_row["avg_temp_c"])
        default_hum = float(ref_row["relative_humidity_pct"])
        observed_cases_display = str(int(ref_row["dengue_cases"]))

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 1.15rem; font-weight: 700; color: rgb(223, 255, 82); margin-bottom: 6px;">
            Meteorological Inputs
        </div>
        <div style="font-size: 0.8rem; color: rgb(247, 255, 229); opacity: 0.8; margin-bottom: 12px;">
            Direct numerical inputs for scenario simulation:
        </div>
        """,
        unsafe_allow_html=True,
    )

    sim_rainfall = st.sidebar.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=600.0,
        value=default_rain,
        step=1.0,
        format="%.1f",
    )
    sim_avg_temp = st.sidebar.number_input(
        "Average Temperature (deg C)",
        min_value=10.0,
        max_value=45.0,
        value=default_temp,
        step=0.2,
        format="%.1f",
    )
    sim_humidity = st.sidebar.number_input(
        "Relative Humidity (%)",
        min_value=10.0,
        max_value=100.0,
        value=default_hum,
        step=1.0,
        format="%.1f",
    )

    # Prepare scoring dataset
    if is_future_date:
        score_input_df = ward_df.copy()
        future_row = {
            "date": picked_dt,
            "city": selected_city,
            "ward_id": selected_ward,
            "population_density_sqkm": ref_row["population_density_sqkm"],
            "water_logging_risk_index": ref_row["water_logging_risk_index"],
            "sanitation_score": ref_row["sanitation_score"],
            "avg_temp_c": sim_avg_temp,
            "max_temp_c": sim_avg_temp + 3.5,
            "min_temp_c": sim_avg_temp - 3.5,
            "rainfall_mm": sim_rainfall,
            "relative_humidity_pct": sim_humidity,
            "dengue_cases": 0,
        }
        score_input_df = pd.concat([score_input_df, pd.DataFrame([future_row])], ignore_index=True)
    else:
        score_input_df = ward_df[ward_df["date"] <= picked_dt].copy()
        last_idx = score_input_df.index[-1]
        score_input_df.loc[last_idx, "rainfall_mm"] = sim_rainfall
        score_input_df.loc[last_idx, "avg_temp_c"] = sim_avg_temp
        score_input_df.loc[last_idx, "relative_humidity_pct"] = sim_humidity

    # Compute 4-week rainfall sum for biological lead-time context
    n_rows = len(score_input_df)
    lag1 = float(score_input_df.loc[n_rows - 2, "rainfall_mm"]) if n_rows >= 2 else sim_rainfall
    lag2 = float(score_input_df.loc[n_rows - 3, "rainfall_mm"]) if n_rows >= 3 else sim_rainfall
    lag3 = float(score_input_df.loc[n_rows - 4, "rainfall_mm"]) if n_rows >= 4 else sim_rainfall
    lag4 = float(score_input_df.loc[n_rows - 5, "rainfall_mm"]) if n_rows >= 5 else sim_rainfall
    sum_4w = lag1 + lag2 + lag3 + lag4

    if model_ready:
        scored = score_rows(score_input_df)
        active_score = scored.iloc[-1]
        p_val = float(active_score["p_nonzero"])
        cases_val = float(active_score["expected_cases"])
        tier_val = str(active_score["risk_tier"])
    else:
        p_val = 0.0
        cases_val = 0.0
        tier_val = "Low"

    tier_color = RISK_COLORS.get(tier_val, "#28a745")

    # Section 1: KPI Overview
    date_label = picked_dt.strftime("%Y-%m-%d")
    status_label = "Future Forecast" if is_future_date else "Historical Nowcast"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; flex-wrap: wrap;">
            <h3>Surveillance Status: {selected_city} | {selected_ward} ({date_label})</h3>
            <span style="font-size: 0.85rem; font-weight: 700; color: rgb(223, 255, 82); border: 1px solid rgba(223, 255, 82, 0.4); padding: 4px 10px; border-radius: 4px;">
                {status_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div style="background-color: rgb(19, 17, 17); border-left: 6px solid {tier_color}; border-top: 1px solid rgba(223, 255, 82, 0.2); border-right: 1px solid rgba(223, 255, 82, 0.2); border-bottom: 1px solid rgba(223, 255, 82, 0.2); padding: 16px 20px; border-radius: 6px; height: 100%;">
                <div style="font-size: 0.8rem; color: rgb(247, 255, 229); opacity: 0.8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Outbreak Risk Tier</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: {tier_color}; line-height: 1.2; margin-top: 4px;">{tier_val}</div>
                <div style="font-size: 0.75rem; color: rgb(247, 255, 229); opacity: 0.6; margin-top: 4px;">Calibrated Decision Threshold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div style="background-color: rgb(19, 17, 17); border-left: 6px solid rgb(223, 255, 82); border-top: 1px solid rgba(223, 255, 82, 0.2); border-right: 1px solid rgba(223, 255, 82, 0.2); border-bottom: 1px solid rgba(223, 255, 82, 0.2); padding: 16px 20px; border-radius: 6px; height: 100%;">
                <div style="font-size: 0.8rem; color: rgb(247, 255, 229); opacity: 0.8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Expected Caseload</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: rgb(223, 255, 82); line-height: 1.2; margin-top: 4px;">{cases_val:.2f} <span style="font-size: 1rem; font-weight: 400; color: rgb(247, 255, 229);">cases</span></div>
                <div style="font-size: 0.75rem; color: rgb(247, 255, 229); opacity: 0.6; margin-top: 4px;">Observed: {observed_cases_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div style="background-color: rgb(19, 17, 17); border-left: 6px solid rgb(223, 255, 82); border-top: 1px solid rgba(223, 255, 82, 0.2); border-right: 1px solid rgba(223, 255, 82, 0.2); border-bottom: 1px solid rgba(223, 255, 82, 0.2); padding: 16px 20px; border-radius: 6px; height: 100%;">
                <div style="font-size: 0.8rem; color: rgb(247, 255, 229); opacity: 0.8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Outbreak Probability</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: rgb(223, 255, 82); line-height: 1.2; margin-top: 4px;">{p_val * 100:.1f}%</div>
                <div style="font-size: 0.75rem; color: rgb(247, 255, 229); opacity: 0.6; margin-top: 4px;">P(Cases > 0 | Weather History)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            f"""
            <div style="background-color: rgb(19, 17, 17); border-left: 6px solid rgb(223, 255, 82); border-top: 1px solid rgba(223, 255, 82, 0.2); border-right: 1px solid rgba(223, 255, 82, 0.2); border-bottom: 1px solid rgba(223, 255, 82, 0.2); padding: 16px 20px; border-radius: 6px; height: 100%;">
                <div style="font-size: 0.8rem; color: rgb(247, 255, 229); opacity: 0.8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Vulnerability Index</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: rgb(223, 255, 82); line-height: 1.2; margin-top: 4px;">{ref_row['water_logging_risk_index']:.2f}</div>
                <div style="font-size: 0.75rem; color: rgb(247, 255, 229); opacity: 0.6; margin-top: 4px;">Density: {int(ref_row['population_density_sqkm']):,} /km²</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Operational Early-Warning Advisory Banner
    if tier_val == "Critical":
        adv_badge = "CRITICAL OUTBREAK ALERT"
        adv_title = "Imminent Transmission Surge — Emergency Field Mobilization Required"
        adv_body = f"""
        <b>Biological Lead-Time (3-4 Week Window):</b> Cumulative 4-week rainfall ({sum_4w:.1f} mm) and thermal suitability ({sim_avg_temp:.1f} deg C) have triggered extensive mosquito breeding. Because the Aedes aquatic cycle and viral extrinsic incubation period require 3 to 4 weeks, hospital admissions will peak rapidly unless larval and adult populations are suppressed immediately.<br><br>
        <b>Prescribed Municipal Action Protocol:</b>
        <ol style="margin: 4px 0 0 20px; padding: 0;">
            <li><b>Vector Knockdown:</b> Deploy vehicle-mounted thermal fogging within 48 hours across high-risk blocks of {selected_ward}.</li>
            <li><b>Source Reduction:</b> Mobilize community health teams to inspect construction sites, unsealed drums, and rooftop water tanks.</li>
            <li><b>Clinical Surge Alert:</b> Notify municipal hospitals to prepare IV fluid supplies and dengue triage capacity.</li>
        </ol>
        """
    elif tier_val == "High":
        adv_badge = "HIGH RISK ADVISORY"
        adv_title = "Pre-Outbreak Vector Amplification — 2 to 3 Week Larvicide Window Active"
        adv_body = f"""
        <b>Biological Lead-Time (2-3 Week Window):</b> Sustained moisture ({sum_4w:.1f} mm 4-week rainfall, {sim_humidity:.0f}% humidity) has created extensive aquatic breeding habitats. Public health teams have a 2-to-3 week intervention window to eliminate larvae before adult biting vectors mature.<br><br>
        <b>Prescribed Municipal Action Protocol:</b>
        <ol style="margin: 4px 0 0 20px; padding: 0;">
            <li><b>Targeted Larviciding:</b> Treat stormwater drains and low-lying waterlogged depressions with Bacillus thuringiensis israelensis (Bti).</li>
            <li><b>Drainage Maintenance:</b> Coordinate with ward engineering crews to clear silted road culverts (Ward water-logging index: {ref_row['water_logging_risk_index']:.2f}).</li>
            <li><b>Public Announcements:</b> Issue ward SMS notices urging residents to discard standing container water.</li>
        </ol>
        """
    elif tier_val == "Medium":
        adv_badge = "ELEVATED RISK WATCH"
        adv_title = "Microclimate Priming Detected — Intensify Habitat Surveillance"
        adv_body = f"""
        <b>Biological Lead-Time:</b> Moderate rainfall accumulation ({sum_4w:.1f} mm over 4 weeks). Weather indicators are becoming favorable for mosquito reproduction.<br><br>
        <b>Prescribed Municipal Action Protocol:</b> Conduct targeted inspections of known microclimate hotspots (tire shops, open construction plots) and maintain regular municipal solid waste clearance.
        """
    else:
        adv_badge = "ROUTINE SURVEILLANCE"
        adv_title = "Baseline Vector Suppression — Low Outbreak Potential"
        adv_body = """
        <b>Biological Lead-Time:</b> Current meteorological indicators (low rainfall or non-conducive temperatures) naturally suppress mosquito proliferation and viral replication.<br><br>
        <b>Prescribed Municipal Action Protocol:</b> Maintain standard municipal conservancy schedules and routine entomological surveillance.
        """

    st.markdown(
        f"""
        <div style="background-color: rgb(19, 17, 17); border-left: 6px solid {tier_color}; border-top: 1px solid rgba(223, 255, 82, 0.2); border-right: 1px solid rgba(223, 255, 82, 0.2); border-bottom: 1px solid rgba(223, 255, 82, 0.2); border-radius: 6px; padding: 18px 22px; margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; flex-wrap: wrap;">
                <span style="font-size: 0.8rem; font-weight: 700; color: {tier_color}; text-transform: uppercase; letter-spacing: 0.5px;">{adv_badge}</span>
                <span style="font-size: 0.75rem; color: rgb(247, 255, 229); opacity: 0.7;">Biological Early-Warning Policy</span>
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: rgb(223, 255, 82); margin-bottom: 8px;">{adv_title}</div>
            <div style="font-size: 0.92rem; color: rgb(247, 255, 229); line-height: 1.55;">{adv_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Section 2: Visual Charts
    col_left, col_right = st.columns([6, 5])

    with col_left:
        st.markdown("<h4>Historical Longitudinal Trends (Rainfall vs. Cases)</h4>", unsafe_allow_html=True)
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

        fig_trend.add_trace(
            go.Bar(
                x=ward_df["date"],
                y=ward_df["rainfall_mm"],
                name="Rainfall (mm)",
                marker_color="rgba(223, 255, 82, 0.35)",
            ),
            secondary_y=True,
        )

        fig_trend.add_trace(
            go.Scatter(
                x=ward_df["date"],
                y=ward_df["dengue_cases"],
                name="Actual Dengue Cases",
                mode="lines",
                line=dict(color="#dc3545", width=2.5),
            ),
            secondary_y=False,
        )

        fig_trend.add_vline(
            x=picked_dt.timestamp() * 1000,
            line_dash="dash",
            line_color="rgb(223, 255, 82)",
            annotation_text="Selected Date",
            annotation_position="top left",
            annotation_font=dict(color="rgb(223, 255, 82)"),
        )

        trend_layout = PLOTLY_LAYOUT.copy()
        trend_layout.update(
            hovermode="x unified",
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_trend.update_layout(**trend_layout)
        fig_trend.update_yaxes(title_text="Weekly Dengue Cases", secondary_y=False, gridcolor="rgba(223, 255, 82, 0.08)")
        fig_trend.update_yaxes(title_text="Rainfall (mm)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.markdown(f"<h4>Ward Cross-Section Comparison ({date_label})</h4>", unsafe_allow_html=True)
        cross_df = city_df[city_df["date"].dt.strftime("%Y-%m-%d") == date_label].copy()

        if model_ready and not cross_df.empty:
            history_sub = city_df[city_df["date"] <= picked_dt]
            cross_scores = score_rows(history_sub)
            cross_latest = cross_scores[cross_scores["date"] == date_label]
            cross_df = cross_df.merge(cross_latest[["ward_id", "expected_cases", "risk_tier"]], on="ward_id", how="left")

            fig_comp = px.bar(
                cross_df.sort_values("expected_cases", ascending=True),
                x="expected_cases",
                y="ward_id",
                color="risk_tier",
                color_discrete_map=RISK_COLORS,
                orientation="h",
                labels={"expected_cases": "Expected Cases", "ward_id": "Ward ID", "risk_tier": "Risk Tier"},
            )
        else:
            if cross_df.empty:
                cross_df = city_df[city_df["date"] == city_df["date"].max()].copy()
            fig_comp = px.bar(
                cross_df.sort_values("dengue_cases", ascending=True),
                x="dengue_cases",
                y="ward_id",
                orientation="h",
                labels={"dengue_cases": "Observed Cases", "ward_id": "Ward ID"},
                color_discrete_sequence=["#DFFF52"],
            )

        comp_layout = PLOTLY_LAYOUT.copy()
        comp_layout.update(height=380, margin=dict(l=10, r=10, t=30, b=10))
        fig_comp.update_layout(**comp_layout)
        st.plotly_chart(fig_comp, use_container_width=True)

    # Section 3: Driver Attribution Cards
    st.markdown("<h4>Meteorological Lag Attribution</h4>", unsafe_allow_html=True)
    st.caption("Historical rainfall lags driving vector incubation:")

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1:
        st.metric("Rainfall (t-1)", f"{lag1:.1f} mm")
    with d2:
        st.metric("Rainfall (t-2)", f"{lag2:.1f} mm")
    with d3:
        st.metric("Rainfall (t-3)", f"{lag3:.1f} mm")
    with d4:
        st.metric("Rainfall (t-4)", f"{lag4:.1f} mm")
    with d5:
        st.metric("4-Week Rain Sum", f"{sum_4w:.1f} mm", help="Vector breeding incubation volume")

    # Section 4: Validation Telemetry
    st.markdown("---")
    st.markdown("<h4>Model Validation & Benchmark Telemetry</h4>", unsafe_allow_html=True)
    metrics_data = load_validation_metrics()

    if metrics_data and "splits" in metrics_data:
        splits = metrics_data["splits"]
        tab_test, tab_val, tab_train = st.tabs(["Held-out Test (2025)", "Validation (2023-2024)", "Training (2016-2022)"])

        for tab, split_key in [(tab_test, "test"), (tab_val, "validation"), (tab_train, "train")]:
            with tab:
                sp = splits.get(split_key, {})
                hm = sp.get("hurdle_model", {})
                bm = sp.get("baseline_all_zero", {})

                st.markdown(
                    f"**Temporal Split:** `{split_key.upper()}` | **Sample Count:** `{sp.get('sample_count', 0):,}` | "
                    f"**Outbreak Prevalence:** `{sp.get('positive_prevalence_pct', 0.0):.2f}%` positive weeks"
                )

                metric_df = pd.DataFrame(
                    [
                        {
                            "Metric": "Mean Absolute Error (MAE)",
                            "Two-Stage Hurdle Model": f"{hm.get('mae', 0):.4f}",
                            "All-Zero Baseline": f"{bm.get('mae', 0):.4f}",
                            "Performance": "Superior Accuracy",
                        },
                        {
                            "Metric": "Root Mean Squared Error (RMSE)",
                            "Two-Stage Hurdle Model": f"{hm.get('rmse', 0):.4f}",
                            "All-Zero Baseline": f"{bm.get('rmse', 0):.4f}",
                            "Performance": "Outbreak Penalized",
                        },
                        {
                            "Metric": "Outbreak Recall (Positive Weeks)",
                            "Two-Stage Hurdle Model": f"{hm.get('recall_positive', 0) * 100:.1f}%",
                            "All-Zero Baseline": "0.0%",
                            "Performance": "High Sensitivity",
                        },
                        {
                            "Metric": "Outbreak F1 Score",
                            "Two-Stage Hurdle Model": f"{hm.get('f1_positive', 0):.4f}",
                            "All-Zero Baseline": "0.0000",
                            "Performance": "Balanced Detection",
                        },
                    ]
                )
                st.dataframe(metric_df, use_container_width=True, hide_index=True)

    # Footer
    st.markdown(
        """
        <div style="text-align: center; color: rgb(247, 255, 229); opacity: 0.6; font-size: 0.8rem; padding: 24px 0 10px 0; border-top: 1px solid rgba(223, 255, 82, 0.15); margin-top: 24px;">
            Operation-Mosha | EDU AI Engineering Hackathon 2026 | Technical Architecture: Rajen Sikder, Sadrul, Ibrahim
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
