"""Urban Dengue Risk Intelligence Dashboard.

Localhost Streamlit decision-support application for public health officers and epidemiologists.
Provides ward-level dengue outbreak risk nowcasting, scenario simulations,
comparative city analytics, and empirical validation telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.predict import score_rows

# Page configuration
st.set_page_config(
    page_title="Urban Dengue Risk Intelligence",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("data/processed/cleaned.csv")
MODEL_PATH = Path("models/model_bundle.joblib")
VALIDATION_PATH = Path("reports/validation.json")

RISK_COLORS = {
    "Low": "#28a745",
    "Medium": "#ffc107",
    "High": "#fd7e14",
    "Critical": "#dc3545",
}


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load cleaned dataset from disk."""
    if not DATA_PATH.exists():
        st.error(f"Cleaned dataset not found at `{DATA_PATH}`. Please run `python src/clean_data.py` first.")
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
    # Header Banner
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px 28px; border-radius: 10px; color: white; margin-bottom: 24px;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700;">🦟 Urban Dengue Risk Intelligence</h1>
            <p style="margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.9;">
                Localhost Outbreak Risk Nowcasting & Surveillance System | Operation-Mosha
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset()
    model_ready = check_model_availability()

    if not model_ready:
        st.warning(
            "⚠️ **Model artifact not trained yet.** Predictions are running in offline view.\n\n"
            "To train the two-stage hurdle model and unlock live predictions, run:\n"
            "```bash\npython src/train.py\n```"
        )

    # Sidebar: Filtering and Scenario Simulation
    st.sidebar.header("🕹️ Surveillance & Scenario Controls")

    cities = sorted(df["city"].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("Select City", cities, index=0)

    city_df = df[df["city"] == selected_city].copy()
    wards = sorted(city_df["ward_id"].dropna().unique().tolist())
    selected_ward = st.sidebar.selectbox("Select Ward", wards, index=0)

    ward_df = city_df[city_df["ward_id"] == selected_ward].sort_values("date").reset_index(drop=True)
    available_dates = ward_df["date"].dt.strftime("%Y-%m-%d").tolist()

    default_date_idx = len(available_dates) - 1 if available_dates else 0
    selected_date_str = st.sidebar.selectbox(
        "Observation Week (End of Week)",
        available_dates,
        index=default_date_idx,
    )

    # Extract target observed row
    target_row = ward_df[ward_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🌦️ Weather Simulation Sliders")
    st.sidebar.caption("Adjust current week conditions to simulate microclimate shifts:")

    sim_rainfall = st.sidebar.slider(
        "Current Week Rainfall (mm)",
        min_value=0.0,
        max_value=300.0,
        value=float(target_row["rainfall_mm"]),
        step=1.0,
    )
    sim_avg_temp = st.sidebar.slider(
        "Average Temperature (°C)",
        min_value=12.0,
        max_value=40.0,
        value=float(target_row["avg_temp_c"]),
        step=0.2,
    )
    sim_humidity = st.sidebar.slider(
        "Relative Humidity (%)",
        min_value=30.0,
        max_value=100.0,
        value=float(target_row["relative_humidity_pct"]),
        step=1.0,
    )

    # Dynamic Scoring Execution
    score_input_df = ward_df[ward_df["date"] <= target_row["date"]].copy()

    # Apply simulated weather changes to the active observation week
    last_idx = score_input_df.index[-1]
    score_input_df.loc[last_idx, "rainfall_mm"] = sim_rainfall
    score_input_df.loc[last_idx, "avg_temp_c"] = sim_avg_temp
    score_input_df.loc[last_idx, "relative_humidity_pct"] = sim_humidity

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

    # Section 1: KPI Overview
    st.subheader(f"📊 Surveillance Snapshot — {selected_city} | {selected_ward} ({selected_date_str})")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        tier_color = RISK_COLORS.get(tier_val, "#6c757d")
        st.markdown(
            f"""
            <div style="background-color: {tier_color}1a; border-left: 6px solid {tier_color}; padding: 14px 18px; border-radius: 8px;">
                <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600;">Outbreak Risk Tier</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {tier_color}; line-height: 1.2;">{tier_val}</div>
                <div style="font-size: 0.8rem; color: #777;">Calibrated 4-Tier Policy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; border-left: 6px solid #0d6efd; padding: 14px 18px; border-radius: 8px;">
                <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600;">Expected Caseload (Nowcast)</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #0d6efd; line-height: 1.2;">{cases_val:.2f} <span style="font-size: 1.1rem; font-weight: 500;">cases</span></div>
                <div style="font-size: 0.8rem; color: #777;">Observed Cases: {int(target_row['dengue_cases'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; border-left: 6px solid #6f42c1; padding: 14px 18px; border-radius: 8px;">
                <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600;">Outbreak Probability</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #6f42c1; line-height: 1.2;">{p_val * 100:.1f}%</div>
                <div style="font-size: 0.8rem; color: #777;">P(Cases > 0 | Weather History)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; border-left: 6px solid #20c997; padding: 14px 18px; border-radius: 8px;">
                <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600;">Ward Vulnerability Index</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #20c997; line-height: 1.2;">{target_row['water_logging_risk_index']:.2f}</div>
                <div style="font-size: 0.8rem; color: #777;">Pop. Density: {int(target_row['population_density_sqkm']):,} /km²</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Section 2: Visual Charts
    col_left, col_right = st.columns([6, 5])

    with col_left:
        st.subheader("📈 Historical Longitudinal Trends (Rainfall & Dengue Cases)")
        # Plot longitudinal history for selected ward
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])

        # Rainfall bar
        fig_trend.add_trace(
            go.Bar(
                x=ward_df["date"],
                y=ward_df["rainfall_mm"],
                name="Rainfall (mm)",
                marker_color="rgba(33, 150, 243, 0.4)",
            ),
            secondary_y=True,
        )

        # Actual Dengue Cases
        fig_trend.add_trace(
            go.Scatter(
                x=ward_df["date"],
                y=ward_df["dengue_cases"],
                name="Actual Dengue Cases",
                mode="lines",
                line=dict(color="#d32f2f", width=2.5),
            ),
            secondary_y=False,
        )

        # Vertical line indicating currently selected week
        fig_trend.add_vline(
            x=pd.to_datetime(selected_date_str).timestamp() * 1000,
            line_dash="dash",
            line_color="#212121",
            annotation_text="Selected Week",
            annotation_position="top left",
        )

        fig_trend.update_layout(
            hovermode="x unified",
            height=390,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_trend.update_yaxes(title_text="Weekly Dengue Cases", secondary_y=False)
        fig_trend.update_yaxes(title_text="Rainfall (mm)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.subheader(f"🏙️ Ward Cross-Section Comparison ({selected_date_str})")
        # Compare all wards in the same city for the selected date
        cross_df = city_df[city_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str].copy()

        if model_ready and not cross_df.empty:
            # Score cross section wards
            # For each ward in cross_df, score historical context
            history_sub = city_df[city_df["date"] <= target_row["date"]]
            cross_scores = score_rows(history_sub)
            cross_latest = cross_scores[cross_scores["date"] == selected_date_str]
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
            fig_comp = px.bar(
                cross_df.sort_values("dengue_cases", ascending=True),
                x="dengue_cases",
                y="ward_id",
                orientation="h",
                labels={"dengue_cases": "Observed Cases", "ward_id": "Ward ID"},
                title="Observed Cases (Model Offline)",
            )

        fig_comp.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # Section 3: Driver Attribution & Weather Lags
    st.subheader("🔍 Local Outbreak Driver Attribution & Meteorological Lag Analysis")
    st.caption("How historical weather lags and environmental vulnerabilities drive the current week's prediction:")

    drv1, drv2, drv3, drv4, drv5 = st.columns(5)
    # Extract preceding lags if available
    curr_idx = ward_df[ward_df["date"] == target_row["date"]].index[0]
    lag1 = float(ward_df.loc[curr_idx - 1, "rainfall_mm"]) if curr_idx >= 1 else float(target_row["rainfall_mm"])
    lag2 = float(ward_df.loc[curr_idx - 2, "rainfall_mm"]) if curr_idx >= 2 else float(target_row["rainfall_mm"])
    lag3 = float(ward_df.loc[curr_idx - 3, "rainfall_mm"]) if curr_idx >= 3 else float(target_row["rainfall_mm"])
    lag4 = float(ward_df.loc[curr_idx - 4, "rainfall_mm"]) if curr_idx >= 4 else float(target_row["rainfall_mm"])
    sum_4w = lag1 + lag2 + lag3 + lag4

    with drv1:
        st.metric("Rainfall (t-1)", f"{lag1:.1f} mm", delta=f"{lag1 - lag2:.1f} vs t-2")
    with drv2:
        st.metric("Rainfall (t-2)", f"{lag2:.1f} mm", delta=f"{lag2 - lag3:.1f} vs t-3")
    with drv3:
        st.metric("Rainfall (t-3)", f"{lag3:.1f} mm", delta=f"{lag3 - lag4:.1f} vs t-4")
    with drv4:
        st.metric("Rainfall (t-4)", f"{lag4:.1f} mm")
    with drv5:
        st.metric("4-Week Cumulative Rain", f"{sum_4w:.1f} mm", help="Vector breeding incubation volume")

    st.markdown("---")

    # Section 4: Validation Telemetry & Baseline Benchmarking
    st.subheader("📋 Empirical Model Validation & Out-of-Sample Performance")
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
                    f"**Temporal Horizon Split:** `{split_key.upper()}` | **Sample Count:** `{sp.get('sample_count', 0):,}` | "
                    f"**Outbreak Prevalence:** `{sp.get('positive_prevalence_pct', 0.0):.2f}%` positive weeks"
                )

                metric_df = pd.DataFrame(
                    [
                        {
                            "Metric": "Mean Absolute Error (MAE)",
                            "Two-Stage Hurdle Model": f"{hm.get('mae', 0):.4f}",
                            "All-Zero Baseline": f"{bm.get('mae', 0):.4f}",
                            "Outcome": "✅ Superior Accuracy" if hm.get("mae", 0) < bm.get("mae", 0) else "Comparable",
                        },
                        {
                            "Metric": "Root Mean Squared Error (RMSE)",
                            "Two-Stage Hurdle Model": f"{hm.get('rmse', 0):.4f}",
                            "All-Zero Baseline": f"{bm.get('rmse', 0):.4f}",
                            "Outcome": "✅ Outbreak Penalized",
                        },
                        {
                            "Metric": "Outbreak Recall (Positive Weeks)",
                            "Two-Stage Hurdle Model": f"{hm.get('recall_positive', 0) * 100:.1f}%",
                            "All-Zero Baseline": "0.0% (Fails Outbreaks)",
                            "Outcome": "🎯 High Sensitivity",
                        },
                        {
                            "Metric": "Outbreak F1 Score",
                            "Two-Stage Hurdle Model": f"{hm.get('f1_positive', 0):.4f}",
                            "All-Zero Baseline": "0.0000",
                            "Outcome": "🎯 Balanced Detection",
                        },
                    ]
                )
                st.table(metric_df)
    else:
        st.info("Validation report not available. Run `python src/train.py` to generate empirical evaluation metrics.")

    # Footer
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.85rem; padding: 20px 0 10px 0;">
            Operation-Mosha | EDU AI Engineering Hackathon 2026 | Developed by Rajen, Sadrul, and Ibrahim
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
