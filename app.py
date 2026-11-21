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

    # Sidebar: Filtering and Navigation
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

    # Section 1: KPI Overview
    st.subheader(f"📊 Surveillance Snapshot — {selected_city} | {selected_ward} ({selected_date_str})")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        tier_val = "Pending"
        tier_color = "#6c757d"
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
                <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600;">Observed Cases</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #0d6efd; line-height: 1.2;">{int(target_row['dengue_cases'])} <span style="font-size: 1.1rem; font-weight: 500;">cases</span></div>
                <div style="font-size: 0.8rem; color: #777;">Recorded Hospital Admissions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; border-left: 6px solid #6f42c1; padding: 14px 18px; border-radius: 8px;">
                <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600;">Rainfall (Observed)</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: #6f42c1; line-height: 1.2;">{target_row['rainfall_mm']:.1f} <span style="font-size: 1.1rem; font-weight: 500;">mm</span></div>
                <div style="font-size: 0.8rem; color: #777;">Observation Week Precipitation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; border-left: 6px solid #20c997; padding: 14px 18px; border-radius: 8px;">
                <div style="font-size: 0.85rem; color: #555; text-transform: uppercase; font-weight: 600;">Water Logging Risk</div>
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
        cross_df = city_df[city_df["date"].dt.strftime("%Y-%m-%d") == selected_date_str].copy()

        fig_comp = px.bar(
            cross_df.sort_values("dengue_cases", ascending=True),
            x="dengue_cases",
            y="ward_id",
            orientation="h",
            labels={"dengue_cases": "Recorded Cases", "ward_id": "Ward ID"},
            title=f"Recorded Ward Cases across {selected_city}",
        )
        fig_comp.update_layout(height=390, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_comp, use_container_width=True)

    # Footer
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.85rem; padding: 40px 0 10px 0;">
            Operation-Mosha | EDU AI Engineering Hackathon 2026 | Developed by Rajen, Sadrul, and Ibrahim
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
