"""Vector Control Action Protocols & Biological Lead-Time Runbook.

Translates meteorological multi-week rainfall lags and temperature profiles
into calibrated municipal public health protocols without emojis.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from src.theme import THEME_CSS, RISK_COLORS

st.set_page_config(
    page_title="Action Protocols | Operation-Mosha",
    layout="wide",
)

st.markdown(THEME_CSS, unsafe_allow_html=True)

# Top Banner
st.markdown(
    """
    <div style="background-color: rgb(19, 17, 17); border: 1px solid rgba(223, 255, 82, 0.3); padding: 24px 28px; border-radius: 8px; margin-bottom: 24px;">
        <div style="font-size: 0.75rem; font-weight: 700; color: rgb(223, 255, 82); letter-spacing: 1.5px; text-transform: uppercase;">
            Operation-Mosha | Operational Guidelines
        </div>
        <h1 style="margin: 8px 0 4px 0;">Vector Control Action Protocols</h1>
        <p style="margin: 0; font-size: 1.05rem; color: rgb(247, 255, 229); opacity: 0.85;">
            Translating the 3-4 Week Biological Lag into Pre-Emptive Municipal Field Interventions
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h3>Epidemiological Basis: The 4-Stage Biological Progression</h3>
    <p style="font-size: 0.95rem; color: rgb(247, 255, 229); opacity: 0.9;">
        Dengue transmission exhibits a mathematically proven 3-to-4 week delay following heavy precipitation.
        Rainfall occurring in the current week does not cause immediate clinical cases; it creates the stagnant aquatic environment required for vector propagation:
    </p>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        """
        <div style="background-color: rgb(19, 17, 17); border: 1px solid rgba(223, 255, 82, 0.25); border-radius: 6px; padding: 16px; height: 100%;">
            <div style="font-size: 0.8rem; font-weight: 700; color: rgb(223, 255, 82); text-transform: uppercase;">Stage 1: Week t-4</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: rgb(247, 255, 229); margin: 4px 0;">Water Accumulation</div>
            <div style="font-size: 0.85rem; color: rgb(247, 255, 229); opacity: 0.8; line-height: 1.4;">
                Heavy rainfall pools in unsealed drums, discarded tires, construction pits, and roof gutters.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div style="background-color: rgb(19, 17, 17); border: 1px solid rgba(223, 255, 82, 0.25); border-radius: 6px; padding: 16px; height: 100%;">
            <div style="font-size: 0.8rem; font-weight: 700; color: rgb(223, 255, 82); text-transform: uppercase;">Stage 2: Week t-3</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: rgb(247, 255, 229); margin: 4px 0;">Larval Metamorphosis</div>
            <div style="font-size: 0.85rem; color: rgb(247, 255, 229); opacity: 0.8; line-height: 1.4;">
                Aedes eggs hatch into larvae and pupae. Development takes 7 to 10 days in warm ambient temperatures.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div style="background-color: rgb(19, 17, 17); border: 1px solid rgba(223, 255, 82, 0.25); border-radius: 6px; padding: 16px; height: 100%;">
            <div style="font-size: 0.8rem; font-weight: 700; color: rgb(223, 255, 82); text-transform: uppercase;">Stage 3: Week t-2</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: rgb(247, 255, 229); margin: 4px 0;">Vector Emergence & EIP</div>
            <div style="font-size: 0.85rem; color: rgb(247, 255, 229); opacity: 0.8; line-height: 1.4;">
                Adult mosquitoes feed. The virus replicates during the 8 to 12 day Extrinsic Incubation Period.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
        <div style="background-color: rgb(19, 17, 17); border-left: 4px solid #dc3545; border-top: 1px solid rgba(223, 255, 82, 0.25); border-right: 1px solid rgba(223, 255, 82, 0.25); border-bottom: 1px solid rgba(223, 255, 82, 0.25); border-radius: 6px; padding: 16px; height: 100%;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #dc3545; text-transform: uppercase;">Stage 4: Week t</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: rgb(247, 255, 229); margin: 4px 0;">Clinical Outbreak Surge</div>
            <div style="font-size: 0.85rem; color: rgb(247, 255, 229); opacity: 0.8; line-height: 1.4;">
                Infectious bites cause human infections. After 4 to 7 days intrinsic incubation, hospital caseloads surge.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

st.markdown("<h3>Calibrated Municipal Policy Matrix</h3>", unsafe_allow_html=True)
st.caption("Standardized public health operating procedures aligned with model risk tiers:")

action_table = pd.DataFrame(
    [
        {
            "Risk Tier": "Low Risk",
            "Threshold": "P < 25% and Cases < 1.0",
            "Intervention Focus": "Baseline Sanitation",
            "Field Protocol": "Routine municipal solid waste clearance, drain desilting, and baseline entomological ovitrap monitoring.",
            "Responsible Authority": "Ward Conservancy Teams",
        },
        {
            "Risk Tier": "Medium Risk",
            "Threshold": "25% <= P < 50% or 1.0 <= Cases < 3.0",
            "Intervention Focus": "Microclimate Priming & Source Reduction",
            "Field Protocol": "Inspect construction sites, tire shops, and low-lying depressions. Remove standing water and issue public awareness notices.",
            "Responsible Authority": "Health Inspectors & Drainage Engineers",
        },
        {
            "Risk Tier": "High Risk",
            "Threshold": "50% <= P < 75% or 3.0 <= Cases < 8.0",
            "Intervention Focus": "Targeted Larviciding Window (2-3 Weeks Lead Time)",
            "Field Protocol": "Deploy biological larvicide (Bti) across all open waterlogged areas. Mobilize door-to-door domestic container inspections.",
            "Responsible Authority": "Vector Control Squads & Community Health Staff",
        },
        {
            "Risk Tier": "Critical Risk",
            "Threshold": "P >= 75% or Cases >= 8.0",
            "Intervention Focus": "Emergency Vector Knockdown & Hospital Surge",
            "Field Protocol": "Immediate vehicle-mounted thermal fogging within 48 hours. Clear building rooftops. Alert municipal hospitals to allocate IV fluid supplies and isolation beds.",
            "Responsible Authority": "Rapid Response Task Force & Hospital Admins",
        },
    ]
)

st.dataframe(action_table, use_container_width=True, hide_index=True)

st.markdown("---")

st.markdown("<h3>Early Warning Threshold Decision Simulator</h3>", unsafe_allow_html=True)
st.caption("Input anticipated meteorological conditions using numeric fields to determine the required alert status:")

col1, col2, col3 = st.columns(3)
with col1:
    test_rain_4w = st.number_input("Cumulative 4-Week Rainfall (mm)", min_value=0.0, max_value=800.0, value=260.0, step=10.0, format="%.1f")
with col2:
    test_temp = st.number_input("Mean Temperature (deg C)", min_value=12.0, max_value=45.0, value=30.5, step=0.5, format="%.1f")
with col3:
    test_hum = st.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=86.0, step=1.0, format="%.1f")

# Decision logic
is_high_rain = test_rain_4w >= 250.0
is_ideal_temp = 26.0 <= test_temp <= 33.0
is_high_hum = test_hum >= 80.0

if is_high_rain and is_ideal_temp and is_high_hum:
    calc_tier = "Critical Risk"
    calc_color = "#dc3545"
    calc_msg = "CRITICAL ALERT: Conditions exceed the triple-threshold (Rain >= 250 mm, Temp 26-33 deg C, Humidity >= 80%). High probability of explosive vector multiplication within 14 to 21 days. Mobilize emergency vector knockdown immediately."
elif (test_rain_4w >= 180.0 and is_ideal_temp) or (is_high_hum and is_high_rain):
    calc_tier = "High Risk"
    calc_color = "#fd7e14"
    calc_msg = "HIGH ALERT: Favorable aquatic breeding conditions. Adult mosquito emergence expected in 10 to 14 days. Mobilize targeted Bti larvicide squads now."
elif test_rain_4w >= 80.0 or test_hum >= 70.0:
    calc_tier = "Medium Risk"
    calc_color = "#ffc107"
    calc_msg = "MEDIUM ALERT: Moderate environmental suitability. Inspect low-lying waterlogged depressions and accelerate solid waste clearing."
else:
    calc_tier = "Low Risk"
    calc_color = "#28a745"
    calc_msg = "LOW BASELINE: Meteorological indicators suppress vector propagation. Standard municipal conservancy operations apply."

st.markdown(
    f"""
    <div style="background-color: rgb(19, 17, 17); border-left: 6px solid {calc_color}; border-top: 1px solid rgba(223, 255, 82, 0.2); border-right: 1px solid rgba(223, 255, 82, 0.2); border-bottom: 1px solid rgba(223, 255, 82, 0.2); padding: 18px 22px; border-radius: 6px; margin-top: 16px;">
        <div style="font-weight: 700; font-size: 1.2rem; color: {calc_color}; text-transform: uppercase; letter-spacing: 0.5px;">{calc_tier}</div>
        <div style="font-size: 0.95rem; color: rgb(247, 255, 229); margin-top: 8px; line-height: 1.5;">{calc_msg}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
