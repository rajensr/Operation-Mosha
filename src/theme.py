"""Shared design tokens, typography, and theme styling for Operation-Mosha.

Strictly adheres to specified typography and color tokens:
- Font: Inter (400, 700)
- Text: rgb(247, 255, 229)
- Background: rgb(0, 41, 46)
- Primary/Accent: rgb(223, 255, 82)
- Secondary: rgb(19, 17, 17)
- Zero Emojis
"""

from __future__ import annotations

from typing import Any

COLOR_TEXT = "rgb(247, 255, 229)"
COLOR_BG = "rgb(0, 41, 46)"
COLOR_PRIMARY = "rgb(223, 255, 82)"
COLOR_SECONDARY = "rgb(19, 17, 17)"
COLOR_ACCENT = "rgb(223, 255, 82)"

RISK_COLORS = {
    "Low": "#28a745",
    "Medium": "#ffc107",
    "High": "#fd7e14",
    "Critical": "#dc3545",
}

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css?family=Inter:700|Inter:400');

:root {
  --text: rgb(247, 255, 229);
  --background: rgb(0, 41, 46);
  --primary: rgb(223, 255, 82);
  --secondary: rgb(19, 17, 17);
  --accent: rgb(223, 255, 82);
}

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  font-family: 'Inter', sans-serif !important;
  font-weight: 400 !important;
  color: rgb(247, 255, 229) !important;
  background-color: rgb(0, 41, 46) !important;
}

h1, h2, h3, h4, h5, [class*="stHeading"] {
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
  color: rgb(223, 255, 82) !important;
  letter-spacing: -0.02em !important;
}

h1 { font-size: clamp(2.2rem, 5vw, 4.210rem) !important; line-height: 1.1 !important; margin-bottom: 0.5rem !important; }
h2 { font-size: clamp(1.8rem, 4vw, 3.158rem) !important; line-height: 1.15 !important; margin-bottom: 0.5rem !important; }
h3 { font-size: clamp(1.4rem, 3vw, 2.369rem) !important; line-height: 1.2 !important; }
h4 { font-size: clamp(1.2rem, 2.5vw, 1.777rem) !important; }
h5 { font-size: clamp(1.05rem, 2vw, 1.333rem) !important; }
small { font-size: 0.750rem !important; }

/* Streamlit Header & Sidebar */
header[data-testid="stHeader"] {
  background-color: transparent !important;
}

[data-testid="stSidebar"] {
  background-color: rgb(19, 17, 17) !important;
  border-right: 1px solid rgba(223, 255, 82, 0.2) !important;
}

[data-testid="stSidebar"] * {
  color: rgb(247, 255, 229) !important;
}

/* Inputs, NumberInputs, Selectboxes, Datepickers */
input, select, textarea, div[data-baseweb="input"], div[data-baseweb="select"] {
  background-color: rgb(19, 17, 17) !important;
  color: rgb(247, 255, 229) !important;
  border: 1px solid rgba(223, 255, 82, 0.3) !important;
  border-radius: 4px !important;
}

div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
  border-color: rgb(223, 255, 82) !important;
  box-shadow: 0 0 0 1px rgb(223, 255, 82) !important;
}

/* Buttons */
button[kind="primary"], .stButton > button {
  background-color: rgb(223, 255, 82) !important;
  color: rgb(19, 17, 17) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 4px !important;
  padding: 8px 18px !important;
  transition: all 0.2s ease-in-out !important;
}

button[kind="primary"]:hover, .stButton > button:hover {
  background-color: rgb(247, 255, 229) !important;
  color: rgb(19, 17, 17) !important;
  transform: translateY(-1px) !important;
}

/* DataFrames and Tables */
div[data-testid="stDataFrame"], table {
  background-color: rgb(19, 17, 17) !important;
  border: 1px solid rgba(223, 255, 82, 0.2) !important;
  border-radius: 6px !important;
}

/* Metric Widgets */
div[data-testid="stMetric"] {
  background-color: rgb(19, 17, 17) !important;
  border: 1px solid rgba(223, 255, 82, 0.25) !important;
  padding: 14px 18px !important;
  border-radius: 6px !important;
}

div[data-testid="stMetricValue"] {
  color: rgb(223, 255, 82) !important;
  font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
  color: rgb(247, 255, 229) !important;
  font-size: 0.85rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.5px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
  background-color: transparent !important;
  color: rgb(247, 255, 229) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
  color: rgb(223, 255, 82) !important;
  border-bottom-color: rgb(223, 255, 82) !important;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
  .stHorizontalBlock {
    flex-direction: column !important;
  }
  h1 { font-size: 2.2rem !important; }
  h2 { font-size: 1.8rem !important; }
  h3 { font-size: 1.4rem !important; }
}
</style>
"""

PLOTLY_LAYOUT: dict[str, Any] = dict(
    paper_bgcolor="rgb(19, 17, 17)",
    plot_bgcolor="rgb(19, 17, 17)",
    font=dict(family="Inter", color="rgb(247, 255, 229)", size=12),
    xaxis=dict(gridcolor="rgba(223, 255, 82, 0.08)", zerolinecolor="rgba(223, 255, 82, 0.15)"),
    yaxis=dict(gridcolor="rgba(223, 255, 82, 0.08)", zerolinecolor="rgba(223, 255, 82, 0.15)"),
    colorway=["#DFFF52", "#28a745", "#fd7e14", "#dc3545", "#38ef7d", "#11998e"],
)
