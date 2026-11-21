"""Inference contract interface for scoring weekly ward dengue risk observations.

Exposes:
    score_rows(df: pd.DataFrame) -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, create_features

DEFAULT_MODEL_PATH = Path("models/model_bundle.joblib")


def _assign_risk_tier(p_nonzero: float, expected_cases: float, thresholds: dict[str, Any]) -> str:
    """Map outbreak probability and expected cases to canonical risk tiers.

    Policy:
        - Low: p < 0.25 and expected_cases < 1.0
        - Medium: 0.25 <= p < 0.50 or 1.0 <= expected_cases < 3.0
        - High: 0.50 <= p < 0.75 or 3.0 <= expected_cases < 8.0
        - Critical: p >= 0.75 or expected_cases >= 8.0
    """
    low_t = thresholds.get("low", {"p_max": 0.25, "cases_max": 1.0})
    med_t = thresholds.get("medium", {"p_max": 0.50, "cases_max": 3.0})
    high_t = thresholds.get("high", {"p_max": 0.75, "cases_max": 8.0})

    if p_nonzero >= high_t["p_max"] or expected_cases >= high_t["cases_max"]:
        return "Critical"
    elif p_nonzero >= med_t["p_max"] or expected_cases >= med_t["cases_max"]:
        return "High"
    elif p_nonzero >= low_t["p_max"] or expected_cases >= low_t["cases_max"]:
        return "Medium"
    else:
        return "Low"


def load_model_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and cache serialized model bundle."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Please train the model first by running:\n"
            "    python src/train.py"
        )
    return joblib.load(model_path)


def score_rows(
    df: pd.DataFrame,
    model_bundle: dict[str, Any] | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """Score weekly ward observations and compute dengue outbreak risk tiers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing weekly ward weather and vulnerability indicators.
    model_bundle : dict, optional
        Preloaded bundle dictionary. If None, loads from `model_path`.
    model_path : Path, default models/model_bundle.joblib
        Path to serialized model bundle.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [ward_id, date, p_nonzero, expected_cases, risk_tier]
    """
    if df.empty:
        return pd.DataFrame(columns=["ward_id", "date", "p_nonzero", "expected_cases", "risk_tier"])

    if model_bundle is None:
        model_bundle = load_model_bundle(model_path)

    clf = model_bundle["classifier"]
    reg = model_bundle["regressor"]
    feature_cols = model_bundle.get("feature_names", FEATURE_COLUMNS)
    thresholds = model_bundle.get("thresholds", {})

    # Compute required feature representations
    feat_df = create_features(df, is_training=False)

    # Extract feature matrix
    X = feat_df[feature_cols]

    # Stage 1: Predict outbreak probability
    p_nonzero = clf.predict_proba(X)[:, 1]

    # Stage 2: Predict conditional severity on positive cases
    sev_pred = reg.predict(X)
    sev_pred = np.clip(sev_pred, a_min=0.0, a_max=None)

    # Combined expected cases: E[y] = P(y > 0) * E[y | y > 0]
    expected_cases = np.clip(p_nonzero * sev_pred, a_min=0.0, a_max=None)

    # Map risk tiers row-wise
    risk_tiers = [
        _assign_risk_tier(p, c, thresholds)
        for p, c in zip(p_nonzero, expected_cases)
    ]

    out_df = pd.DataFrame(
        {
            "ward_id": feat_df["ward_id"].values,
            "date": feat_df["date"].dt.strftime("%Y-%m-%d").values,
            "p_nonzero": np.round(p_nonzero, 4),
            "expected_cases": np.round(expected_cases, 2),
            "risk_tier": risk_tiers,
        }
    )

    return out_df
