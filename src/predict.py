import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from src.features import create_features, FEATURE_COLUMNS

_MODEL_BUNDLE: Optional[Dict[str, Any]] = None
_BUNDLE_PATH = Path("models/model_bundle.joblib")


def load_model_bundle(bundle_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads and caches the model bundle from disk.
    """
    global _MODEL_BUNDLE
    path = bundle_path or _BUNDLE_PATH
    if _MODEL_BUNDLE is not None and bundle_path is None:
        return _MODEL_BUNDLE

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model bundle not found at '{path}'. "
            "Please train the model first by executing: python src/train.py"
        )

    bundle = joblib.load(path)
    if bundle_path is None:
        _MODEL_BUNDLE = bundle
    return bundle


def classify_risk_tier(p_nonzero: float, expected_cases: float, thresholds: Dict[str, Any]) -> str:
    """
    Categorizes risk into Low, Medium, High, or Critical using explicit policy rules.
    """
    p_low = thresholds.get("p_low", 0.25)
    p_med = thresholds.get("p_med", 0.50)
    p_high = thresholds.get("p_high", 0.75)
    cases_low = thresholds.get("cases_low", 1.0)
    cases_med = thresholds.get("cases_med", 3.0)
    cases_high = thresholds.get("cases_high", 8.0)

    if p_nonzero >= p_high or expected_cases >= cases_high:
        return "Critical"
    elif p_nonzero >= p_med or expected_cases >= cases_med:
        return "High"
    elif p_nonzero >= p_low or expected_cases >= cases_low:
        return "Medium"
    else:
        return "Low"


def score_rows(df: pd.DataFrame, bundle_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Inference Contract:
    Input: cleaned weekly ward observations containing weather and vulnerability columns.
    Output: a DataFrame with at least:
        - ward_id
        - date
        - p_nonzero
        - expected_cases
        - risk_tier

    Ensures the dashboard can call this without retraining the model.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["ward_id", "date", "p_nonzero", "expected_cases", "risk_tier"]
        )

    bundle = load_model_bundle(bundle_path)
    classifier = bundle["classifier"]
    regressor = bundle["regressor"]
    feature_names = bundle.get("feature_names", FEATURE_COLUMNS)
    thresholds = bundle.get("thresholds", {})

    # Generate lag and calendar features
    feature_df = create_features(df, is_training=False)

    # Align input matrix with feature names
    X = feature_df[feature_names]

    # Stage 1: Probability of positive outbreak week
    p_nonzero_arr = classifier.predict_proba(X)[:, 1]

    # Stage 2: Expected cases conditional on positive week
    cond_cases_arr = np.clip(regressor.predict(X), a_min=0.0, a_max=None)

    # Composite Expected Count
    expected_cases_arr = np.clip(p_nonzero_arr * cond_cases_arr, a_min=0.0, a_max=None)

    # Risk tiers
    risk_tiers = [
        classify_risk_tier(p, cases, thresholds)
        for p, cases in zip(p_nonzero_arr, expected_cases_arr)
    ]

    out_df = pd.DataFrame(
        {
            "ward_id": feature_df["ward_id"],
            "date": feature_df["date"].dt.strftime("%Y-%m-%d"),
            "p_nonzero": np.round(p_nonzero_arr, 4),
            "expected_cases": np.round(expected_cases_arr, 2),
            "risk_tier": risk_tiers,
        }
    )

    return out_df
