"""Two-stage hurdle model training and evaluation pipeline for dengue outbreak risk.

Stage 1: Outbreak occurrence classifier (P(cases > 0 | X))
Stage 2: Case severity regressor (E[cases | cases > 0, X])
Combined: Expected cases = P(cases > 0 | X) * E[cases | cases > 0, X]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

from src.features import FEATURE_COLUMNS, create_features

# Canonical policy thresholds for risk tiers
RISK_TIER_THRESHOLDS = {
    "low": {"p_max": 0.25, "cases_max": 1.0},
    "medium": {"p_max": 0.50, "cases_max": 3.0},
    "high": {"p_max": 0.75, "cases_max": 8.0},
}


def _evaluate_split(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    cases_pred: np.ndarray,
    split_name: str,
) -> dict[str, Any]:
    """Compute comprehensive regression and classification metrics against all-zero baseline."""
    n_samples = len(y_true)
    y_binary = (y_true > 0).astype(int)
    pos_samples = int(y_binary.sum())
    pos_prevalence = round(float(pos_samples / n_samples) * 100.0, 2) if n_samples > 0 else 0.0

    # All-Zero baseline predictions
    zero_cases = np.zeros_like(y_true, dtype=float)
    zero_binary = np.zeros_like(y_binary, dtype=int)

    base_mae = float(mean_absolute_error(y_true, zero_cases))
    base_rmse = float(np.sqrt(mean_squared_error(y_true, zero_cases)))

    # Trained model metrics
    model_mae = float(mean_absolute_error(y_true, cases_pred))
    model_rmse = float(np.sqrt(mean_squared_error(y_true, cases_pred)))

    # Classification metrics for outbreak presence (using threshold 0.5)
    pred_binary = (p_pred >= 0.5).astype(int)
    if pos_samples > 0:
        prec = float(precision_score(y_binary, pred_binary, zero_division=0))
        rec = float(recall_score(y_binary, pred_binary, zero_division=0))
        f1 = float(f1_score(y_binary, pred_binary, zero_division=0))
    else:
        prec, rec, f1 = 0.0, 0.0, 0.0

    return {
        "split": split_name,
        "sample_count": n_samples,
        "positive_cases_count": pos_samples,
        "positive_prevalence_pct": pos_prevalence,
        "baseline_all_zero": {
            "mae": round(base_mae, 4),
            "rmse": round(base_rmse, 4),
            "recall_positive": 0.0,
            "f1_positive": 0.0,
        },
        "hurdle_model": {
            "mae": round(model_mae, 4),
            "rmse": round(model_rmse, 4),
            "precision_positive": round(prec, 4),
            "recall_positive": round(rec, 4),
            "f1_positive": round(f1, 4),
        },
    }


def train_pipeline(
    input_csv: Path = Path("data/processed/cleaned.csv"),
    model_path: Path = Path("models/model_bundle.joblib"),
    metrics_path: Path = Path("reports/validation.json"),
) -> dict[str, Any]:
    """Train two-stage hurdle model and evaluate chronologically."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {input_csv}")

    print(f"Loading cleaned dataset from {input_csv}...")
    raw_df = pd.read_csv(input_csv)

    print("Engineering lag features (ward-isolated)...")
    feat_df = create_features(raw_df, is_training=True)

    # Chronological splits (strictly avoiding random splitting or future data leakage)
    train_mask = feat_df["date"] <= "2022-12-26"
    val_mask = (feat_df["date"] >= "2023-01-02") & (feat_df["date"] <= "2024-12-30")
    test_mask = feat_df["date"] >= "2025-01-06"

    X_train = feat_df.loc[train_mask, FEATURE_COLUMNS]
    y_train = feat_df.loc[train_mask, "dengue_cases"].values.astype(float)

    X_val = feat_df.loc[val_mask, FEATURE_COLUMNS]
    y_val = feat_df.loc[val_mask, "dengue_cases"].values.astype(float)

    X_test = feat_df.loc[test_mask, FEATURE_COLUMNS]
    y_test = feat_df.loc[test_mask, "dengue_cases"].values.astype(float)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    # Stage 1: Outbreak Occurrence Classifier
    print("Fitting Stage 1 Outbreak Classifier (HistGradientBoostingClassifier)...")
    y_train_binary = (y_train > 0).astype(int)
    clf = HistGradientBoostingClassifier(
        random_state=42,
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
    )
    clf.fit(X_train, y_train_binary)

    # Stage 2: Case Severity Regressor (trained conditional on y > 0)
    print("Fitting Stage 2 Severity Regressor (HistGradientBoostingRegressor on positive cases)...")
    pos_mask = y_train > 0
    X_train_pos = X_train.loc[pos_mask]
    y_train_pos = y_train[pos_mask]

    reg = HistGradientBoostingRegressor(
        random_state=42,
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
    )
    reg.fit(X_train_pos, y_train_pos)

    # Evaluate on all splits
    print("Evaluating model across chronological splits...")
    results: dict[str, Any] = {
        "metadata": {
            "model_architecture": "Two-Stage Hurdle (HistGradientBoostingClassifier + HistGradientBoostingRegressor)",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_features": len(FEATURE_COLUMNS),
            "feature_names": FEATURE_COLUMNS,
            "thresholds": RISK_TIER_THRESHOLDS,
        },
        "splits": {},
    }

    for name, X_s, y_s in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        p_s = clf.predict_proba(X_s)[:, 1]
        sev_s = np.clip(reg.predict(X_s), a_min=0.0, a_max=None)
        expected_s = np.clip(p_s * sev_s, a_min=0.0, a_max=None)

        split_metrics = _evaluate_split(y_s, p_s, expected_s, split_name=name)
        results["splits"][name] = split_metrics

    # Save model bundle artifact
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "classifier": clf,
        "regressor": reg,
        "feature_names": FEATURE_COLUMNS,
        "thresholds": RISK_TIER_THRESHOLDS,
        "metadata": results["metadata"],
        "metrics": results["splits"],
    }
    joblib.dump(bundle, model_path, compress=3)
    print(f"Model bundle saved to: {model_path}")

    # Save metrics JSON
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Validation report saved to: {metrics_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two-stage hurdle dengue risk model.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/cleaned.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/model_bundle.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/validation.json"))
    args = parser.parse_args()

    results = train_pipeline(input_csv=args.input, model_path=args.model, metrics_path=args.metrics)

    print("\n--- Model Training Summary ---")
    for split, m in results["splits"].items():
        hm = m["hurdle_model"]
        bm = m["baseline_all_zero"]
        print(
            f"[{split.upper()}] Samples: {m['sample_count']} | Outbreak Prevalence: {m['positive_prevalence_pct']}%"
        )
        print(f"   Model MAE: {hm['mae']} (Baseline: {bm['mae']}) | RMSE: {hm['rmse']}")
        print(f"   Outbreak Recall: {hm['recall_positive'] * 100:.1f}% | Outbreak F1: {hm['f1_positive']:.3f}")


if __name__ == "__main__":
    main()
