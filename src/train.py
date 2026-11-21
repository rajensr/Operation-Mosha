import json
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
)

from src.features import create_features, FEATURE_COLUMNS


def compute_metrics(y_true: np.ndarray, y_pred_continuous: np.ndarray, y_pred_prob: np.ndarray):
    """
    Computes regression metrics across all records and classification metrics
    on non-zero outbreak weeks.
    """
    mae = mean_absolute_error(y_true, y_pred_continuous)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_continuous))

    y_true_binary = (y_true > 0).astype(int)
    y_pred_binary = (y_pred_prob >= 0.5).astype(int)

    # Guard against split without positive cases
    if y_true_binary.sum() == 0:
        prec, rec, f1 = 0.0, 0.0, 0.0
    else:
        prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "positive_prevalence": round(float(y_true_binary.mean()), 4),
        "total_samples": int(len(y_true)),
        "positive_samples": int(y_true_binary.sum()),
    }


def train_pipeline(data_path: str = "data/processed/cleaned.csv"):
    print("=" * 60)
    print("Starting Dengue Outbreak Hurdle Training Pipeline")
    print("=" * 60)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cleaned dataset not found at '{data_path}'. Run data validation first.")

    raw_df = pd.read_csv(data_path)
    print(f"Loaded raw dataset: {len(raw_df)} rows across {raw_df['ward_id'].nunique()} unique wards.")

    # 1. Feature Engineering with clean initial lag truncation
    df_feat = create_features(raw_df, is_training=True)
    print(f"Engineered feature matrix: {len(df_feat)} rows, {len(FEATURE_COLUMNS)} input features.")

    # 2. Chronological Splitting
    train_mask = df_feat["date"] < "2023-01-01"
    val_mask = (df_feat["date"] >= "2023-01-01") & (df_feat["date"] < "2025-01-01")
    test_mask = df_feat["date"] >= "2025-01-01"

    df_train = df_feat[train_mask].reset_index(drop=True)
    df_val = df_feat[val_mask].reset_index(drop=True)
    df_test = df_feat[test_mask].reset_index(drop=True)

    print(f"Chronological Splits:")
    print(f"  - Train Split (2016 - 2022): {len(df_train)} rows")
    print(f"  - Val Split   (2023 - 2024): {len(df_val)} rows")
    print(f"  - Test Split  (2025):        {len(df_test)} rows")

    X_train, y_train = df_train[FEATURE_COLUMNS], df_train["dengue_cases"].values
    X_val, y_val = df_val[FEATURE_COLUMNS], df_val["dengue_cases"].values
    X_test, y_test = df_test[FEATURE_COLUMNS], df_test["dengue_cases"].values

    # 3. All-Zero Baseline Evaluation
    zero_train_mae = float(mean_absolute_error(y_train, np.zeros_like(y_train)))
    zero_train_rmse = float(np.sqrt(mean_squared_error(y_train, np.zeros_like(y_train))))
    zero_val_mae = float(mean_absolute_error(y_val, np.zeros_like(y_val)))
    zero_val_rmse = float(np.sqrt(mean_squared_error(y_val, np.zeros_like(y_val))))
    zero_test_mae = float(mean_absolute_error(y_test, np.zeros_like(y_test)))
    zero_test_rmse = float(np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test))))

    baselines_zero = {
        "train": {"mae": round(zero_train_mae, 4), "rmse": round(zero_train_rmse, 4), "f1": 0.0, "recall": 0.0},
        "val": {"mae": round(zero_val_mae, 4), "rmse": round(zero_val_rmse, 4), "f1": 0.0, "recall": 0.0},
        "test": {"mae": round(zero_test_mae, 4), "rmse": round(zero_test_rmse, 4), "f1": 0.0, "recall": 0.0},
    }

    # Simple Ridge Baseline
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    ridge_test_pred = np.clip(ridge.predict(X_test), 0, None)
    ridge_test_mae = float(mean_absolute_error(y_test, ridge_test_pred))
    ridge_test_rmse = float(np.sqrt(mean_squared_error(y_test, ridge_test_pred)))

    # 4. Hurdle Stage 1: Occurrence Classifier (Outbreak vs. Non-outbreak)
    y_train_bin = (y_train > 0).astype(int)
    print("\nTraining Stage 1 Classifier: HistGradientBoostingClassifier (occurrence)...")
    clf = HistGradientBoostingClassifier(
        random_state=42,
        class_weight="balanced",
        max_iter=150,
        min_samples_leaf=15,
        learning_rate=0.08,
    )
    clf.fit(X_train, y_train_bin)

    # 5. Hurdle Stage 2: Severity Regressor (Conditional on positive cases)
    pos_mask_train = y_train > 0
    X_train_pos = X_train[pos_mask_train]
    y_train_pos = y_train[pos_mask_train]

    print(f"Training Stage 2 Regressor on {len(X_train_pos)} positive cases...")
    reg = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=42,
        max_iter=150,
        min_samples_leaf=10,
        learning_rate=0.08,
    )
    reg.fit(X_train_pos, y_train_pos)

    # 6. Evaluation across splits
    def predict_hurdle(X):
        p_nonzero = clf.predict_proba(X)[:, 1]
        cond_cases = np.clip(reg.predict(X), 0.0, None)
        expected_cases = np.clip(p_nonzero * cond_cases, 0.0, None)
        return p_nonzero, expected_cases

    p_train, pred_train = predict_hurdle(X_train)
    p_val, pred_val = predict_hurdle(X_val)
    p_test, pred_test = predict_hurdle(X_test)

    train_metrics = compute_metrics(y_train, pred_train, p_train)
    val_metrics = compute_metrics(y_val, pred_val, p_val)
    test_metrics = compute_metrics(y_test, pred_test, p_test)

    print("\n" + "=" * 60)
    print("Validation Results vs. All-Zero Baseline:")
    print("=" * 60)
    print(f"Train - Hurdle MAE: {train_metrics['mae']} | Zero MAE: {baselines_zero['train']['mae']} | F1: {train_metrics['f1']} | Recall: {train_metrics['recall']}")
    print(f"Val   - Hurdle MAE: {val_metrics['mae']} | Zero MAE: {baselines_zero['val']['mae']} | F1: {val_metrics['f1']} | Recall: {val_metrics['recall']}")
    print(f"Test  - Hurdle MAE: {test_metrics['mae']} | Zero MAE: {baselines_zero['test']['mae']} | F1: {test_metrics['f1']} | Recall: {test_metrics['recall']}")
    print(f"Test  - Ridge MAE:  {round(ridge_test_mae, 4)} | Ridge RMSE: {round(ridge_test_rmse, 4)}")

    # 7. Serialize Artifacts
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    risk_thresholds = {
        "p_low": 0.25,
        "p_med": 0.50,
        "p_high": 0.75,
        "cases_low": 1.0,
        "cases_med": 3.0,
        "cases_high": 8.0,
    }

    bundle = {
        "classifier": clf,
        "regressor": reg,
        "feature_names": FEATURE_COLUMNS,
        "thresholds": risk_thresholds,
        "metadata": {
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
        },
    }

    bundle_path = Path("models/model_bundle.joblib")
    joblib.dump(bundle, bundle_path)
    print(f"\nSaved model bundle to: {bundle_path} ({round(os.path.getsize(bundle_path) / (1024 * 1024), 2)} MB)")

    validation_report = {
        "pipeline": "Two-Stage Hurdle Model (HistGradientBoostingClassifier + HistGradientBoostingRegressor)",
        "baselines": {
            "all_zero": baselines_zero,
            "ridge_regression": {"test_mae": round(ridge_test_mae, 4), "test_rmse": round(ridge_test_rmse, 4)},
        },
        "hurdle_model": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "risk_tier_policy": risk_thresholds,
    }

    report_path = Path("reports/validation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    print(f"Saved validation report to: {report_path}")

    return validation_report


if __name__ == "__main__":
    train_pipeline()
