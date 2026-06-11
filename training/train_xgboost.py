"""XGBoost fraud detection classifier training script.

Reads features.parquet produced by the Airflow data ingestion DAG,
trains an XGBoost classifier with SMOTE oversampling, logs everything
to MLflow, and registers a new model version in the Model Registry.

Training does NOT touch the 'champion' alias: promotion is a separate,
gated step (only if PR-AUC improves on the current champion) — run
`make promote` / scripts/promote_model.py, or the retrain DAG's
evaluate_and_promote task. scripts/run_training.sh does this for you.

Run from the repo root:
    training/.venv/Scripts/python training/train_xgboost.py   # Windows
    training/.venv/bin/python training/train_xgboost.py       # Linux/Mac

Requires MLFLOW_TRACKING_URI env var (defaults to http://localhost:5000).
"""

from __future__ import annotations

# ruff: noqa: E402  (sys.path.insert before sibling imports is intentional)
import os
import pickle
import sys
import tempfile
from pathlib import Path

# Ensure sibling modules (evaluate, model_registry) are importable when the
# script is invoked from the repo root (e.g. via Makefile or run_training.sh).
sys.path.insert(0, str(Path(__file__).parent))

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_pr_curve,
    plot_roc_curve,
)
from model_registry import get_latest_version

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"

# V1–V28 (original PCA features) + 4 engineered features.
# Raw Amount and Time are excluded: amount is encoded by amount_log;
# Time is encoded by hour_of_day/is_night.
FEATURE_COLS: list[str] = [f"V{i}" for i in range(1, 29)] + [
    "amount_log",
    "hour_of_day",
    "is_night",
    "v1_v2_interaction",
]
TARGET_COL = "Class"

MODEL_NAME = os.getenv("MODEL_XGBOOST_NAME", "fraud-xgboost")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

RANDOM_STATE = 42
# Three-way split: 60% train / 20% val (threshold tuning) / 20% test (reported
# metrics). Tuning and reporting on the same split would bias metrics upward.
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # fraction of the remaining 80% → 20% of the full dataset


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Features parquet not found at {PARQUET_PATH}. "
            "Run the Airflow data_ingestion_dag first (or trigger it manually)."
        )
    df = pd.read_parquet(PARQUET_PATH)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    print(f"Loaded {len(df):,} rows | fraud: {y.sum():,} ({y.mean()*100:.3f}%)")
    return X, y


def train(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    # SMOTE is the single imbalance strategy: the training data arriving here is
    # already resampled to 1:1, so class weighting (scale_pos_weight) would have
    # nothing to correct — and applied before SMOTE it would double-correct.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("fraud-detection-xgboost")

    X, y = load_data()

    # Split off the held-out test set first, then carve val out of the rest.
    X_rest_df, X_test_df, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_rest_df,
        y_rest,
        test_size=VAL_SIZE,
        stratify=y_rest,
        random_state=RANDOM_STATE,
    )

    # Scale features — fit only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    # SMOTE oversampling on training set only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(
        f"After SMOTE — fraud: {y_train_resampled.sum():,} / "
        f"legit: {(y_train_resampled == 0).sum():,}"
    )

    with mlflow.start_run() as run:
        model = train(X_train_resampled, y_train_resampled)

        # Tune the decision threshold on val, then report metrics on the
        # untouched test set — the logged numbers (and the promotion gate that
        # compares pr_auc) are free of threshold-tuning bias.
        y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
        threshold = find_optimal_threshold(y_val.values, y_val_proba)

        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        metrics = compute_metrics(y_test.values, y_test_proba, threshold=threshold)

        # Log hyperparameters
        mlflow.log_params(
            {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "smote": True,
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
                "n_features": len(FEATURE_COLS),
            }
        )
        mlflow.log_metrics(metrics)

        # Log evaluation plots (test set — same data the metrics report on)
        roc_fig = plot_roc_curve(y_test.values, y_test_proba, title="XGBoost ROC Curve")
        pr_fig = plot_pr_curve(y_test.values, y_test_proba, title="XGBoost PR Curve")
        mlflow.log_figure(roc_fig, "roc_curve.png")
        mlflow.log_figure(pr_fig, "pr_curve.png")
        import matplotlib.pyplot as plt

        plt.close("all")

        # Log the fitted StandardScaler so the serving layer can reproduce the
        # exact same scaling without refitting on production data.
        with tempfile.TemporaryDirectory() as tmp:
            scaler_path = Path(tmp) / "scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact(str(scaler_path), artifact_path="scaler")

        # Register model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=pd.DataFrame([X_val_df.iloc[0]]),
        )

        print(
            f"\nRun {run.info.run_id[:8]}…  "
            f"AUC-ROC={metrics['auc_roc']:.4f}  "
            f"PR-AUC={metrics['pr_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"threshold={metrics['threshold']:.4f}"
        )

        if metrics["auc_roc"] < 0.95:
            print(f"WARNING: AUC-ROC {metrics['auc_roc']:.4f} is below target 0.95")

    # Registration only — the 'champion' alias is NOT moved here. Promotion is
    # a gated release decision (new PR-AUC must beat the current champion's),
    # applied by `make promote` or the retrain DAG. This separation means a bad
    # training run can never silently replace the serving model.
    version = get_latest_version(MODEL_NAME)
    print(
        f"Registered {MODEL_NAME} v{version}. "
        "Run `make promote` (or the retrain DAG) to apply the gated champion promotion."
    )


if __name__ == "__main__":
    main()
