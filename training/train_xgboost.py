"""XGBoost fraud detection classifier training script.

Reads features.parquet produced by the Airflow data ingestion DAG,
trains an XGBoost classifier with SMOTE oversampling, logs everything
to MLflow, and registers the model as 'champion' in the Model Registry.

Run from the repo root:
    training/.venv/Scripts/python training/train_xgboost.py   # Windows
    training/.venv/bin/python training/train_xgboost.py       # Linux/Mac

Requires MLFLOW_TRACKING_URI env var (defaults to http://localhost:5000).
"""

from __future__ import annotations

import os
import sys
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

from evaluate import compute_metrics, find_optimal_threshold, plot_pr_curve, plot_roc_curve
from model_registry import promote_to_champion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"

# V1–V28 (original PCA features) + 5 engineered features.
# Raw Amount and Time are excluded: amount is encoded by amount_log/amount_zscore;
# Time is encoded by hour_of_day/is_night.
FEATURE_COLS: list[str] = (
    [f"V{i}" for i in range(1, 29)]
    + ["amount_log", "amount_zscore", "hour_of_day", "is_night", "v1_v2_interaction"]
)
TARGET_COL = "Class"

MODEL_NAME = os.getenv("MODEL_XGBOOST_NAME", "fraud-xgboost")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

RANDOM_STATE = 42
TEST_SIZE = 0.2


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
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos  # belt-and-suspenders alongside SMOTE

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("fraud-detection-xgboost")

    X, y = load_data()

    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Scale features — fit only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_val_scaled = scaler.transform(X_val_df)

    # SMOTE oversampling on training set only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(
        f"After SMOTE — fraud: {y_train_resampled.sum():,} / "
        f"legit: {(y_train_resampled == 0).sum():,}"
    )

    with mlflow.start_run() as run:
        model = train(X_train_resampled, y_train_resampled)

        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        threshold = find_optimal_threshold(y_val.values, y_pred_proba)
        metrics = compute_metrics(y_val.values, y_pred_proba, threshold=threshold)

        # Log hyperparameters
        mlflow.log_params(
            {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "smote": True,
                "test_size": TEST_SIZE,
                "n_features": len(FEATURE_COLS),
            }
        )
        mlflow.log_metrics(metrics)

        # Log evaluation plots
        roc_fig = plot_roc_curve(y_val.values, y_pred_proba, title="XGBoost ROC Curve")
        pr_fig = plot_pr_curve(y_val.values, y_pred_proba, title="XGBoost PR Curve")
        mlflow.log_figure(roc_fig, "roc_curve.png")
        mlflow.log_figure(pr_fig, "pr_curve.png")
        import matplotlib.pyplot as plt
        plt.close("all")

        # Register model
        model_info = mlflow.xgboost.log_model(
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

    # Promote the newly registered version to champion
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    promote_to_champion(MODEL_NAME, latest.version)


if __name__ == "__main__":
    main()
