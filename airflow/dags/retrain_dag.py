"""Automated model retraining DAG.

Triggered manually from the Airflow UI or on a weekly schedule.
Chain: validate features → train XGBoost → promote to champion if PR-AUC improved.

Prerequisites:
  - data/processed/features.parquet must exist (run data_ingestion DAG first)
  - MLflow tracking server must be reachable (mlflow:5000 inside Docker Compose)

Production note: the train_xgboost task calls the training script via subprocess.
In a production deployment you would replace this with a DockerOperator or
KubernetesPodOperator pointing at a dedicated training image that has xgboost,
torch, and imbalanced-learn installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Paths as seen from inside the Airflow container (volume-mounted).
FEATURES_PATH = Path("/opt/airflow/data/processed/features.parquet")
TRAINING_DIR = Path("/opt/airflow/training")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
XGB_MODEL_NAME = "fraud-xgboost"

EXPECTED_FEATURE_COLS = {f"V{i}" for i in range(1, 29)} | {
    "Amount",
    "Class",
    "amount_log",
    "amount_zscore",
    "hour_of_day",
    "is_night",
    "v1_v2_interaction",
}


def validate_features(**_: object) -> None:
    """Check that features.parquet exists and has the expected schema."""
    import pandas as pd

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features not found: {FEATURES_PATH}\n"
            "Run the data_ingestion DAG first, or `make download-data` on the host."
        )

    df = pd.read_parquet(FEATURES_PATH, columns=list(EXPECTED_FEATURE_COLS))
    missing = EXPECTED_FEATURE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in features.parquet: {missing}")

    fraud_count = int(df["Class"].sum())
    print(
        f"Validation passed: {len(df):,} rows, {fraud_count} frauds "
        f"({fraud_count / len(df) * 100:.3f}%)"
    )


def train_xgboost(**_: object) -> None:
    """Run XGBoost training script via subprocess and log results to MLflow."""
    script = TRAINING_DIR / "train_xgboost.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Training script not found: {script}\n"
            "Ensure the training/ directory is volume-mounted into the Airflow container."
        )

    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": MLFLOW_URI,
        "PYTHONPATH": str(TRAINING_DIR),
    }
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        cwd=str(TRAINING_DIR.parent),
        check=True,
    )
    print(f"Training completed (return code {result.returncode})")


def evaluate_and_promote(**_: object) -> None:
    """Compare the latest registered model vs. the current champion.

    Promotes the new version to 'champion' alias only if its PR-AUC is at
    least as good as the existing champion's. Falls back to unconditional
    promotion when no champion alias exists yet.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # Retrieve the version just registered by the training script.
    versions = client.search_model_versions(f"name='{XGB_MODEL_NAME}'")
    if not versions:
        raise ValueError(
            f"No versions found for model '{XGB_MODEL_NAME}' in MLflow registry."
        )

    latest = max(versions, key=lambda v: int(v.version))
    new_version = latest.version
    new_run = client.get_run(latest.run_id)
    new_pr_auc: float = new_run.data.metrics.get("pr_auc", 0.0)

    # Retrieve current champion metrics (if any).
    try:
        champ = client.get_model_version_by_alias(XGB_MODEL_NAME, "champion")
        champ_run = client.get_run(champ.run_id)
        champ_pr_auc: float = champ_run.data.metrics.get("pr_auc", 0.0)
        champ_version = champ.version
    except Exception:
        champ_pr_auc = 0.0
        champ_version = "none"
        print("No existing champion found — will promote unconditionally.")

    print(
        f"Candidate v{new_version}: PR-AUC={new_pr_auc:.4f} | "
        f"Champion v{champ_version}: PR-AUC={champ_pr_auc:.4f}"
    )

    if new_pr_auc >= champ_pr_auc:
        client.set_registered_model_alias(XGB_MODEL_NAME, "champion", new_version)
        delta = new_pr_auc - champ_pr_auc
        print(
            f"Promoted v{new_version} → champion (+{delta:.4f} PR-AUC vs. previous champion)"
        )
    else:
        print(
            f"New model (PR-AUC={new_pr_auc:.4f}) did not beat champion "
            f"(PR-AUC={champ_pr_auc:.4f}) — keeping existing champion."
        )


with DAG(
    dag_id="retrain",
    description="Retrain XGBoost classifier; promote to champion if PR-AUC improved",
    start_date=datetime(2024, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["phase-6", "training"],
) as dag:
    validate = PythonOperator(
        task_id="validate_features",
        python_callable=validate_features,
    )
    train = PythonOperator(
        task_id="train_xgboost",
        python_callable=train_xgboost,
    )
    promote = PythonOperator(
        task_id="evaluate_and_promote",
        python_callable=evaluate_and_promote,
    )

    validate >> train >> promote
