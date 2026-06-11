"""Model retraining DAG with a gated champion promotion.

Chain: validate features → train XGBoost → promote to champion if PR-AUC improved.

Triggered manually from the Airflow UI (schedule=None). This is a demo stack
that isn't continuously running, so a cron schedule would only produce failed
runs between sessions; in production you would set e.g. schedule="@weekly",
a realistic cadence for fraud model refreshes.

The training script itself only registers a new model version — it never moves
the 'champion' alias. The evaluate_and_promote task here is the single quality
gate deciding whether the new version reaches production.

Prerequisites:
  - data/processed/features.parquet must exist (run data_ingestion DAG first)
  - MLflow tracking server must be reachable (mlflow:5000 inside Docker Compose)
  - ./training is volume-mounted at /opt/airflow/training (docker-compose.yml)
    and the Airflow image includes the training deps (see airflow/Dockerfile)

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
        capture_output=True,
        text=True,
    )
    # Echo the script's output: a subprocess writes to its own stdout/stderr,
    # which bypasses Airflow's task log capture — without this, a training
    # failure would show only "exit status 1" with no traceback.
    print(result.stdout)
    if result.stderr:
        print("--- training stderr ---")
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"train_xgboost.py failed with exit code {result.returncode}"
        )
    print("Training completed")


def evaluate_and_promote(**_: object) -> None:
    """Apply the gated champion promotion to the version just registered.

    Delegates to training/model_registry.promote_champion_if_better — the
    exact same quality gate that `make promote` applies after host-side
    training: the latest version becomes 'champion' only if its PR-AUC is at
    least as good as the current champion's, falling back to unconditional
    promotion when no champion exists yet (the first ever training run).
    """
    import mlflow

    sys.path.insert(0, str(TRAINING_DIR))
    from model_registry import promote_champion_if_better

    mlflow.set_tracking_uri(MLFLOW_URI)
    promote_champion_if_better(XGB_MODEL_NAME)


with DAG(
    dag_id="retrain",
    description="Retrain XGBoost classifier; promote to champion if PR-AUC improved",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # manual trigger only — in production this would be e.g. "@weekly"
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
