"""Data ingestion DAG: validate CSV → engineer features → write Parquet.

Trigger manually from the Airflow UI or via:
    airflow dags trigger data_ingestion

Requires data/raw/creditcard.csv to exist (run `make download-data` first).
Writes engineered features to data/processed/features.parquet.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# Paths as seen from inside the container (volume-mounted).
DATA_RAW = Path("/opt/airflow/data/raw/creditcard.csv")
DATA_PROCESSED = Path("/opt/airflow/data/processed/features.parquet")
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


def validate_csv(**_: object) -> None:
    """Check the raw CSV exists and has the expected shape."""
    import pandas as pd

    if not DATA_RAW.exists():
        raise FileNotFoundError(
            f"Raw data not found: {DATA_RAW}\n"
            "Run `make download-data` on the host, then re-trigger the DAG."
        )

    df = pd.read_csv(DATA_RAW, nrows=5)  # read header only first
    expected_cols = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # Full read for row/fraud count validation
    df = pd.read_csv(DATA_RAW)
    if len(df) < 280_000:
        raise ValueError(f"Expected ~{EXPECTED_ROWS:,} rows, got {len(df):,}")

    fraud_count = int(df["Class"].sum())
    print(
        f"Validation passed: {len(df):,} rows, {fraud_count} frauds "
        f"({fraud_count / len(df) * 100:.3f}%)"
    )


def engineer_and_write(**_: object) -> None:
    """Apply feature engineering and write the result as Parquet."""
    import pandas as pd

    # Ensure the plugins directory is importable inside the container
    plugins_dir = "/opt/airflow/plugins"
    if plugins_dir not in sys.path:
        sys.path.insert(0, plugins_dir)

    from feature_engineering import engineer_features  # type: ignore[import]

    df = pd.read_csv(DATA_RAW)
    df = engineer_features(df)

    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED, index=False)

    new_cols = [
        "amount_log",
        "amount_zscore",
        "hour_of_day",
        "is_night",
        "v1_v2_interaction",
    ]
    print(f"Wrote {len(df):,} rows to {DATA_PROCESSED}. " f"New features: {new_cols}")


with DAG(
    dag_id="data_ingestion",
    description="Validate raw CSV, engineer features, write features.parquet",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # manual trigger only
    catchup=False,
    tags=["phase-2", "ingestion"],
) as dag:
    validate = PythonOperator(
        task_id="validate_csv",
        python_callable=validate_csv,
    )
    engineer = PythonOperator(
        task_id="engineer_and_write",
        python_callable=engineer_and_write,
    )

    validate >> engineer
