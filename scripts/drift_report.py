"""Standalone drift report: compares training reference data vs. recent serving data.

Usage (from project root):
    make drift-report
    # or directly:
    monitoring/evidently/.venv/Scripts/python scripts/drift_report.py

Output: data/reports/drift_report.html

Environment variables:
    EVIDENTLY_REFERENCE_DATA_PATH  path to training parquet (default: data/processed/features.parquet)
    EVIDENTLY_REPORTS_PATH         output directory (default: data/reports)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

REFERENCE_PATH = os.getenv(
    "EVIDENTLY_REFERENCE_DATA_PATH", "data/processed/features.parquet"
)
REPORTS_PATH = os.getenv("EVIDENTLY_REPORTS_PATH", "data/reports")


def load_reference() -> pd.DataFrame:
    path = Path(REFERENCE_PATH)
    if not path.exists():
        print(f"ERROR: reference data not found at {path}", file=sys.stderr)
        print(
            "Run the Airflow ingestion DAG (or make download-data) first.",
            file=sys.stderr,
        )
        sys.exit(1)
    df = pd.read_parquet(path)
    df = df.drop(columns=["Class"], errors="ignore")
    return df


def load_current(reference: pd.DataFrame) -> pd.DataFrame:
    """Load current (serving) data if available; fall back to a 10% sample of reference."""
    current_path = Path(REPORTS_PATH) / "current.parquet"
    if current_path.exists():
        print(f"Loading current data from {current_path}")
        df = pd.read_parquet(current_path)
        return df.drop(columns=["Class"], errors="ignore")
    print(
        f"No current.parquet found at {current_path}. "
        "Using 10% sample of reference as a placeholder."
    )
    return reference.sample(frac=0.1, random_state=42)


def main() -> None:
    print("Loading reference data…")
    reference = load_reference()
    print(f"  {len(reference):,} rows, {reference.shape[1]} features")

    print("Loading current data…")
    current = load_current(reference)
    print(f"  {len(current):,} rows")

    print("Running Evidently DataDriftPreset…")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    out_dir = Path(REPORTS_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drift_report.html"
    report.save_html(str(out_path))
    print(f"Drift report saved → {out_path}")


if __name__ == "__main__":
    main()
