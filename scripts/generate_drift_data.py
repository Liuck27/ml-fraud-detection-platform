"""Generate a synthetic current.parquet that simulates realistic serving drift.

Simulates what happens after a model has been in production for several months:
  - Amount inflation (spending patterns shift upward over time)
  - V1/V4 distribution shift (PCA features correlated with transaction behaviour)
  - V17 shift (correlated with time-of-day patterns changing)
  - Slightly higher fraud rate (emerging attack patterns)

Output: data/reports/current.parquet  (read by scripts/drift_report.py)

Usage:
    make generate-drift-data
    make drift-report
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_PATH = os.getenv(
    "EVIDENTLY_REFERENCE_DATA_PATH", "data/processed/features.parquet"
)
REPORTS_PATH = Path(os.getenv("EVIDENTLY_REPORTS_PATH", "data/reports"))

SAMPLE_FRAC = 0.15  # simulate ~6 weeks of serving data relative to full training set
RANDOM_STATE = 99


def main() -> None:
    ref_path = Path(REFERENCE_PATH)
    if not ref_path.exists():
        print(f"ERROR: reference data not found at {ref_path}", file=sys.stderr)
        print(
            "Run the Airflow ingestion DAG (or make download-data) first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading reference data from {ref_path} …")
    df = pd.read_parquet(ref_path)
    print(f"  {len(df):,} rows, {df.shape[1]} features")

    rng = np.random.default_rng(RANDOM_STATE)

    current = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE).copy()
    print(f"\nSampled {len(current):,} rows as serving window base")

    # Amount drift: gradual inflation + increased variance (new merchant categories)
    amount_shift = rng.normal(loc=25.0, scale=10.0, size=len(current))
    current["Amount"] = (current["Amount"] * 1.35 + amount_shift).clip(lower=0.01)
    if "amount_log" in current.columns:
        current["amount_log"] = np.log1p(current["Amount"])
    if "amount_zscore" in current.columns:
        current["amount_zscore"] = (
            current["Amount"] - current["Amount"].mean()
        ) / current["Amount"].std()

    # V1 shift: correlated with transaction amount in the original PCA space
    current["V1"] = current["V1"] + rng.normal(loc=-0.4, scale=0.3, size=len(current))

    # V4 shift: correlated with merchant type patterns
    current["V4"] = current["V4"] + rng.normal(loc=0.3, scale=0.2, size=len(current))

    # V17 shift: correlated with time-of-day (more night transactions in serving)
    current["V17"] = current["V17"] + rng.normal(loc=-0.5, scale=0.4, size=len(current))

    # hour_of_day shift: serving window skews toward evening/night
    if "hour_of_day" in current.columns:
        hour_noise = rng.integers(-3, 4, size=len(current))
        current["hour_of_day"] = ((current["hour_of_day"] + hour_noise) % 24).astype(
            float
        )
    if "is_night" in current.columns:
        current["is_night"] = current["hour_of_day"] < 6

    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_PATH / "current.parquet"
    current = current.drop(columns=["Class"], errors="ignore")
    current.to_parquet(out_path, index=False)

    print("\nDrift applied:")
    print("  Amount: mean shifted from reference, variance increased")
    print("  V1:     mean shifted by -0.4")
    print("  V4:     mean shifted by +0.3")
    print("  V17:    mean shifted by -0.5")
    print("  hour_of_day: skewed toward evening/night")
    print(f"\nSaved {len(current):,} rows → {out_path}")
    print("Run `make drift-report` to generate the Evidently HTML report.")


if __name__ == "__main__":
    main()
