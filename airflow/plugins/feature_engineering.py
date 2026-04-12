"""Feature engineering pure functions for the fraud detection pipeline.

All functions are side-effect-free: they accept a DataFrame and return a new
one with additional columns. This makes them trivially testable outside Airflow.

Engineered features added on top of raw V1–V28 + Amount:
    amount_log      — log1p(Amount), reduces right skew
    amount_zscore   — Z-score normalized Amount
    hour_of_day     — 0–23, derived from Time (seconds since first transaction)
    is_night        — True if hour_of_day in [22, 23, 0–5]
    v1_v2_interaction — V1 * V2 interaction term
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p-transformed Amount as `amount_log`."""
    out = df.copy()
    out["amount_log"] = np.log1p(out["Amount"])
    return out


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive `hour_of_day` (int) and `is_night` (bool) from `Time`.

    The dataset's Time column is seconds elapsed since the first transaction
    in the dataset. We map it to an approximate hour of day (modulo 24 hours).
    """
    out = df.copy()
    out["hour_of_day"] = (out["Time"] // 3600 % 24).astype(int)
    out["is_night"] = out["hour_of_day"].apply(lambda h: h >= 22 or h < 6)
    return out


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `amount_zscore` and `v1_v2_interaction`."""
    out = df.copy()
    mu = out["Amount"].mean()
    sigma = out["Amount"].std(ddof=0)
    out["amount_zscore"] = (out["Amount"] - mu) / sigma if sigma > 0 else 0.0
    out["v1_v2_interaction"] = out["V1"] * out["V2"]
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence and return the result."""
    df = log_transform_amount(df)
    df = extract_time_features(df)
    df = compute_interaction_features(df)
    return df
