"""
Send a mix of legit and fraud-pattern transactions to the running FastAPI service.

Usage:
    python scripts/populate_metrics.py [--url http://localhost:8000] [--n 80]

This populates Prometheus metrics (scraped by Grafana) and prints one full
/predict response so you can capture it as a screenshot.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Feature templates
# ---------------------------------------------------------------------------

# Derived from real creditcard.csv statistics.
# Legit: most V features near 0, small amounts, normal hours.
# Fraud: V14 strongly negative, V4/V11 shifted, higher amounts.

LEGIT_TEMPLATE: dict[str, float] = {
    "V1": -0.31,
    "V2": 0.47,
    "V3": 1.44,
    "V4": 0.17,
    "V5": -0.27,
    "V6": -0.09,
    "V7": 0.61,
    "V8": 0.03,
    "V9": 0.22,
    "V10": 0.15,
    "V11": 0.55,
    "V12": 0.33,
    "V13": -0.14,
    "V14": 0.31,
    "V15": 0.08,
    "V16": -0.21,
    "V17": -0.07,
    "V18": 0.12,
    "V19": -0.09,
    "V20": 0.04,
    "V21": -0.03,
    "V22": 0.11,
    "V23": -0.04,
    "V24": 0.06,
    "V25": 0.09,
    "V26": -0.13,
    "V27": 0.02,
    "V28": 0.01,
    "Amount": 42.50,
    "Time": 36000.0,
}

FRAUD_TEMPLATE: dict[str, float] = {
    "V1": -3.04,
    "V2": 3.15,
    "V3": -5.09,
    "V4": 3.99,
    "V5": -3.20,
    "V6": 0.31,
    "V7": -1.11,
    "V8": 0.41,
    "V9": 1.02,
    "V10": -5.56,
    "V11": 3.01,
    "V12": -9.25,
    "V13": 1.01,
    "V14": -16.60,
    "V15": -0.34,
    "V16": -7.17,
    "V17": -8.49,
    "V18": -3.09,
    "V19": 0.79,
    "V20": 0.88,
    "V21": 0.58,
    "V22": -0.19,
    "V23": 0.12,
    "V24": 0.37,
    "V25": -0.32,
    "V26": 0.46,
    "V27": 0.29,
    "V28": 0.14,
    "Amount": 312.70,
    "Time": 9600.0,
}


def _jitter(template: dict[str, float], scale: float = 0.15) -> dict[str, float]:
    """Add small random noise to each feature so transactions look distinct."""
    rng = random.Random()
    return {
        k: round(v + rng.gauss(0, abs(v) * scale + 0.01), 4)
        for k, v in template.items()
    }


def build_request(transaction_id: str, is_fraud: bool) -> dict[str, Any]:
    template = FRAUD_TEMPLATE if is_fraud else LEGIT_TEMPLATE
    return {
        "transaction_id": transaction_id,
        "features": _jitter(template),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--n", type=int, default=120, help="Total requests to send (default 120)"
    )
    parser.add_argument(
        "--fraud-rate",
        type=float,
        default=0.15,
        help="Fraction of requests that are fraud-pattern (default 0.15)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between requests (default 2.0). "
        "Must be >0 so requests span multiple Prometheus scrape "
        "intervals — rate() needs at least 2 scrape points to be non-NaN.",
    )
    args = parser.parse_args()

    predict_url = f"{args.url}/predict"
    rng = random.Random(42)
    total_seconds = args.n * args.delay

    print(f"Sending {args.n} requests to {predict_url} …")
    print(f"  Fraud-pattern rate: {args.fraud_rate:.0%}")
    print(f"  Delay between requests: {args.delay}s")
    print(
        f"  Estimated duration: {total_seconds/60:.1f} min — Grafana panels populate live"
    )
    print()

    first_response: dict[str, Any] | None = None
    errors = 0

    for i in range(args.n):
        tid = str(uuid.uuid4())
        is_fraud = rng.random() < args.fraud_rate
        payload = build_request(tid, is_fraud)

        try:
            resp = requests.post(predict_url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            label = "FRAUD" if data["is_fraud"] else "legit"
            model = data.get("model_name", "?")
            prob = data.get("fraud_probability", 0.0)
            print(
                f"  [{i+1:3d}/{args.n}] {label:5s}  p={prob:.3f}  model={model}  id={tid[:8]}"
            )

            if first_response is None:
                first_response = data
                print()
                print("=" * 60)
                print("SAMPLE /predict RESPONSE (screenshot this now):")
                print("=" * 60)
                print(json.dumps(first_response, indent=2, default=str))
                print("=" * 60)
                print()

        except Exception as exc:
            print(f"  [{i+1:3d}/{args.n}] ERROR: {exc}")
            errors += 1

        time.sleep(args.delay)

    print()
    print(f"Done. {args.n - errors}/{args.n} succeeded, {errors} errors.")
    print("Wait ~15s then refresh Grafana — all four panels should now show data.")


if __name__ == "__main__":
    main()
