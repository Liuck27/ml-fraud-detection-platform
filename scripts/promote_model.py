"""Gated champion promotion for the XGBoost fraud model.

Compares the latest registered version's PR-AUC against the current champion's
and moves the 'champion' alias only if it is at least as good (or if no champion
exists yet). Training scripts deliberately never touch the alias themselves —
this script (or the retrain DAG's evaluate_and_promote task) is the only path
to production.

Usage (uses the training venv, which has mlflow installed):
    make promote
    # or directly:
    training/.venv/Scripts/python scripts/promote_model.py   # Windows
    training/.venv/bin/python scripts/promote_model.py       # Linux/Mac

Requires MLFLOW_TRACKING_URI env var (defaults to http://localhost:5000).
"""

from __future__ import annotations

# ruff: noqa: E402  (sys.path.insert before the model_registry import is intentional)
import argparse
import os
import sys
from pathlib import Path

# Make training/model_registry.py importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

import mlflow

from model_registry import promote_champion_if_better


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_XGBOOST_NAME", "fraud-xgboost"),
        help="Registered model name in MLflow (default: fraud-xgboost)",
    )
    parser.add_argument(
        "--metric",
        default="pr_auc",
        help="Run metric the gate compares on (default: pr_auc)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    # "Kept the existing champion" is a successful gate decision, not a
    # failure, so the exit code is 0 either way.
    promote_champion_if_better(args.model_name, metric=args.metric)


if __name__ == "__main__":
    main()
