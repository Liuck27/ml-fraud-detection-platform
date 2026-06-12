"""Gated champion promotion for the XGBoost fraud model.

Compares the latest registered version's PR-AUC against the current champion's
and moves the 'champion' alias only if it is at least as good (or if no champion
exists yet). Training scripts deliberately never touch the alias themselves —
this script (or the retrain DAG's evaluate_and_promote task) is the only path
to production.

--force skips the metric comparison and promotes the latest version
unconditionally. This exists for migrations where the gate's comparison is
meaningless: when the evaluation methodology changes (metrics from old runs
are not comparable to new ones) or the feature schema changes (the old
champion cannot be served by the new code anyway). Normal retrains must go
through the gate.

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

from model_registry import (
    get_latest_version,
    promote_champion_if_better,
    promote_to_champion,
)


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
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Bypass the metric gate and promote the latest version "
            "unconditionally (for methodology or feature-schema migrations only)"
        ),
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    if args.force:
        version = get_latest_version(args.model_name)
        print(
            f"[registry] --force: skipping the {args.metric} gate — old and new "
            "runs are not comparable (methodology or feature-schema change)"
        )
        promote_to_champion(args.model_name, version)
        return

    # "Kept the existing champion" is a successful gate decision, not a
    # failure, so the exit code is 0 either way.
    promote_champion_if_better(args.model_name, metric=args.metric)


if __name__ == "__main__":
    main()
