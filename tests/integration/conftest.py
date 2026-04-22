"""Shared fixtures for integration tests.

These tests require all services to be running:
    docker compose up -d

Run with:
    make test-integration

Each fixture checks connectivity and skips the test if the target
service is unreachable, so a missing service doesn't fail CI with a
confusing connection error.
"""

from __future__ import annotations

import os

import pytest
import requests


def _is_reachable(url: str, timeout: float = 2.0) -> bool:
    """Return True if a GET request to *url* succeeds within *timeout* seconds."""
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------------
# Service base URLs (override via env vars if ports differ)
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
MLFLOW_BASE = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """Base URL for the FastAPI serving service (reachability check only)."""
    if not _is_reachable(f"{API_BASE}/health"):
        pytest.skip(f"FastAPI not reachable at {API_BASE} — run `make up` first")
    return API_BASE


@pytest.fixture(scope="session")
def api_url_with_models(api_base_url: str) -> str:
    """Base URL for the FastAPI service, skipped if no models are loaded.

    Models are absent when training hasn't run yet.  Fix:
        bash scripts/run_training.sh   # train + register champion/challenger
        docker compose restart serving  # reload models from MLflow
    """
    resp = requests.get(f"{api_base_url}/health", timeout=5)
    body = resp.json()
    if body.get("status") == "degraded":
        loaded = {k: v["status"] for k, v in body.get("models", {}).items()}
        pytest.skip(
            f"FastAPI is running but models are not loaded {loaded}.\n"
            "Run `bash scripts/run_training.sh` then `docker compose restart serving`."
        )
    return api_base_url


@pytest.fixture(scope="session")
def mlflow_base_url() -> str:
    """Base URL for the MLflow tracking server."""
    if not _is_reachable(MLFLOW_BASE):
        pytest.skip(f"MLflow not reachable at {MLFLOW_BASE} — run `make up` first")
    return MLFLOW_BASE


@pytest.fixture()
def sample_transaction() -> dict:
    """A complete transaction payload with all 28 V-features + Amount."""
    return {
        "transaction_id": "test-integration-00000000-0000-0000-0000-000000000001",
        "features": {
            **{f"V{i}": float(i) * 0.1 for i in range(1, 29)},
            "Amount": 149.62,
            "Time": 7200.0,
        },
    }
