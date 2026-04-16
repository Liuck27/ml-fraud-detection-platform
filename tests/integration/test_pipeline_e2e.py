"""End-to-end integration tests for the fraud detection pipeline.

Requires all Docker Compose services to be running:
    docker compose up -d

Tests are automatically skipped (not failed) when services are unreachable,
so they never break CI — integration tests only run via `make test-integration`.

What is verified:
  1. POST /predict returns a valid, schema-complete response
  2. GET /metrics exposes the expected Prometheus counters after a prediction
  3. MLflow registry has a 'champion' alias for the XGBoost model
"""

from __future__ import annotations

import requests


# ---------------------------------------------------------------------------
# 1. Prediction endpoint
# ---------------------------------------------------------------------------


def test_predict_returns_valid_response(
    api_url_with_models: str, sample_transaction: dict
) -> None:
    """POST /predict with a valid transaction returns a complete fraud prediction."""
    response = requests.post(
        f"{api_url_with_models}/predict",
        json=sample_transaction,
        timeout=10,
    )

    assert (
        response.status_code == 200
    ), f"Unexpected status: {response.status_code}\n{response.text}"

    body = response.json()

    # Core fields
    assert body["transaction_id"] == sample_transaction["transaction_id"]
    assert isinstance(body["fraud_probability"], float)
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert isinstance(body["model_name"], str) and body["model_name"]
    assert isinstance(body["model_version"], str) and body["model_version"]
    assert isinstance(body["latency_ms"], (int, float)) and body["latency_ms"] >= 0

    # SHAP explanation
    explanation = body.get("explanation")
    assert explanation is not None, "Response missing 'explanation' field"
    top_features = explanation.get("top_features", [])
    assert isinstance(top_features, list)
    for item in top_features:
        assert "feature" in item and "contribution" in item


# ---------------------------------------------------------------------------
# 2. Prometheus metrics endpoint
# ---------------------------------------------------------------------------


def test_metrics_endpoint_has_inference_counters(
    api_base_url: str, sample_transaction: dict
) -> None:
    """GET /metrics exposes Prometheus counters after at least one prediction."""
    # Fire a prediction to ensure counters are non-zero.
    requests.post(f"{api_base_url}/predict", json=sample_transaction, timeout=10)

    response = requests.get(f"{api_base_url}/metrics", timeout=5)
    assert response.status_code == 200

    text = response.text
    assert "inference_total" in text, "/metrics missing 'inference_total' counter"
    assert (
        "inference_latency_seconds" in text
    ), "/metrics missing 'inference_latency_seconds' histogram"


# ---------------------------------------------------------------------------
# 3. MLflow model registry
# ---------------------------------------------------------------------------


def test_mlflow_has_champion_alias(mlflow_base_url: str) -> None:
    """MLflow registry has a 'champion' alias set on the XGBoost model."""
    model_name = "fraud-xgboost"

    # Use the MLflow REST API directly — no mlflow package required in the test venv.
    url = f"{mlflow_base_url}/api/2.0/mlflow/registered-models/alias"
    response = requests.get(
        url, params={"name": model_name, "alias": "champion"}, timeout=5
    )

    assert response.status_code == 200, (
        f"MLflow returned {response.status_code} — is the model registered?\n"
        "Run `bash scripts/run_training.sh` to train and register models."
    )

    body = response.json()
    model_version = body.get("model_version", {})
    assert model_version.get("name") == model_name
    assert model_version.get("version"), "Champion alias exists but version is empty"
