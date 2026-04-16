"""Tests for prediction endpoints: POST /predict and POST /predict/batch."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from serving.tests.conftest import SAMPLE_FEATURES


def _txn(transaction_id: str | None = None, features: dict | None = None) -> dict:
    return {
        "transaction_id": transaction_id or str(uuid.uuid4()),
        "features": features or SAMPLE_FEATURES,
    }


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


def test_predict_happy_path(client: TestClient) -> None:
    resp = client.post("/predict", json=_txn())
    assert resp.status_code == 200
    body = resp.json()
    assert "fraud_probability" in body
    assert isinstance(body["fraud_probability"], float)
    assert "is_fraud" in body
    assert isinstance(body["is_fraud"], bool)
    assert "model_name" in body
    assert "model_version" in body
    assert "explanation" in body
    assert "top_features" in body["explanation"]
    assert "latency_ms" in body
    assert "timestamp" in body


def test_predict_response_schema(client: TestClient) -> None:
    resp = client.post("/predict", json=_txn(transaction_id="fixed-id-abc"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["transaction_id"] == "fixed-id-abc"
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_missing_feature_returns_422(client: TestClient) -> None:
    incomplete = {k: v for k, v in SAMPLE_FEATURES.items() if k != "V1"}
    resp = client.post("/predict", json=_txn(features=incomplete))
    assert resp.status_code == 422


def test_predict_missing_body_returns_422(client: TestClient) -> None:
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------


def test_predict_batch_happy_path(client: TestClient) -> None:
    payload = {"transactions": [_txn(), _txn()]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert "total_latency_ms" in body
    assert "timestamp" in body


def test_predict_batch_single_item(client: TestClient) -> None:
    payload = {"transactions": [_txn()]}
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    assert resp.json()["count"] == 1


def test_predict_batch_empty_returns_422(client: TestClient) -> None:
    resp = client.post("/predict/batch", json={"transactions": []})
    assert resp.status_code == 422


def test_predict_batch_over_limit_returns_422(client: TestClient) -> None:
    transactions = [_txn() for _ in range(1001)]
    resp = client.post("/predict/batch", json={"transactions": transactions})
    assert resp.status_code == 422


def test_predict_batch_item_schema(client: TestClient) -> None:
    t1 = _txn(transaction_id="batch-id-1")
    t2 = _txn(transaction_id="batch-id-2")
    resp = client.post("/predict/batch", json={"transactions": [t1, t2]})
    assert resp.status_code == 200
    ids = [p["transaction_id"] for p in resp.json()["predictions"]]
    assert "batch-id-1" in ids
    assert "batch-id-2" in ids
