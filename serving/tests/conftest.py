"""pytest fixtures for serving tests.

Tests run without a live MLflow server by patching the ModelRegistry singleton
with a fake implementation that returns deterministic predictions.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving.app.main import app
from serving.app.models import loader as loader_module
from serving.app.models import explainer as explainer_module


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

# A complete set of 28 V-features + Amount (no fraud signal)
SAMPLE_FEATURES: dict[str, float] = {
    **{f"V{i}": float(i) * 0.1 for i in range(1, 29)},
    "Amount": 100.0,
    "Time": 7200.0,  # 2 hours → hour_of_day=2, is_night=False
}


def _make_registry(xgb_loaded: bool = True, ae_loaded: bool = True) -> Any:
    """Return a ModelRegistry-like object with pre-configured mock models."""
    reg = MagicMock()

    # Metadata
    reg._xgb_name = "fraud-xgboost"
    reg._xgb_version = "1"
    reg._xgb_threshold = 0.5
    reg._champion_alias = "champion"
    reg._xgb_metrics = {"auc_roc": 0.98, "pr_auc": 0.85, "f1": 0.87}

    reg._ae_name = "fraud-autoencoder"
    reg._ae_version = "1"
    reg._challenger_alias = "challenger"
    reg._ae_metrics = {"auc_roc": 0.95, "pr_auc": 0.74, "f1": 0.78}

    reg.xgb_loaded = xgb_loaded
    reg.ae_loaded = ae_loaded

    # Feature preparation — use real logic so we test the pipeline
    from serving.app.models.loader import ModelRegistry

    real = ModelRegistry()
    reg.prepare_features = real.prepare_features
    reg.prepare_features_batch = real.prepare_features_batch

    # Prediction stubs
    reg.predict_xgb.return_value = (0.03, False)
    reg.predict_ae.return_value = (0.07, False)

    # Scaler stub for SHAP path
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 33))
    reg._xgb_scaler = mock_scaler

    return reg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_registry(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch the global registry singleton with a fake, MLflow-free version."""
    fake = _make_registry()
    monkeypatch.setattr(loader_module, "_registry", fake)
    # Also patch the explainer to avoid SHAP initialisation
    monkeypatch.setattr(explainer_module, "_explainer", None)
    monkeypatch.setattr(
        explainer_module,
        "get_explainer",
        lambda: None,
    )
    return fake


@pytest.fixture()
def client(mock_registry: Any) -> TestClient:
    """TestClient with mocked models — no MLflow required."""
    return TestClient(app, raise_server_exceptions=True)
