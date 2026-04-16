"""SHAP-based feature attribution for XGBoost predictions.

Uses TreeExplainer (exact, fast) to compute per-feature contributions for a
single scaled input row. Only applies to the XGBoost champion — the autoencoder
challenger returns an empty explanation list.
"""

from __future__ import annotations

import logging

import numpy as np
import shap

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Wraps shap.TreeExplainer for XGBoost to return top-k contributions."""

    def __init__(self, model: object, feature_names: list[str]) -> None:
        self._explainer = shap.TreeExplainer(model)
        self._feature_names = feature_names

    def explain(
        self, X_scaled: np.ndarray, top_k: int = 3
    ) -> list[dict[str, float | str]]:
        """Return top-k feature contributions for a single (1-row) scaled array.

        Returns a list of dicts with keys 'feature' and 'contribution', sorted
        by absolute contribution descending.
        """
        shap_values = self._explainer.shap_values(X_scaled)
        # Binary classification: shap_values is shape (1, n_features)
        contributions: np.ndarray = np.asarray(shap_values)[0]
        pairs = sorted(
            zip(self._feature_names, contributions),
            key=lambda x: abs(float(x[1])),
            reverse=True,
        )
        return [
            {"feature": str(f), "contribution": round(float(c), 4)}
            for f, c in pairs[:top_k]
        ]


# Module-level singleton — created lazily after the registry is populated.
_explainer: SHAPExplainer | None = None


def get_explainer() -> SHAPExplainer | None:
    """Return the SHAP explainer, creating it on first call."""
    global _explainer
    if _explainer is None:
        from serving.app.models.loader import FEATURE_COLS, get_registry

        registry = get_registry()
        if not registry.xgb_loaded:
            logger.warning("XGBoost model not loaded — SHAP explainer unavailable")
            return None
        try:
            _explainer = SHAPExplainer(registry._xgb_model, FEATURE_COLS)
            logger.info("SHAP TreeExplainer initialised")
        except Exception:
            logger.exception("Failed to initialise SHAP explainer")
    return _explainer
