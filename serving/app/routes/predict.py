"""Prediction endpoints: POST /predict and POST /predict/batch."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException

from serving.app.config import get_settings
from serving.app.models.ab_testing import route_to_challenger
from serving.app.models.explainer import get_explainer
from serving.app.models.loader import ModelRegistry, get_registry
from serving.app.schemas import (
    BatchPredictionItem,
    BatchRequest,
    BatchResponse,
    Explanation,
    FeatureContribution,
    PredictionResponse,
    TransactionRequest,
)

router = APIRouter()


def _shap_explanation(registry: ModelRegistry, df: pd.DataFrame) -> Explanation:
    """Return SHAP top-features for an XGBoost prediction; empty if unavailable."""
    explainer = get_explainer()
    if explainer is None or registry._xgb_scaler is None:
        return Explanation(top_features=[])
    X_scaled = registry._xgb_scaler.transform(df.values)
    contributions = explainer.explain(X_scaled)
    return Explanation(
        top_features=[FeatureContribution(**c) for c in contributions]
    )


def _select_model(
    transaction_id: str,
    registry: ModelRegistry,
    challenger_fraction: float,
) -> bool:
    """Return True if the transaction should go to the challenger model."""
    use_challenger = route_to_challenger(transaction_id, challenger_fraction)
    # Fallback: use whichever model is available
    if use_challenger and not registry.ae_loaded:
        use_challenger = False
    if not use_challenger and not registry.xgb_loaded:
        use_challenger = True
    return use_challenger


@router.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    registry = get_registry()
    settings = get_settings()

    if not registry.xgb_loaded and not registry.ae_loaded:
        raise HTTPException(status_code=503, detail="No models available")

    use_challenger = _select_model(
        request.transaction_id, registry, settings.ab_challenger_fraction
    )

    t0 = time.perf_counter()
    df = registry.prepare_features(request.features)

    if use_challenger:
        proba, is_fraud = registry.predict_ae(df)
        model_name = f"{registry._ae_name}-{registry._challenger_alias}"
        model_version = registry._ae_version
        explanation = Explanation(top_features=[])
    else:
        proba, is_fraud = registry.predict_xgb(df)
        model_name = f"{registry._xgb_name}-{registry._champion_alias}"
        model_version = registry._xgb_version
        explanation = _shap_explanation(registry, df)

    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        transaction_id=request.transaction_id,
        fraud_probability=round(proba, 6),
        is_fraud=is_fraud,
        model_name=model_name,
        model_version=model_version,
        explanation=explanation,
        latency_ms=round(latency_ms, 3),
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    registry = get_registry()
    settings = get_settings()

    if not registry.xgb_loaded and not registry.ae_loaded:
        raise HTTPException(status_code=503, detail="No models available")

    t_total_start = time.perf_counter()
    predictions: list[BatchPredictionItem] = []

    for txn in request.transactions:
        use_challenger = _select_model(
            txn.transaction_id, registry, settings.ab_challenger_fraction
        )

        t0 = time.perf_counter()
        df = registry.prepare_features(txn.features)

        if use_challenger:
            proba, is_fraud = registry.predict_ae(df)
            model_name = f"{registry._ae_name}-{registry._challenger_alias}"
            model_version = registry._ae_version
        else:
            proba, is_fraud = registry.predict_xgb(df)
            model_name = f"{registry._xgb_name}-{registry._champion_alias}"
            model_version = registry._xgb_version

        latency_ms = (time.perf_counter() - t0) * 1000

        predictions.append(
            BatchPredictionItem(
                transaction_id=txn.transaction_id,
                fraud_probability=round(proba, 6),
                is_fraud=is_fraud,
                model_name=model_name,
                model_version=model_version,
                latency_ms=round(latency_ms, 3),
            )
        )

    total_latency_ms = (time.perf_counter() - t_total_start) * 1000

    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        total_latency_ms=round(total_latency_ms, 3),
        timestamp=datetime.now(timezone.utc),
    )
