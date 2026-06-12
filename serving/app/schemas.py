"""Pydantic request/response schemas for the serving API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class TransactionFeatures(BaseModel):
    """Raw transaction features sent by the client.

    V1–V28 are PCA-transformed features from the dataset.
    Amount and Time are the original transaction fields.
    Time is required: hour_of_day and is_night are derived from it, and a
    silent default would put every timestamp-less transaction at midnight
    (which counts as night) — a distorted feature the model would trust.
    """

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Time: float


class TransactionRequest(BaseModel):
    transaction_id: str
    features: TransactionFeatures


class BatchRequest(BaseModel):
    transactions: Annotated[
        list[TransactionRequest], Field(min_length=1, max_length=1000)
    ]


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class Explanation(BaseModel):
    top_features: list[FeatureContribution]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    model_name: str
    model_version: str
    explanation: Explanation
    latency_ms: float
    timestamp: datetime


class BatchPredictionItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    model_name: str
    model_version: str
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[BatchPredictionItem]
    count: int
    total_latency_ms: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Health / model-info response models
# ---------------------------------------------------------------------------


class LoadedModelInfo(BaseModel):
    name: str
    version: str
    status: str  # "loaded" | "unavailable"


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded"
    models: dict[str, LoadedModelInfo]
    ab_test: dict[str, float]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    role: str  # "champion" | "challenger"
    traffic_percentage: int
    metrics: dict[str, float]


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
