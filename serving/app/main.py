"""FastAPI application entry point for the fraud detection serving service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import mlflow
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from serving.app.config import get_settings
from serving.app.models.explainer import get_explainer
from serving.app.models.loader import get_registry
from serving.app.routes.health import router as health_router
from serving.app.routes.models import router as models_router
from serving.app.routes.predict import router as predict_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info("Loading models from MLflow at %s …", settings.mlflow_tracking_uri)
    get_registry().load(settings)
    # Warm up SHAP explainer so the first /predict request isn't slow
    get_explainer()
    logger.info("Startup complete")
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with A/B testing and SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

# Prometheus metrics exposed at GET /metrics
Instrumentator().instrument(app).expose(app)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(models_router)
