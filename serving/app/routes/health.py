"""Health check endpoint: GET /health."""

from __future__ import annotations

from fastapi import APIRouter

from serving.app.config import get_settings
from serving.app.models.loader import get_registry
from serving.app.schemas import HealthResponse, LoadedModelInfo

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    registry = get_registry()
    settings = get_settings()

    champion_info = LoadedModelInfo(
        name=registry._xgb_name,
        version=registry._xgb_version,
        status="loaded" if registry.xgb_loaded else "unavailable",
    )
    challenger_info = LoadedModelInfo(
        name=registry._ae_name,
        version=registry._ae_version,
        status="loaded" if registry.ae_loaded else "unavailable",
    )

    all_loaded = registry.xgb_loaded and registry.ae_loaded

    return HealthResponse(
        status="healthy" if all_loaded else "degraded",
        models={
            "champion": champion_info,
            "challenger": challenger_info,
        },
        ab_test={
            "champion_traffic": round(1.0 - settings.ab_challenger_fraction, 2),
            "challenger_traffic": round(settings.ab_challenger_fraction, 2),
        },
    )
