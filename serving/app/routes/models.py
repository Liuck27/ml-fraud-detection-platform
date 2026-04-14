"""Model info endpoint: GET /models."""

from __future__ import annotations

from fastapi import APIRouter

from serving.app.config import get_settings
from serving.app.models.loader import get_registry
from serving.app.schemas import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    registry = get_registry()
    settings = get_settings()

    champion_pct = round((1.0 - settings.ab_challenger_fraction) * 100)
    challenger_pct = 100 - champion_pct

    models: list[ModelInfo] = []

    if registry.xgb_loaded:
        models.append(
            ModelInfo(
                name=registry._xgb_name,
                version=registry._xgb_version,
                role=registry._champion_alias,
                traffic_percentage=champion_pct,
                metrics=registry._xgb_metrics,
            )
        )

    if registry.ae_loaded:
        models.append(
            ModelInfo(
                name=registry._ae_name,
                version=registry._ae_version,
                role=registry._challenger_alias,
                traffic_percentage=challenger_pct,
                metrics=registry._ae_metrics,
            )
        )

    return ModelsResponse(models=models)
