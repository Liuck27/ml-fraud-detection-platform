"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Registered model names in the MLflow registry
    model_xgboost_name: str = "fraud-xgboost"
    model_autoencoder_name: str = "fraud-autoencoder"

    # Aliases used to look up the active versions
    model_champion_alias: str = "champion"
    model_challenger_alias: str = "challenger"

    # A/B split: fraction of traffic routed to the challenger (0.0 – 1.0)
    ab_challenger_fraction: float = 0.20


@lru_cache
def get_settings() -> Settings:
    return Settings()
