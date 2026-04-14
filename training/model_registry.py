"""MLflow Model Registry helpers.

Thin wrappers around MlflowClient for promoting models to champion/challenger
aliases and retrieving registry metadata.
"""

from __future__ import annotations

from mlflow.tracking import MlflowClient


def promote_to_champion(model_name: str, version: str | int) -> None:
    """Set the 'champion' alias on the given registered model version."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "champion", str(version))
    print(f"[registry] {model_name} v{version} → champion")


def promote_to_challenger(model_name: str, version: str | int) -> None:
    """Set the 'challenger' alias on the given registered model version."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "challenger", str(version))
    print(f"[registry] {model_name} v{version} → challenger")


def get_latest_version(model_name: str) -> str:
    """Return the version string of the most recently registered model version."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    return max(versions, key=lambda v: int(v.version)).version


def get_champion_run_id(model_name: str) -> str:
    """Return the MLflow run_id for the version currently aliased as 'champion'."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(model_name, "champion")
    return version.run_id
