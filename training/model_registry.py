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


def promote_champion_if_better(
    model_name: str,
    version: str | int | None = None,
    metric: str = "pr_auc",
) -> bool:
    """Promote *version* (default: latest) to 'champion' only if it improves
    on the current champion's *metric*. Returns True if promotion happened.

    This is the quality gate: training scripts only register new versions,
    and promotion is a separate, deliberate step — either this function
    (via scripts/promote_model.py) or the retrain DAG's evaluate_and_promote
    task, which applies the same rule.

    Falls back to unconditional promotion when no champion alias exists yet
    (the first ever training run), so a fresh setup still ends with a
    servable champion.
    """
    client = MlflowClient()
    if version is None:
        version = get_latest_version(model_name)
    candidate = client.get_model_version(model_name, str(version))
    cand_metric: float = client.get_run(candidate.run_id).data.metrics.get(metric, 0.0)

    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
    except Exception:  # MLflow raises RestException when the alias doesn't exist
        print(f"[registry] no champion for '{model_name}' yet — promoting v{version}")
        promote_to_champion(model_name, version)
        return True

    if str(champ.version) == str(version):
        print(f"[registry] v{version} is already champion — nothing to do")
        return False

    champ_metric: float = client.get_run(champ.run_id).data.metrics.get(metric, 0.0)

    if cand_metric >= champ_metric:
        print(
            f"[registry] v{version} {metric}={cand_metric:.4f} >= "
            f"champion v{champ.version} {metric}={champ_metric:.4f} — promoting"
        )
        promote_to_champion(model_name, version)
        return True

    print(
        f"[registry] v{version} {metric}={cand_metric:.4f} < "
        f"champion v{champ.version} {metric}={champ_metric:.4f} — keeping champion"
    )
    return False
