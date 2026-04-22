"""MLflow model loading and feature preparation for the serving layer.

Two models are loaded at startup:
  - champion: XGBoost classifier (mlflow.xgboost format + separate scaler artifact)
  - challenger: PyTorch Autoencoder wrapped as MLflow pyfunc (self-contained)

Feature engineering replicates airflow/plugins/feature_engineering.py exactly so
that single-transaction requests receive the same transforms as batch training data.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler

from serving.app.config import Settings
from serving.app.schemas import TransactionFeatures

logger = logging.getLogger(__name__)

# Feature column order must match training/train_xgboost.py exactly.
FEATURE_COLS: list[str] = [f"V{i}" for i in range(1, 29)] + [
    "amount_log",
    "amount_zscore",
    "hour_of_day",
    "is_night",
    "v1_v2_interaction",
]


class ModelRegistry:
    """Loads and caches champion + challenger models from the MLflow registry."""

    def __init__(self) -> None:
        self._xgb_model: Any = None
        self._xgb_scaler: StandardScaler | None = None
        self._xgb_threshold: float = 0.5
        self._xgb_version: str = "unknown"
        self._xgb_name: str = "fraud-xgboost"
        self._xgb_metrics: dict[str, float] = {}

        self._ae_model: mlflow.pyfunc.PyFuncModel | None = None
        self._ae_version: str = "unknown"
        self._ae_name: str = "fraud-autoencoder"
        self._ae_metrics: dict[str, float] = {}

        self._champion_alias: str = "champion"
        self._challenger_alias: str = "challenger"

    # ------------------------------------------------------------------
    # Startup loading
    # ------------------------------------------------------------------

    def load(self, settings: Settings) -> None:
        """Load champion and challenger from the MLflow registry.

        Errors are logged but not re-raised so the API can start in a
        degraded state and return 503 from /health when models are missing.
        """
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient()

        self._xgb_name = settings.model_xgboost_name
        self._ae_name = settings.model_autoencoder_name
        self._champion_alias = settings.model_champion_alias
        self._challenger_alias = settings.model_challenger_alias

        self._load_xgboost(client, settings)
        self._load_autoencoder(client, settings)

    def _load_xgboost(self, client: MlflowClient, settings: Settings) -> None:
        try:
            uri = (
                f"models:/{settings.model_xgboost_name}@{settings.model_champion_alias}"
            )
            self._xgb_model = mlflow.xgboost.load_model(uri)

            # Retrieve run id so we can fetch the scaler artifact and metrics
            mv = client.get_model_version_by_alias(
                settings.model_xgboost_name, settings.model_champion_alias
            )
            self._xgb_version = mv.version
            run_id = mv.run_id

            # Download and unpickle the scaler logged by train_xgboost.py
            with tempfile.TemporaryDirectory() as tmp:
                local_path = client.download_artifacts(run_id, "scaler/scaler.pkl", tmp)
                with open(local_path, "rb") as f:
                    self._xgb_scaler = pickle.load(f)

            # Load the optimal classification threshold from run metrics
            run_data = client.get_run(run_id).data
            self._xgb_threshold = run_data.metrics.get("threshold", 0.5)
            self._xgb_metrics = {
                k: run_data.metrics[k]
                for k in ("auc_roc", "pr_auc", "f1")
                if k in run_data.metrics
            }

            logger.info(
                "XGBoost champion v%s loaded (threshold=%.4f)",
                self._xgb_version,
                self._xgb_threshold,
            )
        except Exception:
            logger.exception(
                "Failed to load XGBoost champion — API will start degraded"
            )

    def _load_autoencoder(self, client: MlflowClient, settings: Settings) -> None:
        try:
            uri = f"models:/{settings.model_autoencoder_name}@{settings.model_challenger_alias}"
            self._ae_model = mlflow.pyfunc.load_model(uri)

            mv = client.get_model_version_by_alias(
                settings.model_autoencoder_name, settings.model_challenger_alias
            )
            self._ae_version = mv.version
            run_data = client.get_run(mv.run_id).data
            self._ae_metrics = {
                k: run_data.metrics[k]
                for k in ("auc_roc", "pr_auc", "f1")
                if k in run_data.metrics
            }

            logger.info("Autoencoder challenger v%s loaded", self._ae_version)
        except Exception:
            logger.exception(
                "Failed to load Autoencoder challenger — API will start degraded"
            )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def prepare_features(self, features: TransactionFeatures) -> pd.DataFrame:
        """Convert a single TransactionFeatures into a model-ready DataFrame.

        Replicates airflow/plugins/feature_engineering.py transforms:
          - amount_log:       log1p(Amount)
          - hour_of_day:      int(Time // 3600 % 24)
          - is_night:         hour_of_day >= 22 or hour_of_day < 6
          - amount_zscore:    0.0 for single rows (sigma = 0 in a 1-row batch)
          - v1_v2_interaction: V1 * V2
        """
        raw = features.model_dump()
        amount_log = float(np.log1p(raw["Amount"]))
        hour_of_day = int(raw["Time"] // 3600 % 24)
        is_night = float(hour_of_day >= 22 or hour_of_day < 6)
        amount_zscore = 0.0  # single-row; std = 0 → matches pipeline behaviour
        v1_v2_interaction = raw["V1"] * raw["V2"]

        row: dict[str, float] = {f"V{i}": raw[f"V{i}"] for i in range(1, 29)}
        row["amount_log"] = amount_log
        row["amount_zscore"] = amount_zscore
        row["hour_of_day"] = float(hour_of_day)
        row["is_night"] = is_night
        row["v1_v2_interaction"] = v1_v2_interaction

        return pd.DataFrame([row])[FEATURE_COLS]

    def prepare_features_batch(
        self, features_list: list[TransactionFeatures]
    ) -> pd.DataFrame:
        """Prepare a batch of transactions, computing batch-level amount_zscore."""
        rows = []
        for features in features_list:
            raw = features.model_dump()
            hour_of_day = int(raw["Time"] // 3600 % 24)
            rows.append(
                {
                    **{f"V{i}": raw[f"V{i}"] for i in range(1, 29)},
                    "Amount": raw["Amount"],
                    "hour_of_day": float(hour_of_day),
                    "is_night": float(hour_of_day >= 22 or hour_of_day < 6),
                    "v1_v2_interaction": raw["V1"] * raw["V2"],
                }
            )
        df = pd.DataFrame(rows)
        df["amount_log"] = np.log1p(df["Amount"])
        mu = df["Amount"].mean()
        sigma = df["Amount"].std(ddof=0)
        df["amount_zscore"] = (df["Amount"] - mu) / sigma if sigma > 0 else 0.0
        return df[FEATURE_COLS]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_xgb(self, df: pd.DataFrame) -> tuple[float, bool]:
        """Scale features and return (fraud_probability, is_fraud)."""
        assert self._xgb_model is not None and self._xgb_scaler is not None
        X_scaled = self._xgb_scaler.transform(df.values)
        proba = float(self._xgb_model.predict_proba(X_scaled)[0, 1])
        return proba, proba >= self._xgb_threshold

    def predict_ae(self, df: pd.DataFrame) -> tuple[float, bool]:
        """Run the autoencoder pyfunc and return (fraud_probability, is_fraud)."""
        assert self._ae_model is not None
        result = self._ae_model.predict(df)
        proba = float(result["fraud_probability"].iloc[0])
        return proba, proba >= 0.5

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def xgb_loaded(self) -> bool:
        return self._xgb_model is not None and self._xgb_scaler is not None

    @property
    def ae_loaded(self) -> bool:
        return self._ae_model is not None


# Module-level singleton — populated during FastAPI startup.
_registry: ModelRegistry = ModelRegistry()


def get_registry() -> ModelRegistry:
    return _registry
