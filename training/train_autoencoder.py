"""PyTorch Autoencoder for unsupervised fraud anomaly detection.

Trains exclusively on legitimate (non-fraud) transactions so that fraud shows
up as high reconstruction error.  Logs to MLflow and registers the model as
'challenger' in the Model Registry.

Architecture: Input(32) → 64 → 32 → 16 → 32 → 64 → Output(32)

Run from the repo root:
    training/.venv/Scripts/python training/train_autoencoder.py   # Windows
    training/.venv/bin/python training/train_autoencoder.py       # Linux/Mac

Requires MLFLOW_TRACKING_URI env var (defaults to http://localhost:5000).
"""

from __future__ import annotations

# ruff: noqa: E402  (sys.path.insert before sibling imports is intentional)
import os
import pickle
import sys
import tempfile
from pathlib import Path

# Ensure sibling modules (evaluate, model_registry) are importable when the
# script is invoked from the repo root (e.g. via Makefile or run_training.sh).
sys.path.insert(0, str(Path(__file__).parent))

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_pr_curve,
    plot_roc_curve,
)
from model_registry import get_latest_version, promote_to_challenger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
PARQUET_PATH = REPO_ROOT / "data" / "processed" / "features.parquet"

FEATURE_COLS: list[str] = [f"V{i}" for i in range(1, 29)] + [
    "amount_log",
    "hour_of_day",
    "is_night",
    "v1_v2_interaction",
]
TARGET_COL = "Class"

MODEL_NAME = os.getenv("MODEL_AUTOENCODER_NAME", "fraud-autoencoder")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

RANDOM_STATE = 42
# Three-way split: 60% train / 20% val (threshold tuning) / 20% test (reported
# metrics). Tuning and reporting on the same split would bias metrics upward.
TEST_SIZE = 0.2
VAL_SIZE = 0.25  # fraction of the remaining 80% → 20% of the full dataset
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
HIDDEN_DIMS = [64, 32, 16]  # encoder layers; decoder mirrors these


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class FraudAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        # Encoder: input_dim → hidden_dims[-1]
        encoder_layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: mirrors the encoder
        decoder_layers: list[nn.Module] = []
        for h in reversed(hidden_dims[:-1]):
            decoder_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# MLflow pyfunc wrapper
# ---------------------------------------------------------------------------


class AutoencoderPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper that exposes the autoencoder as a fraud scorer.

    Input:  pd.DataFrame with columns matching FEATURE_COLS
    Output: pd.DataFrame with 'fraud_probability' and 'reconstruction_error'
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        def artifact(key: str) -> str:
            # MLflow 2.9 on Windows records the artifact map with os.path.join,
            # so a model trained on Windows hands a Linux loader paths like
            # "artifacts\model.pt". The file itself lands at artifacts/model.pt,
            # only the separator in the string is wrong — normalise it.
            return context.artifacts[key].replace("\\", "/")

        self.model: FraudAutoencoder = torch.jit.load(  # type: ignore[assignment]
            artifact("model_torchscript")
        )
        self.model.eval()
        with open(artifact("scaler_pkl"), "rb") as scaler_file:
            self.scaler: StandardScaler = pickle.load(scaler_file)
        with open(artifact("threshold_txt")) as threshold_file:
            self.threshold = float(threshold_file.read().strip())

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        # Pass the DataFrame, not .values: the scaler was fit on a DataFrame
        # and warns about missing feature names when given a bare array.
        X_scaled = self.scaler.transform(model_input)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            X_recon = self.model(X_tensor).numpy()
        errors = np.mean((X_scaled - X_recon) ** 2, axis=1)
        # Normalise reconstruction error to a [0, 1] fraud probability.
        # Errors above 2× threshold are clipped to 1.
        proba = np.clip(errors / (self.threshold * 2), 0.0, 1.0)
        return pd.DataFrame(
            {"fraud_probability": proba, "reconstruction_error": errors}
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Features parquet not found at {PARQUET_PATH}. "
            "Run the Airflow data_ingestion_dag first."
        )
    df = pd.read_parquet(PARQUET_PATH)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    print(f"Loaded {len(df):,} rows | fraud: {y.sum():,} ({y.mean()*100:.3f}%)")
    return X, y


def train_autoencoder(
    X_train: np.ndarray,
    input_dim: int,
) -> FraudAutoencoder:
    model = FraudAutoencoder(input_dim=input_dim, hidden_dims=HIDDEN_DIMS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / len(X_train)
            print(f"  epoch {epoch+1:3d}/{EPOCHS}  loss={avg:.6f}")
    return model


def reconstruction_errors(model: FraudAutoencoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        X_recon = model(X_tensor).numpy()
    return np.mean((X - X_recon) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("fraud-detection-autoencoder")

    X, y = load_data()

    # Split off the held-out test set first, then carve val out of the rest.
    X_rest_df, X_test_df, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_rest_df,
        y_rest,
        test_size=VAL_SIZE,
        stratify=y_rest,
        random_state=RANDOM_STATE,
    )

    # Scaler fit on non-fraud training data only — mirrors what the model learns
    mask_legit = y_train == 0
    X_train_legit = X_train_df[mask_legit]

    scaler = StandardScaler()
    scaler.fit(X_train_legit)

    X_train_legit_scaled = scaler.transform(X_train_legit)
    X_val_scaled = scaler.transform(X_val_df)
    X_test_scaled = scaler.transform(X_test_df)

    input_dim = X_train_legit_scaled.shape[1]
    print(f"Training autoencoder on {len(X_train_legit_scaled):,} legit transactions…")

    with mlflow.start_run() as run:
        model = train_autoencoder(X_train_legit_scaled, input_dim)

        # Normalise errors to [0, 1] for threshold search via PR curve.
        # We divide by the 99th-percentile error on legitimate val samples
        # to get a stable denominator that isn't pulled up by fraud outliers.
        val_errors = reconstruction_errors(model, X_val_scaled)
        legit_errors = val_errors[y_val.values == 0]
        p99_legit = float(np.percentile(legit_errors, 99))
        val_scores = np.clip(val_errors / (p99_legit * 2 + 1e-8), 0.0, 1.0)

        # Threshold tuned on val; reported metrics come from the untouched
        # test set below (scored with the same frozen val-derived denominator).
        threshold_score = find_optimal_threshold(y_val.values, val_scores)
        # Convert back to a raw-error threshold for the pyfunc wrapper
        threshold_error = threshold_score * (p99_legit * 2)

        test_errors = reconstruction_errors(model, X_test_scaled)
        test_scores = np.clip(test_errors / (p99_legit * 2 + 1e-8), 0.0, 1.0)
        metrics = compute_metrics(y_test.values, test_scores, threshold=threshold_score)

        # Log hyperparameters
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "hidden_dims": str(HIDDEN_DIMS),
                "input_dim": input_dim,
                "threshold_error": round(threshold_error, 6),
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE,
            }
        )
        mlflow.log_metrics(metrics)

        # Log evaluation plots (test set — same data the metrics report on)
        roc_fig = plot_roc_curve(
            y_test.values, test_scores, title="Autoencoder ROC Curve"
        )
        pr_fig = plot_pr_curve(y_test.values, test_scores, title="Autoencoder PR Curve")
        mlflow.log_figure(roc_fig, "roc_curve.png")
        mlflow.log_figure(pr_fig, "pr_curve.png")
        import matplotlib.pyplot as plt

        plt.close("all")

        # Persist artifacts to a temp dir for the pyfunc log_model call.
        # Use Path.as_posix() so MLflow records forward-slash paths in MLmodel —
        # without this, os.path.join on Windows embeds backslashes that break
        # loading inside the Linux serving container.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ts_path = (tmp / "model.pt").as_posix()
            scaler_path = (tmp / "scaler.pkl").as_posix()
            threshold_path = (tmp / "threshold.txt").as_posix()

            scripted = torch.jit.script(model)
            torch.jit.save(scripted, ts_path)

            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

            with open(threshold_path, "w") as f:
                f.write(str(threshold_error))

            artifacts = {
                "model_torchscript": ts_path,
                "scaler_pkl": scaler_path,
                "threshold_txt": threshold_path,
            }

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AutoencoderPyfunc(),
                artifacts=artifacts,
                registered_model_name=MODEL_NAME,
            )

        print(
            f"\nRun {run.info.run_id[:8]}…  "
            f"AUC-ROC={metrics['auc_roc']:.4f}  "
            f"PR-AUC={metrics['pr_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"threshold_error={threshold_error:.4f}"
        )

    # Promote the newly registered version to challenger — unconditionally, and
    # deliberately so: 'challenger' means "the latest candidate under evaluation",
    # so it always points at the newest version. Only the 'champion' alias is
    # protected by a quality gate (see model_registry.promote_champion_if_better).
    promote_to_challenger(MODEL_NAME, get_latest_version(MODEL_NAME))


if __name__ == "__main__":
    main()
