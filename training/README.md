# Training

ML model training scripts for the fraud detection platform.

## Files
- `train_xgboost.py` — XGBoost binary classifier (SMOTE, threshold calibration); registers a new version in MLflow, does NOT promote it
- `train_autoencoder.py` — PyTorch autoencoder for anomaly detection, trained on legit transactions only; registers and points the `challenger` alias at the new version
- `evaluate.py` — shared evaluation utilities (metrics, plots, threshold search)
- `model_registry.py` — MLflow registry helpers, including `promote_champion_if_better`, the quality gate that moves the `champion` alias only when PR-AUC improves

## Local Dev

```bash
# From project root
make venv-training

# Train both models, then apply the gated champion promotion
bash scripts/run_training.sh
# or individually:
make train-xgboost
make train-autoencoder
make promote

# Run tests
make test-training
```

## Environment
Requires `MLFLOW_TRACKING_URI` (defaults to `http://localhost:5000`).
Requires `data/processed/features.parquet` (produced by the Airflow `data_ingestion` DAG).
