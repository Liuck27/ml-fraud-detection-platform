# Training

ML model training scripts for the fraud detection platform.

## Models
- `train_xgboost.py` — XGBoost binary classifier (Phase 4)
- `train_autoencoder.py` — PyTorch autoencoder for anomaly detection (Phase 5)
- `train_isolation_forest.py` — Isolation Forest for anomaly clustering (Phase 4)
- `evaluate.py` — shared evaluation utilities (metrics, plots, thresholds)
- `model_registry.py` — MLflow registry helpers (promote, demote, compare)

## Local Dev

```bash
# From project root
make venv-training

# Run training (requires data in data/raw/creditcard.csv — see Phase 2)
training/.venv/Scripts/python training/train_xgboost.py

# Run tests
make test-training
```

## Environment
Requires `MLFLOW_TRACKING_URI` from `.env`.
Requires dataset at `data/raw/creditcard.csv` (download in Phase 2).
