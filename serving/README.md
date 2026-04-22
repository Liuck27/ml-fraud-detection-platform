# Serving

FastAPI model serving with real-time and batch inference, A/B testing between models.

## Endpoints
- `POST /predict` — single transaction inference
- `POST /predict/batch` — batch inference (up to 1000 transactions)
- `GET /health` — health check with model status
- `GET /models` — loaded models and A/B configuration
- `GET /metrics` — Prometheus metrics

## Local Dev

```bash
# From project root
make venv-serving

# Run the API locally (requires MLflow running — Phase 6)
serving/.venv/Scripts/python -m uvicorn serving.app.main:app --reload --port 8000

# Run tests
make test-serving
```

## Environment
Requires `MLFLOW_TRACKING_URI`, `AB_CHALLENGER_FRACTION`, `MODEL_CHAMPION_ALIAS` from `.env`.
