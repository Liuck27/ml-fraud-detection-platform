# Airflow

Pipeline orchestration for data ingestion, feature engineering, and model retraining.

## DAGs
- `data_ingestion_dag.py` — validates raw data, engineers features, materializes to Feast
- `retrain_dag.py` — triggered by drift detection; retrains and promotes models

## Local Dev

```bash
# From project root
make venv-airflow   # installs Airflow + Feast (~10 min first time)

# Start Airflow (Phase 3 — uncomment services in docker-compose.yml first)
make up
```

## Environment
Requires `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`, `AIRFLOW__CORE__FERNET_KEY` from `.env`.
