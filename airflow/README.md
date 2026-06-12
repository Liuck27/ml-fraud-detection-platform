# Airflow

Pipeline orchestration for data ingestion, feature engineering, and model retraining.

## DAGs
- `data_ingestion_dag.py` — validates the raw CSV, engineers features, writes `data/processed/features.parquet` (manual trigger)
- `retrain_dag.py` — retrains XGBoost and promotes to `champion` only if PR-AUC improved (manual trigger; in production this would run on a schedule, e.g. weekly)

The retrain DAG runs `training/train_xgboost.py` in-process: `./training` is
volume-mounted read-only into the Airflow containers (`docker-compose.yml`),
and the image (`Dockerfile` here) installs the training dependencies pinned to
`training/requirements.txt` versions — minus torch, since only XGBoost is
retrained. In production the training step would instead run in a dedicated
image via DockerOperator/KubernetesPodOperator.

## Local Dev

```bash
# From project root
make venv-airflow   # installs Airflow into airflow/.venv (~5-10 min first time)

# Start the full stack (Airflow UI at http://localhost:8080)
make up
```

## Environment
Requires `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`, `AIRFLOW__CORE__FERNET_KEY` from `.env`.
