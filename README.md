# ML Fraud Detection Platform

End-to-end machine learning platform for detecting fraudulent financial transactions,
built with MLOps best practices: feature store, experiment tracking, streaming pipelines,
A/B testing, and production monitoring.

## Architecture

See `plan.md` for the full architecture diagram and implementation phases.

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | Apache Airflow 2.7, Docker Compose |
| Feature Store | Feast 0.34 + PostgreSQL 15 |
| Training | scikit-learn, XGBoost, PyTorch 2.1 |
| Experiment Tracking | MLflow 2.9 |
| Serving | FastAPI + Uvicorn |
| Streaming | Apache Kafka + Go consumer |
| Monitoring | Prometheus, Grafana, Evidently |

## Quick Start

### Prerequisites
- Docker Desktop (with Docker Compose v2)
- Python 3.11
- Go 1.21 (Phase 8+)
- Git Bash or WSL2 (Windows)

### Phase 1: Start PostgreSQL

```bash
# 1. Copy and configure environment
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD and generate AIRFLOW__CORE__FERNET_KEY:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# 2. Start PostgreSQL
make up-postgres

# 3. Verify (wait ~30s for healthy status)
make ps        # fraud_postgres: running (healthy)
make psql      # \l to list databases, \q to quit

# 4. Set up local dev tools
make venv
```

### Create service venvs

```bash
make venv-all          # all at once (slow — Airflow takes ~10 min)
# or individually:
make venv-training
make venv-serving
```

## Service URLs

| Service | URL | Active from phase |
|---|---|---|
| PostgreSQL | localhost:5432 | Phase 1 |
| MLflow | http://localhost:5000 | Phase 4 |
| Airflow | http://localhost:8080 | Phase 3 |
| FastAPI | http://localhost:8000 | Phase 6 |
| Prometheus | http://localhost:9090 | Phase 9 |
| Grafana | http://localhost:3000 | Phase 9 |

## Available Make Targets

```bash
make help
```
