# ML Fraud Detection Platform — Implementation Plan

## 1. Project Overview

### What It Does

An end-to-end machine learning platform for detecting fraudulent financial transactions. The system ingests transaction data, engineers features, trains multiple ML models (classical + deep learning), serves predictions via real-time and batch APIs, streams transactions through Kafka, and monitors model performance with drift detection — all orchestrated locally via Docker Compose.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose Network                             │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐        │
│  │   Airflow     │────▶│  PostgreSQL   │◀────│    MLflow Server     │        │
│  │ (DAGs: ingest,│     │ (metadata +   │     │ (experiment tracking │        │
│  │  feature eng, │     │  feature      │     │  model registry)     │        │
│  │  retrain)     │     │  store)       │     └──────────┬───────────┘        │
│  └──────┬───────┘     └──────────────┘                 │                    │
│         │                                               │                    │
│         ▼                                               ▼                    │
│  ┌──────────────┐                              ┌──────────────────┐         │
│  │ Feast Feature │                              │  FastAPI Serving  │         │
│  │    Store      │─────────────────────────────▶│  (inference API,  │         │
│  │ (offline +    │   feature retrieval          │   A/B testing)    │         │
│  │  online)      │                              └────────┬─────────┘         │
│  └──────────────┘                                        │                   │
│                                                          │                   │
│  ┌──────────────┐     ┌──────────────┐                   │                   │
│  │Kafka Producer │────▶│    Kafka      │                   │                   │
│  │  (Python,     │     │  (Broker +    │                   │                   │
│  │  simulates    │     │  Zookeeper)   │                   │                   │
│  │  transactions)│     └──────┬───────┘                   │                   │
│  └──────────────┘            │                            │                   │
│                              ▼                            │                   │
│                       ┌──────────────┐                    │                   │
│                       │ Go Consumer   │───────────────────┘                   │
│                       │ (reads txns,  │   HTTP POST /predict                  │
│                       │  calls API,   │                                       │
│                       │  writes       │──────┐                                │
│                       │  results)     │      │                                │
│                       └──────────────┘      ▼                                │
│                                       ┌──────────────┐                       │
│  ┌──────────────┐                     │  Kafka        │                       │
│  │  Evidently    │◀────metrics────────│  (results     │                       │
│  │ (drift        │                    │   topic)      │                       │
│  │  detection)   │                    └──────────────┘                       │
│  └──────┬───────┘                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐     ┌──────────────┐                                      │
│  │  Prometheus   │────▶│   Grafana     │                                      │
│  │ (metrics      │     │ (dashboards)  │                                      │
│  │  collection)  │     └──────────────┘                                      │
│  └──────────────┘                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Services & Communication

| Service | Role | Communicates With |
|---------|------|-------------------|
| **PostgreSQL** | Metadata store (Airflow, MLflow), Feast offline/online store | Airflow, MLflow, Feast, FastAPI |
| **Airflow** (webserver + scheduler + worker) | Orchestrates data ingestion, feature engineering, retraining | PostgreSQL, Feast, MLflow |
| **MLflow Server** | Experiment tracking, model registry, artifact store | PostgreSQL, FastAPI (model loading) |
| **Feast** | Feature store (offline for training, online for serving) | PostgreSQL, Airflow, FastAPI |
| **FastAPI** | Model serving (real-time + batch), A/B testing | MLflow, Feast, Prometheus, Kafka (results) |
| **Kafka + Zookeeper** | Message broker for streaming transactions | Producer, Go Consumer |
| **Kafka Producer** (Python) | Simulates real-time transaction stream | Kafka |
| **Go Consumer** | Consumes transactions, calls FastAPI, writes results | Kafka, FastAPI |
| **Prometheus** | Metrics collection and storage | FastAPI, Evidently, Grafana |
| **Grafana** | Dashboards and alerting | Prometheus |
| **Evidently** | Data/prediction drift detection | FastAPI (via scheduled reports), Prometheus |

---

## 2. Tech Stack

### Core Infrastructure

| Tool | Version | Purpose | Skills Demonstrated |
|------|---------|---------|---------------------|
| Docker + Docker Compose | Latest | Container orchestration | Production deployment |
| PostgreSQL | 15 | Metadata + feature store backend | Database management |

### ML & Data Science

| Tool | Version | Purpose | Skills Demonstrated |
|------|---------|---------|---------------------|
| Python | 3.11 | Primary language | — |
| scikit-learn | 1.3.x | Classical ML (preprocessing, metrics) | Classical ML |
| XGBoost | 2.0.x | Gradient boosted classifier | Classification |
| PyTorch | 2.1.x | Autoencoder for anomaly detection | Deep learning |
| pandas | 2.1.x | Data manipulation | Feature engineering |
| numpy | 1.26.x | Numerical computing | — |

### MLOps

| Tool | Version | Purpose | Skills Demonstrated |
|------|---------|---------|---------------------|
| MLflow | 2.9.x | Experiment tracking + model registry | MLOps, model versioning |
| Apache Airflow | 2.7.x | Pipeline orchestration | Pipeline orchestration |
| Feast | 0.34.x | Feature store | Feature stores |

### Streaming

| Tool | Version | Purpose | Skills Demonstrated |
|------|---------|---------|---------------------|
| Apache Kafka | 3.6.x (Confluent image) | Message streaming | Streaming pipelines |
| Go | 1.21 | Kafka consumer | Go proficiency |
| confluent-kafka-go | v2.3.x | Go Kafka client | — |

### Serving & API

| Tool | Version | Purpose | Skills Demonstrated |
|------|---------|---------|---------------------|
| FastAPI | 0.104.x | Model serving API | Real-time inference |
| Uvicorn | 0.24.x | ASGI server | — |
| Pydantic | 2.5.x | Request/response validation | API design |

### Monitoring & Observability

| Tool | Version | Purpose | Skills Demonstrated |
|------|---------|---------|---------------------|
| Prometheus | 2.48.x | Metrics collection | Production monitoring |
| Grafana | 10.2.x | Dashboards | Observability |
| Evidently | 0.4.x | Drift detection | Drift monitoring |
| prometheus-fastapi-instrumentator | 6.1.x | Auto-instrument FastAPI | — |

---

## 3. Data Strategy

### Dataset

**Kaggle Credit Card Fraud Detection Dataset**
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions, 492 frauds (0.172% — highly imbalanced)
- 30 features: `Time`, `Amount`, and 28 PCA-transformed features (`V1`–`V28`)
- Target: `Class` (0 = legitimate, 1 = fraud)
- License: Open Database License (ODbL)

We will download this CSV and place it in `data/raw/creditcard.csv`.

### Data Schema

```python
# Raw transaction schema
{
    "transaction_id": str,       # Generated UUID
    "Time": float,               # Seconds elapsed from first transaction
    "V1": float,                 # PCA component 1
    # ... V2–V28
    "V28": float,                # PCA component 28
    "Amount": float,             # Transaction amount
    "Class": int                 # 0=legit, 1=fraud (label)
}

# Engineered features (added by feature pipeline)
{
    "amount_log": float,               # log1p(Amount)
    "amount_zscore": float,            # Z-score normalized Amount
    "hour_of_day": int,                # Derived from Time
    "is_night": bool,                  # 22:00–06:00
    "v1_v2_interaction": float,        # V1 * V2
    "amount_rolling_mean_1h": float,   # Rolling mean of amount (1h window)
    "amount_rolling_std_1h": float,    # Rolling std of amount (1h window)
    "transaction_count_1h": int,       # Transaction frequency (1h window)
}
```

### Synthetic Data Generation for Streaming

The Kafka producer will:
1. Load the original dataset
2. Add slight random noise to numerical features (±5% jitter)
3. Assign realistic timestamps (current time with random sub-second offsets)
4. Emit transactions at a configurable rate (default: 10 transactions/second)
5. Optionally inject anomalies by sampling from the fraud class with higher probability

This approach ensures the streaming data is statistically similar to training data while providing continuous flow for the Kafka pipeline.

---

## 4. Component Breakdown

### a) Data Ingestion & Feature Engineering Pipeline (Airflow)

**Purpose:** Orchestrate data loading, validation, feature engineering, and materialization into the Feast feature store.

**Input:** Raw CSV in `data/raw/creditcard.csv`
**Output:** Engineered features materialized in Feast (offline + online stores)

**Key Implementation Details:**
- `data_ingestion_dag.py`: Main DAG with tasks:
  1. `validate_raw_data` — Check file exists, schema validation, null checks
  2. `engineer_features` — Compute derived features (log transforms, rolling aggregations, time features)
  3. `write_to_offline_store` — Write feature DataFrame to Feast offline store (Parquet files)
  4. `materialize_online_store` — Run `feast materialize` to push to online store
- `feature_engineering.py`: Pure functions for feature computation (testable independently)
- Uses SMOTE or class weighting strategies for imbalance (handled in training, noted in features)

**Files:**
```
airflow/
├── dags/
│   ├── data_ingestion_dag.py
│   └── retrain_dag.py
├── plugins/
│   └── feature_engineering.py
└── Dockerfile
```

### b) Model Training & Experiment Tracking (MLflow + PyTorch + scikit-learn)

**Purpose:** Train, evaluate, and register multiple fraud detection models.

**Models:**

1. **XGBoost Classifier** (Classical ML)
   - Binary classification on engineered features
   - Handle class imbalance via `scale_pos_weight`
   - Hyperparameter tuning: learning rate, max depth, n_estimators
   - Metrics: precision, recall, F1, AUC-ROC, PR-AUC

2. **PyTorch Autoencoder** (Deep Learning — Anomaly Detection)
   - Trained on legitimate transactions only
   - Reconstruction error as anomaly score
   - Architecture: Input(30) → 64 → 32 → 16 → 32 → 64 → Output(30)
   - Threshold on reconstruction MSE for fraud classification
   - Export via TorchScript for serving

3. **Isolation Forest** (Unsupervised — Anomaly Clustering)
   - Complementary anomaly detection
   - Used for grouping/clustering anomalous transactions
   - Provides anomaly scores for ensemble potential

**MLflow Integration:**
- All runs logged with hyperparameters, metrics, and model artifacts
- Model registry with aliases: `champion` (production), `challenger` (staging)
- Comparison views across experiments

**Files:**
```
training/
├── train_xgboost.py
├── train_autoencoder.py
├── train_isolation_forest.py
├── evaluate.py              # Shared evaluation utilities
├── model_registry.py        # MLflow registry helpers (promote/demote)
├── requirements.txt
└── notebooks/
    └── eda.ipynb            # Exploratory data analysis
```

### c) Feature Store (Feast)

**Purpose:** Centralized feature management for both training and real-time serving.

**Configuration:**
- **Offline store:** File-based (Parquet) — simple, no extra infra needed
- **Online store:** SQLite (local) — sufficient for portfolio project
- **Registry:** File-based

**Entity:** `transaction` identified by `transaction_id`

**Feature Views:**
- `transaction_features`: Core engineered features
- `transaction_stats`: Rolling aggregation features

**Files:**
```
feature_store/
├── feature_repo/
│   ├── feature_store.yaml    # Feast project config
│   ├── entities.py           # Entity definitions
│   ├── features.py           # Feature view definitions
│   └── data_sources.py       # Data source definitions
└── README.md
```

### d) Kafka Streaming Pipeline

**Purpose:** Simulate real-time transaction processing end-to-end.

**Topics:**
- `transactions` — Raw transaction events (JSON)
- `predictions` — Inference results (JSON)

**Message Schema (transactions topic):**
```json
{
    "transaction_id": "uuid-string",
    "timestamp": "2024-01-15T10:30:00Z",
    "features": {
        "V1": -1.359,
        "V2": -0.073,
        "...": "...",
        "V28": 0.015,
        "Amount": 149.62
    }
}
```

**Message Schema (predictions topic):**
```json
{
    "transaction_id": "uuid-string",
    "timestamp": "2024-01-15T10:30:00.150Z",
    "prediction": 1,
    "fraud_probability": 0.87,
    "model_version": "xgboost-v3",
    "latency_ms": 12.5
}
```

**Kafka Producer (Python):**
- Reads from dataset, adds noise, publishes to `transactions` topic
- Configurable throughput (TPS) and duration
- Graceful shutdown on SIGTERM

**Kafka Consumer (Go):**
- Reads from `transactions` topic (consumer group: `fraud-detector`)
- Calls FastAPI `/predict` endpoint via HTTP
- Publishes result to `predictions` topic
- Idiomatic Go: goroutines for concurrent processing, structured logging, graceful shutdown
- Health check endpoint on `:8081/health`

**Files:**
```
streaming/
├── producer/
│   ├── producer.py
│   ├── requirements.txt
│   └── Dockerfile
└── consumer/
    ├── main.go
    ├── go.mod
    ├── go.sum
    ├── internal/
    │   ├── consumer.go     # Kafka consumer logic
    │   ├── client.go       # HTTP client for inference API
    │   └── producer.go     # Kafka producer for results
    └── Dockerfile
```

### e) Model Serving API (FastAPI)

**Purpose:** Serve predictions via REST API with A/B testing between models.

**Endpoints:**
- `POST /predict` — Single transaction inference
- `POST /predict/batch` — Batch inference (up to 1000 transactions)
- `GET /health` — Health check with model status
- `GET /models` — List loaded models and their versions
- `GET /metrics` — Prometheus metrics endpoint

**A/B Testing Logic:**
- Configuration: traffic split percentage (e.g., 80% XGBoost / 20% Autoencoder)
- Consistent routing: hash of `transaction_id` determines model assignment
- Both models' predictions are logged for offline comparison
- Split configured via environment variable, changeable without restart

**Model Loading:**
- On startup, load `champion` and `challenger` models from MLflow registry
- XGBoost model loaded via MLflow's sklearn flavor
- Autoencoder loaded via TorchScript
- Fallback: if MLflow unavailable, load from local artifact cache

**Files:**
```
serving/
├── app/
│   ├── main.py              # FastAPI app, startup/shutdown
│   ├── routes/
│   │   ├── predict.py       # Prediction endpoints
│   │   ├── health.py        # Health check
│   │   └── models.py        # Model info endpoint
│   ├── models/
│   │   ├── loader.py        # MLflow model loading
│   │   └── ab_testing.py    # A/B routing logic
│   ├── schemas.py           # Pydantic request/response models
│   └── config.py            # Settings via pydantic-settings
├── requirements.txt
├── Dockerfile
└── tests/
    ├── test_predict.py
    ├── test_ab_testing.py
    └── conftest.py
```

### f) Monitoring & Observability

**Purpose:** Track model performance, detect drift, and alert on degradation.

**Components:**

1. **Prometheus Metrics (exported by FastAPI):**
   - `inference_latency_seconds` (histogram) — per model
   - `inference_total` (counter) — per model, per prediction class
   - `inference_errors_total` (counter)
   - `model_confidence_score` (histogram) — per model
   - `ab_test_assignments_total` (counter) — per model variant
   - Custom: `estimated_cost_per_inference` (gauge) — estimated compute cost

2. **Evidently Drift Reports:**
   - Scheduled via Airflow DAG (every N hours)
   - Compares reference data (training set) vs. current serving data
   - Generates HTML reports saved to `data/reports/`
   - Exports drift metrics to Prometheus via pushgateway or file-based export

3. **Grafana Dashboards (provisioned via JSON):**
   - **Model Performance:** Accuracy, F1, precision, recall over time
   - **Inference Metrics:** Latency p50/p95/p99, throughput, error rate
   - **Drift Monitoring:** Feature drift scores, prediction drift
   - **A/B Comparison:** Side-by-side model metrics
   - **System Health:** Request volume, consumer lag, uptime

4. **Automated Retraining Trigger:**
   - Airflow DAG checks drift metrics from Prometheus
   - If drift score exceeds threshold → trigger retrain DAG
   - Simple implementation: cron-scheduled Airflow DAG

**Files:**
```
monitoring/
├── grafana/
│   ├── provisioning/
│   │   ├── dashboards/
│   │   │   ├── dashboard.yml       # Dashboard provisioning config
│   │   │   └── fraud_detection.json # Main dashboard
│   │   └── datasources/
│   │       └── datasource.yml      # Prometheus datasource
│   └── Dockerfile
├── prometheus/
│   └── prometheus.yml              # Scrape configs
├── evidently/
│   ├── drift_report.py             # Generate drift reports
│   └── requirements.txt
└── alerting/
    └── rules.yml                   # Prometheus alerting rules
```

### g) Infrastructure (Docker Compose)

**Services in `docker-compose.yml`:**

| Service | Image | Ports | Dependencies |
|---------|-------|-------|-------------|
| postgres | postgres:15 | 5432 | — |
| airflow-webserver | custom (extends apache/airflow:2.7) | 8080 | postgres |
| airflow-scheduler | custom | — | postgres |
| airflow-init | custom | — | postgres |
| mlflow | custom (Python + mlflow) | 5000 | postgres |
| feast-materialize | custom | — | postgres |
| kafka | confluentinc/cp-kafka:7.5.0 | 9092 | zookeeper |
| zookeeper | confluentinc/cp-zookeeper:7.5.0 | 2181 | — |
| serving | custom (FastAPI) | 8000 | mlflow, postgres |
| kafka-producer | custom (Python) | — | kafka, serving |
| go-consumer | custom (Go) | 8081 | kafka, serving |
| prometheus | prom/prometheus:v2.48.0 | 9090 | serving |
| grafana | grafana/grafana:10.2.0 | 3000 | prometheus |

**Volumes:**
- `postgres_data` — PostgreSQL persistence
- `mlflow_artifacts` — Model artifacts
- `./data` — Raw/processed data mounted to relevant services
- `./monitoring/grafana/provisioning` — Grafana auto-provisioning

**Networks:**
- Single `fraud-detection-net` bridge network

---

## 5. Directory Structure

```
ml-fraud-detection-platform/
├── docker-compose.yml
├── .env                         # Environment variables
├── .env.example                 # Template
├── .gitignore
├── plan.md                      # This file
├── README.md
│
├── data/
│   ├── raw/                     # Original dataset (gitignored)
│   │   └── creditcard.csv
│   ├── processed/               # Engineered features
│   └── reports/                 # Evidently drift reports
│
├── airflow/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── dags/
│   │   ├── data_ingestion_dag.py
│   │   └── retrain_dag.py
│   └── plugins/
│       └── feature_engineering.py
│
├── feature_store/
│   └── feature_repo/
│       ├── feature_store.yaml
│       ├── entities.py
│       ├── features.py
│       └── data_sources.py
│
├── training/
│   ├── train_xgboost.py
│   ├── train_autoencoder.py
│   ├── train_isolation_forest.py
│   ├── evaluate.py
│   ├── model_registry.py
│   ├── requirements.txt
│   └── notebooks/
│       └── eda.ipynb
│
├── serving/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── routes/
│   │   │   ├── predict.py
│   │   │   ├── health.py
│   │   │   └── models.py
│   │   └── models/
│   │       ├── loader.py
│   │       └── ab_testing.py
│   └── tests/
│       ├── conftest.py
│       ├── test_predict.py
│       └── test_ab_testing.py
│
├── streaming/
│   ├── producer/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── producer.py
│   └── consumer/
│       ├── Dockerfile
│       ├── go.mod
│       ├── main.go
│       └── internal/
│           ├── consumer.go
│           ├── client.go
│           └── producer.go
│
├── monitoring/
│   ├── grafana/
│   │   ├── Dockerfile
│   │   └── provisioning/
│   │       ├── dashboards/
│   │       │   ├── dashboard.yml
│   │       │   └── fraud_detection.json
│   │       └── datasources/
│   │           └── datasource.yml
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── evidently/
│       ├── drift_report.py
│       └── requirements.txt
│
├── scripts/
│   ├── download_data.py         # Download Kaggle dataset
│   ├── seed_data.py             # Seed initial data
│   └── run_training.sh          # Convenience script to run all training
│
└── tests/
    ├── integration/
    │   ├── test_pipeline_e2e.py
    │   └── test_kafka_flow.py
    └── conftest.py
```

---

## 6. Implementation Phases

### Phase 1: Project Scaffold & Infrastructure Foundation
**Duration:** ~1 day

**Deliverables:**
- Project directory structure created
- `docker-compose.yml` with PostgreSQL, basic networking
- `.env` / `.env.example` with all config variables
- `.gitignore` covering data files, Python artifacts, Go binaries
- PostgreSQL starts and accepts connections
- Basic `Makefile` or `scripts/` for common operations

**Acceptance Criteria:**
- `docker-compose up postgres` starts successfully
- Can connect to PostgreSQL on `localhost:5432`
- All directories exist with placeholder READMEs where needed

---

### Phase 2: Data Ingestion & Exploratory Analysis
**Duration:** ~1 day

**Deliverables:**
- `scripts/download_data.py` — downloads Kaggle dataset (or manual instructions)
- `training/notebooks/eda.ipynb` — EDA notebook with:
  - Class distribution visualization
  - Feature distributions and correlations
  - Fraud vs. legitimate statistical comparison
  - Missing value analysis
- Data schema validation script

**Acceptance Criteria:**
- Dataset present in `data/raw/creditcard.csv`
- EDA notebook runs end-to-end and produces visualizations
- Dataset has 284,807 rows, 31 columns, 492 fraud cases

---

### Phase 3: Feature Engineering & Feast Feature Store
**Duration:** ~1 day

**Deliverables:**
- `airflow/plugins/feature_engineering.py` — feature engineering functions
- Feast feature repository configured (`feature_store/`)
- Feature definitions for all engineered features
- Airflow DAG for data ingestion + feature engineering
- Airflow added to `docker-compose.yml`
- Features materialized to online + offline stores

**Acceptance Criteria:**
- `docker-compose up airflow-webserver airflow-scheduler` starts Airflow UI on `:8080`
- DAG appears in Airflow UI and can be triggered manually
- DAG runs successfully: raw data → engineered features → Feast store
- Can query features from Feast offline store programmatically
- Unit tests pass for feature engineering functions

---

### Phase 4: Classical ML Training with MLflow
**Duration:** ~1 day

**Deliverables:**
- `training/train_xgboost.py` — XGBoost classifier training script
- `training/train_isolation_forest.py` — Isolation Forest training script
- `training/evaluate.py` — evaluation utilities (metrics, plots)
- MLflow server added to `docker-compose.yml`
- MLflow experiment with logged runs
- Best model registered in MLflow model registry as `champion`

**Acceptance Criteria:**
- MLflow UI accessible at `localhost:5000`
- XGBoost experiment shows: precision, recall, F1, AUC-ROC, PR-AUC, confusion matrix
- Isolation Forest experiment shows: anomaly scores, contamination tuning
- XGBoost achieves AUC-ROC > 0.95 (expected given this dataset)
- Model registered with alias `champion` in registry

---

### Phase 5: PyTorch Autoencoder Training
**Duration:** ~1 day

**Deliverables:**
- `training/train_autoencoder.py` — PyTorch autoencoder training
- Model architecture: encoder-decoder with bottleneck
- Training on legitimate transactions only
- Threshold tuning based on reconstruction error percentiles
- Model exported via TorchScript
- Logged to MLflow with reconstruction error metrics

**Acceptance Criteria:**
- Training converges (loss decreases over epochs)
- Reconstruction error clearly separates fraud vs. legitimate (visualized)
- TorchScript model saved and loadable
- MLflow experiment shows: loss curves, threshold, detection metrics
- Model registered with alias `challenger` in registry

---

### Phase 6: FastAPI Model Serving
**Duration:** ~1 day

**Deliverables:**
- FastAPI application with all endpoints
- Model loading from MLflow registry (XGBoost + Autoencoder)
- Single prediction and batch prediction endpoints
- Health check endpoint
- Pydantic schemas for all request/response models
- Dockerfile for serving container
- Added to `docker-compose.yml`

**Acceptance Criteria:**
- `curl localhost:8000/health` returns healthy status with model info
- `curl -X POST localhost:8000/predict` with valid payload returns prediction
- `curl -X POST localhost:8000/predict/batch` handles multiple transactions
- Response includes: prediction, probability, model version, latency
- Error handling for invalid inputs (400), model not loaded (503)
- Unit tests pass for prediction endpoints

---

### Phase 7: A/B Testing Logic
**Duration:** ~1 day

**Deliverables:**
- A/B testing module: traffic splitting based on transaction ID hash
- Configuration via environment variables (split ratio)
- Both models' predictions logged for every request
- `/models` endpoint shows current A/B configuration
- Metrics per model variant (separate Prometheus labels)

**Acceptance Criteria:**
- With 80/20 split, approximately 80% of requests go to champion, 20% to challenger
- Consistent routing: same `transaction_id` always routes to the same model
- Can change split ratio via env var and restart
- `/models` endpoint shows both models and their traffic percentages
- Tests verify routing consistency and approximate split ratios

---

### Phase 8: Kafka Streaming Pipeline
**Duration:** ~1.5 days

**Deliverables:**
- Kafka + Zookeeper added to `docker-compose.yml`
- Python Kafka producer (`streaming/producer/`)
- Go Kafka consumer (`streaming/consumer/`)
- Producer reads dataset, adds noise, publishes to `transactions` topic
- Consumer reads transactions, calls `/predict`, publishes to `predictions` topic
- Go consumer with goroutines, structured logging, graceful shutdown
- Health check endpoint on Go consumer

**Acceptance Criteria:**
- Kafka broker accessible on `localhost:9092`
- Producer sends transactions at configurable rate
- Go consumer processes transactions and calls serving API
- Results appear on `predictions` topic
- Go consumer handles API failures gracefully (retries, backoff)
- `go vet` and `go test` pass
- Consumer lag is minimal under normal load

---

### Phase 9: Monitoring & Observability
**Duration:** ~1.5 days

**Deliverables:**
- Prometheus configuration scraping FastAPI metrics
- Custom Prometheus metrics in FastAPI (latency, throughput, per-model)
- Evidently drift detection script
- Grafana dashboards (provisioned automatically):
  - Model performance dashboard
  - Inference metrics dashboard
  - Drift monitoring dashboard
  - A/B test comparison dashboard
- Airflow DAG for periodic drift checks
- Alerting rules for drift thresholds

**Acceptance Criteria:**
- Prometheus UI at `localhost:9090` shows metrics from FastAPI
- Grafana at `localhost:3000` loads with pre-configured dashboards
- Dashboard panels display real data after running the Kafka pipeline
- Evidently drift report generates successfully
- Drift metrics exported to Prometheus

---

### Phase 10: Integration Testing & Documentation
**Duration:** ~1 day

**Deliverables:**
- End-to-end integration test: full pipeline from data ingestion to prediction
- Kafka flow integration test
- `README.md` with:
  - Project description and architecture diagram
  - Setup instructions
  - Usage examples with curl commands
  - Screenshots of Grafana dashboards, MLflow UI
  - Design decisions and trade-offs
- Clean up code, add docstrings where needed
- Final git tagging and cleanup

**Acceptance Criteria:**
- `docker-compose up` starts all services successfully
- Full pipeline works: data → features → training → serving → streaming → monitoring
- README is comprehensive and a new user could set up the project
- All tests pass
- No hardcoded secrets or credentials

---

## 7. API Contracts

### POST /predict — Single Transaction Inference

**Request:**
```json
{
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "features": {
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Amount": 149.62
    }
}
```

**Response (200):**
```json
{
    "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
    "prediction": 0,
    "fraud_probability": 0.032,
    "is_fraud": false,
    "model_version": "xgboost-v3",
    "model_name": "xgboost-champion",
    "latency_ms": 4.2,
    "timestamp": "2024-01-15T10:30:00.004Z"
}
```

**Error Response (422):**
```json
{
    "detail": [
        {
            "loc": ["body", "features", "V1"],
            "msg": "field required",
            "type": "value_error.missing"
        }
    ]
}
```

### POST /predict/batch — Batch Inference

**Request:**
```json
{
    "transactions": [
        {
            "transaction_id": "uuid-1",
            "features": { "V1": -1.35, "...": "...", "Amount": 149.62 }
        },
        {
            "transaction_id": "uuid-2",
            "features": { "V1": 1.19, "...": "...", "Amount": 2.69 }
        }
    ]
}
```

**Response (200):**
```json
{
    "predictions": [
        {
            "transaction_id": "uuid-1",
            "prediction": 0,
            "fraud_probability": 0.032,
            "is_fraud": false,
            "model_version": "xgboost-v3",
            "model_name": "xgboost-champion",
            "latency_ms": 4.2
        },
        {
            "transaction_id": "uuid-2",
            "prediction": 1,
            "fraud_probability": 0.94,
            "is_fraud": true,
            "model_version": "xgboost-v3",
            "model_name": "xgboost-champion",
            "latency_ms": 3.8
        }
    ],
    "total_latency_ms": 8.0,
    "count": 2,
    "timestamp": "2024-01-15T10:30:00.008Z"
}
```

### GET /health — Health Check

**Response (200):**
```json
{
    "status": "healthy",
    "models": {
        "champion": {
            "name": "xgboost-fraud-detector",
            "version": "3",
            "status": "loaded"
        },
        "challenger": {
            "name": "autoencoder-fraud-detector",
            "version": "1",
            "status": "loaded"
        }
    },
    "ab_test": {
        "enabled": true,
        "champion_traffic": 0.8,
        "challenger_traffic": 0.2
    }
}
```

**Response (503 — model not loaded):**
```json
{
    "status": "degraded",
    "models": {
        "champion": { "status": "loaded" },
        "challenger": { "status": "error", "error": "Failed to load from registry" }
    }
}
```

### GET /models — Model Information

**Response (200):**
```json
{
    "models": [
        {
            "name": "xgboost-fraud-detector",
            "version": "3",
            "role": "champion",
            "traffic_percentage": 80,
            "type": "xgboost",
            "metrics": {
                "auc_roc": 0.978,
                "f1_score": 0.85,
                "precision": 0.92,
                "recall": 0.79
            }
        },
        {
            "name": "autoencoder-fraud-detector",
            "version": "1",
            "role": "challenger",
            "traffic_percentage": 20,
            "type": "pytorch-autoencoder",
            "metrics": {
                "auc_roc": 0.952,
                "f1_score": 0.78,
                "precision": 0.84,
                "recall": 0.73
            }
        }
    ]
}
```

### Go Consumer Health — GET :8081/health

**Response (200):**
```json
{
    "status": "healthy",
    "kafka_connected": true,
    "messages_processed": 15234,
    "errors": 3,
    "uptime_seconds": 3600
}
```

---

## 8. Testing Strategy

### Unit Tests

| Component | Test File | What's Tested |
|-----------|-----------|---------------|
| Feature Engineering | `airflow/tests/test_feature_engineering.py` | Each feature transform function in isolation |
| Model Evaluation | `training/tests/test_evaluate.py` | Metric computation, threshold selection |
| API Schemas | `serving/tests/test_schemas.py` | Pydantic model validation |
| A/B Testing | `serving/tests/test_ab_testing.py` | Routing consistency, split ratios |
| Prediction Endpoints | `serving/tests/test_predict.py` | Happy path, error cases, batch limits |

### Integration Tests

| Test | What's Verified |
|------|----------------|
| `tests/integration/test_pipeline_e2e.py` | Data ingestion → feature engineering → Feast materialization |
| `tests/integration/test_kafka_flow.py` | Producer → Kafka → Consumer → API → Predictions topic |
| `tests/integration/test_model_serving.py` | Model loading from MLflow → inference → correct response |

### Verification Per Phase

- **Phase 1:** `docker-compose up postgres` → `psql` connection test
- **Phase 2:** Run EDA notebook → verify output cells
- **Phase 3:** Trigger Airflow DAG → verify Feast store populated
- **Phase 4:** Run training script → verify MLflow experiment/metrics
- **Phase 5:** Run autoencoder training → verify TorchScript export loads
- **Phase 6:** `curl` all API endpoints → verify response schemas
- **Phase 7:** Send 1000 requests → verify ~80/20 split in Prometheus metrics
- **Phase 8:** Run producer + consumer → verify messages flow end-to-end
- **Phase 9:** Check Grafana dashboards populate with real metrics
- **Phase 10:** Full `docker-compose up` → end-to-end smoke test

### Test Commands
```bash
# Unit tests
pytest serving/tests/ -v
pytest airflow/tests/ -v
pytest training/tests/ -v

# Go tests
cd streaming/consumer && go test ./... -v

# Integration tests (requires docker-compose up)
pytest tests/integration/ -v

# Linting
ruff check .
go vet ./...
```

---

## 9. README Outline

```markdown
# ML Fraud Detection Platform

> End-to-end machine learning platform for real-time financial fraud detection
> with MLOps best practices, streaming pipelines, and production monitoring.

## Overview
- What this project does (2-3 sentences)
- Key capabilities: real-time + batch inference, A/B testing, drift monitoring
- Architecture diagram (Mermaid)

## Architecture
- System diagram
- Data flow description
- Component interaction

## Tech Stack
- Table with: tool, purpose, link
- Why each was chosen

## Quick Start
### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM recommended
- Kaggle account (for dataset download)

### Setup
1. Clone the repo
2. Copy .env.example → .env
3. Download dataset (script provided)
4. docker-compose up -d
5. Wait for services to initialize
6. Access UIs: Airflow :8080, MLflow :5000, Grafana :3000, API :8000

### Train Models
- Run training scripts (or trigger via Airflow)

### Run Streaming Pipeline
- Start producer and consumer

## Usage Examples
- curl commands for all endpoints
- Screenshots of:
  - Grafana dashboards
  - MLflow experiment tracking
  - Airflow DAG runs

## Design Decisions & Trade-offs
- Why XGBoost + Autoencoder (classical + deep learning)
- Why Feast over ad-hoc feature computation
- Why Go for the consumer
- Simplifications made for portfolio context

## Project Structure
- Directory tree with descriptions

## Future Improvements
- Kubernetes deployment
- CI/CD pipeline for model training
- Online learning / incremental training
- Feature importance monitoring
```

---

## 10. CV Integration

The following bullet points can be added to a CV/resume after completing this project:

1. **"Designed and built an end-to-end ML platform for real-time fraud detection, processing 10+ transactions/second through Kafka streaming pipelines with sub-50ms inference latency"**

2. **"Trained and deployed XGBoost and PyTorch autoencoder models achieving 0.97+ AUC-ROC on imbalanced financial data (0.17% fraud rate), with automated model versioning via MLflow"**

3. **"Implemented A/B testing framework for ML model comparison, enabling data-driven model promotion decisions between classical ML and deep learning approaches"**

4. **"Built feature engineering pipelines orchestrated by Apache Airflow with Feast feature store, ensuring consistent feature computation across training and serving environments"**

5. **"Developed production monitoring stack with Evidently AI drift detection, Prometheus metrics, and Grafana dashboards tracking inference latency (p50/p95/p99), model accuracy, and data drift"**

6. **"Engineered a Go-based Kafka consumer service for high-throughput transaction processing with concurrent goroutine-based architecture and graceful shutdown handling"**

7. **"Established MLOps practices including experiment tracking, model registry with staging/production lifecycle, automated retraining triggers on drift detection, and TorchScript model export for optimized inference"**

8. **"Containerized 12+ microservices via Docker Compose with health checks, volume persistence, and single-command deployment for a complete ML platform stack"**
