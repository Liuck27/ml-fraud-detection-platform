# 01, The Big Picture

> **What this page answers:** What problem does this project solve, what
> services exist, how they talk to each other, and what was deliberately
> left out.

## The problem, in one paragraph

Credit card fraud is roughly **0.17%** of real-world transactions (in the
Kaggle dataset the project uses: 492 frauds out of 284,807 rows). Two
things follow from that:

1. **Accuracy is meaningless.** A model that always predicts "not fraud"
   scores 99.83% accuracy on this dataset and catches zero fraud. The
   project has to evaluate on metrics that actually care about the minority
   class (see [PR-AUC](02-ml-concepts.md#pr-auc-vs-roc-auc)).
2. **Missing a fraud costs more than a false alarm.** A false negative
   (FN) is a stolen card; a false positive (FP) is a slightly annoyed
   customer. The decision threshold should reflect that asymmetry, which
   is why `training/evaluate.py:47-72` optimises a *cost-weighted* score,
   not the default 0.5 cutoff.

On top of those two, fraud patterns drift over time, so the system also
has to detect when incoming data no longer looks like training data
(see [Monitoring](07-monitoring.md)) and retrain when that happens (see
`airflow/dags/retrain_dag.py`).

## The shape of the solution

Six long-running services, all in one `docker-compose.yml`, plus a few
scripts and notebooks for humans to run:

| Service | Role | Port | Defined at |
|---|---|---|---|
| PostgreSQL | Metadata store for Airflow and MLflow | 5432 | `docker-compose.yml:27-48` |
| MLflow | Experiment tracking + model registry | 5000 | `docker-compose.yml:52-74` |
| Airflow (init / webserver / scheduler) | Data pipeline orchestration | 8080 | `docker-compose.yml:79-103` |
| FastAPI serving | Real-time inference + A/B testing + SHAP | 8000 | `docker-compose.yml:171-188` |
| Prometheus | Scrapes and stores metrics | 9090 | `docker-compose.yml:192-205` |
| Grafana | Dashboards and alert visualisation | 3000 | `docker-compose.yml:209-221` |

Kafka, a Zookeeper, and a Go consumer are present as commented-out stubs
(`docker-compose.yml:107-168`). They belong to a "Phase 8" that was
deliberately skipped, see [Out of scope](#whats-deliberately-out-of-scope)
below.

## Data flow, end to end

Follow a single credit card transaction through the system:

```
 data/raw/creditcard.csv  (gitignored, from Kaggle, 284,807 rows)
          |
          |  (1) Airflow data_ingestion DAG validates the CSV and
          |      calls engineer_features() from airflow/plugins/
          v
 data/processed/features.parquet   (raw V1-V28 + 5 engineered features)
          |
          |  (2) training/train_xgboost.py reads the parquet, applies
          |      SMOTE on the training split, trains XGBoost, logs to
          |      MLflow and registers version N with alias 'champion'
          |
          |  (3) training/train_autoencoder.py does the same but trains
          |      on legit transactions only, registers as 'challenger'
          v
 MLflow registry             scaler.pkl artifact + threshold metric
   - fraud-xgboost:@champion       <-- FastAPI pulls these at startup
   - fraud-autoencoder:@challenger
          |
          |  (4) serving/app/main.py lifespan (lines 24-33) loads both
          |      models on startup; SHAP TreeExplainer is warmed up
          v
 FastAPI at :8000
   POST /predict -> _select_model -> predict_xgb or predict_ae -> SHAP
                 -> Prometheus metrics + JSON response
          |
          |  (5) prometheus-fastapi-instrumentator exposes /metrics;
          |      Prometheus scrapes every 15s (prometheus.yml)
          v
 Prometheus TSDB -> Grafana dashboard (4 panels) + alert rules
          |
          |  (6) scripts/drift_report.py is run manually, compares
          |      serving data to training distribution, emits HTML
          v
 data/reports/drift_report.html
```

Each of those six steps has its own wiki page. The numbered references in
this diagram are only meant to help you see the chain; the individual
pages are where the details live.

## Services, in one sentence each

- **PostgreSQL** is the shared backend store for both Airflow metadata
  (DAG runs, task history) and MLflow metadata (experiments, runs, model
  registry). Using one database keeps the stack small.
- **Airflow** has two DAGs: `data_ingestion` (validate CSV, engineer
  features, write Parquet) and `retrain` (validate, retrain XGBoost,
  promote to champion if PR-AUC improved). It doesn't run on a schedule
  in practice, you trigger it manually from the UI.
- **MLflow** tracks *every* training run (parameters, metrics, artifacts
  including plots and the fitted scaler) and holds a model registry with
  two aliases: `champion` (live) and `challenger` (A/B tested against
  the champion).
- **FastAPI** is the inference surface. It loads both models at startup,
  routes requests deterministically to champion or challenger, returns
  SHAP explanations for XGBoost, and exposes Prometheus metrics.
- **Prometheus** scrapes FastAPI every 15 seconds and stores time-series
  metrics. It also evaluates three alert rules (high fraud rate, high
  p99 latency, error spikes) every 15 seconds.
- **Grafana** reads from Prometheus and renders a four-panel dashboard
  (request rate, fraud rate, p99/p50 latency, A/B split). It's
  auto-provisioned from JSON under `monitoring/grafana/provisioning/`.

## The six phases

The project was built in six phases, each landing as one git commit on
`main`. Reading the commit messages gives you the best summary of how it
grew:

| Phase | What landed | Commit |
|---|---|---|
| 1 | Scaffold + Docker Compose + PostgreSQL + Makefile | `b2b1366`-ish |
| 2 | Kaggle download, Airflow ingest DAG, feature engineering, EDA notebook, Airflow in compose | `b2b1366` |
| 3 | XGBoost and Autoencoder training, MLflow registry, scaler + threshold logging | `a82611c` |
| 4 | FastAPI serving, A/B testing, SHAP, Prometheus instrumentation, 14 unit tests | `3888295` |
| 5 | Prometheus server, Grafana dashboard, alert rules, Evidently drift script | `c69e5b0` |
| 6 | CI pipeline, integration tests, retrain DAG, README | `13f0146` |

Run `git log --oneline` to see the current titles. The pattern is
"each phase = one commit" so each phase is easy to diff and study in
isolation.

## What's deliberately out of scope

Knowing *what was chosen not to build* and *why* matters as much as the
build itself.

| Omitted | Why it would have been bad to include here |
|---|---|
| **Kafka + streaming producer + Go consumer** | Adds three containers (Zookeeper, Kafka, producer) to fake a stream that is fundamentally a static CSV. Impressive infra, but buries the ML layer. Left as commented stubs in `docker-compose.yml:107-168`. |
| **Feast (feature store)** | Feast solves training/serving skew across many data sources. This dataset is one CSV, one pipeline. Adding Feast would be cargo-culting. |
| **Kubernetes** | Single-node Docker Compose is honest for the actual scale. K8s YAML doesn't prove anything the project actually needs. |
| **Isolation Forest (third model)** | Supervised (XGBoost) + unsupervised (autoencoder) already cover two paradigms. A third model is diminishing returns. |
| **Real authentication on the API** | The API sits inside `fraud-net`. For a local stack that's fine; [limitations](10-limitations-and-extensions.md) lists this as something to add before production. |

See `plan.md` lines 702-725 for the decision log this table compresses.

## Where to go next

- If you want to understand the ML concepts these services are built on,
  read [02, ML concepts](02-ml-concepts.md) before anything else.
- If you'd rather trace code, [06, Serving API](06-serving-api.md) is
  the most self-contained starting point.
- For the consolidated trade-off list, skip to
  [10, Limitations and extensions](10-limitations-and-extensions.md).
