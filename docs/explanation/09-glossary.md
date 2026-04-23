# 09, Glossary

> **What this page answers:** What does every term in this wiki actually
> mean, in one line? Use it as a quick-reference while reading, not a
> standalone primer.

Entries are grouped by domain. Each term links to the page where the
concept is explained in depth, if one exists.

## ML and statistics

- **Accuracy**, fraction of predictions that are correct. Useless on
  imbalanced data ([02](02-ml-concepts.md#class-imbalance)).
- **Anomaly detection**, flagging rare/unusual points; the autoencoder's
  paradigm ([02](02-ml-concepts.md#autoencoders-for-anomaly-detection)).
- **Autoencoder**, neural network trained to reproduce its input through
  a bottleneck; high reconstruction error = anomaly ([02](02-ml-concepts.md#autoencoders-for-anomaly-detection)).
- **AUC-ROC**, area under the ROC curve. 0.5 = random, 1.0 = perfect;
  insensitive to base rate ([02](02-ml-concepts.md#pr-auc-vs-roc-auc)).
- **Baseline rate**, the fraction of positives in the raw data; 0.172%
  here.
- **Calibration**, whether predicted probabilities match observed
  frequencies (a 0.8 prediction should mean 80% of such cases are fraud).
- **Challenger**, the experimental model tested against the champion on
  a fraction of traffic ([02](02-ml-concepts.md#a-b-testing-in-ml-serving)).
- **Champion**, the production model receiving most traffic
  ([02](02-ml-concepts.md#a-b-testing-in-ml-serving)).
- **Class imbalance**, a dataset where one class (here fraud, 0.17%) is
  far rarer than the other ([02](02-ml-concepts.md#class-imbalance)).
- **Concept drift**, the *relationship* between features and label
  changes; the model's assumptions are no longer valid ([02](02-ml-concepts.md#data-drift-and-concept-drift)).
- **Cost-sensitive learning**, optimising total cost (e.g. `10 × FN + 1 ×
  FP`) instead of error count ([02](02-ml-concepts.md#decision-threshold-tuning)).
- **Cross-validation**, splitting data into multiple train/val folds to
  get stable metrics. Not used in this project (flagged as a limitation).
- **Data drift**, the input feature distribution changes over time; model
  may still be right, just scoring a different population
  ([02](02-ml-concepts.md#data-drift-and-concept-drift)).
- **Decision threshold**, the probability cutoff that converts a score
  into a binary `is_fraud`; default 0.5, tuned here to balance FN vs FP
  cost ([02](02-ml-concepts.md#decision-threshold-tuning)).
- **F1 score**, harmonic mean of precision and recall; single number that
  balances both.
- **False negative (FN)**, model said "legit" but it was fraud; the
  expensive mistake in fraud detection.
- **False positive (FP)**, model said "fraud" but it was legit; customer
  friction.
- **Feature engineering**, deriving new features from raw columns
  ([04](04-data-and-features.md#feature-engineering)).
- **Gradient boosting**, building trees sequentially, each fit on
  previous model's errors ([02](02-ml-concepts.md#gradient-boosted-trees--xgboost)).
- **Interaction feature**, a column formed by combining two features
  (here `V1 * V2`); captures joint effects that linear models miss.
- **KS test (Kolmogorov-Smirnov)**, statistical test comparing two
  empirical distributions; used by Evidently for drift on numeric
  features.
- **Labels**, the ground-truth target variable (`Class`: 0 or 1).
- **MSE (Mean Squared Error)**, average squared difference between
  prediction and target; the autoencoder's loss function.
- **PCA (Principal Component Analysis)**, linear projection that
  concentrates variance into fewer features. The dataset's V1-V28 are
  PCA components of the original private features.
- **PR-AUC**, area under the Precision-Recall curve; the honest metric on
  imbalanced data ([02](02-ml-concepts.md#pr-auc-vs-roc-auc)).
- **Precision**, of everything flagged as fraud, what fraction really was
  fraud? TP / (TP + FP).
- **Probability (fraud_probability)**, the model's estimated chance that
  a transaction is fraud, in `[0, 1]`.
- **PyFunc (MLflow)**, generic wrapper that lets any Python model
  implement a uniform `predict()` interface ([05](05-training.md#the-autoencoderpyfunc-wrapper)).
- **Recall**, of all actual fraud cases, what fraction did we catch? TP /
  (TP + FN).
- **Reconstruction error**, how different an autoencoder's output is from
  its input; high on anomalies ([02](02-ml-concepts.md#autoencoders-for-anomaly-detection)).
- **ROC curve**, plot of TPR vs FPR at varying thresholds; the reference
  binary-classification plot.
- **`scale_pos_weight`**, XGBoost parameter that up-weights minority-class
  errors; here `n_neg / n_pos` ([02](02-ml-concepts.md#class-imbalance)).
- **Scaler (StandardScaler)**, fits per-feature mean and std on training
  data; transforms future data to zero-mean, unit-variance.
- **SHAP (Shapley Additive exPlanations)**, per-feature contribution
  decomposition of a single prediction ([02](02-ml-concepts.md#shap-values)).
- **SMOTE (Synthetic Minority Oversampling Technique)**, synthesizes new
  minority-class examples by interpolating between real ones
  ([02](02-ml-concepts.md#smote)).
- **Stratified split**, train/test split that preserves the class ratio
  in each partition.
- **Supervised learning**, training a model on labelled examples.
- **Target leakage**, when a feature accidentally encodes the label;
  produces too-good-to-be-true metrics.
- **TreeExplainer (SHAP)**, exact, polynomial-time SHAP algorithm for
  tree models like XGBoost.
- **True positive (TP)**, model said "fraud" and it was fraud. The win.
- **True negative (TN)**, model said "legit" and it was legit.
- **Unsupervised learning**, training without labels; autoencoders here.
- **XGBoost**, gradient-boosted tree library, dominant on tabular data
  ([02](02-ml-concepts.md#gradient-boosted-trees--xgboost)).

## MLOps

- **Alias (MLflow)**, mutable pointer to a specific model version (here
  `champion` and `challenger`) so code doesn't hardcode versions
  ([05](05-training.md#aliases-champion-and-challenger)).
- **Artifact (MLflow)**, file logged with a run; includes the model
  binary, scaler pickle, PNG plots, threshold text ([02](02-ml-concepts.md#mlflow-vocabulary)).
- **Batch prediction**, scoring N transactions in one request (here
  1-1000 per call).
- **Drift report**, HTML summary of feature distribution changes between
  reference and current data ([07](07-monitoring.md#evidently-drift-report)).
- **Experiment (MLflow)**, a namespace grouping related runs (here
  `fraud-detection-xgboost`, `fraud-detection-autoencoder`).
- **Feature store**, centralised service (e.g. Feast) that serves feature
  values to both training and serving; deliberately not used here.
- **Inference**, using a trained model to make a prediction.
- **Model registry**, MLflow's catalog of versioned models with aliases
  ([02](02-ml-concepts.md#mlflow-vocabulary)).
- **Promotion**, moving a model version to a new alias (here, `→
  champion`). Manual for first-time models; conditional on PR-AUC for
  retrains.
- **Real-time vs batch serving**, /predict returns one score per
  request; a batch job might score a nightly CSV. This project does
  both.
- **Retraining**, rebuilding a model on fresh data; triggered here by
  the `retrain` DAG.
- **Run (MLflow)**, one execution of a training script; produces
  metrics + params + artifacts.
- **Schema (registered model)**, input/output spec captured at
  `log_model` time via `input_example`.
- **Serving**, running a model behind an API to take live requests (the
  FastAPI layer).
- **Shadow traffic**, sending production requests to a model without
  using its predictions; used for offline evaluation before promotion.
  Not implemented here.
- **Signature (MLflow)**, declarative schema of a model's input/output
  shapes, auto-inferred from `input_example`.
- **Tracking URI**, where MLflow stores its metadata; here
  `http://mlflow:5000` inside Compose, `http://localhost:5000` from a
  host shell.
- **Training-serving skew**, a bug where feature transforms differ
  between training and inference; the `amount_zscore` edge case is a
  known minor instance ([06](06-serving-api.md#feature-prep-the-amount_zscore-quirk)).

## Platform and infra

- **Airflow DAG**, a directed graph of tasks that runs on a schedule or
  trigger. This project has two: `data_ingestion` and `retrain`.
- **Alert rule**, a PromQL expression that fires an alert when true for
  a sustained duration ([07](07-monitoring.md#alert-rules--monitoringalertingrulesyml)).
- **Bridge network**, Docker's default network driver; containers on
  the same bridge resolve each other by name ([03](03-infrastructure.md#the-network-fraud-net)).
- **Compose**, Docker Compose; declarative way to define multi-container
  stacks in one YAML file ([03](03-infrastructure.md#why-docker-compose)).
- **Compose service**, one container declaration in
  `docker-compose.yml`; not the same as a "microservice".
- **Counter (Prometheus)**, a monotonically increasing metric; always
  queried with `rate()`.
- **CI (GitHub Actions)**, the automated pipeline that runs lint,
  typecheck, and tests on push/PR ([08](08-testing-and-ci.md#github-actions--githubworkflowsciyml)).
- **Dashboard (Grafana)**, a collection of panels reading from a
  datasource, here auto-provisioned as JSON.
- **Datasource (Grafana)**, a backend that Grafana queries; here a
  single Prometheus datasource.
- **Endpoint (FastAPI)**, one path + HTTP method combination (e.g.
  `POST /predict`).
- **Evidently**, open-source library for data drift reports.
- **FastAPI**, Python async web framework used for the serving API
  ([06](06-serving-api.md#why-fastapi)).
- **Fernet key**, symmetric encryption key Airflow uses to encrypt
  connection secrets; must be a valid key, not a placeholder.
- **Grafana**, open-source dashboard tool; reads from Prometheus here.
- **Gauge (Prometheus)**, a metric that can go up or down; not used
  custom in this project.
- **Healthcheck**, a periodic probe Compose runs to decide whether a
  container is "healthy"; Postgres uses `pg_isready`.
- **Histogram (Prometheus)**, a bucketed metric for tail-latency
  analysis; `inference_latency_seconds` is this project's one histogram.
- **Instrumentator (prometheus-fastapi-instrumentator)**, FastAPI
  middleware that auto-exposes standard HTTP metrics on `/metrics`.
- **Label (Prometheus)**, a `key="value"` attached to a metric to slice
  it; low-cardinality ones only.
- **Lifespan (FastAPI)**, async context manager that runs at app
  startup/shutdown; loads models here ([06](06-serving-api.md#app-lifecycle--servingappmainpy)).
- **MLflow tracking server**, the process that stores experiment
  metadata in Postgres and artifacts on disk.
- **OpenAPI**, the REST API schema format; FastAPI generates it
  automatically at `/openapi.json` and renders Swagger at `/docs`.
- **Panel (Grafana)**, one chart/stat/pie on a dashboard.
- **Postgres**, shared metadata store for Airflow and MLflow; not used
  for application data.
- **Prometheus**, time-series database and alert engine; scrapes
  `/metrics` endpoints.
- **PromQL**, Prometheus Query Language; used in alert rules and
  Grafana panels.
- **Provisioning (Grafana)**, auto-loading datasources and dashboards
  from YAML/JSON at startup.
- **Pydantic**, Python runtime type-validation library; every API
  schema here is a Pydantic `BaseModel`.
- **Scrape**, Prometheus's periodic HTTP GET against a metrics endpoint.
- **Swagger UI**, the interactive API explorer FastAPI serves at
  `/docs`.
- **TSDB**, Time-Series Database; Prometheus's storage engine.
- **Uvicorn**, ASGI server that runs the FastAPI app.
- **Venv (virtual environment)**, per-service Python environment; this
  project has five of them ([03](03-infrastructure.md#per-service-virtual-environments)).
- **Volume (Docker)**, named persistent storage, decoupled from the
  container lifecycle; `postgres_data` and `mlflow_artifacts` are the
  two here.

## Where to go next

- [02, ML concepts](02-ml-concepts.md) for intuition behind the
  ML-specific terms above.
- [10, Limitations and extensions](10-limitations-and-extensions.md)
  is the last stop, the consolidated "what you'd change" list and
  list.
