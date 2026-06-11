# 04, Data and features

> **What this page answers:** Where the data comes from, what each column
> means, how it moves from CSV to Parquet, and what every engineered
> feature is actually capturing.

If you haven't already, skim [02, ML concepts § Feature engineering](02-ml-concepts.md#feature-engineering-for-tabular-fraud)
first, this page assumes you know *why* feature engineering matters and
focuses on the *what*.

## The dataset

**Kaggle Credit Card Fraud Detection**, authored by Worldline / ULB.

- **Rows**: 284,807 transactions.
- **Frauds**: 492 (0.172%).
- **Time range**: two days of European cardholder transactions.
- **Columns**: 31 total, `Time`, `Amount`, `Class`, and `V1` through `V28`.

### Why the features are called V1 … V28

The original data is sensitive cardholder information, merchant IDs, card
numbers, country codes, transaction categories. Publishing them raw
would break both user privacy and Worldline's contractual obligations.
So the authors applied **PCA** (Principal Component Analysis) to the
real features and published the first 28 principal components as
`V1 … V28`.

Consequences:

- You cannot ask "what does `V4` mean?", it's a linear combination of
  dozens of underlying features, not a real-world quantity.
- **`Amount`** and **`Time`** are *not* PCA-transformed and remain in their
  original units (Euros and seconds elapsed since the first transaction
  in the dataset).
- You cannot do human-intuitive feature engineering on `V1 … V28`. The
  engineered features in this project all derive from `Amount` and
  `Time`, which is the honest limit of what you can do without access
  to the raw data.

### Why this dataset instead of something else

- **Real enough.** It's an actual production payment dataset, not toy
  synthetic data.
- **Small enough.** ~150 MB, 284k rows, fits in memory, trains in
  minutes, fast iteration.
- **Imbalanced enough.** 0.172% positive rate forces you to confront
  real class-imbalance countermeasures
  (SMOTE, threshold tuning, see
  [02 § Class imbalance](02-ml-concepts.md#class-imbalance)) rather than
  faking the problem on a balanced dataset.

**Honest limitations:**

- **No user IDs.** Real fraud systems rely heavily on per-card velocity
  features ("how many transactions in the last 10 minutes on this
  card?"). You can't build those here.
- **No merchant or location data.** Merchant category, country, MCC
  codes are all baked into the PCA components, invisible.
- **Two days only.** Long-horizon drift (weekly, seasonal) can't be
  tested on this dataset.

## Getting the data

`scripts/download_data.py` handles the Kaggle download. Prereqs:

1. A Kaggle account and an API token (download from
   `https://www.kaggle.com/settings`).
2. `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`.

```bash
make download-data
# writes data/raw/creditcard.csv
```

If you already have the CSV (e.g. downloaded manually), drop it at
`data/raw/creditcard.csv` and skip the script, the ingestion DAG does
not care how it got there.

**`data/raw/` is gitignored.** The file is too large to commit and has
a license that doesn't permit redistribution in this repo.

## Postgres schema and `init_db.sql`

The Postgres container is the shared metadata store for Airflow and
MLflow, but project data itself does not live in Postgres, it lives in
Parquet on disk. On first startup, `scripts/init_db.sql` runs
automatically and creates two extra databases:

- **`mlflow_db`**, MLflow tracking and registry tables.
- **`airflow_db`**, Airflow DAG runs, task instances, variables, pools.

The main `fraud_db` (from `POSTGRES_DB` in `.env`) exists but the
current pipeline doesn't write any application tables to it. All
processed features land in a Parquet file, not a SQL table.

That's a design choice worth calling out: for this dataset size,
Parquet is faster to scan, faster to write, easier to version, and
doesn't require the training container to run SQL queries against
Postgres. If rows climbed to hundreds of millions, a real OLAP store
(ClickHouse, DuckDB, BigQuery) would be the right move, not more
Parquet files.

## The `data_ingestion` DAG

File: `airflow/dags/data_ingestion_dag.py` (98 lines).

```
validate_csv  >>  engineer_and_write
```

Trigger it manually from the Airflow UI at `http://localhost:8080`
(login `admin` / `admin`) or on the CLI:

```bash
airflow dags trigger data_ingestion
```

There is no schedule, `schedule=None` at line 85 means manual only.
That's the right call here: re-running against the same CSV
is a no-op; you'd only re-run when the dataset changed.

### Task 1, `validate_csv` (`data_ingestion_dag.py:26-51`)

Fails loudly if the CSV isn't what we expect:

- **File exists.** If not, raises `FileNotFoundError` with a hint to run
  `make download-data` (`data_ingestion_dag.py:30-34`).
- **Schema.** Reads the first 5 rows and checks all 31 expected columns
  are present (`Time`, `Amount`, `Class`, `V1 … V28`).
  Missing columns raise `ValueError` (`data_ingestion_dag.py:36-40`).
- **Row count.** Full read; asserts at least 280,000 rows (loose lower
  bound to tolerate trivial dataset updates).
- **Fraud count.** Logs the actual positive rate, useful sanity
  check that prints e.g. `284,807 rows, 492 frauds (0.173%)`.

### Task 2, `engineer_and_write` (`data_ingestion_dag.py:54-77`)

Reads the raw CSV, applies `engineer_features()` (below), writes Parquet
to `data/processed/features.parquet`. The final DataFrame has the
original 31 columns plus 4 engineered columns (35 total).

### Idempotency

Both tasks are idempotent:

- `validate_csv` reads only. Safe to re-run.
- `engineer_and_write` overwrites `features.parquet`. Re-running produces
  the same output as long as the input CSV hasn't changed.

If you half-ran the DAG and killed the scheduler mid-task, re-triggering
it is safe.

## Feature engineering

File: `airflow/plugins/feature_engineering.py` (56 lines, pure
functions).

These functions deliberately live in `airflow/plugins/` rather than
`airflow/dags/` so they can be imported by other contexts, notably the
FastAPI serving code needs the same transforms at inference time.
`airflow/plugins/` is a directory Airflow auto-adds to `PYTHONPATH`, and
the serving container recreates the same transforms in
`serving/app/models/loader.py` to keep training-serving symmetry.

Each function takes a DataFrame and returns a *new* DataFrame with
additional columns. No mutation, no I/O, trivial to test.

### `log_transform_amount` → `amount_log` (`feature_engineering.py:25-29`)

```python
out["amount_log"] = np.log1p(out["Amount"])
```

- **What**: the natural log of `(Amount + 1)`. `log1p` handles the
  `Amount == 0` case without a divide-by-zero.
- **Why**: transaction amounts are heavily right-skewed, most
  purchases are under €100, a handful are over €10,000. A linear model
  treating a €5,000 transaction as "50× more important than a €100
  one" overfits to rare big purchases. Taking the log compresses that
  tail.
- **When it helps**: tree models (XGBoost) are invariant to monotonic
  transforms, so `amount_log` is redundant for XGBoost in theory. It
  still helps in practice because the autoencoder and SHAP rendering
  benefit from a less skewed scale.

### `extract_time_features` → `hour_of_day`, `is_night` (`feature_engineering.py:32-41`)

```python
out["hour_of_day"] = (out["Time"] // 3600 % 24).astype(int)
out["is_night"] = out["hour_of_day"].apply(lambda h: h >= 22 or h < 6)
```

- **What**: `Time` is "seconds since first transaction in the dataset".
  Dividing by 3600 gives hours; `% 24` folds into a day-of-week-agnostic
  hour. `is_night` is true for 22:00-06:00.
- **Why**: fraud has a diurnal pattern. Card-not-present fraud often
  spikes at night when cardholders are asleep and less likely to notice
  SMS alerts. A "23:48 transaction for €350 in a country the holder
  doesn't live in" is suspicious; the same transaction at 13:00 is
  not.
- **Caveat**: `Time` is relative to the start of the dataset, not a
  real wall clock. Hour-of-day is an *approximation*. In a production
  system you'd use the transaction's actual timezone-aware timestamp.

### `compute_interaction_features` → `v1_v2_interaction` (`feature_engineering.py:44-49`)

```python
out["v1_v2_interaction"] = out["V1"] * out["V2"]
```

- **`v1_v2_interaction`** is `V1 * V2`. Interaction terms let linear
  models capture effects that depend on two features jointly (a
  particular combination of PCA components correlating with fraud).
  For a tree model like XGBoost this is partly redundant because trees
  discover interactions implicitly via splits. It's kept because it's
  a common convention and doesn't hurt.
- **A removed sibling**: this function used to also compute a
  batch-level `amount_zscore` (Amount standardised against the current
  batch's mean/stdev). It was dropped because the statistic depended on
  whatever rows happened to be in the batch — leaking validation rows
  into training, and impossible to reproduce for a single-row serving
  request. `amount_log` plus the model's `StandardScaler` carry the
  same signal without the skew.

### `engineer_features` (`feature_engineering.py:51-56`)

The composition pipeline, calls the three transforms in sequence:

```python
def engineer_features(df):
    df = log_transform_amount(df)
    df = extract_time_features(df)
    df = compute_interaction_features(df)
    return df
```

That's the public API. DAGs, training scripts, and tests all call this
function, not the individual transforms.

### Feature limitations, honestly

- **No user-level aggregates.** Velocity features ("N transactions in
  the last hour on this card", "avg amount last 30 days") are where
  fraud systems earn their keep. You can't build them here, the
  dataset has no user ID.
- **No merchant/geography.** Would be a huge signal. PCA-masked.
- **No time-series targets.** No "is this transaction's country
  different from the last 10 on this card?" because there's no last
  10.

## The EDA notebook

`training/notebooks/eda.ipynb` is a Jupyter notebook for humans, not
part of any pipeline. It exists to make two things vivid:

- **The class imbalance.** A bar chart of class counts that
  stretches the fraud bar to almost invisible, the single most
  important thing a new reader has to internalise.
- **Feature distributions.** Histograms of `Amount`, `amount_log`,
  and the engineered features, conditioned on `Class`. You can see at
  a glance which features separate fraud from legit (V11, V12, V14 are
  particularly clean; V1, V2 are not).
- **Correlation heatmap.** Shows `Class` correlations with V1-V28 so
  you can eyeball which features matter most.

Nothing downstream depends on the notebook, it's diagnostic. It's
committed *executed* — with cell outputs in place — so the charts
render directly on GitHub and a reader sees the plots without needing
the dataset or a Jupyter kernel at all.

## The `retrain` DAG

File: `airflow/dags/retrain_dag.py` (149 lines).

```
validate_features  >>  train_xgboost  >>  evaluate_and_promote
```

Schedule: manual trigger only (`schedule=None` at `retrain_dag.py:136`).
This is a demo stack that isn't continuously running, so a cron schedule
would only accumulate failed runs between sessions. In production you
would set e.g. `@weekly` — a realistic cadence for fraud model
refreshes; the schedule is a one-line change, and the interesting part
(the gated promotion below) is identical either way.

### Task 1, `validate_features` (`retrain_dag.py:53-76`)

Checks `data/processed/features.parquet` exists and has all expected
columns (V1-V28, `Amount`, `Class`, plus the four engineered features
defined at `retrain_dag.py:43-50`). The column check reads the parquet
*schema* (`pyarrow.parquet.read_schema`) rather than loading the data
with `columns=`: passing an expected-column list to `read_parquet`
would make pandas raise its own error on a missing column before the
validation could produce a readable one.

### Task 2, `train_xgboost` (`retrain_dag.py:79-111`)

Runs `training/train_xgboost.py` as a **subprocess** inside the Airflow
container. This works because `./training` is volume-mounted read-only
at `/opt/airflow/training` (`docker-compose.yml`) and the Airflow image
installs the training dependencies — scikit-learn, xgboost,
imbalanced-learn, mlflow, matplotlib, pinned to match
`training/requirements.txt` (`airflow/Dockerfile`). Torch is deliberately
excluded: the DAG only retrains XGBoost.

This is a deliberate shortcut with an honest caveat right in the
docstring (`retrain_dag.py:20-23`):

> In a production deployment you would replace this with a
> DockerOperator or KubernetesPodOperator pointing at a dedicated
> training image that has xgboost, torch, and imbalanced-learn
> installed.

Running training as a subprocess means the Airflow image carries the
training dependencies, which puffs it up. A real production setup
would hand the work off to a separate training container so Airflow
stays a pure orchestrator.

### Task 3, `evaluate_and_promote` (`retrain_dag.py:114-129`)

The interesting part. The training script deliberately only *registers*
a new model version — it never moves the `champion` alias itself (see
[05 § Registration vs. promotion](05-training.md)). This task is the
quality gate, and it delegates to **the same function** host-side runs
use: `promote_champion_if_better` in `training/model_registry.py:43-91`
(importable thanks to the volume mount). The rule:

1. Find the just-registered version of `fraud-xgboost`.
2. If no champion exists yet, promote unconditionally (first run).
3. Otherwise promote only if the new version's PR-AUC is at least as
   good as the current champion's; if not, keep the champion and log why.

One gate, two entry points: `make promote` (called automatically by
`scripts/run_training.sh`) on the host, and this task inside Airflow.
The promotion rule can never silently diverge between the two paths.

This is the production-grade pattern: **don't auto-promote without a
quality gate**. Using the `champion` alias (rather than hardcoding a
version in serving code) means promotion is a single MLflow API call,
the serving container picks up the new version on next restart without
any code change.

**Limitations:**

- Only XGBoost is retrained. The autoencoder is trained once manually
  and not on the retrain schedule.
- No statistical significance test, `>=` on a single PR-AUC number
  can promote a model that's better by noise. A real gate would use
  e.g. bootstrapped CI or a shadow-traffic evaluation.
- No automatic rollback. If a promoted model starts misbehaving in
  production, the operator has to manually move the alias back.

## How training, serving, and the DAGs share the feature definition

The engineered features appear in three places:

1. **`airflow/plugins/feature_engineering.py`**, the canonical
   definition. Called by the ingestion DAG.
2. **`training/train_xgboost.py`** and **`training/train_autoencoder.py`**
  , read the Parquet from step 1, so they inherit the definition
   transitively.
3. **`serving/app/models/loader.py`**, re-implements the same
   transforms on the inference path for single-row requests.

If you change `feature_engineering.py`, you must also update the
serving prep function, or training-serving skew silently breaks model
quality. There's no automatic check for this, a real production
system would use a feature store (like Feast) precisely to remove this
risk. See [02 § MLflow and registry vocabulary](02-ml-concepts.md#mlflow-vocabulary)
for why Feast wasn't worth it here.

## Where to go next

- [05, Training](05-training.md) picks up where the DAG leaves off:
  how the Parquet becomes a model registered in MLflow.
- [06, Serving API](06-serving-api.md) is where the
  training-serving feature prep lives.
