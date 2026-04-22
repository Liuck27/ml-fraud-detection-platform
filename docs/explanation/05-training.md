# 05 — Training

> **What this page answers:** How two very different models are trained on
> the same Parquet file, why MLflow is the bookkeeping layer, and how the
> champion/challenger aliases decide what the API serves.

You should read [02 — ML concepts § SMOTE](02-ml-concepts.md#smote), [§
Autoencoders](02-ml-concepts.md#autoencoders-for-anomaly-detection), and
[§ MLflow vocabulary](02-ml-concepts.md#mlflow-vocabulary) before or
alongside this page.

## Why train two models

The project ships with two fundamentally different models for the same
task:

| Model | Paradigm | Sees labels? | Catches novel fraud? | Explainable? |
|---|---|---|---|---|
| **XGBoost** | Supervised, gradient-boosted trees | Yes | Only patterns like training data | Yes (SHAP) |
| **Autoencoder** | Unsupervised anomaly detection | Only legit at training | Yes, by definition | Not directly |

Supervised models dominate where you have labels — they get calibrated
probabilities and clean SHAP explanations. But their blind spot is
*new* fraud patterns: a model trained on last year's fraud signatures
will miss this year's novel attack. An autoencoder trained only on
legitimate transactions flags *anything unusual*, novel or not, at the
cost of harder-to-explain scores and lower precision.

Running both gives you [A/B testing](02-ml-concepts.md#a-b-testing-in-ml-serving)
and lets you compare them on real traffic. The setup also demonstrates
two paradigms in one repo, which is the portfolio story.

## XGBoost — `training/train_xgboost.py`

Entry point: `training/train_xgboost.py:102` (`main`).

### The feature contract

`FEATURE_COLS` at lines 54-60 is the authoritative list of 33 features:

```python
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + [
    "amount_log", "amount_zscore", "hour_of_day",
    "is_night", "v1_v2_interaction",
]
```

Raw `Amount` and `Time` are intentionally excluded — `amount_log` /
`amount_zscore` and `hour_of_day` / `is_night` encode them in better
forms (see [04 § Feature engineering](04-data-and-features.md#feature-engineering)).
This list must match the serving loader's `FEATURE_COLS` exactly
(`serving/app/models/loader.py:32-38`) — if they drift apart,
training-serving skew breaks predictions silently.

### Data split — `train_xgboost.py:108-110`

```python
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

- **80/20 random split**, stratified on `Class` so both sides see the
  same ~0.17% fraud rate.
- **Not time-aware.** The dataset is chronologically ordered (`Time`
  increases monotonically), so a random split leaks future-into-past
  information. In production you'd split by time (train on days 1-N,
  validate on day N+1). Flagged as a known honest limitation in
  [10 — Limitations](10-limitations-and-extensions.md).

### Scaling — `train_xgboost.py:113-115`

`StandardScaler` fit on the training split only, then used to transform
both splits. The fitted scaler is pickled and logged as an MLflow
artifact (lines 156-160) so the serving container can apply **the exact
same scaling** at inference time, not fit a new scaler on serving data.
This is the standard way to avoid training-serving skew on feature
scaling.

### SMOTE — `train_xgboost.py:118-123`

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
```

SMOTE synthesizes new minority-class samples until the classes are
balanced. See [02 § SMOTE](02-ml-concepts.md#smote) for how it works.

After SMOTE, the training set typically goes from ~227k legit vs ~394
fraud to ~227k vs ~227k. The validation set is **never** SMOTE'd —
that would make metrics meaningless.

### Model config — `train_xgboost.py:88-97`

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=n_neg / n_pos,
    eval_metric="aucpr",
    ...
)
```

- **300 trees, depth 6, lr 0.05.** Standard tabular baseline — not
  aggressively tuned. `n_estimators × learning_rate ~ 15` is the rough
  "how much total signal are we learning" knob.
- **`scale_pos_weight`** is "belt and suspenders" with SMOTE (line 86
  comment). Both are working on the same problem (class imbalance); in
  theory using both is redundant. In practice it's cheap extra
  regularisation toward the minority class. See [02 § Class imbalance](02-ml-concepts.md#class-imbalance).
- **`eval_metric="aucpr"`** tells XGBoost to optimise PR-AUC internally
  when it needs a single metric.

**Not done:** no cross-validation, no Optuna/hyperparameter sweep, no
early stopping. All of those would be the next productivity
investments if you were tuning seriously.

### Threshold calibration — `evaluate.py:47-72` via `train_xgboost.py:129`

```python
threshold = find_optimal_threshold(y_val.values, y_pred_proba)
```

`find_optimal_threshold` sweeps the PR-curve's threshold array and
picks the one minimising total cost:

```python
cost = cost_fp * fp + cost_fn * fn    # default cost_fn=10, cost_fp=1
```

The default 10:1 cost ratio encodes the project's assumption that one
missed fraud is worth 10 customer-friction events. That single choice
drives the downstream operating point and is configurable per-call. See
[02 § Decision threshold tuning](02-ml-concepts.md#decision-threshold-tuning).

The chosen threshold is logged as a metric (`evaluate.py:43`) so the
serving loader can read it back from MLflow instead of hardcoding 0.5.

### What gets logged to MLflow

Inside the `with mlflow.start_run()` block (`train_xgboost.py:125-168`):

- **Parameters** (`log_params` at 133-142): `n_estimators`, `max_depth`,
  `learning_rate`, `smote`, `test_size`, `n_features`.
- **Metrics** (`log_metrics` at 143): `auc_roc`, `pr_auc`, `f1`,
  `precision`, `recall`, `threshold` — all from `compute_metrics`.
- **Figures** (146-149): ROC curve PNG, PR curve PNG.
- **Scaler** (156-160): `scaler.pkl` logged under `artifact_path="scaler"`.
- **Model** (163-168): `mlflow.xgboost.log_model` registers under
  `registered_model_name="fraud-xgboost"` with an `input_example` so
  MLflow captures the signature.

### Registry promotion — `train_xgboost.py:182-187`

After the run closes, the newly registered version is promoted to
`champion`:

```python
latest = max(versions, key=lambda v: int(v.version))
promote_to_champion(MODEL_NAME, latest.version)
```

`promote_to_champion` (at `training/model_registry.py:12-16`) is a
thin wrapper that calls
`client.set_registered_model_alias(model_name, "champion", version)`.

**Important:** the one-shot training script promotes *unconditionally*.
The retrain DAG (`airflow/dags/retrain_dag.py:130-140`) does the same
thing *conditionally* — only if the new version's PR-AUC beats the
existing champion's. See [04 § The retrain DAG](04-data-and-features.md#the-retrain-dag).

### Target metric

The "done" bar is **AUC-ROC > 0.95** (line 178-179 prints a warning
otherwise). In practice the model clears ~0.97 comfortably on this
dataset. PR-AUC is the more honest metric — typically around 0.85 on
the validation split.

## Autoencoder — `training/train_autoencoder.py`

Entry point: `training/train_autoencoder.py:198` (`main`).

Same input features (lines 53-59). Different training philosophy.

### Architecture — `train_autoencoder.py:78-98`

```
Input(33) → 64 → 32 → 16 → 32 → 64 → Output(33)
```

- Six `Linear` layers separated by ReLU, mirrored encoder/decoder.
- Bottleneck is 16 units. The network can only use this compressed
  representation to reconstruct a 33-dim input, so it has to learn the
  structure of *normal* transactions — no capacity for memorising
  outliers.

### Trains only on legitimate data — `train_autoencoder.py:209-213`

```python
mask_legit = y_train == 0
X_train_legit = X_train_df[mask_legit]
scaler = StandardScaler()
scaler.fit(X_train_legit)
```

The fitted scaler **and** the model both see only `Class == 0` rows at
training. Fraud is never used as a training signal; the labels are only
used later on the validation split for threshold calibration and metric
computation. This is what "unsupervised" means in practice here.

### Training loop — `train_autoencoder.py:159-182`

- 50 epochs, batch size 256, AdamW optimizer, lr 1e-3.
- MSE loss between input and reconstruction.
- Standard PyTorch loop — no fancy scheduler, early stopping, or
  gradient clipping.

### From error to probability — `train_autoencoder.py:225-238`

Reconstruction error is just a non-negative real number. To plug the
autoencoder into the same A/B-testing surface as XGBoost, the code
turns errors into a pseudo-probability in `[0, 1]`:

```python
legit_errors = errors[y_val.values == 0]
p99_legit = float(np.percentile(legit_errors, 99))
scores = np.clip(errors / (p99_legit * 2), 0.0, 1.0)
```

- Divide by twice the 99th-percentile legit error (a stable denominator
  that isn't pulled up by fraud outliers).
- Clip to `[0, 1]`.
- Anything above roughly `2 × p99_legit` is a saturation at 1.0.

The threshold search then runs on these normalised scores
(`find_optimal_threshold` at line 234). The raw-error threshold is
stored alongside, because the serving pyfunc wrapper uses raw errors.

### The `AutoencoderPyfunc` wrapper — `train_autoencoder.py:106-138`

MLflow can't natively serve a PyTorch model the same way it serves
XGBoost. You wrap it in an `mlflow.pyfunc.PythonModel` so serving code
can call `.predict(df)` uniformly on either model.

Three artifacts are bundled (lines 267-286):

- `model.pt` — the scripted PyTorch model via `torch.jit.script`
- `scaler.pkl` — the `StandardScaler` fit on legit training data
- `threshold.txt` — the raw-error threshold

At load time (`load_context` at 113-121) all three are restored. At
predict time (123-138):

```python
X_scaled = self.scaler.transform(model_input.values)
recon = self.model(X_tensor).numpy()
errors = np.mean((X_scaled - recon) ** 2, axis=1)
proba = np.clip(errors / (self.threshold * 2), 0.0, 1.0)
```

and returns a DataFrame with `fraud_probability` and
`reconstruction_error` columns.

**Windows-specific gotcha** (line 264-266 comment): `Path.as_posix()` is
called on artifact paths so MLflow records forward slashes in the
`MLmodel` file. Without this, `os.path.join` on Windows bakes in
backslashes and the model fails to load inside the Linux serving
container.

### Registry promotion — `train_autoencoder.py:303-309`

Same pattern as XGBoost but promotes to **`challenger`** instead of
`champion`. The API's A/B router (see [06 § A/B routing](06-serving-api.md))
sends `AB_CHALLENGER_FRACTION` (0.20 by default) of requests to the
challenger.

## MLflow, concretely

See [02 § MLflow and registry vocabulary](02-ml-concepts.md#mlflow-vocabulary)
for definitions. Here's how those concepts map to this codebase.

### Tracking URI

`MLFLOW_TRACKING_URI` is read from the environment at the top of each
training script:

- **From inside Docker Compose**: `http://mlflow:5000` (service DNS).
- **From a host terminal** running `make train-xgboost`:
  `http://localhost:5000`.

If the env var isn't set, both scripts default to `http://localhost:5000`
(lines `train_xgboost.py:64` and `train_autoencoder.py:63`).

### Experiments

Two experiments, each a namespace for runs:

- `fraud-detection-xgboost` (`train_xgboost.py:104`)
- `fraud-detection-autoencoder` (`train_autoencoder.py:200`)

Each `make train-xgboost` or `make train-autoencoder` invocation
creates one new run under the corresponding experiment.

### Registered models

Two registered models, configurable via env (`MODEL_XGBOOST_NAME`,
`MODEL_AUTOENCODER_NAME`), defaulting to:

- `fraud-xgboost`
- `fraud-autoencoder`

Each run ends with `registered_model_name=MODEL_NAME` so the run's
model artifact becomes version N of that registered model.

### Aliases: champion and challenger

These are the production-to-code contract:

| Alias | Meaning | Serving route |
|---|---|---|
| `champion` | The default production model | Receives ~80% of traffic |
| `challenger` | The experimental model | Receives ~20% (via `AB_CHALLENGER_FRACTION`) |

FastAPI resolves `models:/fraud-xgboost@champion` and
`models:/fraud-autoencoder@challenger` at startup. Promoting a new
version to `champion` is one `set_registered_model_alias` call
(`training/model_registry.py:15`) — zero lines of serving code change.
That's the point of aliases.

### The helper module — `training/model_registry.py`

Tiny wrapper around `MlflowClient`:

- `promote_to_champion(model_name, version)` — `:12-16`
- `promote_to_challenger(model_name, version)` — `:19-23`
- `get_latest_version(model_name)` — `:26-32`
- `get_champion_run_id(model_name)` — `:35-39`

The training scripts import from here; the retrain DAG uses the
`MlflowClient` directly. Both patterns coexist; the helpers just keep
the training-script code readable.

## Shared evaluation — `training/evaluate.py`

Used by both training scripts to avoid drift between XGBoost and
autoencoder metric definitions.

| Function | Line | What |
|---|---|---|
| `compute_metrics` | 25-44 | `auc_roc`, `pr_auc`, `f1`, `precision`, `recall`, `threshold` |
| `find_optimal_threshold` | 47-72 | Cost-weighted threshold sweep (default 10:1) |
| `plot_roc_curve` | 75-92 | Matplotlib ROC figure for MLflow |
| `plot_pr_curve` | 95-111 | Matplotlib PR figure for MLflow |

Note line 9-11: `matplotlib.use("Agg")` before any plot import. This
switches to a non-interactive backend so the module is safe in CI and
on headless servers (no X display required).

## Running training

```bash
make train-xgboost        # ~2-3 minutes on a laptop
make train-autoencoder    # ~3-5 minutes (50 epochs on CPU)
make train                # both, sequentially
```

Each command runs inside the `training/.venv` so `torch`, `xgboost`, and
`imblearn` don't pollute the dev environment.

Prerequisite: `data/processed/features.parquet` must exist. If it
doesn't, both scripts raise `FileNotFoundError` with a hint to run the
ingestion DAG (`train_xgboost.py:71-75`, `train_autoencoder.py:147-151`).

## Limitations

- **Random split, not time-based.** Overstated validation metrics on a
  temporally ordered dataset. Biggest honest caveat in the whole
  project.
- **No hyperparameter search.** Both models use hand-picked configs.
  Optuna on the training venv would be one afternoon of work.
- **No cross-validation.** A single 80/20 split. With 492 frauds total,
  CV would give more stable PR-AUC estimates.
- **Autoencoder architecture is generic.** No hyperparameter tuning on
  the bottleneck size, layer count, or dropout. It's fine, not
  optimal.
- **Retraining the autoencoder isn't automated.** The `retrain` DAG
  only retrains XGBoost. Manual invocation of
  `make train-autoencoder` is expected for the challenger.
- **No shadow evaluation.** Champion promotion is decided on validation
  PR-AUC, not on live-traffic shadow scoring.
- **Memory-bound training.** Both scripts read the full Parquet into
  memory. Fine at 284k rows; would need streaming at 100M+.

## Where to go next

- [06 — Serving API](06-serving-api.md) is where these models get
  loaded and actually used — the A/B router, the SHAP explainer, the
  `amount_zscore=0` quirk all live there.
- [07 — Monitoring](07-monitoring.md) covers how runtime metrics reveal
  whether the champion or challenger is winning.
- [02 § MLflow vocabulary](02-ml-concepts.md#mlflow-vocabulary) is the
  primer if the experiment/run/artifact/registry distinction still
  feels fuzzy.
