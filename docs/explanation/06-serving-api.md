# 06, Serving API

> **What this page answers:** How a transaction goes from JSON to a
> scored prediction with SHAP explanation, how the A/B router picks a
> model, and where the quirks (degraded mode,
> degraded-start behaviour) live.

You should have read [02 § A/B testing](02-ml-concepts.md#a-b-testing-in-ml-serving)
and [02 § SHAP](02-ml-concepts.md#shap-values), this page assumes both.

## Why FastAPI

The service is a synchronous inference API with four real endpoints
(`/predict`, `/predict/batch`, `/health`, `/models`) plus the
auto-exposed `/metrics` and `/docs`. Three properties mattered when
picking FastAPI:

- **Pydantic v2 on every request**, so you get runtime validation for
  free (30 floats, the right field names, correct types) and an
  auto-generated OpenAPI schema at `/docs`.
- **Type hints as the API spec.** Function signatures are the contract
 , you change a field in a Pydantic model and the OpenAPI doc, mypy
  check, and runtime validator all update together.
- **Async-capable ASGI server.** Even though the predict code is
  synchronous (XGBoost and PyTorch inference are CPU-bound and don't
  benefit from `async`), the server stays responsive under load and
  integrates cleanly with the Prometheus instrumentator.

Flask would work; FastAPI is a clear upgrade for a typed ML API.

## App lifecycle, `serving/app/main.py`

### Startup: the `lifespan` context manager (`main.py:24-33`)

```python
@asynccontextmanager
async def lifespan(application: FastAPI):
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    get_registry().load(settings)       # pull both models from MLflow
    get_explainer()                     # warm up SHAP TreeExplainer
    yield
```

Why this matters:

- **Models load once at startup**, not per request. The XGBoost model,
  its `StandardScaler`, the autoencoder pyfunc, and their metadata all
  come from MLflow here. See `loader.py:64-79`.
- **SHAP is warmed up on startup.** `shap.TreeExplainer(model)` does
  non-trivial work at first call, pre-computing weights from the
  XGBoost trees. If you skipped the warm-up, the first `/predict`
  request would be 200ms slower than subsequent ones. Warming it up
  avoids that tail.
- **Degraded start is allowed.** `ModelRegistry.load` catches
  exceptions (`loader.py:114-117`, `136-139`) rather than crashing the
  app. If MLflow is down or models aren't registered yet, the API
  starts, `/health` reports `"degraded"`, and `/predict` returns 503.
  This is better than crash-looping because it's debuggable, you
  can still hit `/health` to see what failed.

### Instrumentation, `main.py:44`

```python
Instrumentator().instrument(app).expose(app)
```

`prometheus-fastapi-instrumentator` adds the default HTTP metrics
(request count, in-progress, duration by endpoint/method/status) and
exposes them on `/metrics`. The project's four *custom* metrics live
in `metrics.py` and are emitted from the predict handler manually.

## Request schema, `serving/app/schemas.py`

### Single prediction, `schemas.py:16-60`

```python
class TransactionFeatures(BaseModel):
    V1: float; V2: float; ... V28: float
    Amount: float
    Time: float

class TransactionRequest(BaseModel):
    transaction_id: str
    features: TransactionFeatures
```

Notes:

- `V1`-`V28` are the PCA components (see [04 § The dataset](04-data-and-features.md#the-dataset)).
  All 30 fields are mandatory floats: any missing field returns 422
  with a clear validator error.
- **`Time` is required** (`schemas.py:55`), because `hour_of_day` and
  `is_night` are derived from it. An earlier version defaulted it to
  0.0, which silently placed every timestamp-less transaction at
  midnight — an hour that counts as *night*, so the model saw a
  distorted feature it had learned to trust. A fraud system always has
  a timestamp; requiring the field is the honest contract.
- `transaction_id` is an opaque string, the only thing the A/B router
  uses to bucket.

### Batch, `schemas.py:63-66`

```python
class BatchRequest(BaseModel):
    transactions: Annotated[
        list[TransactionRequest], Field(min_length=1, max_length=1000)
    ]
```

The 1-1000 window is enforced by Pydantic, not hand-coded. Reasoning:

- **Upper bound 1000.** Prevents a single request from monopolising a
  worker and blowing out p99 latency for everyone else.
- **Lower bound 1.** A zero-length batch is meaningless; reject at
  validation instead of returning an empty response.

### Response shapes, `schemas.py:83-111`

- `PredictionResponse` includes `fraud_probability`, `is_fraud`,
  `model_name` (`fraud-xgboost-champion` or
  `fraud-autoencoder-challenger`), `model_version`, full
  `Explanation`, `latency_ms`, and UTC `timestamp`.
- `BatchResponse` is a list of `BatchPredictionItem` (same fields
  minus the SHAP explanation, batch predictions skip SHAP to stay
  fast) plus `count` and `total_latency_ms`.

### The `protected_namespaces` escape hatch

Pydantic v2 reserves the `model_` prefix by default and warns on any
field that starts with it. This codebase has several such fields by
necessity (`model_name`, `model_version`, `model_xgboost_name`), so
every class that declares one clears the namespace — `ModelInfo`,
`PredictionResponse`, and `BatchPredictionItem` set
`model_config = ConfigDict(protected_namespaces=())` (`schemas.py:84`,
`97`, `132`), and `Settings` sets
`protected_namespaces=("settings_",)` in its `SettingsConfigDict`
(`config.py:13-15`). This is the documented work-around; without it,
every import of the app printed six warnings.

## The model registry, `serving/app/models/loader.py`

`ModelRegistry` is the stateful singleton that holds both models,
scaler, threshold, and metadata. Singleton via module-level `_registry`
at `loader.py:202` and `get_registry()` at `:205-206`.

### Feature column contract, `loader.py:32-37`

```python
FEATURE_COLS: list[str] = [f"V{i}" for i in range(1, 29)] + [
    "amount_log", "hour_of_day",
    "is_night", "v1_v2_interaction",
]
```

Must be identical to `training/train_xgboost.py:59-64` and
`training/train_autoencoder.py:53-58`. This is the canonical
feature-order contract; if it drifts, model predictions silently
return nonsense.

### Loading XGBoost, `loader.py:80-117`

```python
uri = f"models:/{settings.model_xgboost_name}@{settings.model_champion_alias}"
self._xgb_model = mlflow.xgboost.load_model(uri)
```

Then three more things happen:

1. **Resolve the alias to a version** (`loader.py:88-91`), so
   `/health` and `/models` can report the current version.
2. **Download the scaler** (`loader.py:95-98`), the pickled
   `StandardScaler` logged at training time. This is critical: using
   the production scaler, fit on training data only, is how we avoid
   training-serving skew on feature scales.
3. **Read the optimal threshold** (`loader.py:101-102`),
   `run_data.metrics.get("threshold", 0.5)`. Falls back to 0.5 if
   missing. This is the cost-weighted threshold from
   `find_optimal_threshold`, reused at inference.

### Loading the Autoencoder, `loader.py:119-139`

```python
uri = f"models:/{settings.model_autoencoder_name}@{settings.model_challenger_alias}"
self._ae_model = mlflow.pyfunc.load_model(uri)
```

Much simpler, the autoencoder is a self-contained pyfunc (its
scaler and threshold are bundled inside the model artifacts; see [05
§ AutoencoderPyfunc](05-training.md#the-autoencoderpyfunc-wrapper)).
No separate artifact download.

### Feature prep: `prepare_features`, `loader.py:145-169`

One method converts a validated `TransactionFeatures` into the
model-ready 32-column DataFrame, replicating the training-time
transforms exactly:

```python
amount_log = float(np.log1p(raw["Amount"]))
hour_of_day = int(raw["Time"] // 3600 % 24)
is_night = float(hour_of_day >= 22 or hour_of_day < 6)
v1_v2_interaction = raw["V1"] * raw["V2"]
```

Every transform here is **row-local** — it depends only on the values
in this one transaction, so a single-row request and a training batch
produce identical features for identical inputs. That property is
deliberate: an earlier `amount_zscore` feature was batch-dependent
(standardised against whatever rows happened to be processed together)
and could not be reproduced for a single-row request, so it was
removed from the feature set entirely. The batch endpoint simply calls
`prepare_features` once per transaction; with only row-local features
there is nothing a batch-level prep could do differently.

### `predict_xgb`, `loader.py:172-179`

```python
X_scaled = self._xgb_scaler.transform(df)
proba = float(self._xgb_model.predict_proba(X_scaled)[0, 1])
return proba, proba >= self._xgb_threshold
```

Scales the 32-feature row using the fitted training scaler, predicts
the positive-class probability, and applies the cost-weighted
threshold to produce `is_fraud`. The scaler receives the DataFrame
itself, not `.values`: it was fit on a named DataFrame at training
time, and handing it a bare array makes sklearn emit a "X does not
have valid feature names" warning on every request.

### `predict_ae`, `loader.py:181-186`

```python
result = self._ae_model.predict(df)
proba = float(result["fraud_probability"].iloc[0])
return proba, proba >= 0.5
```

The autoencoder pyfunc returns its own `fraud_probability` column
(already normalized via the legit-99th-percentile trick, see [05 §
From error to probability](05-training.md#from-error-to-probability)).
Uses a fixed 0.5 threshold on that score. The AE doesn't currently
honour a registry-logged threshold; that would be an easy extension.

## A/B routing, `serving/app/models/ab_testing.py`

```python
def route_to_challenger(transaction_id: str, challenger_fraction: float) -> bool:
    digest = hashlib.md5(transaction_id.encode(), usedforsecurity=False).hexdigest()
    bucket = int(digest, 16) % 100
    return bucket < int(challenger_fraction * 100)
```

Twenty-six lines total. The pattern:

- **MD5 the transaction_id, mod 100, compare.** Same ID always goes to
  the same model (deterministic). No server-side state, no sticky
  sessions, no cookies.
- **`usedforsecurity=False`** is the explicit way to tell Python
  ("Yes, I know MD5 is broken for crypto. I'm using it for
  load-balancing."). This bypasses FIPS mode objections on some
  platforms. It's not a bug; it's the right tool.
- **Uniformly distributed.** MD5 output bytes are indistinguishable
  from uniform random. Taking `int(digest, 16) % 100` gives a nearly
  uniform bucket in `[0, 99]`. Over enough transactions,
  `challenger_fraction = 0.2` produces ~20% challenger traffic; a
  statistical test in `serving/tests/test_ab_testing.py` verifies the
  split is within ±2% on 10,000 samples.

See [02 § A/B testing](02-ml-concepts.md#a-b-testing-in-ml-serving) for
the deeper "why" of champion/challenger.

### The fallback in `_select_model`, `predict.py:46-58`

```python
use_challenger = route_to_challenger(transaction_id, challenger_fraction)
if use_challenger and not registry.ae_loaded:
    use_challenger = False   # AE unavailable → fall back to champion
if not use_challenger and not registry.xgb_loaded:
    use_challenger = True    # champion unavailable → fall back to AE
return use_challenger
```

The A/B decision is a *suggestion*: if the target model isn't loaded,
route to the other one. This keeps the API serving something even in
degraded mode, as long as *one* model is up. If neither is loaded,
the top-level 503 at `predict.py:66-68` fires.

## SHAP explanations, `serving/app/models/explainer.py`

```python
class SHAPExplainer:
    def __init__(self, model, feature_names):
        self._explainer = shap.TreeExplainer(model)
```

`TreeExplainer` is SHAP's exact-for-trees backend, see [02 § SHAP](02-ml-concepts.md#shap-values).
It only works on tree models, which is why only the XGBoost champion
gets real explanations. The autoencoder challenger returns
`Explanation(top_features=[])` (`predict.py:92`).

### Top-k sorting, `explainer.py:33-44`

```python
shap_values = self._explainer.shap_values(X_scaled)
pairs = sorted(
    zip(self._feature_names, contributions),
    key=lambda x: abs(float(x[1])),
    reverse=True,
)
return [{"feature": f, "contribution": round(c, 4)} for f, c in pairs[:top_k]]
```

- Returns the 3 features (by default) with the largest *absolute*
  contribution, so features pushing the prediction *toward legit* show
  up too (negative SHAP value) instead of only those pushing toward
  fraud.
- Rounded to 4 decimals for a clean JSON response.

### Lazy singleton, `explainer.py:48-66`

```python
_explainer: SHAPExplainer | None = None

def get_explainer() -> SHAPExplainer | None:
    global _explainer
    if _explainer is None:
        ... registry = get_registry()
        if not registry.xgb_loaded: return None
        _explainer = SHAPExplainer(registry._xgb_model, FEATURE_COLS)
    return _explainer
```

Created on the first call, cached thereafter. The lifespan hook calls
it once at startup (`main.py:31`) so warm-up happens there. If
XGBoost isn't loaded, `get_explainer()` returns `None` and
`predict.py:37-38` substitutes an empty explanation.

## Custom Prometheus metrics, `serving/app/metrics.py`

Four custom metrics live there:

| Metric | Type | Labels | Emitted at |
|---|---|---|---|
| `inference_latency_seconds` | Histogram | `model_name` | `predict.py:101`, `158` |
| `inference_total` | Counter | `model_name`, `prediction` | `predict.py:102-104`, `159-161` |
| `inference_errors_total` | Counter | `model_name` | `predict.py:67`, `97`, `124`, `154` |
| `ab_test_assignments_total` | Counter | `model_variant` | `predict.py:73-75`, `134-136` |

Histogram buckets (`metrics.py:9`) are `[5ms, 10ms, 25ms, 50ms, 100ms,
250ms, 500ms, 1s]`. Most XGBoost predictions take 5-15ms; SHAP adds
~40-80ms. The 500ms bucket is the alert trigger (see [07 §
Alerts](07-monitoring.md)).

### Why these specific labels

- **`model_name`** combines the MLflow name and the alias,
  `fraud-xgboost-champion`, `fraud-autoencoder-challenger`. Good for
  slicing latency / throughput by which model was used. The error
  counter uses the same values (the full name is resolved before
  predicting, `predict.py:80-87`), so a PromQL join of errors against
  totals matches label-for-label; an earlier version labelled errors
  with the bare alias, which silently broke such joins. The only
  exception is `model_name="unknown"` on the no-models-loaded 503,
  where there is no model to name.
- **`prediction`** is just `"fraud"` / `"legit"`, used by the
  fraud-rate panel in Grafana.
- **`model_variant`** is `"champion"` / `"challenger"`, used by the
  A/B-split pie chart.

All four labels have *low cardinality* (bounded by model count, fraud
label, variant). If you added a `transaction_id` label by mistake,
Prometheus would explode, see [07 § Cardinality risk](07-monitoring.md).

## The predict handler, `serving/app/routes/predict.py:61-115`

End-to-end flow for a single request:

1. **Validate**, FastAPI runs the Pydantic validator on
   `TransactionRequest`; bad input returns 422 before we get here.
2. **Guard against no models** (`:66-68`), return 503 if neither
   model is loaded.
3. **Route** (`:70-75`), call `_select_model`, increment
   `AB_ASSIGNMENTS`.
4. **Prep features** (`:77-78`), apply the engineered features to the
   raw 30 floats.
5. **Resolve the model name** (`:80-87`), so error metrics carry the
   same label scheme as the success metrics.
6. **Predict** (`:89-98`), call the right model's `predict_*`,
   record errors on exception.
7. **Explain** (`:92-95`), SHAP if XGBoost, else empty.
8. **Record metrics** (`:100-104`), latency histogram, prediction
   counter.
9. **Respond**, build and return the `PredictionResponse`.

The batch endpoint (`:118-181`) repeats steps 3-8 in a loop over up
to 1000 transactions, but skips the SHAP explanation and reports a
single `total_latency_ms` plus per-item latencies.

## The other endpoints

### `/health`, `serving/app/routes/health.py`

Returns:

```json
{
  "status": "healthy" | "degraded",
  "models": {"champion": {...}, "challenger": {...}},
  "ab_test": {"champion_traffic": 0.80, "challenger_traffic": 0.20}
}
```

`"degraded"` if either model failed to load. This is the endpoint
integration tests hit to decide whether to run or skip.

### `/models`, `serving/app/routes/models.py`

Lists each loaded model with its MLflow version, alias role, computed
traffic percentage (derived from `AB_CHALLENGER_FRACTION`), and the
training metrics (`auc_roc`, `pr_auc`, `f1`) logged to MLflow. Useful
for a "which model version is in production right now" dashboard
tile.

### `/metrics`

Auto-exposed by the `Instrumentator()` call in `main.py:44`. Default
HTTP metrics plus every custom metric from `metrics.py`. Scraped by
Prometheus every 15s, see [07, Monitoring](07-monitoring.md).

### `/docs`

FastAPI's built-in Swagger UI at `http://localhost:8000/docs`. Free
with FastAPI, uses the OpenAPI schema generated from the Pydantic
models.

## Limitations

- **Single instance, single process.** `docker-compose.yml:122-148`
  runs exactly one `serving` container. No horizontal scaling, no
  rolling deploys. The `SERVING_WORKERS=1` env variable (in
  `.env.example:35`) is the knob to turn it up; the container would
  need a proper process manager (gunicorn with uvicorn workers) to
  take advantage.
- **No authentication on `/predict`.** Anyone on `fraud-net` can
  score. For a local stack that's fine; a real deployment would
  put Authorization header / API key / mutual TLS on the route.
- **No rate limiting.** A misbehaving client can blast requests and
  drown out everyone else.
- **No prediction persistence.** Predictions are returned to the
  caller and counted in Prometheus, not stored. A real production
  system logs every prediction with inputs and outputs for later
  review, essential for debugging and for offline drift/quality
  monitoring.
- **SHAP on every XGBoost request adds ~40-80ms.** Already in the
  histogram buckets. The `explanation` field could be made opt-in via
  a query parameter for latency-sensitive callers.
- **No graceful model reload.** Promoting a new champion in MLflow
  doesn't propagate to a running serving container, you have to
  restart it. A real production system would poll the registry or
  listen for a webhook.

## Where to go next

- [07, Monitoring](07-monitoring.md) picks up where `/metrics` ends
  and shows how Prometheus + Grafana + Evidently turn those numbers
  into operator-facing signals.
- [08, Testing and CI](08-testing-and-ci.md) covers how the predict
  handler is unit-tested with a mocked registry and how integration
  tests exercise the real container.
- [10, Limitations and extensions](10-limitations-and-extensions.md)
  ranks every item in the Limitations section above by impact.
