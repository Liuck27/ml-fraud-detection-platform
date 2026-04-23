# 08, Testing and CI

> **What this page answers:** What the test pyramid looks like for this
> project, why MLflow is mocked in unit tests but real in integration
> tests, and exactly what GitHub Actions does on every push.

## The pyramid

```
        /\
       /  \       Integration (tests/integration/)
      /----\      ~3 tests, real services, run locally only
     /      \
    /--------\    Unit, training (training/tests/)
   /          \   ~7 tests, pure numerics, no I/O
  /------------\  Unit, serving (serving/tests/)
 /              \ ~14 tests, mocked MLflow, run in CI
 ----------------
```

Two conscious choices:

- **Unit tests go wide at the serving layer.** Every endpoint has at
  least one happy path and one validation-failure test. Serving is
  where bugs cost the most.
- **Integration tests stay narrow.** Three tests, one per critical
  integration point (predict end-to-end, Prometheus scrape, MLflow
  registry). Any more would duplicate what unit tests already prove
  without the infrastructure cost.

No end-to-end tests through Airflow, because DAGs are manually
triggered in this project and not part of the serving SLA.

## Serving unit tests, `serving/tests/`

### Shared fixtures, `serving/tests/conftest.py`

This file is the heart of the serving test strategy. Three things
matter:

#### 1. `SAMPLE_FEATURES` (`conftest.py:26-30`)

A fixed dictionary with all 28 V-features plus `Amount=100.0` and
`Time=7200.0`. `Time=7200.0` is exactly 2 hours, that maps to
`hour_of_day=2, is_night=False` (since 2 is between 22 and 6 is False),
which gives tests a deterministic feature-engineering output.

#### 2. The `_make_registry` helper (`conftest.py:33-68`)

```python
reg = MagicMock()
# Metadata: version, threshold, metrics
# Prediction stubs: reg.predict_xgb.return_value = (0.03, False)
# BUT reg.prepare_features = real.prepare_features
```

The key insight: **mock everything that calls MLflow, keep the
feature-engineering code real**. If we mocked `prepare_features` too,
we'd lose coverage of the single most subtle piece of serving code
(the `amount_zscore = 0` logic, the `hour_of_day // 3600 % 24`
formula). By using the real transform, the tests protect against
accidental changes to feature logic.

Mocked pieces:

- `xgb_loaded` / `ae_loaded` flags
- `predict_xgb.return_value = (0.03, False)`, deterministic
- `predict_ae.return_value = (0.07, False)`, deterministic
- `_xgb_scaler.transform.return_value = np.zeros((1, 33))`, for the
  SHAP path

#### 3. The `mock_registry` and `client` fixtures (`conftest.py:76-94`)

`mock_registry` monkeypatches the module-level singleton in
`loader.py` with the fake registry **and** neutralises the SHAP
explainer so tests don't try to create a real `shap.TreeExplainer`
against a MagicMock (which would crash).

`client` wraps the real FastAPI `app` in a `TestClient`. The whole
app, routes, middleware, Pydantic validation, Instrumentator, runs
exactly as it does in production; only the MLflow-backed registry is
fake.

### `test_predict.py`, nine tests

| Test | What it locks in | Line |
|---|---|---|
| `test_predict_happy_path` | All response fields present and correctly typed | 24-37 |
| `test_predict_response_schema` | `transaction_id` echoed, probability in `[0,1]` | 40-45 |
| `test_predict_missing_feature_returns_422` | Drop `V1` → Pydantic rejects | 48-51 |
| `test_predict_missing_body_returns_422` | Empty body → 422 | 54-56 |
| `test_predict_batch_happy_path` | Batch returns list of correct length | 64-72 |
| `test_predict_batch_single_item` | 1-item batch works (min_length=1) | 75-78 |
| `test_predict_batch_empty_returns_422` | 0-item batch rejected | 82-84 |
| `test_predict_batch_over_limit_returns_422` | 1001-item batch rejected (max=1000) | 87-90 |
| `test_predict_batch_item_schema` | Each item preserves its `transaction_id` | 93-100 |

Every test uses `_txn()` (a helper at `test_predict.py:12-16`) that
generates a random `transaction_id` unless one is specified, giving
deterministic tests where it matters and randomised ones where it
doesn't.

### `test_ab_testing.py`, five tests

| Test | What it locks in | Line |
|---|---|---|
| `test_routing_is_deterministic` | Same ID → same result, 100× | 10-15 |
| `test_full_challenger_fraction_routes_all` | `fraction=1.0` → always challenger | 18-21 |
| `test_zero_challenger_fraction_routes_none` | `fraction=0.0` → never challenger | 24-27 |
| `test_approximate_split_ratio` | 10,000 samples, 20% target, ±2% tolerance | 30-38 |
| `test_different_ids_can_split` | 100 IDs at 50% split produce both outcomes | 41-44 |

The statistical test at line 30-38 is the most interesting:

```python
n = 10_000
challenger_count = sum(route_to_challenger(str(uuid.uuid4()), 0.20) for _ in range(n))
actual = challenger_count / n
assert abs(actual - 0.20) < 0.02
```

Ten thousand samples gives ~2% noise on a Bernoulli(0.2) mean, that's
the basis for the ±0.02 tolerance. Smaller `n` (say 1000) would have
noisier variance and flake occasionally; larger `n` (100,000) would
slow CI for no real gain.

## Training unit tests, `training/tests/test_evaluate.py`

Seven tests, all pure numerics, no I/O, no MLflow:

| Test | What it checks | Line |
|---|---|---|
| `test_compute_metrics_returns_expected_keys` | Returns the full metric dict | 50-61 |
| `test_compute_metrics_perfect_classifier` | Perfect separation → all metrics = 1.0 | 64-71 |
| `test_compute_metrics_threshold_stored` | `threshold` arg is echoed back | 74-77 |
| `test_compute_metrics_values_in_valid_range` | All metrics ∈ [0, 1] | 80-85 |
| `test_find_optimal_threshold_returns_float` | Correct type | 93-96 |
| `test_find_optimal_threshold_in_valid_range` | Threshold ∈ [0, 1] | 99-102 |
| `test_find_optimal_threshold_cost_sensitivity` | Higher FN cost → lower threshold | 105-111 |

The cost-sensitivity test is the important one, it encodes the
[02 § Decision threshold tuning](02-ml-concepts.md#decision-threshold-tuning)
theory as a property: `cost_fn=50, cost_fp=1` should always give a
threshold `≤` one from `cost_fn=1, cost_fp=1`. If someone flips the
cost parameters by mistake, this test catches it.

### Why no tests on the training scripts themselves

`train_xgboost.py` and `train_autoencoder.py` are orchestration
scripts, they read Parquet, fit sklearn/XGBoost/PyTorch objects,
log to MLflow. Unit-testing "does SMOTE oversample?" or "does XGBoost
converge?" would be testing the libraries, not this code. The
valuable test boundary is `evaluate.py`, which holds the actual
project logic.

## Integration tests, `tests/integration/`

Three tests in `test_pipeline_e2e.py`, plus a Kafka placeholder
(`test_kafka_flow.py` is a one-line stub for a future Phase 11).

### The reachability pattern, `conftest.py:22-28`

```python
def _is_reachable(url, timeout=2.0):
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.exceptions.RequestException:
        return False
```

Every fixture checks its dependency first; if the service isn't
running, the test is **skipped**, not failed. This means:

- Running `make test-integration` without starting docker compose
  produces a clean pytest report with skips, not noise.
- Integration tests *never* fail CI, they're not wired to CI
  (`.github/workflows/ci.yml` only runs unit tests).
- A degraded MLflow (running but no models registered) gets a
  separate fixture (`api_url_with_models`) that skips with a clear
  "run training first" message.

### The three real tests

**`test_predict_returns_valid_response`** (`test_pipeline_e2e.py:25-56`)
, POSTs a fixed transaction to the live API and asserts the response
is schema-complete (probability in `[0,1]`, `is_fraud` is a bool,
SHAP explanation has `feature` and `contribution` fields on each
item, etc). This is the only test that actually exercises a real
MLflow-loaded XGBoost model + real SHAP.

**`test_metrics_endpoint_has_inference_counters`** (`test_pipeline_e2e.py:64-78`)
, fires a prediction, then hits `/metrics`, and asserts the output
text contains `inference_total` and `inference_latency_seconds`. This
is the cheapest end-to-end check that the Prometheus instrumentation
actually exports what Grafana expects. If the metric names drift in
`serving/app/metrics.py`, this catches it before Prometheus scrapes
an empty target.

**`test_mlflow_has_champion_alias`** (`test_pipeline_e2e.py:86-104`)
, queries the MLflow REST API directly (no `mlflow` SDK in the test
venv) to assert that `fraud-xgboost@champion` is set. This is a
"training happened and promotion worked" canary.

### Why this tiny set is enough

The unit tests already prove the *logic* is correct. The integration
tests prove the three non-trivial integration surfaces work:

1. Real MLflow → real model loading → real SHAP generation.
2. Real `/predict` → Prometheus instrumentator → scrape-shaped text.
3. MLflow registry has the alias the serving layer expects to find.

Adding a Grafana-renders-correctly test would require a browser
driver. Adding an Airflow-DAG-runs test would require triggering the
DAG and waiting. The ROI on either isn't there at this scale.

## The `Makefile` test targets

Already covered in [03 § Testing](03-infrastructure.md#testing--makefile126-144)
but worth restating:

- `make test`, both unit suites (training + serving)
- `make test-serving`, serving only (fast, ~1s)
- `make test-training`, training only (fast, ~1s)
- `make test-integration`, integration, needs docker compose up
- `make check`, `format-check + lint + typecheck + test` (local CI
  parity)

## GitHub Actions, `.github/workflows/ci.yml`

Three jobs, runs on `push` to `main` / `dev` and on PRs to `main`.

### Job 1, `lint` (`ci.yml:10-33`)

Installs `ruff==0.1.9` and `black==23.12.1` directly (no venv), runs:

- `ruff check .`, linting
- `black --check .`, formatting assertion

Both are pinned to specific versions so CI can't break because of a
tool-side release. If you bump the version in `requirements-dev.txt`,
also bump it here.

### Job 2, `typecheck` (`ci.yml:35-63`)

Two mypy passes: one on `serving/app/`, one on `training/`.

**`continue-on-error: true`** (line 39) is the important detail. mypy
failures don't block the CI green check.

Why? Training imports `torch` and `sklearn` whose stubs are partial
or aggressive, and some third-party imports (MLflow, XGBoost) have
stubs that change across versions. Treating mypy as a warning rather
than a gate keeps CI green on upstream stub churn while still
surfacing real issues in the job logs.

When the code stabilises, flip this to `false` and take the signal
seriously. For now it's a tradeoff: signal-rich but not gating.

### Job 3, `test` (`ci.yml:65-91`)

- **`needs: lint`**, only runs if lint passed. No point running tests
  on code that won't merge.
- **Only serving tests.** `pytest serving/tests/ -v --tb=short
  --cov=serving --cov-report=term-missing` (line 91).
- **Why not training tests in CI?** Training tests need the training
  venv (torch, xgboost, imblearn), which is ~1.5 GB to install.
  That's ~5-10 minutes per CI run. Training logic is numerics-only
  and stable; running those tests locally via `make test` is
  sufficient. Move them into CI later if training changes pick up.
- **Coverage.** Printed to stdout but not enforced (no
  `--cov-fail-under`). Useful for eyeballing regressions. Adding a
  `--cov-fail-under=80` flag would make coverage a gate.

### Pip caching

Each job uses `actions/cache@v4` keyed on the hash of its
requirements file. Cache hits take CI from ~3 min to ~45 s. The keys
are scoped per job (`lint-...`, `typecheck-...`, `test-...`) so
installing `ruff` doesn't pollute the typecheck cache.

## What integration tests aren't in CI

Running integration tests in GitHub Actions would require:

1. Building all Docker images in CI (~5-10 minutes).
2. Starting the full compose stack (~30-60s).
3. Running training inside CI (~2-3 minutes).
4. Then the tests.

Total budget: ~15 minutes per CI run. Not worth it at this scale,
`make test-integration` locally after a big change is the pragmatic
approach.

If you wanted to add them, the right move is a nightly scheduled
workflow (`on: schedule: - cron: "0 3 * * *"`) that spins up the
whole stack, not a per-commit check.

## Limitations

- **No coverage floor.** Coverage is reported but not enforced.
  Setting `--cov-fail-under=80` in `ci.yml:91` would make regressions
  impossible to merge.
- **No security scanner.** `pip-audit` exists as a Makefile target
  (`make audit`) but isn't wired to CI. A `pip-audit` job on the
  requirements files would catch CVEs on push.
- **No integration tests in CI.** Discussed above; acceptable for a
  local development, not for production.
- **No mutation testing.** Tests protect against the bugs their
  authors thought of. Running `mutmut` occasionally would surface
  blind spots.
- **Mocked MLflow hides version mismatches.** Unit tests don't catch
  a scenario where MLflow changes its artifact path format or its
  client SDK signature. The integration test at
  `test_mlflow_has_champion_alias` covers the critical path.
- **No performance regression tests.** A 100ms regression on p99
  latency wouldn't fail CI. Adding a simple `pytest-benchmark` check
  on the `prepare_features` function would be quick wins.
- **Kafka integration test is a stub.** `tests/integration/test_kafka_flow.py`
  is one comment line, the pattern is there for a future Phase 11,
  which is deliberately out of scope.

## Where to go next

- [09, Glossary](09-glossary.md) if you want the quick reference of
  every term used anywhere in this wiki.
- [10, Limitations and extensions](10-limitations-and-extensions.md)
  consolidates every limitation section across the wiki into one
  ranked list.
