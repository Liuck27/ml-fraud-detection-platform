# Project Audit Report

Full review of the ML fraud detection platform: training code, serving API, Airflow DAGs,
Docker/compose setup, monitoring, CI, tests, and documentation. Static analysis plus a run
of both unit test suites (14 serving + 7 training, all passing). The original audit did not
run the full Docker stack; the H1 fix work later did (live DAG runs, serving predictions),
which surfaced one additional issue (MLflow artifact misconfiguration, documented under H1).

## Status (updated 2026-06-11)

| Finding | State |
|---|---|
| H1 retrain DAG cannot run | **FIXED** (training mount + deps, py3.11 image; verified end-to-end twice) |
| H2 dead promotion gate | **FIXED** (register-only training, shared `promote_champion_if_better` gate) |
| H3 broken fraud-rate alert | **FIXED** (`sum()` wrappers, promtool-validated) |
| M1 leaky/dead `amount_zscore` | **FIXED** (feature dropped everywhere; both models retrained on 32 features) |
| M2 `scale_pos_weight` no-op | **FIXED** (SMOTE is the single imbalance strategy; param deleted) |
| M3 threshold tuned and evaluated on same split | **FIXED** (60/20/20 split: tune on val, report on test) |
| M4 `Time` default flips `is_night` | **FIXED** (`Time` is now required; missing field returns 422) |
| M5 dead `prepare_features_batch` | **FIXED** (deleted; all serving features are row-local) |
| M6 venv baked into serving image | **FIXED** (`serving/.dockerignore`; image 12.1 -> 9.9 GB, context 647 bytes) |
| M7 CI contradicts docs | **FIXED** (typecheck blocking, training tests in CI) |
| M8 phantom-phase template leftovers | **FIXED** (stubs and Kafka compose blocks deleted) |
| M9 broken README quickstart | **FIXED** (correct Airflow creds, serving restart step) |
| L1 autoencoder dims doc mismatch | **FIXED** (correct value is now 32 after M1; docstring/plan.md updated) |
| L4 `use_label_encoder` no-op param | **FIXED** (deleted) |
| L6 drift generator's `is_night` definition | **FIXED** (matches canonical 22:00-06:00 definition) |
| Bonus: MLflow artifacts written client-side | **FIXED** (proxied `mlflow-artifacts:/` scheme; see H1 resolution) |
| Bonus: pyfunc artifact paths break cross-OS | **FIXED** (see M1 resolution: `load_context` normalises separators) |
| L2 unreachable column check | **FIXED** (schema validated via `pyarrow.parquet.read_schema` first) |
| L3 unpinned shap | **FIXED** (`shap==0.49.1` — the version already deployed) |
| L5 inconsistent error-counter labels | **FIXED** (full `model_name` scheme on all four metrics) |
| L7 EDA notebook committed unexecuted | **FIXED** (executed in place; charts render on GitHub) |
| L8 sklearn feature-names warning | **FIXED** (scalers receive DataFrames; autoencoder retrained as v5) |
| L9 plan.md fallback claim | **FIXED** (now describes degraded mode + 503) |
| L10 O(n*k) threshold sweep | **FIXED** (vectorised cumulative-sum sweep, equivalence-tested) |
| L11 no serving healthcheck | **FIXED** (mlflow + serving healthchecks; `service_healthy` dependency) |

All findings resolved as of 2026-06-11.

**Overall verdict:** the architecture is genuinely sound and matches industry practice.
The scope decisions (no Kafka, no Kubernetes, Evidently as a script) are well-reasoned and
well-documented. The serving code is clean, the A/B routing is correct, the scaler artifact
is properly versioned alongside the model, and the wiki is unusually honest about
limitations. This is a good portfolio project. However, the audit found three High-severity
issues where the code did not do what the documentation claims it does — exactly the kind
of thing a technical interviewer would find — plus a set of Medium issues worth fixing or
at least being able to explain. (All three HIGH findings have since been fixed; see the
status table above and the resolution notes inline.)

Severity levels:

- **HIGH** — broken functionality, or documentation that claims the opposite of what the code does
- **MEDIUM** — real ML/engineering flaws; fix them, or be prepared to defend them
- **LOW** — polish, consistency, hygiene

---

## HIGH severity

### H1. The retrain DAG cannot actually run — FIXED 2026-06-10

> **Resolution (Option A: mount + deps):** `./training` is now volume-mounted read-only at
> `/opt/airflow/training`, and `airflow/Dockerfile` installs the training deps (scikit-learn,
> xgboost, imbalanced-learn, mlflow, matplotlib — no torch) pinned to
> `training/requirements.txt` versions. The base image was switched to
> `apache/airflow:2.7.3-python3.11` because the default ships Python 3.8, too old for those
> pins (3.11 also matches the host training venv exactly). The DAG's `evaluate_and_promote`
> now imports the shared `promote_champion_if_better` instead of an inlined copy, and the
> train task captures subprocess stdout/stderr into the Airflow log. Verified end-to-end
> twice: validate → train → gated promotion all green, new champion (v3, then v4) loaded by
> serving and predicting via `/predict`.
>
> **Discovered while testing (and fixed):** the MLflow server was started with
> `--default-artifact-root ./mlartifacts`, a server-local path. Experiments recorded
> `/app/mlartifacts/<id>` and every client treated it as a path on its *own* machine —
> host training had been silently writing artifacts to `C:\app\mlartifacts` on the
> laptop (safe to delete), and the volume copies that made serving work had been copied
> in manually at some point (identical mtimes). On a Linux host the quickstart would
> crash with `PermissionError: /app`. Fixed by switching the server to proxied artifacts
> (`--serve-artifacts --artifacts-destination ${MLFLOW_ARTIFACT_ROOT}
> --default-artifact-root mlflow-artifacts:/`) so all clients upload/download over HTTP
> through the server; existing experiments were migrated via SQL
> (`UPDATE experiments SET artifact_location = 'mlflow-artifacts:/' || experiment_id`).
> Serving keeps the volume mount only for pre-fix model versions with absolute-path URIs.

`airflow/dags/retrain_dag.py` runs `training/train_xgboost.py` as a subprocess inside the
Airflow container, but:

1. `docker-compose.yml` mounts only `./airflow/dags`, `./airflow/plugins`, and `./data`
   into the Airflow containers. `./training` is never mounted, so
   `/opt/airflow/training/train_xgboost.py` does not exist → task 2 fails with
   `FileNotFoundError` on every run.
2. Even if it were mounted, the Airflow image (`airflow/Dockerfile`) installs only
   `pandas` and `pyarrow`. The training script needs `xgboost`, `scikit-learn`,
   `imbalanced-learn`, `matplotlib`, and `mlflow` → `ImportError`.
3. Task 3 (`evaluate_and_promote`) does `import mlflow` — also not installed in the
   Airflow image → fails even if training were skipped.

The DAG also has `schedule="@weekly"`, so on a long-running stack it will fail visibly
every week in the Airflow UI.

**Fix options** (pick one):
- (a) Mount `./training` in `x-airflow-common` and add the training deps to
  `airflow/Dockerfile` (heavyweight, but makes the demo real).
- (b) Honest-demo route: change the DAG to clearly document that the train task requires
  the training deps, and validate that in code with a clear error message.
- (c) Best practice: use `DockerOperator` to run training in the training image, as the
  DAG docstring itself suggests.

### H2. The champion promotion gate is dead code, and the wiki claims otherwise — FIXED 2026-06-10

**Resolution:** `train_xgboost.py` now only registers; the gate lives in
`model_registry.promote_champion_if_better`, exposed via `scripts/promote_model.py` /
`make promote` and called by `run_training.sh`. The DAG's `evaluate_and_promote` task is
now the real (and only) in-Airflow gate. The retrain DAG was also switched to manual
trigger (`schedule=None`) per the demo-vs-production decision; wiki, READMEs, and
comments updated. The autoencoder keeps unconditional `challenger` promotion by design
(challenger = latest candidate), documented in code and wiki.

Original finding:

`training/train_xgboost.py:181-187` **unconditionally** promotes the just-trained model to
`champion` at the end of every run. The retrain DAG's `evaluate_and_promote` task then
compares "latest version" vs "current champion" — but they are now the same version, so
`new_pr_auc >= champ_pr_auc` is always true (comparing a run to itself). The quality gate
can never reject a model.

Worse, the wiki (`docs/explanation/04-data-and-features.md:316-317`) states: "After
training finishes, the new model is registered in MLflow but *not* automatically promoted"
— this is factually wrong, and it's presented as "the production-grade pattern". If you
say this in an interview and the interviewer opens `train_xgboost.py`, it falls apart.

**Fix:** remove the `promote_to_champion` call from `train_xgboost.py` (make the script
register only), and let promotion happen exclusively in the DAG's gated task. Provide a
`make promote-champion` or a `--promote` flag for the standalone/first-run case. Update
the wiki section. (Same consideration applies to `train_autoencoder.py` and `challenger`,
though there is no gate there to contradict.)

### H3. The `HighFraudRate` Prometheus alert is mathematically always 100% — FIXED 2026-06-10

**Resolution:** expression rewritten with `sum()` on both sides, validated with
`promtool check rules` (3 rules pass). Wiki section in `07-monitoring.md` updated with
an explanation of the PromQL vector-matching pitfall.

Original finding:

`monitoring/alerting/rules.yml`:

```promql
rate(inference_total{prediction="fraud"}[5m]) / rate(inference_total[5m]) > 0.10
```

PromQL binary operations match series label-for-label. The left side has series labeled
`{model_name, prediction="fraud"}`; the right side contains that exact same series, so
each fraud series is divided by itself → the ratio is always 1 whenever any fraud
prediction occurs. The alert fires on any nonzero fraud traffic, not at a 10% rate.

The Grafana dashboard's fraud-rate panel does this correctly (`100 * sum(rate(...))`),
so the fix is to mirror it:

```promql
sum(rate(inference_total{prediction="fraud"}[5m]))
/ sum(rate(inference_total[5m])) > 0.10
```

---

## MEDIUM severity

### M1. `amount_zscore` is leaky in training and dead weight in serving — FIXED 2026-06-11

> **Resolution (drop the feature):** removed from `feature_engineering.py`, both training
> scripts' `FEATURE_COLS`, the serving loader, both DAGs' expected-column lists, the drift
> generator, and all docs. Model input width went 33 -> 32, so both models were retrained
> (XGBoost v5/v6, autoencoder v4) and the data_ingestion DAG re-run to regenerate
> `features.parquet`. Because the old champion's metrics were measured under the old
> methodology (see M3) and the feature schema changed, the champion was moved with a
> one-time `scripts/promote_model.py --force` (new flag, documented for exactly this
> migration case); subsequent retrains go through the normal gate, verified live (the
> retrain DAG trained v6 in-container with metrics identical to the host run and the gate
> promoted it).
>
> **Discovered while retraining (and fixed):** two latent Windows bugs. (1) MLflow 2.9
> records the pyfunc artifact map with `os.path.join`, so a model logged from Windows
> hands the Linux serving container paths like `artifacts\model.pt` — the autoencoder's
> `load_context` now normalises separators (the old challenger only loaded because its
> volume artifacts had been hand-patched). (2) The `→` in `model_registry`'s prints
> crashed `print()` on cp1252 consoles; replaced with ASCII `->`.

In training data, `amount_zscore` is computed by `compute_interaction_features` over the
**entire dataset** before the train/val split (mild data leakage: validation rows
contribute to the mean/std the model trains on). At serving time,
`loader.prepare_features` hardcodes it to `0.0` — so the model receives a feature at
inference that never matches what it saw in training (train/serve skew). The wiki
documents this as a "quirk," but the defensible engineering answer is to fix it:

- **Simplest:** drop `amount_zscore` from `FEATURE_COLS` entirely. It is redundant —
  `amount_log` carries the signal, and the downstream `StandardScaler` already
  standardizes it.
- **Alternative:** freeze training-set mean/std as model artifacts (like the scaler) and
  apply them at serving.

### M2. `scale_pos_weight` is a silent no-op, and SMOTE + class weighting double-counts — FIXED 2026-06-11

> **Resolution (SMOTE only):** the `scale_pos_weight` computation and parameter were
> deleted from `train()`, with a comment stating the single-strategy rationale; the
> no-op `use_label_encoder=False` went with it (L4). The wrong "about 580" claim in
> `docs/explanation/02-ml-concepts.md` is gone — the wiki, README, plan.md, glossary,
> and EDA notebook now all describe SMOTE as the one rebalancing strategy and explain
> why class weighting is deliberately absent.

`train_xgboost.py:train()` computes `scale_pos_weight = n_neg / n_pos` on the **already
SMOTE-resampled** data, where classes are balanced 1:1 — so it is ≈ 1.0 and does nothing.
The comment "belt-and-suspenders alongside SMOTE" describes intent the code doesn't
implement. Conceptually, using both SMOTE *and* `scale_pos_weight` would double-correct
the imbalance anyway. Pick one strategy and make the code say so:

- Either SMOTE alone (delete the `scale_pos_weight` computation), or
- `scale_pos_weight` computed on the original training distribution, without SMOTE
  (in practice this often performs equally well on this dataset and is cheaper).

This is a classic interview probe ("why both?") — currently the code has no good answer.

### M3. Threshold is tuned and evaluated on the same validation set — FIXED 2026-06-11

> **Resolution (three-way split):** both training scripts now split 60/20/20 (stratified;
> test split off first). The threshold (and the autoencoder's p99 score denominator) is
> tuned on val; all logged metrics — including the `pr_auc` the promotion gate compares —
> come from the untouched test split. Honest test metrics after retraining: XGBoost
> AUC-ROC 0.9797 (target > 0.95 still holds), PR-AUC 0.8547, F1 0.8526; autoencoder
> PR-AUC 0.2668 (in line with all previous versions — the ~0.74 figure only ever existed
> in test mocks).

Both training scripts call `find_optimal_threshold(y_val, …)` and then report
`compute_metrics(y_val, …, threshold=…)` on the **same** split. The reported F1/precision/
recall are optimistically biased, and the promotion gate compares models on the same data
their thresholds were tuned on. Standard fix: three-way split (train / val for threshold
tuning / held-out test for reported metrics), or tune the threshold via cross-validation
on the training set.

### M4. `Time` default silently flips `is_night` on, contradicting the schema docs — FIXED 2026-06-11

> **Resolution (make `Time` required):** the `= 0.0` default was removed from
> `TransactionFeatures`; a request without `Time` now returns 422 (verified live). All
> existing examples, tests, and demo scripts already sent `Time`, so nothing else broke.
> Docstring and wiki updated with the rationale (a fraud system always has a timestamp).

`serving/app/schemas.py` documents: "when omitted hour_of_day defaults to 0 and is_night
to False". But hour 0 satisfies `hour_of_day < 6`, so `prepare_features` sets
`is_night=True`. Any client that omits `Time` gets a midnight transaction with a night
flag — a silent feature distortion. Fix: either make `Time` required (cleanest for a
fraud model) or fix the docstring and accept the semantics deliberately.

### M5. `prepare_features_batch` is dead code with divergent semantics — FIXED 2026-06-11

> **Resolution (delete):** removed from the loader and the test conftest. With
> `amount_zscore` gone (M1) every serving feature is row-local, so a batch-level prep
> could not behave differently from per-row prep even in principle; the wiki now states
> that property explicitly.

`POST /predict/batch` loops and calls the single-row `prepare_features` per transaction;
`prepare_features_batch` (which computes `amount_zscore` over the request batch) is never
called by any route. If it were used, the same transaction would get different scores
depending on what else was in the batch. Delete it (or wire it in deliberately after
resolving M1). Dead code with a subtle behavioral difference is a red flag in review.

### M6. `serving/Dockerfile` copies the multi-GB venv into the image — FIXED 2026-06-11

> **Resolution:** added `serving/.dockerignore` excluding `.venv`, `tests`, tool caches,
> and the Dockerfile itself. Verified by rebuilding: build context went from gigabytes
> (minutes to transfer) to 647 bytes, and the image shrank 12.1 GB -> 9.88 GB (what
> remains is the pip-installed runtime — torch, mlflow, shap — which serving legitimately
> needs). Container recreated and verified healthy with both models loaded.

`COPY . /app/serving/` with no `.dockerignore` means `serving/.venv/` (torch ≈ several GB
on disk) and `serving/tests/` are sent as build context and baked into the image. Add a
`serving/.dockerignore` with at least `.venv`, `tests`, `__pycache__`.

### M7. CI contradicts the project's own documentation — FIXED 2026-06-11

> **Resolution:** `continue-on-error: true` removed from the typecheck job — safe because
> the GitHub Actions history shows the job succeeding on every recorded run (the escape
> hatch was masking nothing), and mypy is green locally on both packages. The test job now
> runs `training/tests/` before the serving suite: the only dependency not already pulled
> in by `serving/requirements.txt` was matplotlib (one extra pinned pip install), so the
> old "needs the ~1.5 GB training venv" justification did not hold. Wiki 08 rewritten to
> match; the "typecheck is non-gating" limitation removed from wiki 10.

- `CLAUDE.md` says "mypy is enforced in CI", but `.github/workflows/ci.yml` sets
  `continue-on-error: true` on the typecheck job — it can never fail the build.
- CI runs only `serving/tests/`; `training/tests/test_evaluate.py` is never executed in
  CI (only via `make test` locally), even though it needs nothing heavier than numpy,
  sklearn, and matplotlib.

Fix: run training tests in CI, and either make typecheck blocking or change the docs to
say it's advisory.

### M8. Leftover "phantom phase" artifacts from a project template — FIXED 2026-06-10

> **Resolution:** Deleted `tests/conftest.py` (stub), `tests/integration/test_kafka_flow.py`,
> and `monitoring/evidently/drift_report.py`. Removed the ~64 lines of Zookeeper/Kafka/
> producer/Go-consumer stubs from `docker-compose.yml` and rewrote the stale header.
> Updated all wiki line references shifted by the deletion (01, 03, 06, 08, 10) and removed
> every "commented stubs" / "Phase 8/9/11" mention; rewrote `monitoring/evidently/README.md`
> to point at the real `scripts/drift_report.py` + `make drift-report` flow. The README's
> "why no Kafka" rationale now stands alone, which is the stronger story.

The plan defines 6 phases and explicitly rules Kafka out of scope, yet the repo contains:

- `tests/integration/test_kafka_flow.py` — stub: "Implementation in Phase 11"
- `tests/conftest.py` — stub: "Implementation in Phase 11"
- `monitoring/evidently/drift_report.py` — stub: "Implementation in Phase 9" (the real
  script is `scripts/drift_report.py`)
- `docker-compose.yml` — ~60 lines of commented Zookeeper/Kafka/producer/Go-consumer
  stubs marked "Uncomment in Phase 8", referencing a `streaming/` directory that doesn't
  exist
- `docker-compose.yml` header still says "Phase 1: PostgreSQL only (active)" although all
  services are live

Phases 8/9/11 exist nowhere in the plan. To a careful reader this signals the project was
stamped from a template. Delete the stubs and the Kafka compose blocks (the README already
explains why Kafka was excluded — that's the stronger story), and fix the compose header.

### M9. README quickstart fails at two steps — FIXED 2026-06-10

> **Resolution:** Step 4 now says `admin` / `admin` (matching the user seeded in
> `docker-compose.yml`), and a new step 6 (`docker compose restart serving`) was inserted
> after training, with a comment explaining that models are loaded once at startup.
> The old step 6 (curl /predict) is now step 7.

1. Step 4 says Airflow login is `airflow` / `airflow`; `docker-compose.yml` seeds
   `admin` / `admin`.
2. Step 6 (`curl /predict`) will fail: the serving container started in step 3, **before**
   models existed in MLflow, and it only loads models at startup. The quickstart is
   missing `docker compose restart serving` after step 5. (The integration test fixtures
   know this — the README doesn't.)

The plan's own acceptance criterion is "README quickstart works for a fresh clone."

---

## LOW severity

- **L1.** ~~`train_autoencoder.py` docstring says `Input(33)`, while `plan.md` and
  `CLAUDE.md` say `Input(30) → 64 → …`.~~ FIXED with M1: the correct value is now 32
  (28 V + 4 engineered); docstring, plan.md, and wiki all agree.
- **L2.** ~~`retrain_dag.validate_features` reads the parquet with `columns=list(EXPECTED…)`
  and then checks for missing columns — `read_parquet` would already have raised, so the
  check is unreachable. Read without `columns=` or drop the check.~~ FIXED 2026-06-11: the
  check now reads the parquet *schema* via `pyarrow.parquet.read_schema` (no data load),
  then loads only the `Class` column for the fraud-rate stat — the validation is reachable
  and produces its own readable error.
- **L3.** ~~`serving/requirements.txt` pins everything except `shap>=0.44`, directly under a
  comment explaining why exact pins matter. Pin shap.~~ FIXED 2026-06-11: pinned to
  `shap==0.49.1`, the version the resolver had already installed in both the serving venv
  and the running container — so the pin freezes current behaviour rather than changing it.
- **L4.** ~~`use_label_encoder=False` in `XGBClassifier` is a removed/no-op parameter in
  XGBoost 2.x and produces a "Parameters: { use_label_encoder } are not used" warning.~~
  FIXED with M2 (deleted).
- **L5.** ~~`predict.py` error counter uses label values `"champion"`/`"challenger"` for
  `model_name`, while latency/total counters use full names like
  `fraud-xgboost-champion`. Inconsistent label scheme makes PromQL joins across the
  metrics awkward. Unify.~~ FIXED 2026-06-11: both handlers resolve the full model name
  before predicting, so the error counter shares the exact label values of the
  latency/total metrics (`"unknown"` remains only for the no-models-loaded 503, where
  there is no model to name). The alert rule uses no `model_name` filter, so it is
  unaffected.
- **L6.** ~~`generate_drift_data.py` recomputes `is_night` as `hour_of_day < 6`, dropping
  the 22:00–23:00 hours from the training definition.~~ FIXED with M1 (same file edit):
  now `(hour >= 22) | (hour < 6)`, matching the canonical definition.
- **L7.** ~~The EDA notebook is committed with no executed outputs (all
  `execution_count: null`), so on GitHub it renders without a single chart. For a
  portfolio, commit it executed — the rendered plots are the point.~~ FIXED 2026-06-11:
  executed in place with `jupyter nbconvert --execute` against the real dataset; all
  cells have outputs and the charts render on GitHub.
- **L8.** ~~The StandardScaler is fit on a DataFrame but applied to `df.values` at serving,
  which triggers sklearn's "X does not have valid feature names" warning on every request.~~
  FIXED 2026-06-11: the XGBoost path (loader + SHAP prep) now passes the DataFrame
  itself. The autoencoder's `transform(.values)` lives *inside* the pickled pyfunc, so
  fixing the source required retraining — v5 trained and promoted to challenger
  (PR-AUC 0.2519, within the model's usual 0.23-0.28 band). Remaining note kept on
  purpose: scaling is a no-op for XGBoost (trees are scale-invariant) and is retained
  only for pipeline symmetry with the autoencoder — a deliberate, explainable choice
  rather than a bug.
- **L9.** ~~`plan.md` promises "fallback to local artifact cache if MLflow is unreachable";
  the implementation does degraded mode + 503 instead (which is fine — update the plan).~~
  FIXED 2026-06-11: plan.md now describes the degraded-mode behaviour that is actually
  implemented.
- **L10.** ~~`find_optimal_threshold` re-thresholds the entire score array for every
  candidate threshold — O(n²)-ish on a 57k-row validation set. Works, but a vectorized
  cumulative-sum sweep is the textbook version; cheap interview win.~~ FIXED 2026-06-11:
  rewritten as a sort-once + cumulative-sum sweep. Verified equivalent to the loop on
  ~900 randomised cases (including heavy score ties): identical threshold every time,
  same lowest-threshold tie-breaking.
- **L11.** ~~The serving container has no compose healthcheck and only `service_started`
  dependency on MLflow; a healthcheck on `/health` would make `make up` self-verifying.~~
  FIXED 2026-06-11: MLflow got a healthcheck (python `urllib` probe — the image has no
  curl) and serving depends on it with `service_healthy`, removing the startup race
  against model loading. Serving's own healthcheck parses `/health` JSON and reports
  healthy only when both models are loaded — `docker compose ps` now answers "can it
  actually score", and a model-less fresh stack correctly shows unhealthy.

---

## What is genuinely good (worth saying in interviews)

- Correct leak-avoidance where it matters most: scaler fit on train only, SMOTE applied
  only to the training split, autoencoder trained only on legit transactions, threshold
  derived from a PR-curve cost sweep.
- The fitted scaler and decision threshold are versioned with the model in MLflow and
  reloaded by serving — that is real training/serving-skew thinking (and since M1, every
  serving feature is row-local, so there is no remaining skew).
- Deterministic hash-based A/B routing, correctly implemented and well-tested, including
  a distribution test.
- Degraded-mode startup (API boots without models, 503s cleanly, health endpoint reports
  per-model status) with fallback routing when one model is missing.
- Sensible monitoring stack: custom Prometheus metrics with useful labels, provisioned
  Grafana, alert rules (modulo H3), drift report with a synthetic-drift generator.
- Isolated per-service venvs with documented reasons (Airflow's tightly pinned dependency
  tree) — pinned, mutually consistent ML versions across training and serving.
- Honest scope decisions, documented trade-offs, idempotent DB init, secrets kept out of
  git, clean test suites that actually pass.

## Suggested fix order

1. ~~H2 (promotion gate) — small change, removes the biggest interview risk~~ DONE
2. ~~H3 (alert expression) — one-line fix~~ DONE
3. ~~M8 + M9 (template leftovers + README quickstart) — credibility and first impressions~~ DONE
4. ~~H1 (retrain DAG runtime) — decide the strategy first (DockerOperator vs mounting)~~ DONE (mount + deps)
5. ~~M1 + M2 + M4 + M5 (feature/imbalance correctness) — then retrain and update metrics~~ DONE
6. ~~M3 (proper test split) — retrain once more, report honest metrics~~ DONE (bundled with step 5: one retrain covered both)
7. ~~M6, M7, then the LOW list as time allows~~ ALL DONE — every finding in this audit is
   resolved as of 2026-06-11.
