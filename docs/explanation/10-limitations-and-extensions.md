# 10, Limitations and extensions

> **What this page answers:** What's deliberately imperfect about this
> project, what you'd change before shipping, and what you might build
> next.

This is the most honest page in the wiki. It consolidates every
"Limitations" section from the other pages into one ranked list.

## Known trade-offs, ranked by "would bite in production"

Grouped by severity. Each item links back to the page where the
issue is discussed in context.

### Critical for production, acceptable at this scale

1. **Random train/val/test split instead of time-based.** The dataset
   is chronologically ordered, so a random split leaks future into
   past. Reported metrics are mildly optimistic. The fix is
   `TimeSeriesSplit` or a date cutoff, plus revalidating the PR-AUC
   target. ([05](05-training.md#data-split-train_xgboostpy112-122))
2. **No authentication on `/predict`.** Anyone on `fraud-net` can
   score. For production, add API keys or OAuth2 at the FastAPI
   layer, ingress TLS, and rate limiting. ([06](06-serving-api.md#limitations))
3. **No prediction persistence.** Predictions are counted in
   Prometheus then discarded. Without a prediction log you can't
   audit decisions, compute true quality offline, or feed a drift
   report automatically. Write each prediction to Postgres (or
   Kafka, S3) with inputs, output, model version, and timestamp.
4. **Alerts fire but don't route.** No AlertManager, no Slack/email.
   Fine on a laptop; blind in production. ([07](07-monitoring.md#no-alertmanager))
5. **`.env` is plain text.** Secrets should come from a secret
   manager at runtime, not from a file on disk. ([03](03-infrastructure.md#env-and-secrets-hygiene))

### Significant but manageable

6. **No hyperparameter search.** XGBoost uses a hand-picked config.
   Optuna on `training/.venv` would be a one-afternoon win. ([05](05-training.md#limitations))
7. **No cross-validation.** A single 60/20/20 split is noisy when you
   only have 492 frauds (~98 in the test split). CV would give more
   stable PR-AUC. ([05](05-training.md#limitations))
8. **Single serving instance.** One container, one worker. Any real
   load or high-availability setup needs gunicorn + multiple
   uvicorn workers, then horizontal scaling. ([06](06-serving-api.md#limitations))
9. **No graceful model reload.** Promoting a new champion in MLflow
   doesn't propagate until `docker compose restart serving`.
   A polling or webhook-based reload would fix this. ([06](06-serving-api.md#limitations))
10. **Evidently drift is offline and manual.** `make drift-report`
    when you remember; no ingestion pipeline for serving data; no
    trend across reports. Wire prediction logs, daily report,
    alert. ([07](07-monitoring.md#what-current-data-means-here))
11. **Static alert thresholds.** 10% fraud rate, 500ms p99, 0.1 err/s,
    hand-picked. A real system would learn or compare against
    last-week-same-time. ([07](07-monitoring.md#limitations))
12. **Retrain DAG only retrains XGBoost.** The autoencoder is
    trained manually and not on a schedule. Adding a second branch
    in the DAG is straightforward. ([04](04-data-and-features.md#the-retrain-dag))
13. **Retrain promotion uses a single-number gate.** `new_pr_auc >=
    champion_pr_auc` can promote on noise. Bootstrapped CI, or a
    shadow-traffic eval, would be more robust.
### Minor but worth knowing

14. **`amount_log` is redundant for XGBoost.** Tree models are
    invariant to monotonic transforms. Kept because it helps the
    autoencoder and SHAP readability.
15. **SHAP adds ~40-80ms per request.** Reasonable, but could be
    opt-in via a query parameter.
16. **Autoencoder threshold isn't stored in the registry run metrics
    the same way XGBoost's is.** It lives inside the pyfunc's
    `threshold.txt` artifact instead. Not a bug, just asymmetry.
17. **Cardinality risk on Prometheus labels.** If you ever add a
    label with unbounded values (like `transaction_id`), Prometheus
    dies quickly. ([07](07-monitoring.md#cardinality-risk))
18. **No coverage floor in CI.** Coverage is reported but not
    enforced. `--cov-fail-under=80` would gate it. ([08](08-testing-and-ci.md#limitations))
19. **No security scanner in CI.** `make audit` exists; wiring to CI
    would catch CVEs on push.
20. **No performance regression tests.** A 100ms regression wouldn't
    fail CI. `pytest-benchmark` on hot paths would surface them.

## What you'd change first in a real system

If a team agreed to productionise this tomorrow, the ranked work
order:

1. **Add a prediction log.** Everything else (drift pipelines, offline
   quality, audit) depends on having this data. Postgres table or
   Kafka topic.
2. **Add auth + rate limiting on `/predict`.** API keys minimum;
   OAuth2 / mTLS for real.
3. **Time-based train/val/test split.** Retrain and re-baseline the
   PR-AUC target. Everything upstream of this is optimistic until
   done.
4. **AlertManager + Slack routing.** Alerts that fire but don't page
   anyone are theatre.
5. **Gunicorn + multiple Uvicorn workers.** Instant latency and
   throughput win; lays groundwork for horizontal scaling.
6. **Graceful model reload.** Polling `models:/name@alias` every 60s
   and switching on version change.
7. **Shadow-traffic evaluation before promotion.** Route 100% to
   champion in production but score on challenger for N days and
   compare offline before moving the alias.

## Possible extensions

Directions the project could grow in next:

- **Kafka streaming ingress.** Turn the static batch pipeline into a
  streaming pipeline with a Kafka producer and a Go consumer.
- **Feast feature store.** Eliminate training-serving feature skew and
  serve online features with sub-millisecond reads.
- **Deploy to a cloud.** ECS, GKE, or Cloud Run. Takes the stack from
  compose to managed infra.
- **Hyperparameter sweep with Optuna.** Integrated with MLflow nested
  runs, targeting PR-AUC.
- **Time-aware cross-validation.** Replace the random 60/20/20 split
  with a `TimeSeriesSplit`; the honest PR-AUC is lower but no longer
  optimistic.
- **LLM-assisted fraud reasoning.** A `POST /predict/explain`
  endpoint that feeds SHAP contributions into a small LLM and
  returns a natural-language rationale.
- **AlertManager + Slack.** Small, clean, visible. One container plus
  a two-line webhook config.
- **pgvector for similarity search.** Store recent transactions as
  embeddings and flag new transactions by nearest-neighbour distance
  to past fraud. Complements the two existing models with a
  retrieval-based paradigm.
- **Drift-triggered retraining.** `drift_report.py`, Evidently, if
  drift detected then trigger the retrain DAG. Closes the loop.

## What's explicitly out of scope (summary)

From the original `plan.md` decision log, repeated here for quick
reference:

| Omitted | One-line reason |
|---|---|
| Kafka + Go consumer | Adds three containers to fake a stream over a static CSV |
| Feast feature store | One data source + one pipeline; Feast would be cargo-cult |
| Kubernetes | Single-node Compose is more honest for the actual scale |
| Isolation Forest | XGBoost + autoencoder already cover supervised + unsupervised |
| Real auth on the API | In-network service; auth is the first production add |
| AlertManager + Slack | Alerts visible in Prometheus UI; routing is one step away |
| Hyperparameter sweep | Meaningful only against a time-based split, which is also missing |
| Cross-validation | Same reason as above; next thing to add |

## Where to go next

- Back to [01, Big picture](01-big-picture.md) for the overall
  architecture.
- [02, ML concepts](02-ml-concepts.md) if any ML term on this page
  still feels fuzzy.
- [README](README.md) to see the wiki ToC and reading order.
