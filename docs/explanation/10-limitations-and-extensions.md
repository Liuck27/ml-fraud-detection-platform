# 10 — Limitations and extensions

> **What this page answers:** What's deliberately imperfect about this
> project, what you'd change before shipping, what you might build
> next, and how to talk about it in an interview.

This is the single most honest page in the wiki. If you only read one
page before a recruiter conversation, read this one.

## Known trade-offs, ranked by "would bite in production"

Grouped by severity. Each item links back to the page where the
issue is discussed in context.

### Critical for production, acceptable for portfolio

1. **Training-serving skew on `amount_zscore`.** The single-row
   serving path uses `amount_zscore = 0.0` because a 1-row batch has
   `std = 0`. In production you'd log training-time mean/std and
   apply them to every single-row request. ([06](06-serving-api.md#feature-prep-the-amount_zscore-quirk))
2. **Random train/val split instead of time-based.** The dataset is
   chronologically ordered, so a random split leaks future into past.
   Validation metrics are mildly optimistic. The fix is
   `TimeSeriesSplit` or a date cutoff, plus revalidating the PR-AUC
   target. ([05](05-training.md#data-split--train_xgboostpy108-110))
3. **No authentication on `/predict`.** Anyone on `fraud-net` can
   score. For production, add API keys or OAuth2 at the FastAPI
   layer, ingress TLS, and rate limiting. ([06](06-serving-api.md#limitations))
4. **No prediction persistence.** Predictions are counted in
   Prometheus then discarded. Without a prediction log you can't
   audit decisions, compute true quality offline, or feed a drift
   report automatically. Write each prediction to Postgres (or
   Kafka → S3) with inputs, output, model version, and timestamp.
5. **Alerts fire but don't route.** No AlertManager, no Slack/email.
   Fine for a laptop; blind in production. ([07](07-monitoring.md#no-alertmanager))
6. **`.env` is plain text.** Secrets should come from a secret
   manager at runtime, not from a file on disk. ([03](03-infrastructure.md#env-and-secrets-hygiene))

### Significant but manageable

7. **No hyperparameter search.** XGBoost uses a hand-picked config.
   Optuna on `training/.venv` would be a one-afternoon win. ([05](05-training.md#limitations))
8. **No cross-validation.** A single 80/20 split is noisy when you
   only have 492 frauds. CV would give more stable PR-AUC. ([05](05-training.md#limitations))
9. **Single serving instance.** One container, one worker. Any real
   load or high-availability story needs gunicorn + multiple
   uvicorn workers, then horizontal scaling. ([06](06-serving-api.md#limitations))
10. **No graceful model reload.** Promoting a new champion in MLflow
    doesn't propagate until `docker compose restart serving`.
    A polling or webhook-based reload would fix this. ([06](06-serving-api.md#limitations))
11. **Evidently drift is offline and manual.** `make drift-report`
    when you remember; no ingestion pipeline for serving data; no
    trend across reports. Wire prediction logs → daily report →
    alert. ([07](07-monitoring.md#what-current-data-means-here))
12. **Static alert thresholds.** 10% fraud rate, 500ms p99, 0.1 err/s
    — hand-picked. A real system would learn or compare against
    last-week-same-time. ([07](07-monitoring.md#limitations))
13. **Retrain DAG only retrains XGBoost.** The autoencoder is
    trained manually and not on a schedule. Adding a second branch
    in the DAG is straightforward. ([04](04-data-and-features.md#the-retrain-dag))
14. **Retrain promotion uses a single-number gate.** `new_pr_auc >=
    champion_pr_auc` can promote on noise. Bootstrapped CI, or a
    shadow-traffic eval, would be more robust.
15. **Typecheck job is non-gating.** `continue-on-error: true` on
    mypy. Fine while type coverage is mixed; should flip to `false`
    once stable. ([08](08-testing-and-ci.md#job-2--typecheck-ciyml35-63))

### Minor but worth knowing

16. **Batch endpoint doesn't use `prepare_features_batch`.** Each
    row goes through `prepare_features` independently, so
    `amount_zscore` is always 0 in batches. Low-impact. ([06](06-serving-api.md#batch-prepare_features_batch--loaderpy172-194))
17. **`amount_log` is redundant for XGBoost.** Tree models are
    invariant to monotonic transforms. Kept because it helps the
    autoencoder and SHAP readability.
18. **SHAP adds ~40-80ms per request.** Reasonable, but could be
    opt-in via a query parameter.
19. **Autoencoder threshold isn't stored in the registry run metrics
    the same way XGBoost's is.** It lives inside the pyfunc's
    `threshold.txt` artifact instead. Not a bug, just asymmetry.
20. **Cardinality risk on Prometheus labels.** If you ever add a
    label with unbounded values (like `transaction_id`), Prometheus
    dies quickly. ([07](07-monitoring.md#cardinality-risk))
21. **No coverage floor in CI.** Coverage is reported but not
    enforced. `--cov-fail-under=80` would gate it. ([08](08-testing-and-ci.md#limitations))
22. **No security scanner in CI.** `make audit` exists; wiring to CI
    would catch CVEs on push.
23. **No performance regression tests.** A 100ms regression wouldn't
    fail CI. `pytest-benchmark` on hot paths would surface them.

## What you'd change first in a real system

If a team agreed to productionise this tomorrow, the ranked work
order:

1. **Add a prediction log.** Everything else (drift pipelines, offline
   quality, audit) depends on having this data. Postgres table or
   Kafka topic.
2. **Fix `amount_zscore` training-serving skew.** Log training-time
   mean/std; use them at inference.
3. **Add auth + rate limiting on `/predict`.** API keys minimum;
   OAuth2 / mTLS for real.
4. **Time-based train/val split.** Retrain and re-baseline the
   PR-AUC target. Everything upstream of this is optimistic until
   done.
5. **AlertManager + Slack routing.** Alerts that fire but don't page
   anyone are theatre.
6. **Gunicorn + multiple Uvicorn workers.** Instant latency and
   throughput win; lays groundwork for horizontal scaling.
7. **Graceful model reload.** Polling `models:/name@alias` every 60s
   and switching on version change.
8. **Shadow-traffic evaluation before promotion.** Route 100% to
   champion in production but score on challenger for N days and
   compare offline before moving the alias.

## Possible portfolio extensions

Things you could build *next* to extend the story, each with a
one-line recruiter pitch:

- **Kafka streaming ingress** — "Turned the static batch pipeline
  into a streaming pipeline with Kafka producer + Go consumer;
  shows event-driven architecture on top of the same ML stack." The
  commented stubs at `docker-compose.yml:107-168` are the starting
  point.
- **Feast feature store** — "Added Feast to eliminate training-
  serving feature skew and serve online features with sub-millisecond
  reads." Good for discussing feature store trade-offs.
- **Deploy to a cloud** — ECS / GKE / Cloud Run. Shows you can go
  from compose to managed infra. Pick one that matches the job
  you're applying for.
- **Hyperparameter sweep with Optuna** — "Added a 100-trial Optuna
  sweep; integrated with MLflow nested runs; improved PR-AUC by
  3-5%." Quantifiable.
- **Time-aware cross-validation** — "Replaced the random 80/20
  split with a TimeSeriesSplit; the honest PR-AUC is lower but no
  longer optimistic."
- **LLM-assisted fraud reasoning** — "Added a `POST
  /predict/explain` endpoint that feeds SHAP contributions into a
  small LLM and returns a natural-language rationale." Current-year
  addition.
- **Alertmanager + Slack** — small, clean, visible. Adds one
  container plus a two-line webhook config.
- **pgvector for similarity search** — store recent transactions as
  embeddings, flag new transactions by nearest-neighbour distance
  to past fraud. Complements the two existing models as a third
  paradigm (retrieval-based).
- **Drift-triggered retraining** — `drift_report.py` → Evidently →
  if drift detected → trigger the retrain DAG. Closes the loop.

## Interview talking-points cheat-sheet

Seven bullets you can say out loud, in order, to summarise the
project in under two minutes:

1. **The problem is imbalanced binary classification** — 0.17%
   fraud in a 284,807-row dataset, so accuracy is meaningless and
   PR-AUC is the honest metric. I handle the imbalance with three
   layered techniques: SMOTE on the training split,
   `scale_pos_weight` inside XGBoost, and a cost-weighted threshold
   calibration where missing a fraud is 10× worse than a false
   alarm.
2. **Two models, two paradigms, one API.** A supervised XGBoost as
   the champion and a PyTorch autoencoder as the unsupervised
   challenger. Running both lets me A/B test and demonstrates
   that supervised and anomaly-detection approaches are
   complementary.
3. **A/B routing is deterministic.** MD5 of the transaction ID modulo
   100, so the same transaction always hits the same model — this
   is what makes fair offline comparison possible.
4. **SHAP is warmed up at startup** and added to every XGBoost
   response. I use `TreeExplainer` because it's exact and
   polynomial-time on tree models; the autoencoder returns empty
   because `TreeExplainer` only works on trees.
5. **MLflow aliases, not versions, are the deploy contract.** Serving
   resolves `models:/fraud-xgboost@champion` at startup, so
   promoting a new version is one MLflow API call — zero serving
   code change. The retrain DAG promotes conditionally: new
   PR-AUC must beat the champion.
6. **Monitoring stack is Prometheus + Grafana + Evidently.**
   Prometheus does live rates and tail latency with a 15s scrape;
   Grafana renders four auto-provisioned panels; Evidently is the
   offline drift report. They're separate because they answer
   different questions at different cadences.
7. **I know what I cut.** No Kafka, no Feast, no Kubernetes, no
   Isolation Forest. Each was considered and rejected because it
   would add infra without moving the ML story forward. The
   commented stubs in `docker-compose.yml` show exactly where Kafka
   would plug in — they're a portfolio signal that I made a
   deliberate choice, not that I didn't know.

## What's explicitly out of scope (summary)

From the original `plan.md` decision log (lines 702-725), repeated
here for a recruiter conversation:

| Omitted | One-line reason |
|---|---|
| Kafka + Go consumer | Adds three containers to fake a stream over a static CSV |
| Feast feature store | One data source + one pipeline; Feast would be cargo-cult |
| Kubernetes | Single-node Compose is more honest for the actual scale |
| Isolation Forest | XGBoost + autoencoder already cover supervised + unsupervised |
| Real auth on the API | In-network portfolio service; auth is the first production add |
| AlertManager + Slack | Alerts visible in Prometheus UI; routing is one-step away |
| Hyperparameter sweep | Meaningful only against a time-based split, which is also missing |
| Cross-validation | Same reason as above; next thing to add |

## Where to go next

- Back to [01 — Big picture](01-big-picture.md) for the overall
  architecture.
- [02 — ML concepts](02-ml-concepts.md) if any ML term on this page
  still feels fuzzy.
- [README](README.md) to see the wiki ToC and reading order.
