# 07 — Monitoring

> **What this page answers:** How raw Prometheus metrics become a Grafana
> dashboard, what each alert actually means, and when to reach for the
> Evidently drift report instead.

Familiarity with [02 § Data drift](02-ml-concepts.md#data-drift-and-concept-drift)
and [06 § Custom Prometheus metrics](06-serving-api.md#custom-prometheus-metrics--servingappmetricspy)
is useful before diving in.

## The monitoring stack at a glance

```
 FastAPI (/metrics on :8000)
         |
         |  scrape every 15s
         v
    Prometheus  (TSDB, alert evaluation)  <--- rules.yml
         |
         |  HTTP queries via PromQL
         v
      Grafana  (4-panel dashboard) <--- provisioning YAML + JSON
```

- **Prometheus** is the time-series database. It scrapes FastAPI's
  `/metrics` endpoint on a schedule, stores the result, and evaluates
  alert rules.
- **Grafana** is the viewer. It queries Prometheus over HTTP using
  PromQL and renders panels.
- **Evidently** is a *separate*, *offline* diagnostic tool. It doesn't
  plug into Prometheus at all — it reads Parquet files and emits an
  HTML report.

Running `make up-monitoring` starts Prometheus + Grafana; `make up`
starts everything.

## A one-minute refresher on metric types

Three metric types show up here:

- **Counter** — monotonically increasing count. Only goes up, never
  down. Good for "total requests", "total errors". You always query
  counters via `rate()` to get a per-second derivative.
- **Gauge** — a value that can go up or down. This project doesn't
  define any custom gauges, but Prometheus-fastapi-instrumentator
  exposes some for in-flight requests.
- **Histogram** — a set of predefined buckets plus a sum and count.
  Enables `histogram_quantile(0.99, ...)` to estimate tail latencies
  cheaply. This project's `inference_latency_seconds` is a histogram.

## Prometheus config — `monitoring/prometheus/prometheus.yml`

Only 17 lines:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/rules.yml

scrape_configs:
  - job_name: prometheus
    static_configs: [{targets: [localhost:9090]}]
  - job_name: fraud_serving
    static_configs: [{targets: [serving:8000]}]
    metrics_path: /metrics
```

Key points:

- **`scrape_interval: 15s`** — Prometheus reads `/metrics` four times a
  minute. Higher resolution (5s or 1s) exists but would 3-15× the TSDB
  size for no practical gain here.
- **`evaluation_interval: 15s`** — alerts are re-evaluated every 15s.
  The `for:` clause on each alert is what adds delay to actual firing.
- **`targets: [serving:8000]`** — uses Docker DNS inside `fraud-net`.
  `serving` resolves to the FastAPI container. From outside the
  compose network (like your browser), you'd use `localhost:8000`
  instead.
- **Prometheus scrapes itself.** The `job_name: prometheus` entry lets
  you see Prometheus's own memory usage and scrape duration metrics —
  useful when you need to debug "is the dashboard slow because
  Prometheus is slow?".

### No `AlertManager`

The `rules.yml` file defines alerts, but there's no AlertManager
service in `docker-compose.yml`. That means alerts fire *in
Prometheus* (visible at `http://localhost:9090/alerts`) but they don't
route anywhere — no Slack, no PagerDuty, no email. This is intentional
for a portfolio project: AlertManager adds a container and an
integration step, and the alerts are best seen live in the Prometheus
UI anyway.

## Alert rules — `monitoring/alerting/rules.yml`

Three alerts, all under one group `fraud_detection`. Read
`rules.yml:4-33`.

### `HighFraudRate` — `rules.yml:4-13`

```yaml
expr: |
  rate(inference_total{prediction="fraud"}[5m])
  / rate(inference_total[5m]) > 0.10
for: 2m
severity: warning
```

- **What it means.** Over the last 5 minutes, more than 10% of
  predictions came back as fraud, and this has been true continuously
  for at least 2 minutes.
- **Why 10%.** The real-world baseline from the training set is
  0.17%. Even with classifier miscalibration, a healthy system runs
  at a few percent. 10% is the "definitely something's wrong" line —
  either traffic is unusually fraudulent, or the model is
  over-predicting (e.g. after a bad deploy).
- **Why 2m.** One-off burst of fraud requests (e.g. someone testing
  the API with obvious-fraud inputs) shouldn't page. Two minutes of
  *sustained* elevation is the signal worth investigating.
- **Honest critique.** At 0.17% baseline, 10% is a very wide margin.
  A production system might set this at 2% and tune it against the
  real rate. 10% works for a portfolio because synthetic test traffic
  would never cross it accidentally.

### `HighInferenceLatency` — `rules.yml:15-24`

```yaml
expr: histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m])) > 0.5
for: 1m
severity: warning
```

- **What it means.** The estimated p99 latency over the last 5
  minutes exceeds 500ms, sustained for 1 minute.
- **Why p99, not average.** Average latency hides tail behaviour. A
  model with mostly-5ms predictions plus a few pathological 2-second
  ones looks fine on average but is terrible for users. p99 forces
  the tail into view.
- **Why 500ms.** XGBoost + SHAP is typically 5-80ms. 500ms would mean
  something is clearly wrong — GC pause, disk IO, something hanging
  the worker.
- **Honest critique.** `histogram_quantile` is an estimate based on
  the histogram buckets defined in `serving/app/metrics.py:9`
  (`[5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s]`). Everything
  above 500ms falls into the 1s bucket, so p99 > 500ms could really
  be p99 = 700ms or p99 = 990ms — you can't tell. If you wanted finer
  detail at the tail, you'd add buckets at 700ms, 850ms, 1500ms, 3s.

### `InferenceErrorSpike` — `rules.yml:26-33`

```yaml
expr: rate(inference_errors_total[5m]) > 0.1
for: 1m
severity: critical
```

- **What it means.** More than 0.1 errors per second, sustained 1
  minute. At 1 request/s that's 10% error rate.
- **Why `critical`**. Errors mean the model threw an exception — bad
  input that escaped validation, a loaded model is corrupt, OOM in
  SHAP. These are never normal.
- **Honest critique.** An absolute rate (not a ratio against total
  requests) is ambiguous. If you're doing 100 req/s, 0.1 err/s is
  0.1% (tolerable). If you're doing 0.5 req/s, 0.1 err/s is 20%
  (broken). A ratio-based expression (like the fraud rate above)
  would be more robust.

## Grafana — `monitoring/grafana/provisioning/`

Fully auto-provisioned. Two YAML files point Grafana at its config:

- **`datasources/datasource.yml`** (9 lines) — registers Prometheus at
  `http://prometheus:9090` (Docker DNS; `fraud-net` resolves this).
  Marked `isDefault: true` so dashboard panels can omit the datasource
  ref.
- **`dashboards/dashboard.yml`** (7 lines) — tells Grafana to load any
  JSON file under `dashboards/` into a folder named `Monitoring`.

The actual dashboard is a single JSON file,
`dashboards/fraud_detection.json`, containing four panels.

### Panel 1 — Request Rate (`fraud_detection.json:10-33`)

```promql
rate(inference_total[1m])
```

Time series, one line per `(model_name, prediction)` combination. Shows
requests/second broken down by champion vs challenger and fraud vs
legit. The shape of this panel immediately tells you: is traffic
steady? Is the champion getting most of it (it should, at 80/20)? Are
fraud predictions a thin strip at the bottom (yes, baseline) or
anything else (suspicious)?

### Panel 2 — Fraud Rate (`fraud_detection.json:35-69`)

```promql
100 * rate(inference_total{prediction="fraud"}[5m]) / rate(inference_total[5m])
```

Single-value stat panel. Thresholds: green < 5%, yellow 5-10%, red >
10%. The red threshold lines up with the `HighFraudRate` alert.

### Panel 3 — p99 Inference Latency (`fraud_detection.json:71-107`)

Two queries on the same panel:

- **p99** — `histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m])) * 1000`
- **p50** — same with `0.50`

Plotted in milliseconds. Having both on one chart makes it easy to see
whether the tail is diverging from the median (sign of GC, concurrent
contention, etc). Thresholds: green < 200ms, yellow 200-500ms, red >
500ms (matches the alert rule).

### Panel 4 — A/B Traffic Split (`fraud_detection.json:109-134`)

```promql
ab_test_assignments_total
```

Pie chart, instant query (not rate). Shows cumulative counts of
`model_variant="champion"` vs `"challenger"`. At steady state the pie
should match `AB_CHALLENGER_FRACTION` (80/20 by default). If the
split drifts, your A/B router has a bug.

### Dashboard refresh

`refresh: "10s"` at the bottom of the JSON (line 136). Panels auto-
refresh every 10 seconds while open. The time window defaults to the
last hour.

### Logging in

`http://localhost:3000`, username from `GF_SECURITY_ADMIN_USER`
(default `admin`), password from `GF_SECURITY_ADMIN_PASSWORD` (default
`change_me_grafana`). Set these in `.env` before exposing to anyone.

## Evidently drift report — `scripts/drift_report.py`

Offline and manual. Triggered with `make drift-report`.

### What it does (`drift_report.py:59-76`)

1. Loads `data/processed/features.parquet` as **reference data**
   (the training distribution).
2. Looks for `data/reports/current.parquet` as **current data** (what
   serving has been seeing). If absent, falls back to a 10% sample of
   the reference data — a sanity check that the report runs, not a
   meaningful drift signal.
3. Drops the `Class` column (drift on the label isn't what we want
   here — we want feature drift).
4. Runs Evidently's `DataDriftPreset`, which statistically compares
   each feature's distribution across the two datasets.
5. Saves an HTML report to `data/reports/drift_report.html`.

Open the HTML in a browser. Each feature shows:

- A histogram overlay of reference vs current.
- A test name (KS test for numeric features, chi-squared for
  categorical) and a p-value.
- A `drift detected` flag when the p-value falls below a threshold
  (~0.05 by default).

### What "current data" means here

There's no automatic pipeline that writes `current.parquet`. The
intended workflow is:

1. Serving runs for a while.
2. *You* (or a cron / Airflow task that isn't implemented yet) dump
   the recent serving inputs to `data/reports/current.parquet`.
3. `make drift-report` compares it against the reference.

In a production system, FastAPI would write every prediction to a
Kafka topic or a SQL table; a nightly job would aggregate the last N
days into `current.parquet`; `drift-report` would run as an Airflow
DAG. This project stops short of that pipeline on purpose — the
report is a *template* for how you'd wire it, not a running pipeline.

### Why KS and chi-squared

See [02 § Data drift and concept drift](02-ml-concepts.md#data-drift-and-concept-drift)
for the intuition. Short version:

- **Kolmogorov-Smirnov (numeric):** "Are the two empirical CDFs of
  this feature close enough to come from the same distribution?"
- **Chi-squared (categorical):** "Do the category counts look like
  draws from the same multinomial?"

Both output a test statistic and a p-value. Evidently uses default
thresholds and tags features as drifted. You can always open the HTML,
look at the overlay histograms, and decide for yourself whether a
"drift detected" flag is real or just noise at small sample size.

## Why Evidently is offline, not wired into Prometheus

Three reasons:

1. **Different cadence.** Prometheus scrapes every 15s; drift
   detection is a daily or weekly exercise. Running a K-S test every
   15s on 30 features is wasteful.
2. **Different storage.** Drift needs a distribution snapshot
   (thousands of samples), not a single scalar. Prometheus histograms
   can approximate but are lossy — Evidently wants the raw values.
3. **Different audience.** Live metrics are for on-call. Drift reports
   are for the ML team doing a quarterly review or investigating a
   suspected model quality drop.

Building a "drift detected → Prometheus metric → alert" bridge is
straightforward, but the right cadence for that alert is days, not
seconds.

## Runbook: what each signal means

| Symptom | Likely cause | Where to look |
|---|---|---|
| `HighFraudRate` firing | Model over-predicts; bad deploy; real attack | `/models` to check current versions; EDA notebook on recent inputs |
| `HighInferenceLatency` firing | GC pause; SHAP on many requests; oversized batches | Grafana p99/p50 split; container memory |
| `InferenceErrorSpike` firing | Feature schema mismatch; model loaded in degraded state | FastAPI logs; `/health` endpoint |
| Fraud rate quietly at 0% | `is_fraud` threshold too high; champion promotion used wrong threshold | `/models` → check `threshold` metric logged in MLflow |
| A/B split drifted from 80/20 | A/B router bug; one model in degraded state | `/health` — if one model is `unavailable`, the fallback in `_select_model` sends all traffic to the other |
| Evidently report shows drift on `amount_log` | Payment patterns changed (seasonal, holidays) | Consider retrain |
| Drift on V1–V28 | Real population shift in underlying features | Likely retrain; don't just re-scale |

## Limitations

- **Scrape interval granularity.** 15 seconds is good for rates, bad
  for debugging single slow requests. Individual slow calls are lost
  in the aggregation.
- **Static alert thresholds.** 10% fraud rate, 500ms p99, 0.1 err/s —
  hardcoded. A real system would learn baselines or compare against
  last-week-same-time.
- **No AlertManager.** Alerts fire but don't route. Add AlertManager
  + a Slack integration for anything serious.
- **No distributed tracing.** A slow `/predict` call doesn't produce
  a span waterfall. Would add OpenTelemetry if you wanted to see
  `prepare_features` vs `predict_xgb` vs `SHAP` timings per request.
- **Evidently is entirely offline.** No automatic ingestion of serving
  traffic, no trend chart across reports, no regression test.
- **No prediction log persistence.** Everything is aggregated into
  Prometheus; individual predictions are gone once counted. Makes root
  cause analysis harder.
- **Cardinality risk.** Every `model_name` value becomes a separate
  time series. Adding `transaction_id` as a label would create one
  series per transaction, which would kill Prometheus within minutes.
  Only add labels with bounded, low cardinality.
- **Grafana password is in `.env`.** Fine for dev. In production this
  lives in a secret manager and the admin is SSO-backed.

## Where to go next

- [08 — Testing and CI](08-testing-and-ci.md) for how the metrics
  module is tested and how integration tests verify the scrape path
  end-to-end.
- [06 § Custom Prometheus metrics](06-serving-api.md#custom-prometheus-metrics--servingappmetricspy)
  is the source-of-truth on the four metrics this dashboard reads.
- [10 — Limitations and extensions](10-limitations-and-extensions.md)
  covers what you'd add first — AlertManager + Slack routing is
  usually top of the list.
