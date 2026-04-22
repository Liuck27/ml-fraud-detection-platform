# Project Wiki

An in-depth, file-by-file walkthrough of the ML Fraud Detection Platform —
written for someone who built the project but wants to **really** understand
every component before showing it off.

## Who this is for

You, a junior software engineer with solid Python and Docker fundamentals,
but limited background in ML. This wiki assumes you can read code, run
`docker compose`, and navigate a repo — but it will not assume you know what
SHAP, SMOTE, PR-AUC, or an autoencoder are. Every ML concept is introduced
before it's used.

## What this wiki does not do

- It does not replace `README.md` at the repo root — that's the recruiter-
  facing landing page. This wiki is the "sit down and learn the system"
  document.
- It does not replace `plan.md` — that's the original spec. This wiki
  explains **why** those decisions were made and where they show up in
  code, with working file/line references.
- It does not include tutorials for setting up Python, Docker, or Git.

## How to read this

Read in order the first time — each page builds on the previous one:

1. [01 — Big picture](01-big-picture.md) — what the system is, why it
   exists, and how the services connect.
2. [02 — ML concepts primer](02-ml-concepts.md) — just enough ML theory
   to understand every model, metric, and monitoring choice in the
   project. Self-contained; come back to it as a reference.
3. [03 — Infrastructure](03-infrastructure.md) — Docker Compose, the
   Makefile, per-service virtual environments, and secrets hygiene.
4. [04 — Data and features](04-data-and-features.md) — the Kaggle
   dataset, the ingestion DAG, and each engineered feature.
5. [05 — Training](05-training.md) — the XGBoost and Autoencoder
   training scripts, MLflow tracking, and the model registry.
6. [06 — Serving API](06-serving-api.md) — FastAPI app lifecycle,
   Pydantic schemas, the model loader, A/B routing, SHAP, and
   Prometheus instrumentation.
7. [07 — Monitoring](07-monitoring.md) — Prometheus metrics and
   alerts, Grafana panels, and Evidently drift reports.
8. [08 — Testing and CI](08-testing-and-ci.md) — unit tests,
   integration tests, and the GitHub Actions workflow.
9. [09 — Glossary](09-glossary.md) — a one-line definition for every
   term in the wiki. Use it as a quick reference.
10. [10 — Limitations and extensions](10-limitations-and-extensions.md)
    — honest trade-offs, what's deliberately out of scope, and what
    you'd build next. Ends with an interview-ready cheat-sheet.

## The system at a glance

```
          +------------+       +--------+       +------------+
   CSV -> |  Airflow   | --->  | Parquet| --->  |  Training  |
          |  (ingest)  |       |  file  |       | (XGB + AE) |
          +------------+       +--------+       +-----+------+
                                                      |
                                                      v
                                               +------+-------+
                                               |    MLflow    |
                                               | (experiments |
                                               |  + registry) |
                                               +------+-------+
                                                      |
                                                      v
                    +-------------+  scrapes  +-------+-------+
                    | Prometheus  | <-------- |   FastAPI     |
                    +------+------+  /metrics | (/predict ++) |
                           |                  +-------+-------+
                           v                          |
                    +------+------+                   |
                    |   Grafana   |                   |
                    +-------------+                   |
                                                      v
                                               +------+-------+
                                               |   Evidently  |
                                               |  drift (HTML)|
                                               +--------------+
```

Solid arrows are automatic. The Evidently box is run manually (`make
drift-report`) — it's a diagnostic, not a production pipeline. See
[01 — Big picture](01-big-picture.md) for the full story.

## Conventions in this wiki

- **File paths** are written from the repo root:
  `serving/app/routes/predict.py`.
- **Line references** are `file:line_number` (e.g.
  `serving/app/main.py:24`) or a range (`:59-110`). They were accurate
  at the time of writing — if they drift, use them as directions, not
  GPS coordinates.
- **Cross-page links** are relative: `[SHAP](02-ml-concepts.md#shap)`.
  You can click through on GitHub.
- Every architectural decision ends with a **Limitations** or
  **Trade-offs** section. If you find one that doesn't, the page is
  incomplete — open an issue on yourself.

## Where to start contributing to this wiki later

If you learn something new about your own project, update the relevant
page, then update the page's "last reviewed" sense by glancing at the
line references and fixing any that drifted. Keep the wiki honest;
a wiki full of plausible-but-wrong statements is worse than no wiki.
