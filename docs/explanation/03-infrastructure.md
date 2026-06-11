# 03, Infrastructure

> **What this page answers:** How the stack runs on one laptop, why every
> service has its own Python environment, what the `Makefile` is actually
> for, and where secrets live.

Read [01, Big picture](01-big-picture.md) first if you haven't, this page
assumes you already know the six services.

## Why Docker Compose

The whole stack, Postgres, MLflow, Airflow (three containers), FastAPI,
Prometheus, Grafana, comes up with `docker compose up -d`. That single
command is the headline feature:

- **Reproducible.** Anyone who clones the repo and has Docker installed
  gets the same stack. No "install this version of Postgres", no Airflow
  on the host Python, no MLflow server to babysit.
- **Mirrors production shape.** Each service is a container with its own
  image, ports, volumes, and restart policy. If you later move to
  Kubernetes or ECS, the unit of work (a container) is already the same.
- **Obvious wiring.** `docker-compose.yml` is a single ~170-line file.
  You can read the whole architecture from it in ten minutes.

What Compose gives up versus something like Kubernetes:

- **Single host.** Everything runs on one machine. No horizontal scaling,
  no multi-node failover.
- **No autoscaling.** The FastAPI container is one process; to scale you'd
  rewrite this as K8s (Deployment + Service + HPA) or ECS.
- **No secret management.** Secrets live in a plain `.env` file that you
  have to remember not to commit.

At this scale, those trade-offs are correct: readability beats
theoretical scale.

## `docker-compose.yml` walkthrough

All services share one bridge network (`fraud-net`) so they can resolve
each other by service name (`postgres`, `mlflow`, etc). The file is at
`docker-compose.yml`; line ranges below are stable at time of writing.

### PostgreSQL, `docker-compose.yml:30-51`

```yaml
postgres:
  image: postgres:15
  ports: ["${POSTGRES_PORT:-5432}:5432"]
  volumes:
    - postgres_data:/var/lib/postgresql/data
    - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql:ro
  healthcheck: pg_isready ...
```

- **One database, three schemas/DBs**, Postgres hosts the Airflow
  metadata DB, the MLflow tracking DB, and the project's own `fraud_db`.
  `scripts/init_db.sql` runs once on first start and creates the extra
  databases MLflow/Airflow need.
- **Healthcheck**, the `pg_isready` loop at lines 44-49 is what every
  other service waits on via `depends_on: condition: service_healthy`.
  Without this, MLflow and Airflow would race Postgres on the first
  `docker compose up` and crash-loop.
- **Volume `postgres_data`** (declared at line 173), named volume, not a
  bind mount, so the data persists across `docker compose down` but gets
  wiped by `docker compose down -v` (that's what `make down-volumes`
  does).

### MLflow, `docker-compose.yml:55-89`

```yaml
mlflow:
  build:
    context: .
    dockerfile: Dockerfile.mlflow
  volumes:
    - mlflow_artifacts:/app/mlartifacts
  command: mlflow server --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} ...
```

- **Custom image** (`Dockerfile.mlflow`) instead of the stock MLflow image
  so psycopg2 is included, MLflow's Postgres backend needs it.
- **Proxied artifacts.** `--serve-artifacts` plus
  `--default-artifact-root mlflow-artifacts:/` makes the server the single
  gateway for artifacts: clients (host training, the retrain DAG, serving)
  see `mlflow-artifacts:/` URIs and upload/download over HTTP; the server
  stores the files under `--artifacts-destination` (the
  `mlflow_artifacts` volume). This matters because the alternative — a
  local path as `--default-artifact-root` — gets recorded into each
  experiment and interpreted by every *client* as a path on its own
  machine, scattering artifacts across hosts (a classic MLflow
  misconfiguration).
- **`mlflow_artifacts` volume**, this named volume is shared with the
  serving container (`docker-compose.yml:132`). See
  [Shared artifacts volume](#shared-artifacts-volume-why) below.

### Airflow, `docker-compose.yml:94-118`

Three containers built from the same image (`airflow/Dockerfile`) using a
YAML anchor (`x-airflow-common: &airflow-common` at lines 9-22) to avoid
repetition:

- **`airflow-init`** (87-91), runs `airflow db migrate` then creates an
  admin user. `restart: "no"` because this is a one-shot task; the other
  two wait for it to complete successfully before starting.
- **`airflow-webserver`** (93-102), UI at `:8080`. Login: `admin` /
  `admin` (dev only; spelled out in the init command at line 90).
- **`airflow-scheduler`** (104-111), the process that actually runs DAGs.
  Without it, DAGs appear in the UI but never execute.

DAG and plugin code is bind-mounted from the host (`./airflow/dags`,
`./airflow/plugins`) so you can edit a DAG on your laptop and see the
scheduler pick it up without rebuilding the image. The training code is
also mounted, read-only, at `/opt/airflow/training` so the retrain DAG
can run `train_xgboost.py` and import the shared promotion gate; the
image (`airflow/Dockerfile`) installs the training dependencies (minus
torch — the DAG only retrains XGBoost), pinned to the same versions as
`training/requirements.txt`.

### FastAPI serving, `docker-compose.yml:122-148`

```yaml
serving:
  build:
    context: ./serving
    dockerfile: Dockerfile
  ports: ["${SERVING_PORT:-8000}:8000"]
  volumes:
    - mlflow_artifacts:/app/mlartifacts   # shared with mlflow
  depends_on:
    postgres: { condition: service_healthy }
    mlflow:   { condition: service_healthy }
  healthcheck:
    test: ["CMD", "python", "-c", "... status == 'healthy' ..."]
```

- **`depends_on` waits for MLflow to be ready, not just started.**
  MLflow has its own compose healthcheck (a python `urllib` probe of
  its `/health` endpoint — the image has no curl), and serving uses
  `condition: service_healthy` on it. That matters because serving
  loads its models once, at startup: with the old `service_started`
  condition it could race a still-booting MLflow, fail the load, and
  sit in degraded mode until someone restarted it.
- **Serving's own healthcheck checks model state, not just liveness.**
  `GET /health` always returns 200; the probe parses the JSON and
  exits nonzero unless `status` is `"healthy"` (both models loaded).
  So `docker compose ps` tells you whether the API can actually score,
  which makes `make up` self-verifying. On a fresh clone with no
  trained models the container reports *unhealthy* — that is accurate,
  not a bug: train, then restart serving (the README quickstart's
  order).
- **Port 8000**, both `/predict` and `/metrics` live on it. Prometheus
  scrapes `serving:8000/metrics` every 15s.
- **Code is baked into the image, not bind-mounted.** The Dockerfile's
  `COPY . /app/serving/` means a serving-code change requires
  `docker compose build serving` — a restart alone keeps running the
  old code. `serving/.dockerignore` keeps the local `.venv` (several
  GB, mostly torch), `tests/`, and tool caches out of the build
  context; without it the venv was being copied into the image on
  every build, adding ~2 GB of dead weight and making the context
  transfer take minutes instead of milliseconds.

### Prometheus, `docker-compose.yml:152-165`

```yaml
prometheus:
  image: prom/prometheus:v2.48.0
  volumes:
    - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    - ./monitoring/alerting/rules.yml:/etc/prometheus/rules.yml:ro
  command:
    - --storage.tsdb.retention.time=${PROMETHEUS_RETENTION_DAYS:-15}d
```

- **Config is read-only bind-mounted.** Editing
  `monitoring/prometheus/prometheus.yml` on the host and restarting the
  container is the workflow, no rebuild needed.
- **15-day retention.** TSDB is on the container's ephemeral storage, not
  a named volume. That's an intentional trade-off: if
  you want Prometheus data to survive `docker compose down -v`, add a
  `prometheus_data` volume.

### Grafana, `docker-compose.yml:169-181`

```yaml
grafana:
  image: grafana/grafana:10.2.0
  volumes:
    - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
```

- **Auto-provisioning.** Datasource (Prometheus) and dashboard are
  defined as YAML/JSON under `monitoring/grafana/provisioning/` and
  loaded on startup. No manual "add data source" dance after every
  fresh spin-up.
- **Admin credentials** come from `GF_SECURITY_ADMIN_USER` and
  `GF_SECURITY_ADMIN_PASSWORD` in `.env`.

## Shared artifacts volume (why)

Two services mount the same named volume `mlflow_artifacts`:

| Service | Mount | Line |
|---|---|---|
| `mlflow` | `mlflow_artifacts:/app/mlartifacts` | `docker-compose.yml:65` |
| `serving` | `mlflow_artifacts:/app/mlartifacts` | `docker-compose.yml:132` |

The MLflow server stores all artifacts (the XGBoost booster, the
autoencoder weights, `scaler.pkl`) in this volume — it is the server's
`--artifacts-destination`. Since the move to proxied artifacts (see the
MLflow section above), models trained after that change carry
`mlflow-artifacts:/` URIs and serving fetches them **over HTTP through
the MLflow server** at startup — the volume mount in `serving` is not
involved for those loads.

The mount remains in `serving` for backward compatibility: model
versions registered *before* the proxy fix have absolute-path artifact
URIs (`/app/mlartifacts/...`), and for those the path must be visible
inside the serving container for the load to work. Once every serving
model has been retrained under the proxied scheme, the mount could be
dropped from `serving` — which is exactly what you'd do to split the
two services onto different hosts.

## The network: `fraud-net`

One bridge network, defined at `docker-compose.yml:183-186`:

```yaml
networks:
  fraud-net:
    name: fraud-detection-net
    driver: bridge
```

Every service joins it (you'll see `networks: [fraud-net]` on each one).
Docker's built-in DNS means each service name becomes a resolvable
hostname inside the network:

- From `serving`, `mlflow:5000` resolves to the MLflow container.
- From `prometheus`, `serving:8000` resolves to FastAPI.
- From `airflow-*`, `postgres:5432` resolves to the DB.

This is why `.env` has `MLFLOW_TRACKING_URI=http://mlflow:5000` and
`POSTGRES_HOST=postgres`, those hostnames only work *inside* the
network. From your laptop's browser you use `localhost:5000`, which
hits the port-mapped side (`5000:5000`).

## Per-service virtual environments

Every service has its own `requirements.txt` and its own `.venv/`:

| Context | Requirements | Venv path | Make target |
|---|---|---|---|
| Dev tools (ruff, black, mypy, pytest, pip-audit) | `requirements-dev.txt` | `.venv/` | `make venv` |
| Training | `training/requirements.txt` | `training/.venv/` | `make venv-training` |
| Serving | `serving/requirements.txt` | `serving/.venv/` | `make venv-serving` |
| Airflow | `airflow/requirements.txt` | `airflow/.venv/` | `make venv-airflow` |
| Evidently | `monitoring/evidently/requirements.txt` | `monitoring/evidently/.venv/` | `make venv-evidently` |

### Why five venvs instead of one

- **Airflow's constraints are brutal.** It pins a huge transitive tree
  (Celery, Flask, SQLAlchemy versions, etc). Letting it share an env
  with modern PyTorch / XGBoost / FastAPI almost always triggers
  a resolver conflict.
- **Training vs serving drift.** Training needs `torch`, `xgboost`, and
  `imblearn` (heavy). Serving only needs `mlflow` client + `xgboost`
  runtime + `shap` + `fastapi`. Separating them keeps the serving image
  small and fast to rebuild.
- **Evidently is version-sensitive.** Its metric API has changed a lot
  across minor versions; pinning it in its own venv means the training
  venv doesn't have to accommodate an Evidently-compatible pandas
  release.
- **CI can parallelize.** Lint and typecheck run on the dev venv; unit
  tests run per-service. No single monster pip install.

The `Makefile` (lines 6-18) detects whether you're on Windows
(`OS=Windows_NT` in Git Bash) and picks
`.venv/Scripts/python` vs `.venv/bin/python` automatically, so the same
targets work on either platform.

**Limitation:** five venvs means five pip installs. `make venv-all`
takes a while on a cold machine (the Airflow one alone is 5-10 minutes
because Airflow's dependency graph is enormous). It's a one-time cost,
but worth noting.

## The `Makefile`: the human API

If Compose is the machine API, the `Makefile` is the human one. It groups
targets by concern with comment headers (`## ── Infrastructure ──`,
`## ── Testing ──`, ...) so `make help` renders a clean menu.

### Infrastructure, `Makefile:27-53`

| Target | What it does |
|---|---|
| `up` | `docker compose up -d`, start everything |
| `up-postgres` | Postgres only (leftover from Phase 1 when it was all you had) |
| `up-monitoring` | `prometheus` + `grafana` only (Phase 5 isolation) |
| `down` | Stop; keep volumes |
| `down-volumes` | Stop and wipe volumes (irreversible data loss) |
| `logs` | `docker compose logs -f` |
| `ps` | `docker compose ps` |
| `psql` | `psql` shell into the Postgres container |

### Virtual environments, `Makefile:57-92`

`make venv`, `make venv-training`, `make venv-serving`, `make
venv-airflow`, `make venv-evidently`, plus `make venv-all` to create
every one and `make clean-venvs` to nuke them.

### Code quality, `Makefile:96-122`

| Target | Runs under | What it does |
|---|---|---|
| `lint` / `lint-fix` | dev | Ruff check / auto-fix |
| `format` / `format-check` | dev | Black format / format check |
| `typecheck` | dev + training | mypy on `serving/app/` and `training/` |
| `audit` | dev | `pip-audit` on every requirements file |

`typecheck` deliberately runs the training venv for the training
checker, because training imports (torch, sklearn) are not installed in
the dev venv.

### Testing, `Makefile:126-144`

| Target | Runs under | What it does |
|---|---|---|
| `test` | training + serving | All unit tests |
| `test-serving` | serving | Just `serving/tests/` |
| `test-training` | training | Just `training/` |
| `test-integration` | dev | `tests/integration/` (needs docker compose services running) |
| `check` | multiple | `format-check + lint + typecheck + test`, the CI equivalent |

`make check` is the single command you run before pushing: if it
passes locally, CI will pass.

### Training, `Makefile:146-161`

| Target | What it does |
|---|---|
| `train-xgboost` | Train XGBoost, log to MLflow, register a new version (no promotion) |
| `train-autoencoder` | Train AE, log to MLflow, register as `challenger` |
| `promote` | Move `champion` to the latest XGBoost version, only if its PR-AUC is at least as good |
| `train` | Run both trainings, then the gated `promote` |

### Data + monitoring, `Makefile:163-178`

| Target | What it does |
|---|---|
| `download-data` | Fetch Kaggle dataset (needs `KAGGLE_*` in `.env`) |
| `drift-report` | Render Evidently HTML to `data/reports/drift_report.html` |

## `.env` and secrets hygiene

Two files, one committed, one not:

- **`.env.example`**, committed. Every variable the stack reads, with
  placeholder values like `change_me_postgres`. A new contributor copies
  this to `.env` and fills in real values.
- **`.env`**, gitignored. The real values.

### What goes in `.env` (grouped as in `.env.example`)

| Group | Key variables |
|---|---|
| Postgres | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `DATABASE_URL` |
| MLflow | `MLFLOW_TRACKING_URI`, `MLFLOW_BACKEND_STORE_URI`, `MLFLOW_ARTIFACT_ROOT` |
| Airflow | `AIRFLOW__CORE__EXECUTOR`, `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`, `AIRFLOW__CORE__FERNET_KEY`, `AIRFLOW__WEBSERVER__SECRET_KEY` |
| Serving | `SERVING_PORT`, `AB_CHALLENGER_FRACTION`, `MODEL_*_NAME`, `MODEL_*_ALIAS` |
| Prometheus | `PROMETHEUS_PORT`, `PROMETHEUS_RETENTION_DAYS` |
| Grafana | `GRAFANA_PORT`, `GF_SECURITY_ADMIN_USER`, `GF_SECURITY_ADMIN_PASSWORD` |
| Evidently | `EVIDENTLY_REFERENCE_DATA_PATH`, `EVIDENTLY_REPORTS_PATH` |
| Kaggle | `KAGGLE_USERNAME`, `KAGGLE_KEY` |

### How services pick them up

- `docker-compose.yml` uses `env_file: .env` on every service that needs
  secrets (lines 13, 31, 58, 177, 213).
- `${VAR:-default}` syntax in the compose file (e.g.
  `${POSTGRES_PORT:-5432}`) gives a fallback when the variable is unset,
  so the stack can still come up with sane defaults during early
  development.
- Application code reads them via `os.environ` / `pydantic-settings`
  (see `serving/app/config.py`).

### Fernet key and webserver secret

`AIRFLOW__CORE__FERNET_KEY` must be a real Fernet key, not the
placeholder. The `.env.example` shows how to generate one:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

If you forget, Airflow won't start, it fails hard on an invalid Fernet
key rather than silently running with a bad one, which is the right
default.

## Limitations

- **No orchestrator.** Single-host Docker Compose. If one container
  dies hard (OOM, segfault loop), only `restart: unless-stopped`
  brings it back, and only on that host.
- **Ports are hardcoded.** `localhost:5432`, `:5000`, `:8080`, `:8000`,
  `:9090`, `:3000`. If anything else on your machine already uses one
  of them, you have to change it in `.env` (the compose file reads
  `${POSTGRES_PORT:-5432}` so it's one line to override).
- **No TLS.** Everything is HTTP inside the network and on localhost.
  Fine for a laptop stack; do not expose any of these ports to the
  public internet.
- **No auth on MLflow, Prometheus, or the FastAPI `/predict`
  endpoint.** Grafana has a login, Airflow has a login, the others
  don't. See [10, Limitations and extensions](10-limitations-and-extensions.md)
  for what you'd add first.
- **`.env` is plain text.** Sufficient for dev; in a real deployment
  you'd use a secret manager (AWS Secrets Manager, GCP Secret Manager,
  Vault) and inject at runtime.
- **One Postgres for everything.** Airflow metadata, MLflow metadata,
  and any app data share one instance. That's fine at this scale; in
  production you'd separate at least Airflow's metadata DB.

## Where to go next

- [04, Data and features](04-data-and-features.md) covers the dataset,
  the ingestion DAG, and what every engineered feature actually means.
- [06, Serving API](06-serving-api.md) dives into how the FastAPI
  container talks to MLflow and renders SHAP responses.
- [10, Limitations and extensions](10-limitations-and-extensions.md)
  has the consolidated "what I'd change before shipping" list.
