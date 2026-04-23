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
- **Obvious wiring.** `docker-compose.yml` is a single ~230-line file.
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

### PostgreSQL, `docker-compose.yml:27-48`

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
- **Healthcheck**, the `pg_isready` loop at lines 41-46 is what every
  other service waits on via `depends_on: condition: service_healthy`.
  Without this, MLflow and Airflow would race Postgres on the first
  `docker compose up` and crash-loop.
- **Volume `postgres_data`** (declared at line 229), named volume, not a
  bind mount, so the data persists across `docker compose down` but gets
  wiped by `docker compose down -v` (that's what `make down-volumes`
  does).

### MLflow, `docker-compose.yml:52-74`

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
- **`--serve-artifacts`** (line 70) makes MLflow the single source of
  truth for artifact downloads. The serving container doesn't need direct
  filesystem access to fetch models by URI; it talks HTTP to MLflow. But
  see the next point, it also has filesystem access, for a reason.
- **`mlflow_artifacts` volume**, this named volume is shared with the
  serving container (`docker-compose.yml:181`). See
  [Shared artifacts volume](#shared-artifacts-volume-why) below.

### Airflow, `docker-compose.yml:79-103`

Three containers built from the same image (`airflow/Dockerfile`) using a
YAML anchor (`x-airflow-common: &airflow-common` at lines 9-22) to avoid
repetition:

- **`airflow-init`** (79-83), runs `airflow db migrate` then creates an
  admin user. `restart: "no"` because this is a one-shot task; the other
  two wait for it to complete successfully before starting.
- **`airflow-webserver`** (85-94), UI at `:8080`. Login: `admin` /
  `admin` (dev only; spelled out in the init command at line 82).
- **`airflow-scheduler`** (96-103), the process that actually runs DAGs.
  Without it, DAGs appear in the UI but never execute.

DAG and plugin code is bind-mounted from the host (`./airflow/dags`,
`./airflow/plugins`) so you can edit a DAG on your laptop and see the
scheduler pick it up without rebuilding the image.

### FastAPI serving, `docker-compose.yml:171-188`

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
    mlflow:   { condition: service_started }
```

- **`depends_on` is weak**, `service_started` on MLflow only waits for
  the container to exist, not for MLflow to be ready to serve requests.
  The startup code in `serving/app/main.py:24-33` is what actually
  retries model loads; this is documented on the serving page.
- **Port 8000**, both `/predict` and `/metrics` live on it. Prometheus
  scrapes `serving:8000/metrics` every 15s.

### Prometheus, `docker-compose.yml:192-205`

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

### Grafana, `docker-compose.yml:209-221`

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

### Kafka / Zookeeper / producers, `docker-compose.yml:107-168`

Left as commented stubs. See [01, Big picture § Out of scope](01-big-picture.md#whats-deliberately-out-of-scope)
for why.

## Shared artifacts volume (why)

Two services mount the same named volume `mlflow_artifacts`:

| Service | Mount | Line |
|---|---|---|
| `mlflow` | `mlflow_artifacts:/app/mlartifacts` | `docker-compose.yml:62` |
| `serving` | `mlflow_artifacts:/app/mlartifacts` | `docker-compose.yml:181` |

The MLflow server writes model artifacts (the XGBoost booster, the
autoencoder weights, `scaler.pkl`) to `/app/mlartifacts`. When FastAPI
starts up and calls `mlflow.pyfunc.load_model("models:/fraud-xgboost@champion")`,
MLflow's client library resolves that URI to an artifact path, and
because the path is visible inside the serving container too, the load
is essentially a local file read, no HTTP copy.

This is a slightly unusual choice. The alternative is pulling artifacts
over HTTP via `--serve-artifacts`, which works but is slower and adds a
dependency on MLflow being up *healthy* (not just started) at serving
startup. Sharing the volume keeps startup fast and deterministic.

**Limitation:** the two containers are now coupled by shared filesystem.
If you ever split them onto different hosts, the serving container has
to switch to HTTP artifact fetching and you'd drop the
`volumes: mlflow_artifacts:...` from the `serving` service.

## The network: `fraud-net`

One bridge network, defined at `docker-compose.yml:223-226`:

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

### Training, `Makefile:148-157`

| Target | What it does |
|---|---|
| `train-xgboost` | Train XGBoost, log to MLflow, promote to `champion` |
| `train-autoencoder` | Train AE, log to MLflow, register as `challenger` |
| `train` | Run both in sequence |

### Data + monitoring, `Makefile:161-171`

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
