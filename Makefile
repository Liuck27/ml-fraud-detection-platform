SHELL := bash
.DEFAULT_GOAL := help

# Python binary — call the venv Python directly (no shell activation needed).
# Works on Windows (Git Bash) with Scripts/ and on Linux/Mac with bin/.
ifeq ($(OS),Windows_NT)
    PYTHON          := .venv/Scripts/python
    PYTHON_TRAINING := training/.venv/Scripts/python
    PYTHON_SERVING  := serving/.venv/Scripts/python
    PYTHON_AIRFLOW  := airflow/.venv/Scripts/python
    PYTHON_EVIDENTLY:= monitoring/evidently/.venv/Scripts/python
else
    PYTHON          := .venv/bin/python
    PYTHON_TRAINING := training/.venv/bin/python
    PYTHON_SERVING  := serving/.venv/bin/python
    PYTHON_AIRFLOW  := airflow/.venv/bin/python
    PYTHON_EVIDENTLY:= monitoring/evidently/.venv/bin/python
endif

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}'

# ── Infrastructure ─────────────────────────────────────────────────────────────

.PHONY: up
up: ## Start all active docker-compose services
	docker compose up -d

.PHONY: up-postgres
up-postgres: ## Start PostgreSQL only (Phase 1)
	docker compose up -d postgres

.PHONY: down
down: ## Stop containers (keeps volumes)
	docker compose down

.PHONY: down-volumes
down-volumes: ## Stop containers AND delete all volumes — resets all data
	docker compose down -v

.PHONY: logs
logs: ## Tail logs for all running services
	docker compose logs -f

.PHONY: ps
ps: ## Show status of all compose services
	docker compose ps

.PHONY: psql
psql: ## Open a psql shell into the running postgres container
	docker compose exec postgres psql -U $${POSTGRES_USER:-fraud_user} -d $${POSTGRES_DB:-fraud_db}

# ── Python Virtual Environments ────────────────────────────────────────────────

.PHONY: venv
venv: ## Create the root dev-tools venv (ruff, black, mypy, pytest, pip-audit)
	python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

.PHONY: venv-training
venv-training: ## Create the training venv (torch, xgboost, sklearn, mlflow...)
	python -m venv training/.venv
	$(PYTHON_TRAINING) -m pip install --upgrade pip
	$(PYTHON_TRAINING) -m pip install -r training/requirements.txt

.PHONY: venv-serving
venv-serving: ## Create the serving venv (fastapi, mlflow, prometheus...)
	python -m venv serving/.venv
	$(PYTHON_SERVING) -m pip install --upgrade pip
	$(PYTHON_SERVING) -m pip install -r serving/requirements.txt

.PHONY: venv-airflow
venv-airflow: ## Create the airflow venv — slow (~5-10 min) due to Airflow's large dependency graph
	python -m venv airflow/.venv
	$(PYTHON_AIRFLOW) -m pip install --upgrade pip
	$(PYTHON_AIRFLOW) -m pip install -r airflow/requirements.txt

.PHONY: venv-evidently
venv-evidently: ## Create the Evidently monitoring venv
	python -m venv monitoring/evidently/.venv
	$(PYTHON_EVIDENTLY) -m pip install --upgrade pip
	$(PYTHON_EVIDENTLY) -m pip install -r monitoring/evidently/requirements.txt

.PHONY: venv-all
venv-all: venv venv-training venv-serving venv-airflow venv-evidently ## Create ALL venvs

.PHONY: clean-venvs
clean-venvs: ## Remove all .venv directories (run venv-all afterwards to rebuild)
	rm -rf .venv training/.venv serving/.venv airflow/.venv monitoring/evidently/.venv

# ── Code Quality ───────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Run ruff linter
	$(PYTHON) -m ruff check .

.PHONY: lint-fix
lint-fix: ## Run ruff with auto-fix
	$(PYTHON) -m ruff check --fix .

.PHONY: format
format: ## Run black formatter
	$(PYTHON) -m black .

.PHONY: format-check
format-check: ## Check formatting without modifying files (for CI)
	$(PYTHON) -m black --check .

.PHONY: typecheck
typecheck: ## Run mypy on serving and training
	$(PYTHON_SERVING) -m mypy serving/app/
	$(PYTHON_TRAINING) -m mypy training/

.PHONY: audit
audit: ## Run pip-audit security scan on all requirements files
	$(PYTHON) -m pip_audit -r training/requirements.txt
	$(PYTHON) -m pip_audit -r serving/requirements.txt
	$(PYTHON) -m pip_audit -r airflow/requirements.txt
	$(PYTHON) -m pip_audit -r monitoring/evidently/requirements.txt

# ── Testing ────────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run all unit tests
	$(PYTHON_TRAINING) -m pytest training/ -v
	$(PYTHON_SERVING) -m pytest serving/tests/ -v

.PHONY: test-serving
test-serving: ## Run serving unit tests only
	$(PYTHON_SERVING) -m pytest serving/tests/ -v

.PHONY: test-training
test-training: ## Run training unit tests only
	$(PYTHON_TRAINING) -m pytest training/ -v

.PHONY: test-integration
test-integration: ## Run integration tests (requires docker compose services up)
	$(PYTHON) -m pytest tests/integration/ -v

.PHONY: check
check: format-check lint typecheck test ## Run all checks — equivalent to CI

# ── Training ───────────────────────────────────────────────────────────────

.PHONY: train-xgboost
train-xgboost: ## Train XGBoost classifier and register as champion in MLflow
	$(PYTHON_TRAINING) training/train_xgboost.py

.PHONY: train-autoencoder
train-autoencoder: ## Train Autoencoder and register as challenger in MLflow
	$(PYTHON_TRAINING) training/train_autoencoder.py

.PHONY: train
train: train-xgboost train-autoencoder ## Train both models (XGBoost + Autoencoder)

# ── Data ───────────────────────────────────────────────────────────────────────

.PHONY: download-data
download-data: ## Download Kaggle fraud dataset (set KAGGLE_USERNAME + KAGGLE_KEY in .env)
	$(PYTHON) scripts/download_data.py

# ── Cleanup ────────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove Python cache and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
