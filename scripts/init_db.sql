-- Runs automatically on first `docker compose up postgres`.
-- Creates additional databases used by later-phase services.
-- The main POSTGRES_DB (fraud_db) is already created by the postgres image.
-- Uses idempotent pattern so re-running against an existing volume is safe.

SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec

SELECT 'CREATE DATABASE airflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow_db')\gexec

GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO fraud_user;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO fraud_user;
