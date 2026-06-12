# Evidently

Data drift detection for the fraud detection platform. This directory
holds the isolated venv and requirements for Evidently; the report
script itself lives at `scripts/drift_report.py`.

## Usage

```bash
# From project root
make venv-evidently

# Generate synthetic "current" data with injected drift, then the HTML report
make generate-drift-data
make drift-report
# Output: data/reports/drift_report.html
```

## Configuration (via .env, both optional)
- `EVIDENTLY_REFERENCE_DATA_PATH` — training reference parquet (default: `data/processed/features.parquet`)
- `EVIDENTLY_REPORTS_PATH` — output directory (default: `data/reports`)
