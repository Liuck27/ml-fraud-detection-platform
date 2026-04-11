# Evidently

Data and prediction drift detection for the fraud detection platform.
Generates HTML reports and exports drift metrics to Prometheus.

## Local Dev

```bash
# From project root
make venv-evidently

# Run drift report (Phase 9 — requires serving data)
monitoring/evidently/.venv/Scripts/python monitoring/evidently/drift_report.py
```

## Configuration (via .env)
- `EVIDENTLY_REFERENCE_DATA_PATH` — path to training reference dataset
- `EVIDENTLY_REPORTS_PATH` — output directory for HTML reports
