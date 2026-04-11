# Feature Store

Feast feature repository for the fraud detection platform.
Manages offline (Parquet) and online (SQLite) feature stores.

## Entities
- `transaction` — identified by `transaction_id`

## Feature Views
- `transaction_features` — core engineered features (log transforms, time features)
- `transaction_stats` — rolling aggregation features (1h window)

## Usage

```bash
# Materialize features to online store (run from feature_store/feature_repo/)
feast materialize-incremental $(date +%Y-%m-%dT%H:%M:%S)
```
