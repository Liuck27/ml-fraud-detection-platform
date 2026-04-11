# Kafka Producer

Simulates a real-time transaction stream by reading the credit card dataset,
adding noise, and publishing to the `transactions` Kafka topic.

## Local Dev

```bash
# From project root
make venv-producer

# Run producer (requires Kafka running — Phase 8)
streaming/producer/.venv/Scripts/python streaming/producer/producer.py
```

## Configuration (via .env)
- `PRODUCER_TRANSACTIONS_PER_SECOND` — throughput (default: 10)
- `PRODUCER_FRAUD_INJECTION_RATE` — fraction of synthetic fraud events (default: 0.01)
- `PRODUCER_DATA_PATH` — path to creditcard.csv
