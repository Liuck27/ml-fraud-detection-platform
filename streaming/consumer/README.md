# Go Kafka Consumer

Reads transactions from the `transactions` topic, calls the FastAPI `/predict`
endpoint, and publishes results to the `predictions` topic.
Exposes a health check at `:8081/health`.

## Local Dev

```bash
# Requires Go 1.21+
make go-build   # builds fraud-consumer binary
make go-test
make go-vet
```

## Configuration (via environment / .env)
- `KAFKA_BOOTSTRAP_SERVERS`
- `KAFKA_TOPIC_TRANSACTIONS` / `KAFKA_TOPIC_PREDICTIONS`
- `KAFKA_CONSUMER_GROUP_ID`
- `GO_CONSUMER_SERVING_URL`
- `GO_CONSUMER_HEALTH_PORT`
