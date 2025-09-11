# Kafka Integration (Incremental)

This project includes optional Kafka integration for streaming normalized market data from exchange WebSocket feeds.

## Overview
- Producer: Publishes `ticker` and `orderbook` messages to Kafka using `confluent-kafka`.
- Consumer: Minimal example demonstrates at-least-once consumption with manual commits.
- Toggle: Enable via `KAFKA_ENABLED=true`. When enabled, the Kafka publish callback is auto-wired inside `WebSocketDataFeed`.

## Install
- Ensure Python deps are installed (adds `confluent-kafka`):
  - `pip install -r requirements.txt`
  - Note: `confluent-kafka` requires `librdkafka` on some platforms. See https://github.com/confluentinc/librdkafka for details.

## Configuration
Environment variables (defaults in `src/config/kafka_config.py`):
- `KAFKA_ENABLED` (default `false`)
- `KAFKA_BROKERS` (default `localhost:9092`)
- `KAFKA_CLIENT_ID` (default `bot-v6`)
- `KAFKA_SECURITY_PROTOCOL`, `KAFKA_SASL_MECHANISM`, `KAFKA_SASL_USERNAME`, `KAFKA_SASL_PASSWORD` (optional for SASL_SSL)
- `KAFKA_TICKER_TOPIC` (default `ticks.v1.raw`)
- `KAFKA_ORDERBOOK_TOPIC` (default `orderbook.v1.raw`)
- `KAFKA_COMPRESSION` (default `zstd`), `KAFKA_LINGER_MS` (default `10`), `KAFKA_BATCH_SIZE` (default `131072`)

## Local Kafka via Docker Compose
- Start Kafka locally (single broker, KRaft mode):
  - `docker compose -f docker-compose.kafka.yml up -d`
- Verify (optional): list containers and ensure `kafka` is healthy.
- Stop:
  - `docker compose -f docker-compose.kafka.yml down`

## Run Producer
- Example (Binance BTCUSDT):
  - `KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 python -m scripts.run_kafka_producer`

## Run Consumer
- Example:
  - `KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 python -m scripts.run_kafka_consumer`

## Topics & Keys
- Topics: `ticks.v1.raw` for ticker, `orderbook.v1.raw` for order book.
- Key: `${exchange}|${symbol}` to preserve per-symbol ordering.

## Startup Check
- Quick connectivity: `PYTHONPATH=. KAFKA_ENABLED=true python -m scripts.kafka_check`

## Topic Management
- List topics: `PYTHONPATH=. KAFKA_ENABLED=true python -m scripts.kafka_topics --list`
- Create topic: `PYTHONPATH=. KAFKA_ENABLED=true python -m scripts.kafka_topics --create ticks.v1.raw --partitions 6 --retention-ms 86400000`

## Metrics (Prometheus)
- Enable metrics HTTP exporter:
  - `KAFKA_METRICS_ENABLED=true KAFKA_METRICS_PORT=8000 ...`
- Producer/consumer scripts start the exporter if enabled.
- Exposed counters:
  - `kafka_messages_published_total{topic,type,exchange}`
  - `kafka_delivery_errors_total{topic}`
  - `kafka_producer_queue_drops_total{topic}`
  - `kafka_consumer_commits_total{topic}`
  - `kafka_consumer_errors_total{topic}`
- Validate exporter:
  - `curl -s localhost:8000/metrics | grep kafka_`

## Validation
- Run all smokes (auto-detects Docker):
  - `PYTHONPATH=. python -m scripts.kafka_smoke_all`
- E2E smoke: `KAFKA_ENABLED=true PYTHONPATH=. python -m scripts.kafka_e2e_smoke`
- Smoke test without Kafka (verifies wiring):
  - `PYTHONPATH=. python scripts/validate_kafka_smoke.py`
  - Expected output includes:
    - `Publisher-disabled-ok`
    - `Callbacks: 0` (when `confluent-kafka` is not installed)
    - `Metrics-server-ok`
- End-to-end with local Kafka:
  1. `docker compose -f docker-compose.kafka.yml up -d`
  2. `KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 PYTHONPATH=. python -m scripts.run_kafka_producer`
  3. In another shell: `KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 PYTHONPATH=. python -m scripts.run_kafka_consumer`
  4. Check metrics: `curl -s localhost:8000/metrics | grep kafka_messages_published_total`
  5. Inspect DLQ (if enabled):
     - `KAFKA_ENABLED=true KAFKA_DLQ_ENABLED=true PYTHONPATH=. python -m scripts.dlq_inspector --max 5`

## Dead-Letter Queue (DLQ)
- Enable: `KAFKA_DLQ_ENABLED=true`
- Topic: `KAFKA_DLQ_TOPIC` (default `events.v1.dlq`)
- Behavior:
  - Producer: standard delivery retries and error logging; DLQ is focused on consumer processing errors.
  - Consumer: on processing exceptions, publishes original message bytes to DLQ with headers `source_topic`, `error_type` and commits offset to avoid poison-message loops.
- Metrics:
  - `kafka_dlq_published_total{dlq_topic,source_topic,error_type}`

## Notes & Next Steps
- Start with JSON payloads. Consider Avro/Protobuf + Schema Registry later.
- Consider replay tooling for DLQ and a small inspector to triage.
- Use `acks=all`, idempotent producer, `linger.ms` batching for throughput.
