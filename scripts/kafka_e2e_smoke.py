"""
Kafka E2E smoke test: produce and consume a single message on a test topic.
Usage:
  PYTHONPATH=. KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 python -m scripts.kafka_e2e_smoke
"""
from __future__ import annotations

import json
import os
import time
import uuid
import logging
import sys

try:
    from confluent_kafka import Producer, Consumer
except Exception:
    Producer = None  # type: ignore
    Consumer = None  # type: ignore

from src.config.kafka_config import kafka_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kafka_e2e_smoke")


def main() -> None:
    if Producer is None or Consumer is None:
        logger.error("confluent-kafka not installed")
        sys.exit(2)
    if not kafka_config.enabled:
        logger.error("KAFKA_ENABLED must be true for E2E smoke test")
        sys.exit(2)

    topic = os.getenv("KAFKA_SMOKE_TOPIC", "botv6.test.v1")
    payload = {"type": "ticker", "symbol": "TESTUSDT", "price": 1.23}
    key = f"smoke-{uuid.uuid4()}".encode("utf-8")

    p = Producer({"bootstrap.servers": kafka_config.brokers})
    c = Consumer({
        "bootstrap.servers": kafka_config.brokers,
        "group.id": f"bot-v6-smoke-{uuid.uuid4()}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    c.subscribe([topic])

    p.produce(topic=topic, key=key, value=json.dumps(payload).encode("utf-8"))
    p.flush(5)

    deadline = time.time() + 5
    while time.time() < deadline:
        msg = c.poll(0.5)
        if msg is None:
            continue
        if msg.error():
            logger.error("Consumer error: %s", msg.error())
            continue
        logger.info("Received message on %s key=%s", msg.topic(), msg.key())
        print("E2E-SMOKE-OK")
        c.close()
        return

    logger.error("E2E smoke message not received in time")
    c.close()
    sys.exit(1)


if __name__ == "__main__":
    main()
