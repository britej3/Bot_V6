"""
Kafka connectivity check: verify broker metadata fetch succeeds quickly.
Usage:
  PYTHONPATH=. KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 python -m scripts.kafka_check
"""
from __future__ import annotations

import sys
import time
import logging

try:
    from confluent_kafka import Producer
except Exception:
    Producer = None  # type: ignore

from src.config.kafka_config import kafka_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kafka_check")


def main() -> None:
    if Producer is None:
        logger.error("confluent-kafka not installed")
        sys.exit(2)
    if not kafka_config.enabled:
        logger.error("KAFKA_ENABLED must be true for connectivity check")
        sys.exit(2)

    conf = {
        "bootstrap.servers": kafka_config.brokers,
        "client.id": kafka_config.client_id + "-check",
        "socket.timeout.ms": 2000,
        "message.timeout.ms": 3000,
    }
    if kafka_config.security_protocol:
        conf["security.protocol"] = kafka_config.security_protocol
    if kafka_config.sasl_mechanism:
        conf["sasl.mechanism"] = kafka_config.sasl_mechanism
    if kafka_config.sasl_username is not None:
        conf["sasl.username"] = kafka_config.sasl_username
    if kafka_config.sasl_password is not None:
        conf["sasl.password"] = kafka_config.sasl_password

    p = Producer(conf)
    t0 = time.time()
    md = p.list_topics(timeout=3.0)
    dt = time.time() - t0
    logger.info("Fetched metadata from %s broker(s) in %.3fs", len(md.brokers), dt)


if __name__ == "__main__":
    main()
