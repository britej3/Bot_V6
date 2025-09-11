"""
Consume messages from Kafka topics and log them.
Usage:
    KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 python -m scripts.run_kafka_consumer
"""
from __future__ import annotations

import logging
import time
import json
import os
import signal

from src.data_pipeline.kafka_consumer_service import KafkaConsumerService
from src.config.kafka_config import kafka_config
from src.monitoring.metrics_kafka import start_metrics_server


def _setup_logging():
    if os.getenv("KAFKA_JSON_LOGS", "false").lower() == "true":
        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps({
                    "level": record.levelname,
                    "name": record.name,
                    "message": record.getMessage(),
                })
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

_setup_logging()
logger = logging.getLogger("run_kafka_consumer")


def main() -> None:
    if kafka_config.metrics_enabled:
        logger.info("Starting Prometheus metrics server on port %s", kafka_config.metrics_port)
        start_metrics_server(kafka_config.metrics_port)

    svc = KafkaConsumerService()
    svc.start()

    stop = False
    def _handle_sig(*_):
        nonlocal stop
        logger.info("Received shutdown signal; stopping consumer...")
        stop = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: _handle_sig())

    try:
        while not stop:
            svc.poll_once(timeout=1.0)
            time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("Stopping consumer...")
    finally:
        svc.close()


if __name__ == "__main__":
    main()
