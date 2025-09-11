"""
Minimal Kafka consumer service for demonstration.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

try:
    from confluent_kafka import Consumer
except Exception:  # pragma: no cover
    Consumer = None  # type: ignore

from src.config.kafka_config import kafka_config
from src.monitoring.metrics_kafka import (
    KAFKA_CONSUMER_COMMITS,
    KAFKA_CONSUMER_ERRORS,
)
from src.data_pipeline.kafka_io import DLPusher

logger = logging.getLogger(__name__)


class KafkaConsumerService:
    def __init__(self, group_id: str = "bot-v6-consumer", topics: List[str] | None = None) -> None:
        self._assigned = []
        self._since_commit = 0
        self._last_commit_ts = 0.0
        if Consumer is None:
            raise RuntimeError("confluent-kafka is not installed")
        if not kafka_config.enabled:
            raise RuntimeError("Kafka is disabled via config")

        conf: Dict[str, Any] = {
            "bootstrap.servers": kafka_config.brokers,
            "group.id": group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "enable.partition.eof": False,
            "session.timeout.ms": 10000,
            "max.poll.interval.ms": 300000,
        }
        if kafka_config.security_protocol:
            conf["security.protocol"] = kafka_config.security_protocol
        if kafka_config.sasl_mechanism:
            conf["sasl.mechanism"] = kafka_config.sasl_mechanism
        if kafka_config.sasl_username is not None:
            conf["sasl.username"] = kafka_config.sasl_username
        if kafka_config.sasl_password is not None:
            conf["sasl.password"] = kafka_config.sasl_password

        self.consumer = Consumer(conf)
        self.topics = topics or [kafka_config.ticker_topic, kafka_config.orderbook_topic]
        self._dlq = DLPusher(kafka_config)

    def start(self) -> None:
        logger.info("Subscribing to topics: %s", self.topics)
        self.consumer.subscribe(self.topics, on_assign=self._on_assign, on_revoke=self._on_revoke)

    def _on_assign(self, consumer, partitions):  # type: ignore
        logger.info("Partitions assigned: %s", partitions)
        self._assigned = partitions

    def _on_revoke(self, consumer, partitions):  # type: ignore
        logger.info("Partitions revoked: %s", partitions)
        self._assigned = []

    def poll_once(self, timeout: float = 1.0) -> None:
        msg = self.consumer.poll(timeout)
        if msg is None:
            return
        if msg.error():
            logger.error("Consumer error: %s", msg.error())
            try:
                KAFKA_CONSUMER_ERRORS.labels(topic=msg.topic() if msg.topic() else "unknown").inc()
            except Exception:
                pass
            return
        try:
            payload = json.loads(msg.value())
            # Placeholder: add real processing here
            logger.debug("Consumed message key=%s topic=%s %s", msg.key(), msg.topic(), payload)
            self._since_commit += 1
            now_ms = __import__("time").time() * 1000.0
            should_commit = False
            if kafka_config.commit_every_n > 0 and self._since_commit >= kafka_config.commit_every_n:
                should_commit = True
            if kafka_config.commit_interval_ms > 0 and (now_ms - self._last_commit_ts) >= kafka_config.commit_interval_ms:
                should_commit = True
            if kafka_config.commit_every_n == 0 and kafka_config.commit_interval_ms == 0:
                should_commit = True  # default behavior: commit per message
            if should_commit:
                self.consumer.commit(asynchronous=False)
                self._since_commit = 0
                self._last_commit_ts = now_ms
                try:
                    KAFKA_CONSUMER_COMMITS.labels(topic=msg.topic()).inc()
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Error processing message: %s", e)
            # DLQ path: publish original bytes with error metadata, then commit to avoid loops
            try:
                if kafka_config.dlq_enabled:
                    self._dlq.publish(
                        source_topic=msg.topic() or "unknown",
                        key=msg.key(),
                        payload_bytes=msg.value(),
                        error_type=type(e).__name__,
                        error_reason=str(e)[:256],
                    )
                    # Commit offset even on failure to avoid poison loops
                    self.consumer.commit(asynchronous=False)
            except Exception:
                # Do not crash on DLQ failures
                logger.exception("Failed to publish to DLQ")

    def close(self) -> None:
        try:
            self.consumer.close()
        except Exception:
            pass

    # Optional backpressure helpers (no overhead unless used)
    def pause(self) -> None:
        try:
            if kafka_config.pause_resume_enabled and self._assigned:
                self.consumer.pause(self._assigned)
                logger.info("Paused partitions: %s", self._assigned)
        except Exception:
            logger.exception("Pause failed")

    def resume(self) -> None:
        try:
            if kafka_config.pause_resume_enabled and self._assigned:
                self.consumer.resume(self._assigned)
                logger.info("Resumed partitions: %s", self._assigned)
        except Exception:
            logger.exception("Resume failed")

