"""
Kafka I/O helpers for publishing normalized market data.
"""
from __future__ import annotations

import json
import logging
import time
import atexit
from typing import Any, Dict, Optional, Callable

try:
    from confluent_kafka import Producer
except Exception:  # pragma: no cover
    Producer = None  # type: ignore

from src.config.kafka_config import kafka_config, KafkaConfig
from src.monitoring.metrics_kafka import (
    KAFKA_MESSAGES_PUBLISHED,
    KAFKA_DELIVERY_ERRORS,
    KAFKA_PRODUCER_QUEUE_DROPS,
    KAFKA_DLQ_PUBLISHED,
)

logger = logging.getLogger(__name__)


def _build_producer_config(cfg: KafkaConfig) -> Dict[str, Any]:
    conf: Dict[str, Any] = {
        "bootstrap.servers": cfg.brokers,
        "client.id": cfg.client_id,
        "enable.idempotence": True,
        "acks": "all",
        "compression.type": cfg.compression_type,
        "linger.ms": cfg.linger_ms,
        "batch.size": cfg.batch_size,
        "max.in.flight.requests.per.connection": 5,
        "retries": cfg.retries,
        "delivery.timeout.ms": 120000,
        "socket.keepalive.enable": True,
    }
    if cfg.security_protocol:
        conf["security.protocol"] = cfg.security_protocol
    if cfg.sasl_mechanism:
        conf["sasl.mechanism"] = cfg.sasl_mechanism
    if cfg.sasl_username is not None:
        conf["sasl.username"] = cfg.sasl_username
    if cfg.sasl_password is not None:
        conf["sasl.password"] = cfg.sasl_password
    return conf


class KafkaPublisher:
    """Lightweight Kafka publisher for normalized ticker/orderbook messages."""

    def __init__(self, cfg: Optional[KafkaConfig] = None) -> None:
        self.cfg = cfg or kafka_config
        if not self.cfg.enabled:
            logger.info("KafkaPublisher initialized but disabled via config")
        if Producer is None and self.cfg.enabled:
            raise RuntimeError("confluent-kafka is not installed but Kafka is enabled")
        self._producer = None

    @property
    def producer(self):
        if self._producer is None and self.cfg.enabled:
            conf = _build_producer_config(self.cfg)
            logger.info("Creating Kafka producer with brokers=%s", self.cfg.brokers)
            self._producer = Producer(conf)
        return self._producer

    def _serialize(self, payload: Dict[str, Any]) -> bytes:
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")

    def _on_delivery(self, err, msg):  # type: ignore
        if err is not None:
            logger.error("Kafka delivery failed for %s [%s]: %s", msg.topic(), msg.partition(), err)
            try:
                KAFKA_DELIVERY_ERRORS.labels(topic=msg.topic()).inc()
            except Exception:
                pass
        else:
            # Extract lightweight labels from headers to avoid JSON parsing
            mtype = "unknown"
            exch = "unknown"
            try:
                hdrs = dict((k, v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else (v or "")) for k, v in (msg.headers() or []))
                mtype = hdrs.get("type", mtype) or mtype
                exch = hdrs.get("exchange", exch) or exch
            except Exception:
                pass
            try:
                KAFKA_MESSAGES_PUBLISHED.labels(topic=msg.topic(), type=mtype, exchange=exch).inc()
            except Exception:
                pass
            # Publish latency from opaque start time if provided
            try:
                start_ts = msg.opaque()
                if isinstance(start_ts, (int, float)):
                    from src.monitoring.metrics_kafka import KAFKA_PUBLISH_LATENCY_SECONDS
                    KAFKA_PUBLISH_LATENCY_SECONDS.labels(topic=msg.topic(), type=mtype, exchange=exch).observe(max(0.0, time.time() - float(start_ts)))
            except Exception:
                pass
            logger.debug("Kafka delivered to %s [%s] @ %s", msg.topic(), msg.partition(), msg.offset())

    def publish(self, exchange: str, data: Dict[str, Any]) -> None:
        if not self.cfg.enabled:
            return
        p = self.producer
        if p is None:
            return
        msg_type = data.get("type")
        symbol = data.get("symbol") or data.get("s") or "unknown"
        key = f"{exchange}|{symbol}".encode("utf-8")
        topic = self.cfg.ticker_topic if msg_type == "ticker" else self.cfg.orderbook_topic
        try:
            payload = {"exchange": exchange, **data}
            headers = [("type", (msg_type or "unknown").encode("utf-8")), ("exchange", str(exchange).encode("utf-8"))]
            p.produce(topic=topic, key=key, value=self._serialize(payload), headers=headers, on_delivery=self._on_delivery, opaque=time.time())
        except BufferError:
            logger.warning("Producer queue is full; dropping message for %s %s", exchange, symbol)
            try:
                KAFKA_PRODUCER_QUEUE_DROPS.labels(topic=topic).inc()
            except Exception:
                pass
        except Exception as e:
            logger.exception("Error producing Kafka message: %s", e)

    def flush(self, timeout: float = 2.0) -> None:
        if self._producer is not None:
            self._producer.flush(timeout)


class DLPusher:
    """Publish failed events to a DLQ topic with error metadata (optional)."""

    def __init__(self, cfg: Optional[KafkaConfig] = None) -> None:
        self.cfg = cfg or kafka_config
        if not (self.cfg.enabled and self.cfg.dlq_enabled):
            return
        if Producer is None:
            raise RuntimeError("confluent-kafka is not installed but DLQ is enabled")
        self._producer = Producer(_build_producer_config(self.cfg))

    def publish(self, source_topic: str, key: Optional[bytes], payload_bytes: Optional[bytes], error_type: str, error_reason: str) -> None:
        if not (self.cfg.enabled and self.cfg.dlq_enabled):
            return
        try:
            headers = [
                ("source_topic", source_topic.encode("utf-8")),
                ("error_type", error_type.encode("utf-8")),
            ]
            self._producer.produce(
                topic=self.cfg.dlq_topic,
                key=key,
                value=payload_bytes,
                headers=headers,
            )
            try:
                KAFKA_DLQ_PUBLISHED.labels(dlq_topic=self.cfg.dlq_topic, source_topic=source_topic, error_type=error_type).inc()
            except Exception:
                pass
        except Exception:
            logger.exception("Failed to publish to DLQ")

    def flush(self, timeout: float = 2.0) -> None:
        if hasattr(self, "_producer") and self._producer is not None:
            self._producer.flush(timeout)


def make_kafka_callback(publisher: Optional[KafkaPublisher] = None) -> Callable[[str, Dict[str, Any]], None]:
    pub = publisher or KafkaPublisher()

    # Ensure producer is flushed on process exit
    try:
        atexit.register(lambda: pub.flush(2.0))
    except Exception:
        pass

    def callback(exchange: str, payload: Dict[str, Any]) -> None:
        try:
            pub.publish(exchange, payload)
        except Exception:
            logger.exception("Kafka callback encountered an error")
    return callback
