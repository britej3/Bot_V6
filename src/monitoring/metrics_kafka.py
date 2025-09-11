"""
Prometheus metrics for Kafka publishing and consumption.
"""
from __future__ import annotations

from typing import Optional
from prometheus_client import Counter, Histogram, start_http_server

# Counters
KAFKA_MESSAGES_PUBLISHED = Counter(
    "kafka_messages_published_total",
    "Total number of Kafka messages successfully published",
    labelnames=("topic", "type", "exchange"),
)

KAFKA_DELIVERY_ERRORS = Counter(
    "kafka_delivery_errors_total",
    "Total number of Kafka delivery errors",
    labelnames=("topic",),
)

KAFKA_PRODUCER_QUEUE_DROPS = Counter(
    "kafka_producer_queue_drops_total",
    "Messages dropped due to full producer queue",
    labelnames=("topic",),
)

KAFKA_CONSUMER_COMMITS = Counter(
    "kafka_consumer_commits_total",
    "Total number of successful consumer commits",
    labelnames=("topic",),
)

# Histograms (low-cardinality, lightweight)
KAFKA_PUBLISH_LATENCY_SECONDS = Histogram(
    "kafka_publish_latency_seconds",
    "End-to-end publish latency (produce to delivery callback)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    labelnames=("topic", "type", "exchange"),
)

KAFKA_CONSUME_PROCESS_SECONDS = Histogram(
    "kafka_consume_process_seconds",
    "Consumer processing time per message",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
    labelnames=("topic",),
)

KAFKA_CONSUMER_ERRORS = Counter(
    "kafka_consumer_errors_total",
    "Total number of consumer errors",
    labelnames=("topic",),
)

KAFKA_DLQ_PUBLISHED = Counter(
    "kafka_dlq_published_total",
    "Total number of messages published to DLQ",
    labelnames=("dlq_topic", "source_topic", "error_type"),
)


def start_metrics_server(port: int) -> None:
    """Start Prometheus metrics HTTP server on given port (idempotent)."""
    # start_http_server is idempotent per process and safe to call once
    start_http_server(port)
