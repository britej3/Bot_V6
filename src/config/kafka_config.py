"""
Kafka configuration for real-time streaming integration.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
import os


class KafkaConfig(BaseModel):
    """Kafka client and topic configuration."""

    enabled: bool = Field(default=False, description="Enable Kafka publishing")
    brokers: str = Field(default="localhost:9092", description="Bootstrap servers")
    client_id: str = Field(default="bot-v6", description="Kafka client.id")

    # Security (for cloud providers)
    security_protocol: Optional[str] = Field(default=None, description="PLAINTEXT or SASL_SSL")
    sasl_mechanism: Optional[str] = Field(default=None, description="PLAIN, SCRAM-SHA-256, SCRAM-SHA-512")
    sasl_username: Optional[str] = Field(default=None, description="SASL username")
    sasl_password: Optional[str] = Field(default=None, description="SASL password")

    # Topics
    ticker_topic: str = Field(default="ticks.v1.raw", description="Topic for ticker events")
    orderbook_topic: str = Field(default="orderbook.v1.raw", description="Topic for orderbook events")

    # Producer tuning
    compression_type: str = Field(default="zstd", description="Compression: zstd|lz4|snappy|gzip|none")
    linger_ms: int = Field(default=10, description="Batch linger in ms")
    batch_size: int = Field(default=131072, description="Batch size bytes (128 KiB)")
    retries: int = Field(default=2147483647, description="Max retries for delivery")

    # Metrics
    metrics_enabled: bool = Field(default=False, description="Expose Prometheus metrics HTTP endpoint")
    metrics_port: int = Field(default=8000, description="Prometheus metrics port")

    # DLQ (Dead Letter Queue)
    dlq_enabled: bool = Field(default=False, description="Enable publishing failed messages to DLQ")
    dlq_topic: str = Field(default="events.v1.dlq", description="DLQ topic for failed events")

    # Consumer commit behavior (opt-in)
    commit_every_n: int = Field(default=0, description="Commit after N messages; 0 = per message")
    commit_interval_ms: int = Field(default=0, description="Commit at least every N ms; 0 = immediate")

    # Logging (opt-in)
    json_logs: bool = Field(default=False, description="Enable JSON logging in scripts")

    # Backpressure helpers exposed (no runtime effect unless used)
    pause_resume_enabled: bool = Field(default=False, description="Expose pause/resume helpers")

    @classmethod
    def from_env(cls) -> "KafkaConfig":
        return cls(
            enabled=os.getenv("KAFKA_ENABLED", "false").lower() == "true",
            brokers=os.getenv("KAFKA_BROKERS", "localhost:9092"),
            client_id=os.getenv("KAFKA_CLIENT_ID", "bot-v6"),
            security_protocol=os.getenv("KAFKA_SECURITY_PROTOCOL"),
            sasl_mechanism=os.getenv("KAFKA_SASL_MECHANISM"),
            sasl_username=os.getenv("KAFKA_SASL_USERNAME"),
            sasl_password=os.getenv("KAFKA_SASL_PASSWORD"),
            ticker_topic=os.getenv("KAFKA_TICKER_TOPIC", "ticks.v1.raw"),
            orderbook_topic=os.getenv("KAFKA_ORDERBOOK_TOPIC", "orderbook.v1.raw"),
            compression_type=os.getenv("KAFKA_COMPRESSION", "zstd"),
            linger_ms=int(os.getenv("KAFKA_LINGER_MS", "10")),
            batch_size=int(os.getenv("KAFKA_BATCH_SIZE", "131072")),
            retries=int(os.getenv("KAFKA_RETRIES", "2147483647")),
            metrics_enabled=os.getenv("KAFKA_METRICS_ENABLED", "false").lower() == "true",
            metrics_port=int(os.getenv("KAFKA_METRICS_PORT", "8000")),
            dlq_enabled=os.getenv("KAFKA_DLQ_ENABLED", "false").lower() == "true",
            dlq_topic=os.getenv("KAFKA_DLQ_TOPIC", "events.v1.dlq"),
            commit_every_n=int(os.getenv("KAFKA_COMMIT_EVERY_N", "0")),
            commit_interval_ms=int(os.getenv("KAFKA_COMMIT_INTERVAL_MS", "0")),
            json_logs=os.getenv("KAFKA_JSON_LOGS", "false").lower() == "true",
            pause_resume_enabled=os.getenv("KAFKA_PAUSE_RESUME_ENABLED", "false").lower() == "true",
        )


# Default instance from environment
kafka_config = KafkaConfig.from_env()
