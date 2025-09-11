"""
Lightweight validation of Kafka integration without requiring a running Kafka cluster.
- Verifies KafkaPublisher works when disabled
- Ensures WebSocketDataFeed initializes with KAFKA_ENABLED=true and gracefully handles missing producer
- Starts Prometheus metrics server on an ephemeral port
"""
from __future__ import annotations

import os


def main() -> None:
    os.environ["KAFKA_ENABLED"] = "false"
    from src.data_pipeline.kafka_io import KafkaPublisher

    pub = KafkaPublisher()
    pub.publish("binance", {"type": "ticker", "symbol": "BTCUSDT", "price": 50000})
    print("Publisher-disabled-ok")

    os.environ["KAFKA_ENABLED"] = "true"
    from src.data_pipeline.websocket_feed import WebSocketConfig, WebSocketDataFeed

    cfg = WebSocketConfig(
        exchange_name="binance",
        ws_url="wss://stream.binance.com:9443/ws",
        symbols=["BTCUSDT"],
        channels=["ticker"],
    )
    feed = WebSocketDataFeed([cfg])
    print("Callbacks:", len(feed.data_callbacks))

    from src.monitoring.metrics_kafka import start_metrics_server

    start_metrics_server(8021)
    print("Metrics-server-ok")


if __name__ == "__main__":
    main()
