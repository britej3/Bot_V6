"""
Run a WebSocket feed and publish normalized messages to Kafka (if enabled).
Usage:
    KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 python -m scripts.run_kafka_producer
"""
from __future__ import annotations

import asyncio
import logging
import json
import os
import signal
from typing import List

from src.data_pipeline.websocket_feed import WebSocketConfig, WebSocketDataFeed
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
logger = logging.getLogger("run_kafka_producer")


async def main() -> None:
    # Start metrics if enabled
    if kafka_config.metrics_enabled:
        logger.info("Starting Prometheus metrics server on port %s", kafka_config.metrics_port)
        start_metrics_server(kafka_config.metrics_port)

    symbols: List[str] = ["BTCUSDT"]
    configs = [
        WebSocketConfig(
            exchange_name="binance",
            ws_url="wss://stream.binance.com:9443/ws",
            symbols=symbols,
            channels=["ticker", "orderbook"],
        )
    ]

    feed = WebSocketDataFeed(configs)
    # Kafka callback is auto-wired in WebSocketDataFeed when KAFKA_ENABLED=true
    if kafka_config.enabled:
        logger.info("Kafka enabled; publishing will be active via auto-wired callback")
    else:
        logger.info("Kafka disabled; feed will run without publishing")

    # Graceful shutdown signals
    stop_event = asyncio.Event()
    def _handle_sig(*_):
        logger.info("Received shutdown signal; stopping feed...")
        stop_event.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _handle_sig())

    # Run feed until stopped
    task = asyncio.create_task(feed.start())
    await stop_event.wait()
    await feed.stop()
    await asyncio.sleep(0.1)
    try:
        task.cancel()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
