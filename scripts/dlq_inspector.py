"""
DLQ Inspector: Consume and print messages from the configured DLQ topic.

Usage examples:
  PYTHONPATH=. KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 KAFKA_DLQ_ENABLED=true \
    python -m scripts.dlq_inspector --max 20

Flags:
  --max N          Stop after N messages (default: 10). Use 0 for endless.
  --timeout SEC    Poll timeout seconds (default: 1.0)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

try:
    from confluent_kafka import Consumer
except Exception:  # pragma: no cover
    Consumer = None  # type: ignore

from src.config.kafka_config import kafka_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dlq_inspector")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=10)
    p.add_argument("--timeout", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    if Consumer is None:
        logger.error("confluent-kafka not installed")
        sys.exit(2)
    if not (kafka_config.enabled and kafka_config.dlq_enabled):
        logger.error("Enable KAFKA_ENABLED=true and KAFKA_DLQ_ENABLED=true to inspect DLQ")
        sys.exit(2)

    args = parse_args()

    conf: dict[str, Any] = {
        "bootstrap.servers": kafka_config.brokers,
        "group.id": "bot-v6-dlq-inspector",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
        "enable.partition.eof": False,
        "session.timeout.ms": 10000,
    }

    c = Consumer(conf)
    c.subscribe([kafka_config.dlq_topic])
    seen = 0
    logger.info("Inspecting DLQ topic=%s (max=%s, timeout=%.1fs)", kafka_config.dlq_topic, args.max, args.timeout)
    try:
        while True:
            msg = c.poll(args.timeout)
            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue
            # Print a concise dump
            headers = {k: (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v) for k, v in (msg.headers() or [])}
            key = (msg.key() or b"").decode("utf-8", errors="ignore")
            body = msg.value() or b""
            preview = body[:256].decode("utf-8", errors="ignore")
            print("=== DLQ MESSAGE ===")
            print(f"offset={msg.offset()} partition={msg.partition()} topic={msg.topic()}")
            print(f"key={key}")
            print(f"headers={headers}")
            # Try to pretty-print JSON if possible
            try:
                obj = json.loads(preview)
                print("payload_json=", json.dumps(obj, indent=2)[:512])
            except Exception:
                print("payload_preview=", preview)
            print()
            seen += 1
            if args.max and seen >= args.max:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            c.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
