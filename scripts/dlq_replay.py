"""
DLQ Replay: Re-publish messages from DLQ to their original topic.

Usage:
  PYTHONPATH=. KAFKA_ENABLED=true KAFKA_DLQ_ENABLED=true KAFKA_BROKERS=localhost:9092 \
    python -m scripts.dlq_replay --max 100 --dry-run false

Flags:
  --max N         Max messages to process (0 = unlimited)
  --timeout S     Poll timeout (default 1.0)
  --dry-run BOOL  If true, only prints what would be re-published (default true)
  --to-topic T    Override replay target topic instead of headers
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Optional

try:
    from confluent_kafka import Consumer, Producer
except Exception:
    Consumer = None  # type: ignore
    Producer = None  # type: ignore

from src.config.kafka_config import kafka_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dlq_replay")


def parse_bool(s: str) -> bool:
    return s.lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max", type=int, default=10)
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--dry-run", type=parse_bool, default=True)
    p.add_argument("--to-topic", type=str, default=None)
    return p.parse_args()


def main() -> None:
    if Consumer is None or Producer is None:
        logger.error("confluent-kafka not installed")
        sys.exit(2)
    if not (kafka_config.enabled and kafka_config.dlq_enabled):
        logger.error("Enable KAFKA_ENABLED=true and KAFKA_DLQ_ENABLED=true for replay")
        sys.exit(2)

    args = parse_args()

    consumer_conf: dict[str, Any] = {
        "bootstrap.servers": kafka_config.brokers,
        "group.id": "bot-v6-dlq-replay",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
    producer_conf: dict[str, Any] = {
        "bootstrap.servers": kafka_config.brokers,
        "client.id": kafka_config.client_id + "-replay",
    }

    c = Consumer(consumer_conf)
    p = Producer(producer_conf)
    c.subscribe([kafka_config.dlq_topic])

    processed = 0
    try:
        while True:
            if args.max and processed >= args.max:
                break
            msg = c.poll(args.timeout)
            if msg is None:
                continue
            if msg.error():
                logger.error("Consumer error: %s", msg.error())
                continue
            headers = {k: (v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v) for k, v in (msg.headers() or [])}
            source = headers.get("source_topic") or "unknown"
            target = args.to_topic or source
            key = msg.key()
            val = msg.value()
            if args.dry_run:
                print(f"DRY-RUN replay to {target} key={key} headers={headers}")
            else:
                p.produce(topic=target, key=key, value=val)
                print(f"Replayed message to {target}")
            processed += 1
        p.flush(5)
        c.commit(asynchronous=False)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            c.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
