"""
Kafka topic manager: list and create topics with retention/partitions.

Examples:
  PYTHONPATH=. KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 \
    python -m scripts.kafka_topics --list

  PYTHONPATH=. KAFKA_ENABLED=true KAFKA_BROKERS=localhost:9092 \
    python -m scripts.kafka_topics --create ticks.v1.raw --partitions 6 --retention-ms 86400000
"""
from __future__ import annotations

import argparse
import logging
import sys

try:
    from confluent_kafka.admin import AdminClient, NewTopic
except Exception:
    AdminClient = None  # type: ignore
    NewTopic = None  # type: ignore

from src.config.kafka_config import kafka_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kafka_topics")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="List existing topics")
    p.add_argument("--create", type=str, help="Topic name to create")
    p.add_argument("--partitions", type=int, default=3)
    p.add_argument("--replication", type=int, default=1)
    p.add_argument("--retention-ms", type=int, default=None)
    return p.parse_args()


def main() -> None:
    if AdminClient is None or NewTopic is None:
        logger.error("confluent-kafka (admin) not installed")
        sys.exit(2)
    if not kafka_config.enabled:
        logger.error("KAFKA_ENABLED must be true for topic management")
        sys.exit(2)

    client = AdminClient({"bootstrap.servers": kafka_config.brokers})

    args = parse_args()

    if args.list:
        md = client.list_topics(timeout=5)
        for t in sorted(md.topics.keys()):
            print(t)
        return

    if args.create:
        configs = {}
        if args.retention_ms is not None:
            configs["retention.ms"] = str(args.retention_ms)
        new_topics = [NewTopic(args.create, num_partitions=args.partitions, replication_factor=args.replication, config=configs or None)]
        fs = client.create_topics(new_topics)
        for topic, f in fs.items():
            try:
                f.result()
                logger.info("Created topic %s", topic)
            except Exception as e:
                logger.error("Failed to create topic %s: %s", topic, e)


if __name__ == "__main__":
    main()
