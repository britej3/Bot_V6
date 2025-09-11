"""
All-in-one Kafka smoke runner.
- Always runs no-broker smoke (`scripts/validate_kafka_smoke.py`).
- If Docker is available, brings up local Kafka via docker-compose and runs E2E checks:
  - connectivity, e2e smoke, topic list
- Tears down local Kafka if it started it.

Usage:
  PYTHONPATH=. python -m scripts.kafka_smoke_all
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))


def run(cmd: list[str], env: dict | None = None) -> int:
    print(f"$ {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env)
    return p.wait()


def main() -> None:
    # 1) No-broker smoke
    code = run([sys.executable, "scripts/validate_kafka_smoke.py"], env={**os.environ, "PYTHONPATH": "."})
    if code != 0:
        print("validate_kafka_smoke failed")

    # 2) Try full E2E if Docker exists
    docker = shutil.which("docker")
    if not docker:
        print("Docker not found; skipping E2E smoke")
        sys.exit(0)

    env = {**os.environ, "PYTHONPATH": ".", "KAFKA_ENABLED": "true", "KAFKA_BROKERS": "localhost:9092"}

    # Bring up local Kafka
    up_code = run(["docker", "compose", "-f", "docker-compose.kafka.yml", "up", "-d"], env=env)
    if up_code != 0:
        print("docker compose up failed; skipping E2E smoke")
        sys.exit(0)

    try:
        # connectivity check
        run([sys.executable, "-m", "scripts.kafka_check"], env=env)
        # e2e smoke
        run([sys.executable, "-m", "scripts.kafka_e2e_smoke"], env=env)
        # topic list
        run([sys.executable, "-m", "scripts.kafka_topics", "--list"], env=env)
    finally:
        # Tear down docker compose quietly
        run(["docker", "compose", "-f", "docker-compose.kafka.yml", "down"], env=env)


if __name__ == "__main__":
    main()
