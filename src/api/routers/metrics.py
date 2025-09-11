from fastapi import APIRouter
from typing import Dict
from datetime import datetime, timezone

metrics_router = APIRouter()

_METRICS = {
    "requests": 0,
    "backtests_run": 0,
}


@metrics_router.get("/v1/metrics")
async def get_metrics() -> Dict:
    """Basic system metrics placeholder (extend with real counters)."""
    _METRICS["requests"] += 1
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": _METRICS,
    }

