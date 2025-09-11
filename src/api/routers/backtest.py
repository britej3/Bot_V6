from fastapi import APIRouter
from typing import Dict
import numpy as np

from src.validation.walk_forward_backtester import WalkForwardBacktester

backtest_router = APIRouter()


@backtest_router.post("/v1/backtest/walk-forward/synthetic")
async def run_synthetic_walk_forward(n_samples: int = 2000,
                                     n_features: int = 20,
                                     n_splits: int = 5,
                                     purge: int = 20) -> Dict:
    """Run a synthetic walk-forward backtest to validate pipeline health."""
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features))
    y = (rng.random(n_samples) > 0.5).astype(int)

    backtester = WalkForwardBacktester()
    report = backtester.run(X, y, n_splits=n_splits, purge=purge)
    return report

