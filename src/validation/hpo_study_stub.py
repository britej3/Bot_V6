"""
HPO Study Stub
==============

Lightweight wrapper to demonstrate how an Optuna study could be integrated
around the walk-forward backtester without adding heavy dependencies.

For now, it runs the backtester once and returns the aggregate metrics.
Extend later to iterate over param grids or wire Optuna.
"""

from typing import Dict
import numpy as np

from src.validation.walk_forward_backtester import WalkForwardBacktester


def run_hpo_stub(n_samples: int = 1000,
                 n_features: int = 20,
                 n_splits: int = 3,
                 purge: int = 10) -> Dict:
    rng = np.random.default_rng(7)
    X = rng.random((n_samples, n_features))
    y = (rng.random(n_samples) > 0.5).astype(int)

    wf = WalkForwardBacktester()
    report = wf.run(X, y, n_splits=n_splits, purge=purge)
    return {
        "best_params": {},
        "report": report,
    }


if __name__ == "__main__":
    print(run_hpo_stub()["report"].get("aggregate", {}))

