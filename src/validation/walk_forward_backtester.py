"""
Walk-Forward Backtester (Purged) for XGBoost Ensemble
=====================================================

Minimal, dependency-light walk-forward backtester wired to
`src/learning/xgboost_ensemble.XGBoostEnsemble`.

Features:
- Purged walk-forward splits to reduce leakage
- Uses existing `train_ensemble` to evaluate per split
- Aggregates standard metrics across splits

Usage example:
    from src.validation.walk_forward_backtester import WalkForwardBacktester
    import numpy as np

    # X: (n_samples, n_features), y: (n_samples,) binary labels
    X = np.random.rand(2000, 20)
    y = (np.random.rand(2000) > 0.5).astype(int)

    backtester = WalkForwardBacktester()
    report = backtester.run(X, y, n_splits=5, purge=20)
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.config.trading_config import AdvancedTradingConfig, get_trading_config
from src.memory.memory_service import MemoryService
from src.database.manager import DatabaseManager
from src.database.redis_manager import RedisManager
import asyncio
from src.learning.xgboost_ensemble import XGBoostEnsemble


@dataclass
class SplitResult:
    split_index: int
    metrics: Dict[str, float]
    train_size: int
    val_size: int


class WalkForwardBacktester:
    """Purged walk-forward backtester using XGBoostEnsemble."""

    def __init__(self, config: AdvancedTradingConfig | None = None,
                 memory: MemoryService | None = None):
        self.config = config or get_trading_config()
        self.memory = memory

        # Lazy-init default MemoryService if not provided
        if self.memory is None:
            self._db = DatabaseManager()  # default SQLite
            self._redis = RedisManager()
            # Note: caller must connect managers when used in app; for module usage
            # we operate without strict connection to avoid env issues.
            self.memory = MemoryService(self._redis, self._db, default_ttl_seconds=300, max_list_length=100)

    def _generate_splits(self, n_samples: int, n_splits: int, purge: int) -> List[Tuple[slice, slice]]:
        """Generate purged walk-forward train/val index slices.

        Args:
            n_samples: total number of samples
            n_splits: number of splits (>=2)
            purge: number of samples to skip between train and validation

        Returns:
            List of (train_slice, val_slice)
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        # Evenly sized validation windows
        val_len = max(1, n_samples // (n_splits + 1))
        splits: List[Tuple[slice, slice]] = []

        start = 0
        for i in range(n_splits):
            train_end = start + val_len * i
            val_start = min(train_end + purge, n_samples - val_len)
            val_end = min(val_start + val_len, n_samples)

            if val_start <= 0 or val_end <= val_start or train_end <= 0:
                continue

            train_slice = slice(0, train_end)
            val_slice = slice(val_start, val_end)
            splits.append((train_slice, val_slice))

        if not splits:
            # Fallback single split: 70/30 with purge
            train_end = int(n_samples * 0.7)
            val_start = min(train_end + purge, n_samples - 1)
            val_end = n_samples
            splits.append((slice(0, train_end), slice(val_start, val_end)))

        return splits

    def run(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, purge: int = 10) -> Dict[str, object]:
        """Run walk-forward backtest.

        Args:
            X: features array, shape (n_samples, n_features)
            y: binary labels, shape (n_samples,)
            n_splits: number of walk-forward splits
            purge: gap between train and validation windows

        Returns:
            Report dict with per-split metrics and aggregate averages.
        """
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("Invalid shapes: X must be 2D and y 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_samples = X.shape[0]
        splits = self._generate_splits(n_samples, n_splits, purge)

        split_results: List[SplitResult] = []
        similar_for_last: List[Dict[str, object]] = []

        for idx, (tr_slice, va_slice) in enumerate(splits):
            X_train, y_train = X[tr_slice], y[tr_slice]
            X_val, y_val = X[va_slice], y[va_slice]

            # Initialize a fresh ensemble per split
            ensemble = XGBoostEnsemble(self.config)
            metrics = ensemble.train_ensemble(X_train, y_train, X_val, y_val)

            split_results.append(
                SplitResult(
                    split_index=idx,
                    metrics=metrics,
                    train_size=X_train.shape[0],
                    val_size=X_val.shape[0],
                )
            )

            # Persist representative embeddings for regime retrieval (best-effort)
            try:
                tr_sig = X_train.mean(axis=0).tolist()
                va_sig = X_val.mean(axis=0).tolist()
                meta = {
                    "split": idx,
                    "train_size": int(X_train.shape[0]),
                    "val_size": int(X_val.shape[0]),
                    "metrics": metrics,
                }
                similar_for_last = self._persist_and_query_signatures(tr_sig, va_sig, idx, meta)
            except Exception:
                # Non-blocking persistence
                similar_for_last = []

        # Aggregate metrics
        agg: Dict[str, float] = {}
        if split_results:
            keys = split_results[0].metrics.keys()
            for k in keys:
                vals = [sr.metrics.get(k, np.nan) for sr in split_results]
                # Filter NaNs just in case
                vals = [v for v in vals if isinstance(v, (float, int))]
                if vals:
                    agg[k] = float(np.mean(vals))

        return {
            "n_samples": n_samples,
            "n_splits": len(split_results),
            "purge": purge,
            "splits": [
                {
                    "split": sr.split_index,
                    "train_size": sr.train_size,
                    "val_size": sr.val_size,
                    "metrics": sr.metrics,
                }
                for sr in split_results
            ],
            "aggregate": agg,
            # Optional: similarity lookup for last validation window signature
            # (best-effort; empty if not available)
            "similar_regimes": similar_for_last,
        }

    def _persist_and_query_signatures(self, tr_sig, va_sig, split_idx: int, meta: Dict[str, object]):
        """Best-effort sync wrapper to persist train/val signatures and query similar regimes.

        Returns a small list of similar items (id, score, metadata) for the validation signature.
        """
        async def _do():
            try:
                await self.memory.upsert_embedding("wfa_train", f"split-{split_idx}", tr_sig, meta)
                await self.memory.upsert_embedding("wfa_val", f"split-{split_idx}", va_sig, meta)
                neighbors = await self.memory.query_similar("wfa_val", va_sig, k=5)
                out = []
                for item, score in neighbors:
                    out.append({
                        "id": item.id,
                        "namespace": item.namespace,
                        "item_id": item.item_id,
                        "score": float(score),
                        "metadata": item.metadata,
                    })
                return out
            except Exception:
                return []

        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # In an active event loop; schedule persistence and return without blocking
                loop.create_task(self.memory.upsert_embedding("wfa_train", f"split-{split_idx}", tr_sig, meta))
                loop.create_task(self.memory.upsert_embedding("wfa_val", f"split-{split_idx}", va_sig, meta))
                return []
            else:
                return asyncio.run(_do())
        except Exception:
            return []



if __name__ == "__main__":
    print("Walk-Forward Backtester module. Import and use `WalkForwardBacktester.run(...)`.")
