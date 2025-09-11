"""
Enhanced Performance Layer for CryptoScalp AI
Phase 1: Core Performance Revolution
"""

from .jax_ensemble import UltraFastTradingEnsemble, JAXPerformanceMonitor
from .polars_pipeline import FeaturePipeline, PolarsAccelerator

# Future modules (Phase 1 implemented)
# from .redis_cache import TradingCache, CacheManager
# from .duckdb_analytics import QuantAnalytics, DuckDBManager
# from .talib_indicators import OptimizedIndicators, IndicatorManager

__all__ = [
    'UltraFastTradingEnsemble',
    'JAXPerformanceMonitor',
    'FeaturePipeline',
    'PolarsAccelerator',
    # Future components:
    # 'TradingCache',
    # 'CacheManager',
    # 'QuantAnalytics',
    # 'DuckDBManager',
    # 'OptimizedIndicators',
    # 'IndicatorManager'
]
