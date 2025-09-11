"""
High-Frequency Trading Engine Package
=====================================

This package provides ultra-low latency trading capabilities with real exchange integration.
"""

from .ultra_low_latency_engine import (
    UltraLowLatencyTradingEngine,
    ExecutionConfig,
    MarketData,
    Order,
    OrderSide,
    OrderType,
    ExecutionResult,
    create_ultra_low_latency_engine,
    get_ultra_low_latency_engine,
    initialize_ultra_low_latency_engine,
    shutdown_ultra_low_latency_engine
)
from .exchange_connector import (
    ExchangeConnector,
    ExchangeConfig,
    create_exchange_connector
)

__all__ = [
    'UltraLowLatencyTradingEngine',
    'ExecutionConfig',
    'ExchangeConfig',
    'MarketData',
    'Order',
    'OrderSide',
    'OrderType',
    'ExecutionResult',
    'ExchangeConnector',
    'create_ultra_low_latency_engine',
    'create_exchange_connector',
    'get_ultra_low_latency_engine',
    'initialize_ultra_low_latency_engine',
    'shutdown_ultra_low_latency_engine'
]