"""
CryptoScalp AI Data Pipeline Module

This module handles multi-source data acquisition from cryptocurrency exchanges
including Binance, OKX, and Bybit with real-time WebSocket connections and
failover mechanisms.
"""

from .data_loader import MultiSourceDataLoader
from .websocket_feed import WebSocketDataFeed
from .data_validator import DataValidator

__all__ = [
    'MultiSourceDataLoader',
    'WebSocketDataFeed',
    'DataValidator'
]