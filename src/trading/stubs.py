"""
Development Stubs for Optional Dependencies
===========================================

This module provides stub implementations for optional dependencies
to enable development and testing without requiring all packages to be installed.

Usage:
    Import this module early in your application to provide fallbacks
    for missing optional dependencies.
"""

import sys
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

# Stub for slowapi if not available
try:
    from slowapi import Limiter
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.responses import JSONResponse
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False

    # Create stub classes
    class Limiter:
        def __init__(self, **kwargs):
            pass
        def limit(self, limit_str):
            def decorator(func):
                return func
            return decorator

    class SlowAPIMiddleware:
        def __init__(self, app, **kwargs):
            self.app = app

    def get_remote_address(request):
        return "127.0.0.1"

    class RateLimitExceeded(Exception):
        pass

    class JSONResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

    # Make stubs available at module level
    sys.modules['slowapi'] = MagicMock()
    sys.modules['slowapi.middleware'] = MagicMock()
    sys.modules['slowapi.util'] = MagicMock()
    sys.modules['slowapi.errors'] = MagicMock()
    sys.modules['slowapi.responses'] = MagicMock()

# Stub for ccxt if not available
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

    class ccxt:
        """Stub CCXT module"""
        @staticmethod
        def binance(config=None):
            return MagicMock()

        @staticmethod
        def okx(config=None):
            return MagicMock()

        @staticmethod
        def bybit(config=None):
            return MagicMock()

        @staticmethod
        def coinbase(config=None):
            return MagicMock()

        @staticmethod
        def kraken(config=None):
            return MagicMock()

    # Create stub ccxt.pro module
    class ccxt_pro:
        """Stub CCXT Pro module"""
        pass

    sys.modules['ccxt'] = ccxt
    sys.modules['ccxt.pro'] = ccxt_pro

# Stub for nautilus_trader if not available
try:
    import nautilus_trader
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False

    # Create comprehensive nautilus stubs
    class nautilus_trader:
        class core:
            class nautilus_pyo3:
                class AccountBalance(MagicMock): pass
                class Order(MagicMock): pass
                class Position(MagicMock): pass
                class Trade(MagicMock): pass
                class Bar(MagicMock): pass
                class QuoteTick(MagicMock): pass
                class TradeTick(MagicMock): pass

        class model:
            class enums:
                class OrderType:
                    MARKET = "MARKET"
                    LIMIT = "LIMIT"
                    STOP_MARKET = "STOP_MARKET"
                    TRAILING_STOP = "TRAILING_STOP"
                    ICEBERG = "ICEBERG"

                class OrderSide:
                    BUY = "BUY"
                    SELL = "SELL"

                class TimeInForce:
                    GTC = "GTC"
                    IOC = "IOC"
                    FOK = "FOK"

                class OrderStatus:
                    SUBMITTED = "SUBMITTED"
                    FILLED = "FILLED"
                    CANCELLED = "CANCELLED"
                    REJECTED = "REJECTED"

                class PositionSide:
                    LONG = "LONG"
                    SHORT = "SHORT"

            class identifiers:
                class AccountId(MagicMock):
                    def __init__(self, value): self.value = value

                class Venue(MagicMock):
                    def __init__(self, value): self.value = value

                class InstrumentId(MagicMock):
                    def __init__(self, value): self.value = value

                class Symbol(MagicMock):
                    def __init__(self, value): self.value = value

                class ClientOrderId(MagicMock):
                    def __init__(self, value): self.value = value

                class PositionId(MagicMock):
                    def __init__(self, value): self.value = value

                class StrategyId(MagicMock):
                    def __init__(self, value): self.value = value

            class objects:
                class Price(MagicMock):
                    def __init__(self, value, precision=2):
                        self.value = value
                        self.precision = precision

                class Quantity(MagicMock):
                    def __init__(self, value, precision=8):
                        self.value = value
                        self.precision = precision

                class Money(MagicMock):
                    def __init__(self, value, currency):
                        self.value = value
                        self.currency = currency

                class Currency(MagicMock):
                    def __init__(self, code):
                        self.code = code

            class orders:
                class MarketOrder(MagicMock): pass
                class LimitOrder(MagicMock): pass
                class StopMarketOrder(MagicMock): pass
                class StopLimitOrder(MagicMock): pass
                class TrailingStopMarketOrder(MagicMock): pass
                class TrailingStopLimitOrder(MagicMock): pass

            class position:
                class Position(MagicMock): pass

            class events:
                class OrderEvent(MagicMock): pass
                class PositionEvent(MagicMock): pass

    sys.modules['nautilus_trader'] = nautilus_trader
    sys.modules['nautilus_trader.core'] = nautilus_trader.core
    sys.modules['nautilus_trader.core.nautilus_pyo3'] = nautilus_trader.core.nautilus_pyo3
    sys.modules['nautilus_trader.model'] = nautilus_trader.model
    sys.modules['nautilus_trader.model.enums'] = nautilus_trader.model.enums
    sys.modules['nautilus_trader.model.identifiers'] = nautilus_trader.model.identifiers
    sys.modules['nautilus_trader.model.objects'] = nautilus_trader.model.objects
    sys.modules['nautilus_trader.model.orders'] = nautilus_trader.model.orders
    sys.modules['nautilus_trader.model.position'] = nautilus_trader.model.position
    sys.modules['nautilus_trader.model.events'] = nautilus_trader.model.events

# Stub for websockets if not available
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

    class websockets:
        """Stub websockets module"""
        class WebSocketServerProtocol(MagicMock): pass
        class WebSocketClientProtocol(MagicMock): pass

    sys.modules['websockets'] = websockets

# Stub for src.monitoring.performance_tracker if not available
try:
    from src.monitoring.performance_tracker import PerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False

    class PerformanceTracker(MagicMock):
        """Stub PerformanceTracker"""
        def __init__(self):
            super().__init__()
            self.record_metric = MagicMock()
            self.get_metrics = MagicMock(return_value={})
            self.reset = MagicMock()

    # Add to sys.modules for import
    if 'src.monitoring' not in sys.modules:
        sys.modules['src.monitoring'] = MagicMock()
    if 'src.monitoring.performance_tracker' not in sys.modules:
        sys.modules['src.monitoring.performance_tracker'] = MagicMock()
        sys.modules['src.monitoring.performance_tracker'].PerformanceTracker = PerformanceTracker

def get_dependency_status() -> Dict[str, bool]:
    """Get status of optional dependencies"""
    return {
        'slowapi': SLOWAPI_AVAILABLE,
        'ccxt': CCXT_AVAILABLE,
        'nautilus_trader': NAUTILUS_AVAILABLE,
        'websockets': WEBSOCKETS_AVAILABLE,
        'performance_tracker': PERFORMANCE_TRACKER_AVAILABLE
    }

def log_dependency_status():
    """Log the status of optional dependencies"""
    status = get_dependency_status()
    print("📦 Optional Dependency Status:")
    for dep, available in status.items():
        status_icon = "✅" if available else "❌"
        print(f"   {status_icon} {dep}: {'Available' if available else 'Stubbed'}")

# Auto-log status when module is imported
if __name__ != "__main__":
    log_dependency_status()