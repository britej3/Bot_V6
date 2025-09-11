"""
Architecture Demo: Core Components Implementation Examples

This file demonstrates key architectural patterns used in the autonomous trading bot system.
"""

import asyncio
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 1. Event-Driven Architecture Pattern
class Event:
    """Base event class"""
    def __init__(self, event_type: str, data: Dict):
        self.type = event_type
        self.data = data
        self.timestamp = asyncio.get_event_loop().time()


class EventBus:
    """Central event bus for event-driven communication"""
    def __init__(self):
        self._subscribers: Dict[str, List] = {}
    
    def subscribe(self, event_type: str, handler):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers"""
        if event.type in self._subscribers:
            for handler in self._subscribers[event.type]:
                await handler(event)


# 2. Microservices Pattern with Service Interface
class Service(ABC):
    """Abstract base class for all services"""
    
    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self.is_running = False
    
    @abstractmethod
    async def start(self):
        """Start the service"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the service"""
        pass
    
    @abstractmethod
    async def handle_event(self, event: Event):
        """Handle incoming events"""
        pass


# 3. Strategy Pattern for Trading Strategies
class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    @abstractmethod
    def generate_signal(self, market_data: Dict) -> Dict:
        """Generate trading signal based on market data"""
        pass


class MovingAverageCrossoverStrategy(TradingStrategy):
    """Simple moving average crossover strategy"""
    
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, market_data: Dict) -> Dict:
        """Generate buy/sell signal based on moving average crossover"""
        # Simplified implementation
        if market_data.get('short_ma', 0) > market_data.get('long_ma', 0):
            return {'action': 'BUY', 'confidence': 0.7}
        elif market_data.get('short_ma', 0) < market_data.get('long_ma', 0):
            return {'action': 'SELL', 'confidence': 0.7}
        else:
            return {'action': 'HOLD', 'confidence': 0.5}


# 4. Observer Pattern for Risk Management
class RiskObserver(ABC):
    """Abstract base class for risk observers"""
    
    @abstractmethod
    def on_risk_event(self, event: Dict):
        """Handle risk events"""
        pass


class RiskManager:
    """Risk management system implementing observer pattern"""
    
    def __init__(self):
        self.observers: List[RiskObserver] = []
        self.risk_limits = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_daily_loss': 0.02,    # 2% daily loss limit
            'max_drawdown': 0.1        # 10% drawdown limit
        }
    
    def attach(self, observer: RiskObserver):
        """Attach a risk observer"""
        self.observers.append(observer)
    
    def check_order_risk(self, order: Dict) -> bool:
        """Check if order complies with risk limits"""
        # Simplified risk check
        position_size = order.get('quantity', 0) * order.get('price', 0)
        
        if position_size > self.risk_limits['max_position_size']:
            self._notify_observers({
                'type': 'RISK_VIOLATION',
                'message': 'Position size exceeds limit',
                'order': order
            })
            return False
        
        return True
    
    def _notify_observers(self, event: Dict):
        """Notify all observers of a risk event"""
        for observer in self.observers:
            observer.on_risk_event(event)


# 5. Factory Pattern for Exchange Connections
class ExchangeConnection(ABC):
    """Abstract base class for exchange connections"""
    
    @abstractmethod
    async def place_order(self, order: Dict) -> Dict:
        """Place an order on the exchange"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data for a symbol"""
        pass


class BinanceConnection(ExchangeConnection):
    """Binance exchange connection"""
    
    async def place_order(self, order: Dict) -> Dict:
        """Place an order on Binance"""
        # Simulated implementation
        logger.info(f"Placing order on Binance: {order}")
        return {
            'order_id': 'binance_12345',
            'status': 'FILLED',
            'filled_quantity': order.get('quantity', 0)
        }
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data from Binance"""
        # Simulated implementation
        return {
            'symbol': symbol,
            'price': 50000.0,
            'volume': 1000.0
        }


class CoinbaseConnection(ExchangeConnection):
    """Coinbase exchange connection"""
    
    async def place_order(self, order: Dict) -> Dict:
        """Place an order on Coinbase"""
        # Simulated implementation
        logger.info(f"Placing order on Coinbase: {order}")
        return {
            'order_id': 'coinbase_12345',
            'status': 'FILLED',
            'filled_quantity': order.get('quantity', 0)
        }
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get market data from Coinbase"""
        # Simulated implementation
        return {
            'symbol': symbol,
            'price': 50100.0,
            'volume': 500.0
        }


class ExchangeConnectionFactory:
    """Factory for creating exchange connections"""
    
    @staticmethod
    def create_connection(exchange_name: str) -> ExchangeConnection:
        """Create an exchange connection based on name"""
        if exchange_name.lower() == 'binance':
            return BinanceConnection()
        elif exchange_name.lower() == 'coinbase':
            return CoinbaseConnection()
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")


# 6. Decorator Pattern for Rate Limiting
def rate_limit(calls_per_second: int):
    """Decorator for rate limiting function calls"""
    def decorator(func):
        last_called = [0.0]
        
        async def wrapper(*args, **kwargs):
            elapsed = asyncio.get_event_loop().time() - last_called[0]
            left_to_wait = 1.0 / calls_per_second - elapsed
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            ret = await func(*args, **kwargs)
            last_called[0] = asyncio.get_event_loop().time()
            return ret
        return wrapper
    return decorator


# 7. Circuit Breaker Pattern for Fault Tolerance
class CircuitBreaker:
    """Circuit breaker for handling service failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == 'OPEN':
            if asyncio.get_event_loop().time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


# Example usage and integration
async def main():
    """Demonstrate the architecture patterns"""
    # Create event bus
    event_bus = EventBus()
    
    # Create risk manager
    risk_manager = RiskManager()
    
    # Create exchange connections
    binance_conn = ExchangeConnectionFactory.create_connection('binance')
    coinbase_conn = ExchangeConnectionFactory.create_connection('coinbase')
    
    # Example market data
    market_data = {
        'symbol': 'BTC/USD',
        'short_ma': 49500,
        'long_ma': 49000,
        'price': 50000
    }
    
    # Example order
    order = {
        'symbol': 'BTC/USD',
        'quantity': 0.1,
        'price': 50000
    }
    
    # Apply risk checks
    if risk_manager.check_order_risk(order):
        # Place order on exchange
        result = await binance_conn.place_order(order)
        logger.info(f"Order result: {result}")
    
    # Generate trading signal
    strategy = MovingAverageCrossoverStrategy()
    signal = strategy.generate_signal(market_data)
    logger.info(f"Trading signal: {signal}")


if __name__ == "__main__":
    asyncio.run(main())