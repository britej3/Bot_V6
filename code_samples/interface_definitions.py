"""
Interface Definitions for Core Components

This file defines the interface contracts for the core components of the autonomous trading bot.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio


# 1. Data Structures

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    exchange: str
    timestamp: float
    price: float
    volume: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None


@dataclass
class Order:
    """Standardized order structure"""
    symbol: str
    exchange: str
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit', 'stop_loss', etc.
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'gtc'  # 'gtc', 'ioc', 'fok'


@dataclass
class Position:
    """Standardized position structure"""
    symbol: str
    exchange: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float


@dataclass
class ExecutionReport:
    """Standardized execution report structure"""
    order_id: str
    status: str  # 'filled', 'partially_filled', 'rejected', 'cancelled'
    filled_quantity: float
    avg_fill_price: float
    fees: float
    timestamp: float


# 2. Component Interfaces

class MarketDataProvider(ABC):
    """Interface for market data providers"""
    
    @abstractmethod
    async def get_current_price(self, symbol: str, exchange: str) -> float:
        """Get current price for a symbol"""
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, exchange: str) -> Dict[str, List[List[float]]]:
        """Get order book for a symbol"""
        pass
    
    @abstractmethod
    async def subscribe_market_data(self, symbols: List[str], callback):
        """Subscribe to real-time market data updates"""
        pass


class OrderExecutionService(ABC):
    """Interface for order execution services"""
    
    @abstractmethod
    async def place_order(self, order: Order) -> ExecutionReport:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, exchange: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, exchange: str) -> ExecutionReport:
        """Get order status"""
        pass


class RiskManagementService(ABC):
    """Interface for risk management services"""
    
    @abstractmethod
    async def check_pre_trade_risk(self, order: Order) -> Dict[str, Any]:
        """Perform pre-trade risk check"""
        pass
    
    @abstractmethod
    async def update_position_risk(self, position: Position) -> Dict[str, Any]:
        """Update position risk metrics"""
        pass
    
    @abstractmethod
    async def get_portfolio_risk(self) -> Dict[str, Any]:
        """Get portfolio-level risk metrics"""
        pass


class Strategy(ABC):
    """Interface for trading strategies"""
    
    @abstractmethod
    async def generate_signal(self, market_data: MarketData) -> Dict[str, Any]:
        """Generate trading signal based on market data"""
        pass
    
    @abstractmethod
    async def manage_position(self, position: Position) -> Optional[Order]:
        """Manage existing position"""
        pass


class BacktestingService(ABC):
    """Interface for backtesting services"""
    
    @abstractmethod
    async def run_backtest(self, strategy: Strategy, start_time: float, 
                          end_time: float, symbols: List[str]) -> Dict[str, Any]:
        """Run backtest for strategy"""
        pass
    
    @abstractmethod
    async def optimize_parameters(self, strategy: Strategy, parameters: List[str],
                                 ranges: Dict[str, tuple]) -> Dict[str, Any]:
        """Optimize strategy parameters"""
        pass


class MonitoringService(ABC):
    """Interface for monitoring services"""
    
    @abstractmethod
    async def record_metric(self, component: str, metric_name: str, value: float):
        """Record a metric"""
        pass
    
    @abstractmethod
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for active alerts"""
        pass
    
    @abstractmethod
    async def log_event(self, level: str, message: str, context: Dict[str, Any]):
        """Log an event"""
        pass


# 3. Event Definitions

class EventType(Enum):
    """Standard event types"""
    MARKET_DATA_UPDATE = "market_data_update"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    RISK_ALERT = "risk_alert"
    STRATEGY_SIGNAL = "strategy_signal"
    SYSTEM_HEALTH = "system_health"


@dataclass
class Event:
    """Standardized event structure"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float
    source: str


class EventBus(ABC):
    """Interface for event bus"""
    
    @abstractmethod
    async def publish(self, event: Event):
        """Publish an event"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: EventType, handler):
        """Subscribe to an event type"""
        pass


# 4. Configuration Interfaces

class ConfigProvider(ABC):
    """Interface for configuration providers"""
    
    @abstractmethod
    async def get_config(self, component: str) -> Dict[str, Any]:
        """Get configuration for a component"""
        pass
    
    @abstractmethod
    async def update_config(self, component: str, config: Dict[str, Any]):
        """Update configuration for a component"""
        pass


# 5. Database Interfaces

class TimeSeriesDatabase(ABC):
    """Interface for time-series databases"""
    
    @abstractmethod
    async def write_market_data(self, data: List[MarketData]):
        """Write market data"""
        pass
    
    @abstractmethod
    async def read_market_data(self, symbol: str, exchange: str, 
                              start_time: float, end_time: float) -> List[MarketData]:
        """Read market data"""
        pass


class RelationalDatabase(ABC):
    """Interface for relational databases"""
    
    @abstractmethod
    async def save_order(self, order: Order, execution: ExecutionReport):
        """Save order and execution data"""
        pass
    
    @abstractmethod
    async def get_positions(self, status: str = 'open') -> List[Position]:
        """Get positions"""
        pass


# 6. Exchange Interfaces

class ExchangeAPI(ABC):
    """Interface for exchange APIs"""
    
    @abstractmethod
    async def connect(self):
        """Connect to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Dict[str, Any]:
        """Place an order on the exchange"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on the exchange"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass


# Example implementation of a simple component
class SimpleMarketDataProvider(MarketDataProvider):
    """Simple implementation of market data provider"""
    
    async def get_current_price(self, symbol: str, exchange: str) -> float:
        # Simulated implementation
        return 50000.0
    
    async def get_order_book(self, symbol: str, exchange: str) -> Dict[str, List[List[float]]]:
        # Simulated implementation
        return {
            'bids': [[49999.0, 1.0], [49998.0, 2.0]],
            'asks': [[50001.0, 1.0], [50002.0, 2.0]]
        }
    
    async def subscribe_market_data(self, symbols: List[str], callback):
        # Simulated implementation
        pass


# Example usage
async def example_usage():
    """Example of how to use the interfaces"""
    # Create a market data provider
    md_provider = SimpleMarketDataProvider()
    
    # Get current price
    price = await md_provider.get_current_price('BTC/USD', 'binance')
    print(f"Current price: {price}")
    
    # Get order book
    order_book = await md_provider.get_order_book('BTC/USD', 'binance')
    print(f"Order book: {order_book}")


if __name__ == "__main__":
    asyncio.run(example_usage())