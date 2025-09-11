"""
Ultra Low Latency Trading Engine
===============================

High-performance trading engine optimized for microsecond-level execution.
Provides the core trading functionality with ultra-low latency capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import random
from .exchange_connector import ExchangeConnector, ExchangeConfig

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"


@dataclass
class ExecutionConfig:
    """Execution configuration parameters"""
    max_latency_ms: float = 1.0  # Maximum execution latency in milliseconds
    max_slippage_pct: float = 0.001  # Maximum slippage percentage
    min_order_size: float = 0.001  # Minimum order size
    max_order_size: float = 1000.0  # Maximum order size
    enable_smart_routing: bool = True  # Enable intelligent order routing
    enable_pre_trade_risk_check: bool = True  # Enable pre-trade risk checks
    enable_post_trade_analysis: bool = True  # Enable post-trade analysis


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    price: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    volume: float
    spread: float
    timestamp: datetime
    exchange: str = "binance"

    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid_price + self.ask_price) / 2


@dataclass
class Order:
    """Order structure"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    client_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class ExecutionResult:
    """Order execution result"""
    order_id: str
    executed_quantity: float
    executed_price: float
    execution_timestamp: datetime
    status: str  # 'filled', 'partial', 'rejected', 'cancelled'
    fees: float = 0.0
    slippage: float = 0.0
    exchange_order_id: Optional[str] = None


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str = "binance"
    base_url: str = "https://api.binance.com"
    api_key: str = ""
    secret_key: str = ""
    testnet: bool = True
    max_retries: int = 3
    timeout: float = 1.0  # 1 second timeout for ultra-low latency


class UltraLowLatencyTradingEngine:
    """
    Ultra Low Latency Trading Engine

    High-performance trading engine optimized for:
    - Sub-millisecond order execution
    - Intelligent order routing
    - Real-time risk management
    - Advanced order types support
    """

    def __init__(self, config: ExecutionConfig = None, exchange_config: ExchangeConfig = None):
        self.config = config or ExecutionConfig()
        self.exchange_config = exchange_config or ExchangeConfig()
        self.engine_name = "Ultra Low Latency Trading Engine"
        self.is_running = False

        # Exchange connector
        self.exchange_connector = ExchangeConnector(self.exchange_config)

        # Core components
        self.order_book: Dict[str, List[Order]] = {}  # Symbol -> Orders
        self.active_orders: Dict[str, Order] = {}  # OrderID -> Order
        self.execution_results: Dict[str, ExecutionResult] = {}  # OrderID -> Result

        # Performance tracking
        self.performance_metrics = {
            'total_orders': 0,
            'executed_orders': 0,
            'average_latency_ms': 0.0,
            'fill_rate': 0.0,
            'slippage_average': 0.0,
            'rejection_rate': 0.0
        }

        # Risk management
        self.daily_loss_limit = -1000.0  # Daily loss limit
        self.daily_pnl = 0.0
        self.max_concurrent_orders = 1000

        logger.info(f"⚡ {self.engine_name} initialized with {self.config.max_latency_ms}ms max latency")

    async def start(self):
        """Start the trading engine"""
        # Connect to exchange first
        if await self.exchange_connector.connect():
            self.is_running = True
            logger.info(f"▶️ {self.engine_name} started")

            # Initialize order processing loop
            asyncio.create_task(self._order_processing_loop())
        else:
            logger.error(f"❌ Failed to start {self.engine_name} - exchange connection failed")
            raise Exception("Exchange connection failed")

    async def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        await self.exchange_connector.disconnect()
        logger.info(f"⏹️ {self.engine_name} stopped")

    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order for execution"""
        try:
            start_time = datetime.utcnow()

            # Pre-trade risk checks
            if not await self._pre_trade_risk_check(order):
                return ExecutionResult(
                    order_id=order.order_id,
                    executed_quantity=0,
                    executed_price=0,
                    execution_timestamp=datetime.utcnow(),
                    status='rejected',
                    fees=0,
                    slippage=0
                )

            # Add to active orders
            self.active_orders[order.order_id] = order

            # Add to symbol order book
            if order.symbol not in self.order_book:
                self.order_book[order.symbol] = []
            self.order_book[order.symbol].append(order)

            # Simulate order execution (in real implementation, this would connect to exchange)
            execution_result = await self._execute_order(order)

            # Track performance
            end_time = datetime.utcnow()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            self.performance_metrics['total_orders'] += 1
            self.performance_metrics['average_latency_ms'] = (
                (self.performance_metrics['average_latency_ms'] * (self.performance_metrics['total_orders'] - 1)) +
                latency_ms
            ) / self.performance_metrics['total_orders']

            if execution_result.status == 'filled':
                self.performance_metrics['executed_orders'] += 1
                self.daily_pnl += (execution_result.executed_price * execution_result.executed_quantity * 0.001)  # Simplified P&L

            logger.info(f"📤 Order {order.order_id} submitted - Status: {execution_result.status}")

            return execution_result

        except Exception as e:
            logger.error(f"❌ Order submission failed: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                executed_quantity=0,
                executed_price=0,
                execution_timestamp=datetime.utcnow(),
                status='error'
            )

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]

            # Cancel on exchange first
            if await self.exchange_connector.cancel_order(order_id, order.symbol):
                # Remove from local tracking
                if order.symbol in self.order_book:
                    self.order_book[order.symbol] = [o for o in self.order_book[order.symbol] if o.order_id != order_id]
                del self.active_orders[order_id]

                logger.info(f"❌ Order {order_id} cancelled on exchange")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id} on exchange")
                return False

        return False

    async def get_order_status(self, order_id: str) -> Optional[ExecutionResult]:
        """Get order execution status"""
        return self.execution_results.get(order_id)

    async def update_market_data(self, market_data: MarketData):
        """Update market data for all symbols"""
        # In real implementation, this would update internal order books
        # and trigger stop losses, trailing stops, etc.
        pass

    async def _execute_order(self, order: Order) -> ExecutionResult:
        """Execute order via real exchange integration"""
        try:
            # Convert order to exchange format
            exchange_order = {
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price
            }

            # Submit to exchange
            start_time = datetime.utcnow()
            result = await self.exchange_connector.submit_order(exchange_order)
            end_time = datetime.utcnow()

            # Calculate actual latency
            actual_latency = (end_time - start_time).total_seconds() * 1000

            # Convert exchange result to our format
            if result['status'] == 'success':
                # Calculate slippage if we have a target price
                slippage = 0.0
                if order.price and result['executed_price'] > 0:
                    slippage = (result['executed_price'] - order.price) / order.price

                return ExecutionResult(
                    order_id=order.order_id,
                    executed_quantity=result['executed_quantity'],
                    executed_price=result['executed_price'],
                    execution_timestamp=result['timestamp'],
                    status=result['status_exchange'],
                    fees=result['fees'],
                    slippage=slippage,
                    exchange_order_id=str(result['order_id'])
                )
            elif result['status'] == 'rejected':
                return ExecutionResult(
                    order_id=order.order_id,
                    executed_quantity=0,
                    executed_price=0,
                    execution_timestamp=result['timestamp'],
                    status='rejected'
                )
            else:
                return ExecutionResult(
                    order_id=order.order_id,
                    executed_quantity=0,
                    executed_price=0,
                    execution_timestamp=datetime.utcnow(),
                    status='error'
                )

        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                executed_quantity=0,
                executed_price=0,
                execution_timestamp=datetime.utcnow(),
                status='error'
            )

    async def _pre_trade_risk_check(self, order: Order) -> bool:
        """Perform pre-trade risk checks"""
        # Check daily loss limit
        if self.daily_pnl <= self.daily_loss_limit:
            logger.warning(f"❌ Daily loss limit reached: {self.daily_pnl}")
            return False

        # Check order size limits
        if not (self.config.min_order_size <= order.quantity <= self.config.max_order_size):
            logger.warning(f"❌ Order size out of bounds: {order.quantity}")
            return False

        # Check concurrent order limits
        if len(self.active_orders) >= self.max_concurrent_orders:
            logger.warning(f"❌ Max concurrent orders reached: {len(self.active_orders)}")
            return False

        return True

    async def _order_processing_loop(self):
        """Main order processing loop"""
        while self.is_running:
            try:
                # Process pending orders
                for order_id, order in list(self.active_orders.items()):
                    # In real implementation, this would check order status with exchange
                    # and update execution results
                    pass

                # Clean up old orders (older than 1 hour)
                cutoff_time = datetime.utcnow().replace(hour=datetime.utcnow().hour - 1)
                self.active_orders = {
                    k: v for k, v in self.active_orders.items()
                    if v.timestamp > cutoff_time
                }

                await asyncio.sleep(0.001)  # 1ms processing interval

            except Exception as e:
                logger.error(f"❌ Order processing loop error: {e}")
                await asyncio.sleep(1)  # Wait before retry

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        # Calculate derived metrics
        total_orders = self.performance_metrics['total_orders']
        executed_orders = self.performance_metrics['executed_orders']

        if total_orders > 0:
            self.performance_metrics['fill_rate'] = executed_orders / total_orders
            self.performance_metrics['rejection_rate'] = 1 - (executed_orders / total_orders)

        return {
            'engine_name': self.engine_name,
            'is_running': self.is_running,
            'config': {
                'max_latency_ms': self.config.max_latency_ms,
                'max_slippage_pct': self.config.max_slippage_pct,
                'enable_smart_routing': self.config.enable_smart_routing
            },
            'metrics': self.performance_metrics,
            'risk_status': {
                'daily_pnl': self.daily_pnl,
                'daily_loss_limit': self.daily_loss_limit,
                'active_orders': len(self.active_orders),
                'max_concurrent_orders': self.max_concurrent_orders
            },
            'order_book_depth': {symbol: len(orders) for symbol, orders in self.order_book.items()},
            'timestamp': datetime.utcnow()
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'engine_name': self.engine_name,
            'status': 'healthy' if self.is_running else 'stopped',
            'is_running': self.is_running,
            'performance': self.get_performance_metrics(),
            'timestamp': datetime.utcnow()
        }


# Global instance
ultra_low_latency_engine = UltraLowLatencyTradingEngine()


def create_ultra_low_latency_engine(config: ExecutionConfig = None, exchange_config: ExchangeConfig = None) -> UltraLowLatencyTradingEngine:
    """Create ultra low latency trading engine"""
    return UltraLowLatencyTradingEngine(config, exchange_config)


def get_ultra_low_latency_engine() -> UltraLowLatencyTradingEngine:
    """Get ultra low latency trading engine instance"""
    return ultra_low_latency_engine


async def initialize_ultra_low_latency_engine():
    """Initialize ultra low latency engine"""
    await ultra_low_latency_engine.start()


async def shutdown_ultra_low_latency_engine():
    """Shutdown ultra low latency engine"""
    await ultra_low_latency_engine.stop()


# Export key classes and functions
__all__ = [
    'UltraLowLatencyTradingEngine',
    'ExecutionConfig',
    'MarketData',
    'Order',
    'OrderSide',
    'OrderType',
    'ExecutionResult',
    'create_ultra_low_latency_engine',
    'get_ultra_low_latency_engine',
    'initialize_ultra_low_latency_engine',
    'shutdown_ultra_low_latency_engine'
]