"""
Base Strategy Class for Trading Strategies
Provides common functionality and validation for all trading strategies
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio

from nautilus_trader.core.nautilus_pyo3 import (
    Strategy as NautilusStrategy,
    Bar,
    QuoteTick,
    TradeTick
)
from nautilus_trader.model.enums import (
    OrderType,
    OrderSide,
    TimeInForce,
    OrderStatus
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder

from src.config.trading_config import AdvancedTradingConfig, get_trading_config

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when strategy validation fails"""
    pass


class RiskError(Exception):
    """Raised when risk management rules are violated"""
    pass


class StrategyState:
    """Represents the current state of a trading strategy"""

    def __init__(self):
        self.is_active = False
        self.is_initialized = False
        self.emergency_stop = False
        self.warmup_completed = False
        self.current_position = 0.0
        self.entry_price = 0.0
        self.last_trade_time = None
        self.daily_pnl = 0.0
        self.daily_start_time = datetime.utcnow()

        # Performance metrics
        self.trades_executed = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 10000.0

        # Risk management
        self.consecutive_losses = 0
        self.circuit_breakers = {
            'max_consecutive_losses': False,
            'high_volatility_pause': False,
            'data_quality_issues': False,
            'model_performance_degraded': False
        }


class BaseTradingStrategy(NautilusStrategy, ABC):
    """
    Base class for all trading strategies

    Provides common functionality including:
    - Validation and error handling
    - Risk management
    - Performance tracking
    - Circuit breakers
    - Logging and monitoring
    """

    def __init__(self, config: Optional[AdvancedTradingConfig] = None):
        super().__init__()

        # Configuration with validation
        self.config = config or get_trading_config()
        self._validate_config()

        # Strategy state
        self.state = StrategyState()

        # Trading parameters
        self.min_trade_interval = timedelta(milliseconds=self.config.min_trade_interval_ms)
        self.warmup_bars = 100

        # Pending operations
        self.pending_orders = []
        self.active_predictions = []

        # Validation
        self.validation_errors = []

        logger.info(f"🧠 {self.__class__.__name__} initialized")

    def _validate_config(self):
        """Validate strategy configuration"""
        required_attrs = [
            'symbol', 'mode', 'risk_per_trade_pct', 'max_position_size_btc',
            'max_drawdown_pct', 'min_confidence_threshold'
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValidationError(f"Missing required config attribute: {attr}")

        # Validate numeric ranges
        if not 0 < self.config.risk_per_trade_pct <= 0.1:
            raise ValidationError("Risk per trade must be between 0 and 10%")

        if not 0 < self.config.max_drawdown_pct <= 0.5:
            raise ValidationError("Max drawdown must be between 0 and 50%")

        if not 0.1 <= self.config.min_confidence_threshold <= 0.99:
            raise ValidationError("Confidence threshold must be between 0.1 and 0.99")

        logger.info("✅ Configuration validation passed")

    async def initialize(self):
        """Initialize the strategy - override in subclasses"""
        try:
            self.state.is_initialized = True
            logger.info(f"✅ {self.__class__.__name__} initialization completed")

        except Exception as e:
            logger.error(f"❌ Strategy initialization failed: {e}")
            raise

    def on_start(self):
        """Called when the strategy starts"""
        logger.info(f"🚀 {self.__class__.__name__} started")
        self.state.is_active = True
        self._log_configuration()

    def on_stop(self):
        """Called when the strategy stops"""
        logger.info(f"🛑 {self.__class__.__name__} stopping...")
        self._log_final_performance()
        logger.info(f"✅ {self.__class__.__name__} stopped")

    def on_bar(self, bar: Bar):
        """Called on each new bar - main strategy logic"""
        try:
            # Emergency stop check
            if self.state.emergency_stop:
                logger.warning("🚨 Emergency stop activated")
                return

            # Skip warmup period
            if not self.state.warmup_completed:
                if len(self.cache.bar_cache) < self.warmup_bars:
                    return
                self.state.warmup_completed = True
                logger.info("✅ Warmup completed, starting trading")

            # Process bar - implemented by subclasses
            self._process_bar(bar)

        except Exception as e:
            logger.error(f"❌ Error in on_bar: {e}")
            self._add_validation_error(f"on_bar error: {e}")

    def on_quote_tick(self, tick: QuoteTick):
        """Called on quote tick updates"""
        try:
            self._process_quote_tick(tick)
        except Exception as e:
            logger.error(f"Error processing quote tick: {e}")

    def on_trade_tick(self, tick: TradeTick):
        """Called on trade tick updates"""
        try:
            self._process_trade_tick(tick)
        except Exception as e:
            logger.error(f"Error processing trade tick: {e}")

    def on_order(self, order):
        """Called when an order update is received"""
        try:
            # Update pending orders
            self.pending_orders = [o for o in self.pending_orders if o['order_id'] != order.client_order_id]

            if order.status == OrderStatus.FILLED:
                logger.info(f"📈 Order filled: {order.client_order_id}")
                self._record_trade(order)

            elif order.status in [OrderStatus.REJECTED, OrderStatus.CANCELED]:
                logger.warning(f"❌ Order failed: {order.client_order_id} - {order.status}")
                self._handle_failed_order(order)

        except Exception as e:
            logger.error(f"Error processing order update: {e}")

    @abstractmethod
    def _process_bar(self, bar: Bar):
        """Process bar data - must be implemented by subclasses"""
        pass

    def _process_quote_tick(self, tick: QuoteTick):
        """Process quote tick data - can be overridden by subclasses"""
        pass

    def _process_trade_tick(self, tick: TradeTick):
        """Process trade tick data - can be overridden by subclasses"""
        pass

    def _execute_buy_order(self, position_size: float, price: float, timestamp: datetime):
        """Execute buy order with validation"""
        try:
            self._validate_order_execution('BUY', position_size, price)

            instrument_id = InstrumentId.from_str(f"{self.config.symbol.upper()}-PERP.BINANCE")
            quantity = Quantity.from_str(str(round(position_size, 6)))

            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=instrument_id,
                order_side=OrderSide.BUY,
                quantity=quantity,
                time_in_force=TimeInForce.GTC,
                post_only=False,
                reduce_only=False,
                quote_quantity=False,
            )

            self.submit_order(order)

            # Update position tracking
            self.state.current_position += position_size
            self.state.entry_price = price
            self.state.last_trade_time = timestamp

            # Add to pending orders for tracking
            self.pending_orders.append({
                'order_id': order.client_order_id,
                'side': 'BUY',
                'size': position_size,
                'price': price,
                'timestamp': timestamp
            })

            logger.info(f"📈 BUY order: {quantity} at ${price:.2f}")

        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
            raise

    def _execute_sell_order(self, position_size: float, price: float, timestamp: datetime):
        """Execute sell order with validation"""
        try:
            self._validate_order_execution('SELL', position_size, price)

            instrument_id = InstrumentId.from_str(f"{self.config.symbol.upper()}-PERP.BINANCE")
            quantity = Quantity.from_str(str(round(position_size, 6)))

            order = MarketOrder(
                trader_id=self.trader_id,
                strategy_id=self.id,
                instrument_id=instrument_id,
                order_side=OrderSide.SELL,
                quantity=quantity,
                time_in_force=TimeInForce.GTC,
                post_only=False,
                reduce_only=False,
                quote_quantity=False,
            )

            self.submit_order(order)

            # Update position tracking
            self.state.current_position -= position_size
            self.state.entry_price = price
            self.state.last_trade_time = timestamp

            # Add to pending orders for tracking
            self.pending_orders.append({
                'order_id': order.client_order_id,
                'side': 'SELL',
                'size': position_size,
                'price': price,
                'timestamp': timestamp
            })

            logger.info(f"📉 SELL order: {quantity} at ${price:.2f}")

        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
            raise

    def _validate_order_execution(self, side: str, position_size: float, price: float):
        """Validate order execution parameters"""
        if position_size <= 0:
            raise ValidationError("Position size must be positive")

        if price <= 0:
            raise ValidationError("Price must be positive")

        if position_size > self.config.max_position_size_btc:
            raise RiskError(f"Position size {position_size} exceeds maximum {self.config.max_position_size_btc}")

        # Check trade interval
        if self.state.last_trade_time:
            time_since_last_trade = datetime.utcnow() - self.state.last_trade_time
            if time_since_last_trade < self.min_trade_interval:
                raise ValidationError("Trade interval too short")

        # Check circuit breakers
        if any(self.state.circuit_breakers.values()):
            raise RiskError("Circuit breaker activated")

    def _calculate_position_size(self, confidence: float, price: float) -> float:
        """Calculate position size with risk management"""
        try:
            # Base risk calculation
            account_balance = 10000.0  # This should be dynamic in production
            risk_amount = account_balance * self.config.risk_per_trade_pct

            # Stop distance (0.5% of current price)
            stop_distance = price * 0.005

            if stop_distance == 0:
                return 0.0

            base_position = risk_amount / stop_distance

            # Adjust by confidence
            confidence_factor = max(0.5, min(1.0, (confidence - 0.5) / 0.5))
            position_size = min(base_position * confidence_factor, self.config.max_position_size_btc)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _record_trade(self, order):
        """Record trade for performance tracking"""
        try:
            self.state.trades_executed += 1

            # Calculate P&L (simplified)
            if hasattr(order, 'filled_quantity') and order.filled_quantity:
                fill_price = float(order.filled_price) if hasattr(order, 'filled_price') else self.state.entry_price
                pnl = (fill_price - self.state.entry_price) * self.state.current_position
                self.state.total_pnl += pnl

                # Update win/loss tracking
                if pnl > 0:
                    self.state.winning_trades += 1
                    self.state.consecutive_losses = 0
                else:
                    self.state.consecutive_losses += 1

                # Update circuit breakers
                self._update_circuit_breakers(pnl > 0)

            # Update daily P&L
            self.state.daily_pnl = self.state.total_pnl

        except Exception as e:
            logger.error(f"Error recording trade: {e}")

    def _handle_failed_order(self, order):
        """Handle failed order"""
        try:
            self._update_circuit_breakers(False)
            logger.warning(f"Order {order.client_order_id} failed with status {order.status}")

        except Exception as e:
            logger.error(f"Error handling failed order: {e}")

    def _update_circuit_breakers(self, trade_result: Optional[bool] = None):
        """Update circuit breaker states"""
        try:
            # Update consecutive losses
            if trade_result is not None:
                if trade_result:
                    self.state.consecutive_losses = 0
                else:
                    self.state.consecutive_losses += 1

            if self.state.consecutive_losses >= self.config.max_consecutive_losses:
                self.state.circuit_breakers['max_consecutive_losses'] = True

        except Exception as e:
            logger.error(f"Error updating circuit breakers: {e}")

    def _add_validation_error(self, error: str):
        """Add validation error"""
        self.validation_errors.append(f"{datetime.utcnow()}: {error}")

        # Keep only recent errors
        if len(self.validation_errors) > 10:
            self.validation_errors.pop(0)

    def _log_configuration(self):
        """Log strategy configuration"""
        logger.info("⚙️  Strategy Configuration:"        logger.info(f"   Symbol: {self.config.symbol}")
        logger.info(f"   Mode: {self.config.mode}")
        logger.info(f"   Risk per trade: {self.config.risk_per_trade_pct:.2%}")
        logger.info(f"   Max position size: {self.config.max_position_size_btc} BTC")
        logger.info(f"   Max drawdown: {self.config.max_drawdown_pct:.2%}")
        logger.info(f"   Min confidence: {self.config.min_confidence_threshold:.2f}")

    def _log_final_performance(self):
        """Log final performance metrics"""
        try:
            win_rate = 0.0
            if self.state.trades_executed > 0:
                win_rate = self.state.winning_trades / self.state.trades_executed

            logger.info("📊 Final Performance Summary:"            logger.info(f"   Total Trades: {self.state.trades_executed}")
            logger.info(f"   Winning Trades: {self.state.winning_trades}")
            logger.info(f"   Win Rate: {win_rate:.2%}")
            logger.info(f"   Total P&L: {self.state.total_pnl:.4f} BTC")
            logger.info(f"   Max Drawdown: {self.state.max_drawdown:.2%}")
            logger.info(f"   Final Position: {self.state.current_position:.6f} BTC")

        except Exception as e:
            logger.error(f"Error logging final performance: {e}")

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get comprehensive strategy status"""
        return {
            'is_active': self.state.is_active,
            'is_initialized': self.state.is_initialized,
            'emergency_stop': self.state.emergency_stop,
            'trades_executed': self.state.trades_executed,
            'current_position': self.state.current_position,
            'total_pnl': self.state.total_pnl,
            'win_rate': (self.state.winning_trades / max(self.state.trades_executed, 1)),
            'max_drawdown': self.state.max_drawdown,
            'last_trade_time': self.state.last_trade_time,
            'entry_price': self.state.entry_price,
            'daily_pnl': self.state.daily_pnl,
            'consecutive_losses': self.state.consecutive_losses,
            'circuit_breakers': self.state.circuit_breakers.copy(),
            'pending_orders': len(self.pending_orders),
            'validation_errors': self.validation_errors.copy()
        }

    def emergency_stop(self):
        """Activate emergency stop"""
        self.state.emergency_stop = True
        logger.critical("🚨 Emergency stop activated")

    def reset_circuit_breakers(self):
        """Reset circuit breakers"""
        self.state.circuit_breakers = {
            'max_consecutive_losses': False,
            'high_volatility_pause': False,
            'data_quality_issues': False,
            'model_performance_degraded': False
        }
        logger.info("🔄 Circuit breakers reset")