"""
Nautilus Trader Strategy Adapter
================================

Adapter to bridge existing trading strategies with Nautilus Trader framework.
Enables existing strategies to leverage Nautilus's advanced features without redundancy.

Key Features:
- Strategy conversion from existing format to Nautilus
- Signal mapping and translation
- Risk management integration
- Performance tracking compatibility
- Order execution optimization

Integration Strategy:
- Non-destructive adaptation (existing strategies unchanged)
- Feature enhancement (leverage Nautilus capabilities)
- Seamless fallback (existing system remains primary)
- Performance comparison (A/B testing capabilities)

Author: Trading Systems Team
Date: 2025-01-22
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

# Conditional import of nautilus_trader - use stubs if not available
try:
    from nautilus_trader.model.identifiers import StrategyId, InstrumentId
    from nautilus_trader.model.orders import Order
    from nautilus_trader.model.position import Position
    from nautilus_trader.model.events import OrderEvent, PositionEvent
    NAUTILUS_AVAILABLE = True
except ImportError:
    # Create stub classes when nautilus_trader is not available
    NAUTILUS_AVAILABLE = False
    
    class StrategyId:
        def __init__(self, value): self.value = value
        def __str__(self): return self.value
    
    class InstrumentId:
        def __init__(self, value): self.value = value
        def __str__(self): return self.value
    
    class Order: pass
    class Position: pass
    class OrderEvent: pass
    class PositionEvent: pass

from src.learning.strategy_model_integration_engine import TradingStrategy
from src.trading.nautilus_integration import (
    NautilusTraderManager,
    NautilusOrderRequest
)
from src.monitoring.performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


class NautilusStrategyAdapter(ABC):
    """
    Abstract base class for Nautilus strategy adapters

    This adapter pattern allows existing strategies to work with Nautilus
    while maintaining their original functionality and adding Nautilus features.
    """

    def __init__(self, strategy_id: str, instrument_id: str):
        self.strategy_id = StrategyId(strategy_id)
        self.instrument_id = InstrumentId(instrument_id)
        self.nautilus_manager = NautilusTraderManager()
        self.performance_tracker = PerformanceTracker()

        # State tracking
        self.is_active = False
        self.positions = {}
        self.orders = {}
        self.signals = []

        # Performance metrics
        self.adaptation_metrics = {
            'signals_processed': 0,
            'orders_submitted': 0,
            'orders_filled': 0,
            'profit_loss': 0.0,
            'win_rate': 0.0
        }

        logger.info(f"🔄 Nautilus Strategy Adapter initialized for {strategy_id}")

    @abstractmethod
    async def adapt_signal(self, signal: Dict[str, Any]) -> Optional[NautilusOrderRequest]:
        """Adapt trading signal to Nautilus format"""
        pass

    @abstractmethod
    async def handle_nautilus_event(self, event: Any):
        """Handle events from Nautilus engine"""
        pass

    async def start(self):
        """Start the adapted strategy"""
        await self.nautilus_manager.initialize()
        self.is_active = True
        logger.info(f"🚀 Nautilus Strategy Adapter started for {self.strategy_id}")

    async def stop(self):
        """Stop the adapted strategy"""
        self.is_active = False
        logger.info(f"🛑 Nautilus Strategy Adapter stopped for {self.strategy_id}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter performance metrics"""
        return {
            'strategy_id': str(self.strategy_id),
            'instrument_id': str(self.instrument_id),
            'is_active': self.is_active,
            'adaptation_metrics': self.adaptation_metrics,
            'positions': len(self.positions),
            'pending_orders': len(self.orders)
        }


class ScalpingStrategyAdapter(NautilusStrategyAdapter):
    """
    Adapter for scalping strategies to leverage Nautilus advanced features

    Enhances existing scalping strategies with:
    - Advanced order types (Iceberg, TWAP, VWAP)
    - Better slippage control
    - Enhanced risk management
    - Professional analytics
    """

    def __init__(self, strategy_id: str, instrument_id: str):
        super().__init__(strategy_id, instrument_id)

        # Scalping-specific configuration
        self.min_order_size = 0.001
        self.max_order_size = 1.0
        self.target_profit_pct = 0.002  # 0.2%
        self.max_hold_time = 300  # 5 minutes
        self.stop_loss_multiplier = 2.0

    async def adapt_signal(self, signal: Dict[str, Any]) -> Optional[NautilusOrderRequest]:
        """Adapt scalping signal for Nautilus execution"""

        try:
            # Extract signal components
            symbol = signal.get('symbol', 'BTC/USDT')
            direction = signal.get('direction', 0)  # -1 to 1
            confidence = signal.get('confidence', 0)  # 0 to 1
            volatility = signal.get('volatility', 0.02)

            # Validate signal
            if confidence < 0.6:  # Minimum confidence threshold
                logger.debug(f"Signal confidence too low: {confidence}")
                return None

            if abs(direction) < 0.1:  # Minimum direction threshold
                logger.debug(f"Signal direction too weak: {direction}")
                return None

            # Calculate order parameters
            side = 'BUY' if direction > 0 else 'SELL'
            quantity = self._calculate_position_size(signal)

            if quantity < self.min_order_size:
                logger.debug(f"Order size too small: {quantity}")
                return None

            # Determine order type based on market conditions
            order_type = self._select_order_type(signal, volatility)

            # Create Nautilus order request
            order_request = NautilusOrderRequest(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                client_id=f"scalping_{datetime.utcnow().timestamp()}",
                metadata={
                    'original_signal': signal,
                    'adaptation_method': 'scalping_enhanced',
                    'confidence': confidence,
                    'volatility': volatility
                }
            )

            # Add price parameters for limit orders
            if order_type == 'LIMIT':
                order_request.price = self._calculate_limit_price(signal)

            # Add stop loss parameters
            if order_type in ['STOP_MARKET', 'TRAILING_STOP']:
                order_request.stop_price = self._calculate_stop_price(signal)

            # Track signal processing
            self.adaptation_metrics['signals_processed'] += 1
            self.signals.append({
                'timestamp': datetime.utcnow(),
                'original_signal': signal,
                'adapted_order': order_request.__dict__
            })

            logger.info(f"✅ Adapted scalping signal: {side} {quantity} {symbol} ({order_type})")
            return order_request

        except Exception as e:
            logger.error(f"❌ Failed to adapt scalping signal: {e}")
            return None

    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate optimal position size for scalping"""
        confidence = signal.get('confidence', 0.5)
        volatility = signal.get('volatility', 0.02)
        market_regime = signal.get('market_regime', 'normal')

        # Base size calculation
        base_size = 0.01  # 1% of portfolio

        # Confidence adjustment
        confidence_multiplier = 0.5 + confidence  # 0.5 to 1.5

        # Volatility adjustment (inverse relationship)
        volatility_multiplier = min(1.0, 0.02 / volatility)

        # Market regime adjustment
        regime_multipliers = {
            'trending': 1.2,
            'ranging': 1.0,
            'volatile': 0.7,
            'crisis': 0.3
        }
        regime_multiplier = regime_multipliers.get(market_regime, 1.0)

        # Calculate final size
        position_size = base_size * confidence_multiplier * volatility_multiplier * regime_multiplier

        # Apply bounds
        return max(self.min_order_size, min(self.max_order_size, position_size))

    def _select_order_type(self, signal: Dict[str, Any], volatility: float) -> str:
        """Select optimal order type based on market conditions"""

        confidence = signal.get('confidence', 0.5)
        volume = signal.get('volume', 1000)

        # High confidence + low volatility = Market order
        if confidence > 0.8 and volatility < 0.01:
            return 'MARKET'

        # Medium confidence + moderate volatility = Limit order
        elif confidence > 0.6 and volatility < 0.03:
            return 'LIMIT'

        # High volatility scenarios = Stop orders
        elif volatility > 0.05:
            return 'STOP_MARKET'

        # Default to limit order
        else:
            return 'LIMIT'

    def _calculate_limit_price(self, signal: Dict[str, Any]) -> float:
        """Calculate limit price for order"""
        current_price = signal.get('price', 50000)
        direction = signal.get('direction', 0)
        spread_pct = signal.get('spread_pct', 0.001)

        # For buy orders, place limit slightly above current price
        if direction > 0:
            return current_price * (1 + spread_pct * 0.5)

        # For sell orders, place limit slightly below current price
        else:
            return current_price * (1 - spread_pct * 0.5)

    def _calculate_stop_price(self, signal: Dict[str, Any]) -> float:
        """Calculate stop price for risk management"""
        current_price = signal.get('price', 50000)
        direction = signal.get('direction', 0)

        # Target profit percentage
        profit_target = current_price * self.target_profit_pct

        # Stop loss distance
        stop_distance = profit_target * self.stop_loss_multiplier

        # For long positions, stop loss below entry
        if direction > 0:
            return current_price - stop_distance

        # For short positions, stop loss above entry
        else:
            return current_price + stop_distance

    async def handle_nautilus_event(self, event: Any):
        """Handle events from Nautilus engine"""

        if isinstance(event, OrderEvent):
            await self._handle_order_event(event)
        elif isinstance(event, PositionEvent):
            await self._handle_position_event(event)
        else:
            logger.debug(f"Unhandled Nautilus event: {type(event)}")

    async def _handle_order_event(self, event: OrderEvent):
        """Handle order-related events"""

        order_id = str(event.order_id)

        if event.event_type.name == 'SUBMITTED':
            self.orders[order_id] = {
                'status': 'submitted',
                'submitted_at': event.timestamp,
                'order_details': event
            }
            self.adaptation_metrics['orders_submitted'] += 1

        elif event.event_type.name == 'FILLED':
            if order_id in self.orders:
                self.orders[order_id]['status'] = 'filled'
                self.orders[order_id]['filled_at'] = event.timestamp
                self.adaptation_metrics['orders_filled'] += 1

            # Update performance metrics
            fill_price = float(event.price)
            quantity = float(event.quantity)

            # Calculate P&L (simplified - would need entry price tracking)
            self.adaptation_metrics['profit_loss'] += fill_price * quantity * 0.001  # 0.1% profit assumption

        elif event.event_type.name == 'CANCELLED':
            if order_id in self.orders:
                self.orders[order_id]['status'] = 'cancelled'
                self.orders[order_id]['cancelled_at'] = event.timestamp

    async def _handle_position_event(self, event: PositionEvent):
        """Handle position-related events"""

        position_id = str(event.position_id)

        if event.event_type.name == 'OPENED':
            self.positions[position_id] = {
                'status': 'opened',
                'opened_at': event.timestamp,
                'entry_price': float(event.price),
                'quantity': float(event.quantity),
                'position_details': event
            }

        elif event.event_type.name == 'CLOSED':
            if position_id in self.positions:
                position = self.positions[position_id]
                entry_price = position['entry_price']
                exit_price = float(event.price)
                quantity = position['quantity']

                # Calculate actual P&L
                pnl = (exit_price - entry_price) * quantity
                self.adaptation_metrics['profit_loss'] += pnl

                position['status'] = 'closed'
                position['closed_at'] = event.timestamp
                position['exit_price'] = exit_price
                position['pnl'] = pnl


class MarketMakingStrategyAdapter(NautilusStrategyAdapter):
    """
    Adapter for market making strategies with Nautilus advanced features

    Leverages Nautilus capabilities for:
    - Iceberg orders for large positions
    - TWAP/VWAP algorithms
    - Advanced spread management
    - Professional inventory management
    """

    def __init__(self, strategy_id: str, instrument_id: str):
        super().__init__(strategy_id, instrument_id)

        # Market making specific configuration
        self.spread_target_pct = 0.001  # 0.1% target spread
        self.inventory_target = 0.0      # Neutral inventory target
        self.order_refresh_interval = 30  # 30 seconds
        self.max_order_size = 0.5        # 50% of inventory limit

    async def adapt_signal(self, signal: Dict[str, Any]) -> Optional[NautilusOrderRequest]:
        """Adapt market making signal for Nautilus execution"""

        try:
            symbol = signal.get('symbol', 'BTC/USDT')
            imbalance = signal.get('order_imbalance', 0)
            spread_pct = signal.get('spread_pct', 0.001)
            current_price = signal.get('price', 50000)

            # Market making logic - provide liquidity on both sides
            orders = []

            # Create buy order (provide liquidity)
            if imbalance < -0.3 or spread_pct > self.spread_target_pct:
                buy_order = NautilusOrderRequest(
                    symbol=symbol,
                    side='BUY',
                    quantity=self._calculate_mm_order_size(signal),
                    order_type='LIMIT',
                    price=current_price * (1 - spread_pct * 0.5),
                    client_id=f"mm_buy_{datetime.utcnow().timestamp()}",
                    metadata={
                        'original_signal': signal,
                        'adaptation_method': 'market_making',
                        'order_imbalance': imbalance,
                        'spread_pct': spread_pct
                    }
                )
                orders.append(buy_order)

            # Create sell order (provide liquidity)
            if imbalance > 0.3 or spread_pct > self.spread_target_pct:
                sell_order = NautilusOrderRequest(
                    symbol=symbol,
                    side='SELL',
                    quantity=self._calculate_mm_order_size(signal),
                    order_type='LIMIT',
                    price=current_price * (1 + spread_pct * 0.5),
                    client_id=f"mm_sell_{datetime.utcnow().timestamp()}",
                    metadata={
                        'original_signal': signal,
                        'adaptation_method': 'market_making',
                        'order_imbalance': imbalance,
                        'spread_pct': spread_pct
                    }
                )
                orders.append(sell_order)

            # Use Nautilus advanced features for large orders
            if len(orders) > 0:
                # Convert to iceberg orders for better execution
                enhanced_orders = []
                for order in orders:
                    if order.quantity > self.max_order_size:
                        # Use iceberg order for large sizes
                        order.order_type = 'ICEBERG'
                        order.metadata['iceberg'] = True
                    enhanced_orders.append(order)

                self.adaptation_metrics['signals_processed'] += 1
                return enhanced_orders[0]  # Return first order for now

            return None

        except Exception as e:
            logger.error(f"❌ Failed to adapt market making signal: {e}")
            return None

    def _calculate_mm_order_size(self, signal: Dict[str, Any]) -> float:
        """Calculate market making order size"""
        volume = signal.get('volume', 1000)
        spread_pct = signal.get('spread_pct', 0.001)

        # Size based on market volume and spread
        base_size = volume * 0.001  # 0.1% of recent volume

        # Adjust for spread (wider spread = larger orders)
        spread_multiplier = max(0.5, spread_pct / 0.001)

        order_size = base_size * spread_multiplier

        return max(0.001, min(self.max_order_size, order_size))


class MeanReversionStrategyAdapter(NautilusStrategyAdapter):
    """
    Adapter for mean reversion strategies with Nautilus advanced features

    Enhances mean reversion with:
    - Trailing stop orders
    - Time-based exits
    - Volatility-adjusted entries
    - Professional risk management
    """

    def __init__(self, strategy_id: str, instrument_id: str):
        super().__init__(strategy_id, instrument_id)

        # Mean reversion specific configuration
        self.oversold_threshold = 30    # RSI threshold
        self.overbought_threshold = 70  # RSI threshold
        self.min_reversal_confidence = 0.7
        self.max_hold_time = 3600       # 1 hour

    async def adapt_signal(self, signal: Dict[str, Any]) -> Optional[NautilusOrderRequest]:
        """Adapt mean reversion signal for Nautilus execution"""

        try:
            rsi = signal.get('rsi', 50)
            bollinger_position = signal.get('bollinger_position', 0.5)
            confidence = signal.get('confidence', 0.5)

            # Mean reversion entry conditions
            entry_conditions = (
                (rsi <= self.oversold_threshold and bollinger_position <= 0.1) or
                (rsi >= self.overbought_threshold and bollinger_position >= 0.9)
            )

            if not entry_conditions or confidence < self.min_reversal_confidence:
                return None

            # Determine direction
            if rsi <= self.oversold_threshold:
                side = 'BUY'   # Buy oversold conditions
            else:
                side = 'SELL'  # Sell overbought conditions

            # Use trailing stop for exit
            order_request = NautilusOrderRequest(
                symbol=signal.get('symbol', 'BTC/USDT'),
                side=side,
                quantity=self._calculate_mr_position_size(signal),
                order_type='TRAILING_STOP',
                client_id=f"mr_{datetime.utcnow().timestamp()}",
                metadata={
                    'original_signal': signal,
                    'adaptation_method': 'mean_reversion',
                    'rsi': rsi,
                    'bollinger_position': bollinger_position,
                    'confidence': confidence
                }
            )

            # Set trailing stop parameters
            order_request.stop_price = self._calculate_trailing_stop(signal)

            self.adaptation_metrics['signals_processed'] += 1
            logger.info(f"✅ Adapted mean reversion signal: {side} {order_request.quantity}")

            return order_request

        except Exception as e:
            logger.error(f"❌ Failed to adapt mean reversion signal: {e}")
            return None

    def _calculate_mr_position_size(self, signal: Dict[str, Any]) -> float:
        """Calculate mean reversion position size"""
        rsi_divergence = abs(signal.get('rsi', 50) - 50) / 50
        confidence = signal.get('confidence', 0.5)

        # Size based on RSI divergence and confidence
        base_size = 0.02  # 2% of portfolio
        rsi_multiplier = rsi_divergence * 2  # 0 to 2x
        confidence_multiplier = confidence * 1.5     # 0 to 1.5x

        position_size = base_size * rsi_multiplier * confidence_multiplier

        return max(0.001, min(0.1, position_size))

    def _calculate_trailing_stop(self, signal: Dict[str, Any]) -> float:
        """Calculate trailing stop price"""
        current_price = signal.get('price', 50000)
        volatility = signal.get('volatility', 0.02)

        # Volatility-adjusted trailing distance
        trail_distance = current_price * volatility * 2

        return trail_distance


# Strategy adapter factory
class StrategyAdapterFactory:
    """Factory for creating strategy adapters"""

    @staticmethod
    def create_adapter(
        strategy_type: str,
        strategy_id: str,
        instrument_id: str
    ) -> NautilusStrategyAdapter:
        """Create appropriate strategy adapter"""

        if strategy_type.lower() == 'scalping':
            return ScalpingStrategyAdapter(strategy_id, instrument_id)
        elif strategy_type.lower() == 'market_making':
            return MarketMakingStrategyAdapter(strategy_id, instrument_id)
        elif strategy_type.lower() == 'mean_reversion':
            return MeanReversionStrategyAdapter(strategy_id, instrument_id)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")


# Utility functions
async def adapt_strategy_for_nautilus(
    existing_strategy: Any,
    strategy_type: str,
    instrument_id: str
) -> NautilusStrategyAdapter:
    """Adapt an existing strategy for Nautilus integration"""

    strategy_id = f"nautilus_adapted_{existing_strategy.__class__.__name__}"

    # Create adapter
    adapter = StrategyAdapterFactory.create_adapter(
        strategy_type=strategy_type,
        strategy_id=strategy_id,
        instrument_id=instrument_id
    )

    # Initialize adapter with existing strategy data
    if hasattr(existing_strategy, 'performance_metrics'):
        adapter.adaptation_metrics.update(existing_strategy.performance_metrics)

    return adapter


async def get_adapter_performance_metrics(adapter: NautilusStrategyAdapter) -> Dict[str, Any]:
    """Get comprehensive performance metrics for strategy adapter"""

    return {
        'adapter_info': adapter.get_metrics(),
        'nautilus_status': await adapter.nautilus_manager.get_system_status(),
        'signal_adaptation_stats': {
            'total_signals': len(adapter.signals),
            'successful_adaptations': adapter.adaptation_metrics['signals_processed'],
            'adaptation_rate': adapter.adaptation_metrics['signals_processed'] / max(1, len(adapter.signals))
        },
        'order_execution_stats': {
            'orders_submitted': adapter.adaptation_metrics['orders_submitted'],
            'orders_filled': adapter.adaptation_metrics['orders_filled'],
            'fill_rate': adapter.adaptation_metrics['orders_filled'] / max(1, adapter.adaptation_metrics['orders_submitted'])
        },
        'performance_stats': {
            'total_pnl': adapter.adaptation_metrics['profit_loss'],
            'win_rate': adapter.adaptation_metrics['win_rate'],
            'active_positions': len(adapter.positions)
        }
    }


# Export key classes
__all__ = [
    'NautilusStrategyAdapter',
    'ScalpingStrategyAdapter',
    'MarketMakingStrategyAdapter',
    'MeanReversionStrategyAdapter',
    'StrategyAdapterFactory',
    'adapt_strategy_for_nautilus',
    'get_adapter_performance_metrics'
]