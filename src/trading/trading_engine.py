"""
Trading Engine Module
====================

Basic trading engine interface for integration with risk management and strategy systems.
This module provides the core trading engine class that integrates with adaptive risk
management and dynamic strategy switching.

Key Features:
- Integration with AdaptiveRiskManager
- Dynamic strategy switching support
- Position management
- Order execution interface

Author: Integration Team
Date: 2025-08-25
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class TradingEngine:
    """
    Core trading engine for executing trades and managing positions.
    Integrates with adaptive risk management and dynamic strategy switching.
    """
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        
        # Integration components
        self.risk_manager = None
        self.strategy_manager = None
        
        # Status
        self.is_running = False
        self.last_update_time = time.time()
        
        logger.info("Trading Engine initialized")
    
    def set_risk_manager(self, risk_manager) -> None:
        """Set the adaptive risk manager"""
        self.risk_manager = risk_manager
        logger.info("Risk manager integrated with trading engine")
    
    def set_strategy_manager(self, strategy_manager) -> None:
        """Set the dynamic strategy manager"""
        self.strategy_manager = strategy_manager
        logger.info("Strategy manager integrated with trading engine")
    
    def start(self) -> bool:
        """Start the trading engine"""
        try:
            self.is_running = True
            self.last_update_time = time.time()
            logger.info("Trading engine started")
            return True
        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the trading engine"""
        try:
            self.is_running = False
            logger.info("Trading engine stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop trading engine: {e}")
            return False
    
    def submit_order(self, order: Order) -> Optional[str]:
        """Submit a trading order"""
        if not self.is_running:
            logger.warning("Cannot submit order: engine not running")
            return None
        
        # Check risk limits if risk manager is available
        if self.risk_manager:
            try:
                # Simulate risk check
                if hasattr(self.risk_manager, 'check_order_risk'):
                    risk_approved = self.risk_manager.check_order_risk(order)
                    if not risk_approved:
                        logger.warning(f"Order rejected by risk manager: {order}")
                        order.status = OrderStatus.REJECTED
                        return None
            except Exception as e:
                logger.error(f"Risk check failed: {e}")
        
        # Generate order ID
        self.order_counter += 1
        order.order_id = f"ORD_{self.order_counter:06d}"
        
        # Store order
        self.orders[order.order_id] = order
        
        # Simulate order execution (in real implementation, this would be async)
        self._simulate_order_execution(order)
        
        logger.info(f"Order submitted: {order.order_id}")
        return order.order_id
    
    def _simulate_order_execution(self, order: Order) -> None:
        """Simulate order execution (for testing purposes)"""
        try:
            # Simulate successful execution
            order.status = OrderStatus.FILLED
            
            # Update positions
            existing_position = self.positions.get(order.symbol)
            
            if existing_position:
                # Update existing position
                if order.side == OrderSide.BUY:
                    new_quantity = existing_position.quantity + order.quantity
                else:
                    new_quantity = existing_position.quantity - order.quantity
                
                if new_quantity != 0:
                    # Update position
                    avg_price = ((existing_position.quantity * existing_position.entry_price) + 
                               (order.quantity * (order.price or existing_position.current_price))) / (
                               existing_position.quantity + order.quantity)
                    existing_position.quantity = new_quantity
                    existing_position.entry_price = avg_price
                else:
                    # Close position
                    del self.positions[order.symbol]
            else:
                # Create new position
                if order.quantity > 0:
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        quantity=order.quantity if order.side == OrderSide.BUY else -order.quantity,
                        entry_price=order.price or 50000,  # Default price for simulation
                        current_price=order.price or 50000
                    )
            
            logger.info(f"Order executed: {order.order_id}")
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            order.status = OrderStatus.REJECTED
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
    
    def close_position(self, symbol: str) -> bool:
        """Close position for specific symbol"""
        position = self.positions.get(symbol)
        if not position:
            logger.warning(f"No position found for {symbol}")
            return False
        
        # Create closing order
        side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
        close_order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=abs(position.quantity)
        )
        
        order_id = self.submit_order(close_order)
        return order_id is not None
    
    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """Update market prices for position valuation"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                
                # Calculate unrealized PnL
                if position.quantity > 0:
                    position.unrealized_pnl = (price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - price) * abs(position.quantity)
        
        self.last_update_time = time.time()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status"""
        total_positions = len(self.positions)
        total_orders = len(self.orders)
        active_orders = len([o for o in self.orders.values() if o.status == OrderStatus.PENDING])
        
        return {
            "is_running": self.is_running,
            "total_positions": total_positions,
            "total_orders": total_orders,
            "active_orders": active_orders,
            "last_update_time": self.last_update_time,
            "risk_manager_connected": self.risk_manager is not None,
            "strategy_manager_connected": self.strategy_manager is not None
        }


# Factory function
def create_trading_engine() -> TradingEngine:
    """Factory function to create trading engine"""
    return TradingEngine()


# Demo and testing
if __name__ == "__main__":
    print("🏗️ Trading Engine Demo")
    print("=" * 30)
    
    # Create trading engine
    engine = create_trading_engine()
    
    # Start engine
    engine.start()
    
    # Create sample order
    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1
    )
    
    # Submit order
    order_id = engine.submit_order(order)
    print(f"Order submitted: {order_id}")
    
    # Check status
    status = engine.get_engine_status()
    print(f"Engine status: {status}")
    
    # Update prices
    engine.update_market_prices({"BTCUSDT": 51000})
    
    # Check positions
    positions = engine.get_positions()
    print(f"Positions: {positions}")
    
    print("\n🎯 Trading Engine - Basic functionality validated")