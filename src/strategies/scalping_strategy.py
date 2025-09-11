"""
Scalping Strategy Implementation
===============================

High-frequency scalping strategy optimized for crypto futures trading.
Focuses on capturing small price movements with tight spreads and quick execution.

Key Features:
- Ultra-low latency execution optimized for scalping
- Dynamic spread management based on volatility
- Risk-aware position sizing
- Integration with adaptive risk management
- Market microstructure analysis

Author: Trading Strategy Team
Date: 2025-08-25
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ScalpingMode(Enum):
    """Scalping operation modes"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


class MarketMicrostructure(Enum):
    """Market microstructure conditions"""
    TIGHT_SPREAD = "tight_spread"
    WIDE_SPREAD = "wide_spread"
    HIGH_VOLUME = "high_volume"
    LOW_VOLUME = "low_volume"
    TRENDING = "trending"
    CHOPPY = "choppy"


@dataclass
class ScalpingConfig:
    """Configuration for scalping strategy"""
    mode: ScalpingMode = ScalpingMode.BALANCED
    max_position_size: float = 0.1
    target_spread_bps: float = 2.0  # Target spread in basis points
    max_hold_time: float = 30.0     # Maximum hold time in seconds
    profit_target_bps: float = 5.0  # Profit target in basis points
    stop_loss_bps: float = 3.0      # Stop loss in basis points
    min_volume_threshold: float = 1000.0  # Minimum volume threshold
    max_slippage_bps: float = 1.0   # Maximum acceptable slippage
    
    # Risk management
    daily_loss_limit: float = 0.02  # Daily loss limit as fraction of capital
    max_consecutive_losses: int = 3
    cooldown_period: float = 60.0   # Cooldown after max losses


@dataclass
class ScalpingSignal:
    """Scalping trading signal"""
    direction: int  # 1 for long, -1 for short, 0 for neutral
    confidence: float
    size: float
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    urgency: float = 0.5  # 0 to 1, higher means more urgent
    expected_hold_time: float = 15.0  # Expected hold time in seconds
    microstructure: Optional[MarketMicrostructure] = None


@dataclass
class MarketData:
    """Market data for scalping analysis"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last_price: float
    volume: float
    bid_size: float
    ask_size: float
    spread_bps: float = 0.0
    
    def __post_init__(self):
        if self.bid > 0 and self.ask > 0:
            self.spread_bps = ((self.ask - self.bid) / self.last_price) * 10000


class ScalpingStrategy:
    """
    Advanced scalping strategy for high-frequency crypto trading.
    Optimized for capturing small price movements with minimal risk.
    """
    
    def __init__(self, config: ScalpingConfig):
        self.config = config
        self.is_active = False
        
        # Performance tracking
        self.trades_today = 0
        self.pnl_today = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = None
        
        # Market analysis
        self.price_history = []
        self.volume_history = []
        self.spread_history = []
        
        # Risk management
        self.daily_loss_exceeded = False
        self.in_cooldown = False
        self.cooldown_end_time = None
        
        logger.info(f"Scalping strategy initialized with mode: {config.mode.value}")
    
    def analyze_market_microstructure(self, market_data: MarketData) -> MarketMicrostructure:
        """Analyze market microstructure conditions"""
        
        # Analyze spread
        if market_data.spread_bps < self.config.target_spread_bps:
            spread_condition = MarketMicrostructure.TIGHT_SPREAD
        else:
            spread_condition = MarketMicrostructure.WIDE_SPREAD
        
        # Analyze volume
        if market_data.volume > self.config.min_volume_threshold:
            volume_condition = MarketMicrostructure.HIGH_VOLUME
        else:
            volume_condition = MarketMicrostructure.LOW_VOLUME
        
        # Simple price trend analysis
        if len(self.price_history) >= 10:
            recent_prices = self.price_history[-10:]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            if abs(price_trend) > 0.001:  # 0.1% movement
                trend_condition = MarketMicrostructure.TRENDING
            else:
                trend_condition = MarketMicrostructure.CHOPPY
        else:
            trend_condition = MarketMicrostructure.CHOPPY
        
        # Return the most restrictive condition
        if spread_condition == MarketMicrostructure.WIDE_SPREAD:
            return spread_condition
        elif volume_condition == MarketMicrostructure.LOW_VOLUME:
            return volume_condition
        else:
            return trend_condition
    
    def calculate_position_size(self, market_data: MarketData, 
                              microstructure: MarketMicrostructure) -> float:
        """Calculate optimal position size based on market conditions"""
        
        base_size = self.config.max_position_size
        
        # Adjust based on microstructure
        if microstructure == MarketMicrostructure.TIGHT_SPREAD:
            size_multiplier = 1.2
        elif microstructure == MarketMicrostructure.WIDE_SPREAD:
            size_multiplier = 0.6
        elif microstructure == MarketMicrostructure.HIGH_VOLUME:
            size_multiplier = 1.1
        elif microstructure == MarketMicrostructure.LOW_VOLUME:
            size_multiplier = 0.7
        else:
            size_multiplier = 0.8
        
        # Adjust based on recent performance
        if self.consecutive_losses > 0:
            size_multiplier *= (0.8 ** self.consecutive_losses)
        
        # Adjust based on spread quality
        if market_data.spread_bps > self.config.target_spread_bps * 2:
            size_multiplier *= 0.5
        
        return min(base_size * size_multiplier, self.config.max_position_size)
    
    def generate_signal(self, market_data: MarketData) -> Optional[ScalpingSignal]:
        """Generate scalping signal based on market conditions"""
        
        # Check if strategy is active and not in cooldown
        if not self.is_active or self.in_cooldown:
            return None
        
        # Check daily loss limit
        if self.daily_loss_exceeded:
            return None
        
        # Update market analysis
        self.price_history.append(market_data.last_price)
        self.volume_history.append(market_data.volume)
        self.spread_history.append(market_data.spread_bps)
        
        # Keep only recent history
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]
            self.volume_history = self.volume_history[-100:]
            self.spread_history = self.spread_history[-100:]
        
        # Analyze market microstructure
        microstructure = self.analyze_market_microstructure(market_data)
        
        # Skip trading in unfavorable conditions
        if microstructure in [MarketMicrostructure.WIDE_SPREAD, MarketMicrostructure.LOW_VOLUME]:
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(market_data, microstructure)
        
        if position_size < 0.01:  # Minimum position size
            return None
        
        # Generate signal based on market conditions
        signal = self._generate_scalping_signal(market_data, microstructure, position_size)
        
        return signal
    
    def _generate_scalping_signal(self, market_data: MarketData, 
                                 microstructure: MarketMicrostructure,
                                 position_size: float) -> Optional[ScalpingSignal]:
        """Generate specific scalping signal based on strategy mode"""
        
        if self.config.mode == ScalpingMode.AGGRESSIVE:
            return self._aggressive_scalping_signal(market_data, microstructure, position_size)
        elif self.config.mode == ScalpingMode.CONSERVATIVE:
            return self._conservative_scalping_signal(market_data, microstructure, position_size)
        elif self.config.mode == ScalpingMode.BALANCED:
            return self._balanced_scalping_signal(market_data, microstructure, position_size)
        elif self.config.mode == ScalpingMode.ADAPTIVE:
            return self._adaptive_scalping_signal(market_data, microstructure, position_size)
        else:
            return None
    
    def _aggressive_scalping_signal(self, market_data: MarketData,
                                   microstructure: MarketMicrostructure,
                                   position_size: float) -> Optional[ScalpingSignal]:
        """Aggressive scalping signal - quick entries and exits"""
        
        # Look for momentum in tight spread conditions
        if len(self.price_history) < 5:
            return None
        
        recent_prices = self.price_history[-5:]
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        if abs(price_momentum) < 0.0005:  # Minimum momentum threshold
            return None
        
        direction = 1 if price_momentum > 0 else -1
        confidence = min(abs(price_momentum) * 1000, 0.8)  # Scale momentum to confidence
        
        # Aggressive targets
        profit_target_bps = self.config.profit_target_bps * 0.8
        stop_loss_bps = self.config.stop_loss_bps * 0.6
        
        entry_price = market_data.ask if direction > 0 else market_data.bid
        target_price = entry_price * (1 + direction * profit_target_bps / 10000)
        stop_price = entry_price * (1 - direction * stop_loss_bps / 10000)
        
        return ScalpingSignal(
            direction=direction,
            confidence=confidence,
            size=position_size * 1.2,  # Larger size for aggressive mode
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            urgency=0.8,
            expected_hold_time=10.0,
            microstructure=microstructure
        )
    
    def _conservative_scalping_signal(self, market_data: MarketData,
                                     microstructure: MarketMicrostructure,
                                     position_size: float) -> Optional[ScalpingSignal]:
        """Conservative scalping signal - patient entries with good risk/reward"""
        
        # Only trade in very favorable conditions
        if microstructure != MarketMicrostructure.TIGHT_SPREAD:
            return None
        
        if len(self.price_history) < 20:
            return None
        
        # Look for mean reversion opportunities
        recent_prices = self.price_history[-20:]
        mean_price = np.mean(recent_prices)
        current_deviation = (market_data.last_price - mean_price) / mean_price
        
        if abs(current_deviation) < 0.001:  # Minimum deviation
            return None
        
        # Mean reversion signal
        direction = -1 if current_deviation > 0 else 1
        confidence = min(abs(current_deviation) * 500, 0.6)
        
        # Conservative targets
        profit_target_bps = self.config.profit_target_bps * 1.5
        stop_loss_bps = self.config.stop_loss_bps * 0.8
        
        entry_price = market_data.ask if direction > 0 else market_data.bid
        target_price = entry_price * (1 + direction * profit_target_bps / 10000)
        stop_price = entry_price * (1 - direction * stop_loss_bps / 10000)
        
        return ScalpingSignal(
            direction=direction,
            confidence=confidence,
            size=position_size * 0.8,  # Smaller size for conservative mode
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            urgency=0.3,
            expected_hold_time=25.0,
            microstructure=microstructure
        )
    
    def _balanced_scalping_signal(self, market_data: MarketData,
                                 microstructure: MarketMicrostructure,
                                 position_size: float) -> Optional[ScalpingSignal]:
        """Balanced scalping signal - combination of momentum and mean reversion"""
        
        if len(self.price_history) < 10:
            return None
        
        # Analyze both momentum and mean reversion
        short_term_prices = self.price_history[-5:]
        medium_term_prices = self.price_history[-10:]
        
        momentum = (short_term_prices[-1] - short_term_prices[0]) / short_term_prices[0]
        mean_price = np.mean(medium_term_prices)
        deviation = (market_data.last_price - mean_price) / mean_price
        
        # Combined signal
        signal_strength = abs(momentum) + abs(deviation) * 0.5
        
        if signal_strength < 0.0008:
            return None
        
        # Direction based on stronger signal
        if abs(momentum) > abs(deviation):
            direction = 1 if momentum > 0 else -1
            confidence = min(signal_strength * 800, 0.7)
        else:
            direction = -1 if deviation > 0 else 1
            confidence = min(signal_strength * 600, 0.6)
        
        entry_price = market_data.ask if direction > 0 else market_data.bid
        target_price = entry_price * (1 + direction * self.config.profit_target_bps / 10000)
        stop_price = entry_price * (1 - direction * self.config.stop_loss_bps / 10000)
        
        return ScalpingSignal(
            direction=direction,
            confidence=confidence,
            size=position_size,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            urgency=0.5,
            expected_hold_time=15.0,
            microstructure=microstructure
        )
    
    def _adaptive_scalping_signal(self, market_data: MarketData,
                                 microstructure: MarketMicrostructure,
                                 position_size: float) -> Optional[ScalpingSignal]:
        """Adaptive scalping signal - adjusts to current market conditions"""
        
        # Adapt strategy based on recent performance
        if self.pnl_today > 0:
            # If profitable, use balanced approach
            return self._balanced_scalping_signal(market_data, microstructure, position_size)
        elif self.consecutive_losses == 0:
            # If no recent losses, try aggressive
            return self._aggressive_scalping_signal(market_data, microstructure, position_size)
        else:
            # If recent losses, be conservative
            return self._conservative_scalping_signal(market_data, microstructure, position_size)
    
    def update_trade_result(self, pnl: float, trade_duration: float) -> None:
        """Update strategy performance with trade result"""
        
        self.trades_today += 1
        self.pnl_today += pnl
        self.last_trade_time = time.time()
        
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check daily loss limit
        if abs(self.pnl_today) > self.config.daily_loss_limit:
            self.daily_loss_exceeded = True
            self.is_active = False
            logger.warning(f"Daily loss limit exceeded: {self.pnl_today:.4f}")
        
        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            self.enter_cooldown()
        
        logger.info(f"Trade result: PnL={pnl:.4f}, Duration={trade_duration:.1f}s, "
                   f"Consecutive losses={self.consecutive_losses}")
    
    def enter_cooldown(self) -> None:
        """Enter cooldown period after consecutive losses"""
        self.in_cooldown = True
        self.cooldown_end_time = time.time() + self.config.cooldown_period
        logger.info(f"Entering cooldown for {self.config.cooldown_period} seconds")
    
    def check_cooldown(self) -> None:
        """Check if cooldown period has ended"""
        if self.in_cooldown and time.time() > self.cooldown_end_time:
            self.in_cooldown = False
            self.consecutive_losses = 0
            logger.info("Cooldown period ended, resuming trading")
    
    def start(self) -> None:
        """Start the scalping strategy"""
        self.is_active = True
        self.daily_loss_exceeded = False
        logger.info("Scalping strategy started")
    
    def stop(self) -> None:
        """Stop the scalping strategy"""
        self.is_active = False
        logger.info("Scalping strategy stopped")
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of each trading day)"""
        self.trades_today = 0
        self.pnl_today = 0.0
        self.consecutive_losses = 0
        self.daily_loss_exceeded = False
        self.in_cooldown = False
        logger.info("Daily statistics reset")
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy status"""
        return {
            "is_active": self.is_active,
            "mode": self.config.mode.value,
            "trades_today": self.trades_today,
            "pnl_today": self.pnl_today,
            "consecutive_losses": self.consecutive_losses,
            "in_cooldown": self.in_cooldown,
            "daily_loss_exceeded": self.daily_loss_exceeded,
            "cooldown_end_time": self.cooldown_end_time,
            "last_trade_time": self.last_trade_time
        }


# Factory function
def create_scalping_strategy(mode: ScalpingMode = ScalpingMode.BALANCED) -> ScalpingStrategy:
    """Factory function to create scalping strategy"""
    config = ScalpingConfig(mode=mode)
    return ScalpingStrategy(config)


# Demo and testing
if __name__ == "__main__":
    print("⚡ Scalping Strategy Demo")
    print("=" * 30)
    
    # Create strategy
    strategy = create_scalping_strategy(ScalpingMode.BALANCED)
    strategy.start()
    
    # Create sample market data
    market_data = MarketData(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        bid=50000.0,
        ask=50002.0,
        last_price=50001.0,
        volume=1500.0,
        bid_size=2.5,
        ask_size=3.0
    )
    
    # Generate signal
    signal = strategy.generate_signal(market_data)
    
    if signal:
        print(f"Signal generated:")
        print(f"  Direction: {signal.direction}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Size: {signal.size:.3f}")
        print(f"  Entry: {signal.entry_price}")
        print(f"  Target: {signal.target_price}")
        print(f"  Stop: {signal.stop_price}")
    else:
        print("No signal generated")
    
    # Show strategy status
    status = strategy.get_strategy_status()
    print(f"\nStrategy Status: {status}")
    
    print("\n⚡ Scalping Strategy - Implementation complete")