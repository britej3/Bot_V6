"""
Trailing Take Profit System for Autonomous Crypto Scalping
==========================================================

This module implements an intelligent trailing take profit system that dynamically
adjusts profit targets based on market momentum, volatility, and tick-level price action.
Designed for ultra-high frequency crypto scalping with microsecond precision.

Key Features:
- Dynamic trailing based on market momentum
- Tick-level precision adjustments
- Volatility-adaptive profit targets
- Integration with ML model predictions
- Multi-strategy profit optimization
- Real-time profit lock mechanisms

Author: Autonomous Systems Team
Date: 2025-01-22
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time
import statistics
from datetime import datetime, timedelta
import math

# Import existing components
try:
    from .dynamic_leveraging_system import (
        DynamicLeverageManager, LeverageDecision, TickData, 
        MarketMicrostructure, LeverageMode
    )
    from .adaptive_risk_management import (
        AdaptiveRiskManager, MarketCondition, MarketRegime, 
        PortfolioRiskMetrics, RiskLevel
    )
    RISK_COMPONENTS_AVAILABLE = True
except ImportError:
    # Define placeholder types when imports not available
    DynamicLeverageManager = Any
    LeverageDecision = Any
    TickData = Any
    MarketMicrostructure = Any
    LeverageMode = Any
    AdaptiveRiskManager = Any
    MarketCondition = Any
    MarketRegime = Any
    PortfolioRiskMetrics = Any
    RiskLevel = Any
    RISK_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TrailingMode(Enum):
    """Different trailing take profit modes"""
    FIXED_PERCENTAGE = "fixed_percentage"      # Fixed % from entry
    ATR_BASED = "atr_based"                   # ATR-based trailing
    MOMENTUM_ADAPTIVE = "momentum_adaptive"    # Adapt to momentum
    VOLATILITY_SCALED = "volatility_scaled"    # Scale with volatility
    ML_PREDICTED = "ml_predicted"             # ML model predictions
    TICK_OPTIMIZED = "tick_optimized"         # Tick-level optimization


class ProfitLockLevel(Enum):
    """Profit locking levels"""
    NONE = "none"                 # No profit locked
    PARTIAL_25 = "partial_25"     # 25% profit locked
    PARTIAL_50 = "partial_50"     # 50% profit locked
    PARTIAL_75 = "partial_75"     # 75% profit locked
    FULL_BREAKEVEN = "full_breakeven"  # Locked at breakeven
    GUARANTEED = "guaranteed"      # Guaranteed profit locked


@dataclass
class TrailingConfig:
    """Configuration for trailing take profit system"""
    # Base trailing settings
    initial_profit_target: float = 0.003      # 0.3% initial target
    minimum_profit_target: float = 0.001      # 0.1% minimum
    maximum_profit_target: float = 0.02       # 2% maximum
    
    # Trailing parameters
    trailing_percentage: float = 0.3          # 30% of profit to trail
    acceleration_factor: float = 0.02         # Acceleration on momentum
    deceleration_factor: float = 0.05         # Deceleration on reversal
    
    # Tick-level settings
    tick_sensitivity: float = 0.0001          # Minimum tick movement
    momentum_window: int = 10                 # Ticks for momentum calculation
    reversal_detection_ticks: int = 5         # Ticks to detect reversal
    
    # Profit locking
    profit_lock_threshold: float = 0.002      # Lock profit above 0.2%
    partial_lock_percentage: float = 0.5      # Lock 50% of position
    breakeven_buffer: float = 0.0005          # Buffer above entry
    
    # ML integration
    use_ml_predictions: bool = True           # Use ML for profit targets
    prediction_weight: float = 0.4           # Weight of ML predictions
    confidence_threshold: float = 0.7        # Minimum ML confidence


@dataclass
class TrailingState:
    """Current state of trailing take profit"""
    position_id: str
    entry_price: float
    current_price: float
    initial_target: float
    current_target: float
    highest_profit: float
    locked_profit: float
    lock_level: ProfitLockLevel
    trailing_active: bool
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class ProfitDecision:
    """Decision result from trailing system"""
    action: str                    # "HOLD", "TAKE_PARTIAL", "TAKE_FULL"
    target_price: float
    profit_percentage: float
    confidence: float
    reasoning: str
    lock_level: ProfitLockLevel
    timestamp: datetime = field(default_factory=datetime.now)


class MomentumAnalyzer:
    """Analyzes price momentum for trailing adjustments"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)
        self.momentum_history = deque(maxlen=window_size)
        
    def update_price(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update with new price data"""
        self.price_history.append({
            'price': price,
            'timestamp': timestamp or datetime.now()
        })
        
        # Calculate momentum
        if len(self.price_history) >= 2:
            momentum = self._calculate_momentum()
            self.momentum_history.append(momentum)
    
    def _calculate_momentum(self) -> float:
        """Calculate current price momentum"""
        if len(self.price_history) < 2:
            return 0.0
        
        # Simple momentum: rate of price change
        recent_prices = [p['price'] for p in list(self.price_history)[-5:]]
        if len(recent_prices) < 2:
            return 0.0
        
        # Linear regression slope as momentum indicator
        x = np.arange(len(recent_prices))
        y = np.array(recent_prices)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope / recent_prices[-1]  # Normalized momentum
        return 0.0
    
    def get_momentum_strength(self) -> float:
        """Get current momentum strength (0-1)"""
        if not self.momentum_history:
            return 0.0
        
        current_momentum = self.momentum_history[-1]
        return min(1.0, abs(current_momentum) * 1000)  # Scale momentum
    
    def detect_reversal(self) -> bool:
        """Detect momentum reversal"""
        if len(self.momentum_history) < 3:
            return False
        
        recent_momentum = list(self.momentum_history)[-3:]
        
        # Check for sign change in momentum
        return (recent_momentum[0] > 0 and recent_momentum[-1] < 0) or \
               (recent_momentum[0] < 0 and recent_momentum[-1] > 0)


class VolatilityTracker:
    """Tracks market volatility for adaptive trailing"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
        
    def update_price(self, price: float) -> None:
        """Update with new price and calculate return"""
        if hasattr(self, 'last_price') and self.last_price:
            log_return = np.log(price / self.last_price)
            self.returns.append(log_return)
        self.last_price = price
    
    def get_current_volatility(self) -> float:
        """Get current volatility estimate"""
        if len(self.returns) < 10:
            return 0.02  # Default volatility
        
        returns_array = np.array(self.returns)
        return float(np.std(returns_array) * np.sqrt(1440))  # Minute-based annualized


class MLProfitPredictor:
    """ML model for predicting optimal profit targets"""
    
    def __init__(self):
        self.prediction_cache = {}
        self.last_prediction_time = None
        
    def predict_optimal_target(self, 
                             current_state: TrailingState,
                             market_condition: MarketCondition,
                             tick_data: Optional[TickData] = None) -> Tuple[float, float]:
        """Predict optimal profit target using ML features"""
        
        # Extract features for ML prediction
        features = self._extract_features(current_state, market_condition, tick_data)
        
        # Simple ML prediction (would be replaced with trained model)
        base_target = current_state.initial_target
        
        # Adjust based on market regime
        regime_multiplier = self._get_regime_multiplier(market_condition.regime)
        
        # Adjust based on volatility
        volatility_multiplier = 1.0 + market_condition.volatility * 2
        
        # Calculate predicted target
        predicted_target = base_target * regime_multiplier * volatility_multiplier
        
        # Confidence based on market conditions
        confidence = market_condition.confidence * 0.8  # Conservative
        
        return predicted_target, confidence
    
    def _extract_features(self, state: TrailingState, condition: MarketCondition, tick: Optional[TickData]) -> np.ndarray:
        """Extract features for ML model"""
        features = [
            state.current_price / state.entry_price - 1,  # Current profit
            state.highest_profit,                         # Max profit achieved
            condition.volatility,                         # Market volatility
            condition.trend_strength,                     # Trend strength
            condition.confidence                          # Market confidence
        ]
        
        if tick:
            features.extend([
                tick.spread / tick.mid_price,             # Relative spread
                tick.bid_size / (tick.bid_size + tick.ask_size),  # Order imbalance
                tick.volume                               # Current volume
            ])
        else:
            features.extend([0.001, 0.5, 1000])         # Default values
        
        return np.array(features)
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get profit target multiplier based on market regime"""
        multipliers = {
            MarketRegime.TRENDING: 1.5,      # Higher targets in trends
            MarketRegime.VOLATILE: 0.7,      # Lower targets in volatility
            MarketRegime.BULL_RUN: 1.8,      # Highest targets in bull runs
            MarketRegime.CRASH: 0.3,         # Minimal targets in crashes
            MarketRegime.RANGE_BOUND: 0.9,   # Moderate targets in ranges
            MarketRegime.NORMAL: 1.0,        # Base targets
            MarketRegime.RECOVERY: 1.2       # Moderate increase in recovery
        }
        return multipliers.get(regime, 1.0)


class TrailingTakeProfitSystem:
    """Main trailing take profit system"""
    
    def __init__(self, 
                 config: Optional[TrailingConfig] = None,
                 leverage_manager: Optional[DynamicLeverageManager] = None,
                 risk_manager: Optional[AdaptiveRiskManager] = None):
        
        self.config = config or TrailingConfig()
        self.leverage_manager = leverage_manager
        self.risk_manager = risk_manager
        
        # Initialize components
        self.momentum_analyzer = MomentumAnalyzer()
        self.volatility_tracker = VolatilityTracker()
        self.ml_predictor = MLProfitPredictor()
        
        # Active positions tracking
        self.active_positions: Dict[str, TrailingState] = {}
        self.decision_history = deque(maxlen=1000)
        
        # Performance tracking
        self.profit_statistics = {
            'total_profits_taken': 0,
            'average_profit': 0.0,
            'maximum_profit': 0.0,
            'profit_efficiency': 0.0
        }
        
        logger.info("Trailing Take Profit System initialized")
    
    def add_position(self, 
                    position_id: str,
                    entry_price: float,
                    initial_target: Optional[float] = None,
                    market_condition: Optional[MarketCondition] = None) -> TrailingState:
        """Add new position for trailing management"""
        
        # Calculate initial target
        if initial_target is None:
            if market_condition:
                initial_target = self._calculate_adaptive_target(entry_price, market_condition)
            else:
                initial_target = entry_price * (1 + self.config.initial_profit_target)
        
        # Create trailing state
        trailing_state = TrailingState(
            position_id=position_id,
            entry_price=entry_price,
            current_price=entry_price,
            initial_target=initial_target,
            current_target=initial_target,
            highest_profit=0.0,
            locked_profit=0.0,
            lock_level=ProfitLockLevel.NONE,
            trailing_active=False
        )
        
        self.active_positions[position_id] = trailing_state
        
        logger.info(f"Added position {position_id} for trailing: entry=${entry_price:.4f}, target=${initial_target:.4f}")
        
        return trailing_state
    
    def update_position(self,
                       position_id: str,
                       current_price: float,
                       market_condition: Optional[MarketCondition] = None,
                       tick_data: Optional[TickData] = None,
                       mode: TrailingMode = TrailingMode.MOMENTUM_ADAPTIVE) -> ProfitDecision:
        """Update position and get trailing decision"""
        
        if position_id not in self.active_positions:
            raise ValueError(f"Position {position_id} not found in trailing system")
        
        state = self.active_positions[position_id]
        
        # Update position state
        state.current_price = current_price
        state.last_update = datetime.now()
        
        # Update momentum and volatility trackers
        self.momentum_analyzer.update_price(current_price)
        self.volatility_tracker.update_price(current_price)
        
        # Calculate current profit
        current_profit = (current_price - state.entry_price) / state.entry_price
        state.highest_profit = max(state.highest_profit, current_profit)
        
        # Get trailing decision based on mode
        decision = self._calculate_trailing_decision(state, market_condition, tick_data, mode)
        
        # Update trailing target
        state.current_target = decision.target_price
        
        # Handle profit locking
        self._handle_profit_locking(state, decision)
        
        # Store decision
        self.decision_history.append(decision)
        
        return decision
    
    def _calculate_adaptive_target(self, entry_price: float, market_condition: MarketCondition) -> float:
        """Calculate adaptive initial profit target"""
        
        base_target = self.config.initial_profit_target
        
        # Adjust for market regime
        if market_condition.regime == MarketRegime.TRENDING:
            base_target *= 1.5
        elif market_condition.regime == MarketRegime.VOLATILE:
            base_target *= 0.7
        elif market_condition.regime == MarketRegime.BULL_RUN:
            base_target *= 2.0
        elif market_condition.regime == MarketRegime.CRASH:
            base_target *= 0.3
        
        # Adjust for volatility
        volatility_factor = 1.0 + market_condition.volatility
        base_target *= volatility_factor
        
        # Apply limits
        base_target = max(self.config.minimum_profit_target,
                         min(self.config.maximum_profit_target, base_target))
        
        return entry_price * (1 + base_target)
    
    def _calculate_trailing_decision(self,
                                   state: TrailingState,
                                   market_condition: Optional[MarketCondition],
                                   tick_data: Optional[TickData],
                                   mode: TrailingMode) -> ProfitDecision:
        """Calculate trailing decision based on specified mode"""
        
        current_profit = (state.current_price - state.entry_price) / state.entry_price
        
        if mode == TrailingMode.MOMENTUM_ADAPTIVE:
            return self._momentum_adaptive_trailing(state, current_profit, market_condition)
        elif mode == TrailingMode.VOLATILITY_SCALED:
            return self._volatility_scaled_trailing(state, current_profit, market_condition)
        elif mode == TrailingMode.ML_PREDICTED and self.config.use_ml_predictions:
            return self._ml_predicted_trailing(state, current_profit, market_condition, tick_data)
        elif mode == TrailingMode.TICK_OPTIMIZED:
            return self._tick_optimized_trailing(state, current_profit, tick_data)
        else:
            return self._fixed_percentage_trailing(state, current_profit)
    
    def _volatility_scaled_trailing(self,
                                   state: TrailingState,
                                   current_profit: float,
                                   market_condition: Optional[Any]) -> ProfitDecision:
        """Trailing scaled by market volatility"""
        
        if market_condition and hasattr(market_condition, 'volatility'):
            volatility = market_condition.volatility
        else:
            volatility = 0.02  # Default volatility
        
        # Scale trailing distance with volatility
        base_trailing = self.config.trailing_percentage
        volatility_factor = 1.0 + volatility * 5  # Higher volatility = looser trailing
        trailing_distance = base_trailing * volatility_factor
        
        if current_profit > 0:
            pullback_target = state.current_price * (1 - trailing_distance)
            new_target = max(pullback_target, state.entry_price * (1 + self.config.minimum_profit_target))
        else:
            new_target = state.initial_target
        
        return ProfitDecision(
            action="HOLD",
            target_price=new_target,
            profit_percentage=current_profit,
            confidence=0.6,
            reasoning=f"Volatility-scaled trailing (vol: {volatility:.3f})",
            lock_level=ProfitLockLevel.NONE
        )
    
    def _momentum_adaptive_trailing(self,
                                  state: TrailingState,
                                  current_profit: float,
                                  market_condition: Optional[MarketCondition]) -> ProfitDecision:
        """Trailing based on momentum analysis"""
        
        momentum_strength = self.momentum_analyzer.get_momentum_strength()
        reversal_detected = self.momentum_analyzer.detect_reversal()
        
        # Calculate adaptive trailing distance
        base_trailing = self.config.trailing_percentage
        
        if momentum_strength > 0.7:
            # Strong momentum, trail closer
            trailing_distance = base_trailing * 0.5
            confidence = 0.8
            reasoning = f"Strong momentum ({momentum_strength:.2f}), tight trailing"
        elif reversal_detected:
            # Reversal detected, consider taking profit
            trailing_distance = base_trailing * 2.0
            confidence = 0.9
            reasoning = "Momentum reversal detected, loose trailing"
        else:
            # Normal momentum, standard trailing
            trailing_distance = base_trailing
            confidence = 0.6
            reasoning = "Normal momentum, standard trailing"
        
        # Calculate new target
        if current_profit > 0:
            pullback_target = state.current_price * (1 - trailing_distance)
            new_target = max(pullback_target, state.entry_price * (1 + self.config.minimum_profit_target))
        else:
            new_target = state.initial_target
        
        # Determine action
        action = "HOLD"
        if current_profit >= self.config.profit_lock_threshold:
            if reversal_detected:
                action = "TAKE_PARTIAL"
            elif momentum_strength < 0.3:
                action = "TAKE_FULL"
        
        return ProfitDecision(
            action=action,
            target_price=new_target,
            profit_percentage=current_profit,
            confidence=confidence,
            reasoning=reasoning,
            lock_level=ProfitLockLevel.NONE
        )
    
    def _ml_predicted_trailing(self,
                             state: TrailingState,
                             current_profit: float,
                             market_condition: Optional[MarketCondition],
                             tick_data: Optional[TickData]) -> ProfitDecision:
        """Trailing based on ML predictions"""
        
        if not market_condition:
            return self._fixed_percentage_trailing(state, current_profit)
        
        # Get ML prediction
        predicted_target, ml_confidence = self.ml_predictor.predict_optimal_target(
            state, market_condition, tick_data
        )
        
        # Combine with traditional trailing
        traditional_decision = self._momentum_adaptive_trailing(state, current_profit, market_condition)
        
        # Weight combination
        ml_weight = self.config.prediction_weight
        traditional_weight = 1 - ml_weight
        
        combined_target = (predicted_target * ml_weight + 
                          traditional_decision.target_price * traditional_weight)
        
        combined_confidence = (ml_confidence * ml_weight + 
                             traditional_decision.confidence * traditional_weight)
        
        return ProfitDecision(
            action=traditional_decision.action,
            target_price=combined_target,
            profit_percentage=current_profit,
            confidence=combined_confidence,
            reasoning=f"ML-enhanced trailing (ML conf: {ml_confidence:.2f})",
            lock_level=ProfitLockLevel.NONE
        )
    
    def _tick_optimized_trailing(self,
                               state: TrailingState,
                               current_profit: float,
                               tick_data: Optional[TickData]) -> ProfitDecision:
        """Trailing optimized for tick-level movements"""
        
        if not tick_data:
            return self._fixed_percentage_trailing(state, current_profit)
        
        # Analyze tick-level dynamics
        tick_momentum = (tick_data.last_price - state.entry_price) / state.entry_price
        spread_factor = tick_data.spread / tick_data.mid_price
        
        # Adjust trailing based on spread and momentum
        if spread_factor < 0.001:  # Tight spread
            trailing_factor = 0.2   # Tight trailing
        elif spread_factor > 0.005:  # Wide spread
            trailing_factor = 0.5   # Loose trailing
        else:
            trailing_factor = 0.3   # Standard trailing
        
        # Calculate tick-optimized target
        tick_target = state.current_price * (1 - trailing_factor * self.config.trailing_percentage)
        
        return ProfitDecision(
            action="HOLD",
            target_price=max(tick_target, state.entry_price * (1 + self.config.minimum_profit_target)),
            profit_percentage=current_profit,
            confidence=0.7,
            reasoning=f"Tick-optimized trailing (spread: {spread_factor:.4f})",
            lock_level=ProfitLockLevel.NONE
        )
    
    def _fixed_percentage_trailing(self, state: TrailingState, current_profit: float) -> ProfitDecision:
        """Simple fixed percentage trailing"""
        
        if current_profit > 0:
            pullback_target = state.current_price * (1 - self.config.trailing_percentage)
            new_target = max(pullback_target, state.entry_price * (1 + self.config.minimum_profit_target))
        else:
            new_target = state.initial_target
        
        return ProfitDecision(
            action="HOLD",
            target_price=new_target,
            profit_percentage=current_profit,
            confidence=0.5,
            reasoning="Fixed percentage trailing",
            lock_level=ProfitLockLevel.NONE
        )
    
    def _handle_profit_locking(self, state: TrailingState, decision: ProfitDecision) -> None:
        """Handle profit locking mechanisms"""
        
        current_profit = decision.profit_percentage
        
        # Determine lock level based on profit
        if current_profit >= self.config.profit_lock_threshold * 4:
            decision.lock_level = ProfitLockLevel.PARTIAL_75
            state.locked_profit = current_profit * 0.75
        elif current_profit >= self.config.profit_lock_threshold * 2:
            decision.lock_level = ProfitLockLevel.PARTIAL_50
            state.locked_profit = current_profit * 0.5
        elif current_profit >= self.config.profit_lock_threshold:
            decision.lock_level = ProfitLockLevel.PARTIAL_25
            state.locked_profit = current_profit * 0.25
        else:
            decision.lock_level = ProfitLockLevel.NONE
            state.locked_profit = 0.0
        
        state.lock_level = decision.lock_level
    
    def remove_position(self, position_id: str, exit_price: float, reason: str = "manual") -> Dict[str, Any]:
        """Remove position and calculate final statistics"""
        
        if position_id not in self.active_positions:
            return {"error": f"Position {position_id} not found"}
        
        state = self.active_positions[position_id]
        
        # Calculate final profit
        final_profit = (exit_price - state.entry_price) / state.entry_price
        
        # Update statistics
        self.profit_statistics['total_profits_taken'] += 1
        self.profit_statistics['average_profit'] = (
            (self.profit_statistics['average_profit'] * (self.profit_statistics['total_profits_taken'] - 1) + final_profit) /
            self.profit_statistics['total_profits_taken']
        )
        self.profit_statistics['maximum_profit'] = max(self.profit_statistics['maximum_profit'], final_profit)
        
        # Calculate efficiency (actual vs highest possible)
        efficiency = final_profit / state.highest_profit if state.highest_profit > 0 else 0
        self.profit_statistics['profit_efficiency'] = (
            (self.profit_statistics['profit_efficiency'] * (self.profit_statistics['total_profits_taken'] - 1) + efficiency) /
            self.profit_statistics['total_profits_taken']
        )
        
        # Remove position
        del self.active_positions[position_id]
        
        result = {
            "position_id": position_id,
            "entry_price": state.entry_price,
            "exit_price": exit_price,
            "final_profit": final_profit,
            "highest_profit": state.highest_profit,
            "efficiency": efficiency,
            "reason": reason,
            "timestamp": datetime.now()
        }
        
        logger.info(f"Position {position_id} closed: profit={final_profit:.4f}, efficiency={efficiency:.2%}")
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        
        active_count = len(self.active_positions)
        total_unrealized = sum(
            (state.current_price - state.entry_price) / state.entry_price
            for state in self.active_positions.values()
        )
        
        return {
            "active_positions": active_count,
            "total_unrealized_profit": total_unrealized,
            "profit_statistics": self.profit_statistics.copy(),
            "decision_history_size": len(self.decision_history),
            "momentum_strength": self.momentum_analyzer.get_momentum_strength(),
            "current_volatility": self.volatility_tracker.get_current_volatility(),
            "last_update": datetime.now()
        }


# Factory function
def create_trailing_system(custom_config: Optional[Dict[str, Any]] = None,
                          leverage_manager: Optional[DynamicLeverageManager] = None,
                          risk_manager: Optional[AdaptiveRiskManager] = None) -> TrailingTakeProfitSystem:
    """Create trailing take profit system with custom configuration"""
    
    config = TrailingConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return TrailingTakeProfitSystem(
        config=config,
        leverage_manager=leverage_manager,
        risk_manager=risk_manager
    )


# Demo and testing
if __name__ == "__main__":
    print("🎯 Trailing Take Profit System Demo")
    print("=" * 50)
    
    # Create trailing system
    trailing_system = create_trailing_system({
        'initial_profit_target': 0.005,  # 0.5%
        'trailing_percentage': 0.25      # 25% trailing
    })
    
    # Mock market condition
    class MockMarketCondition:
        def __init__(self):
            self.regime = MarketRegime.TRENDING
            self.volatility = 0.03
            self.confidence = 0.8
            self.trend_strength = 0.7
    
    market_condition = MockMarketCondition()
    
    # Add test position
    position_id = "BTC_SCALP_001"
    entry_price = 50000.0
    
    state = trailing_system.add_position(
        position_id=position_id,
        entry_price=entry_price,
        market_condition=market_condition
    )
    
    print(f"📊 Added position: {position_id}")
    print(f"   Entry: ${entry_price:.2f}")
    print(f"   Initial Target: ${state.initial_target:.2f}")
    
    # Simulate price movements
    price_scenarios = [
        (50150, "Small gain"),
        (50300, "Medium gain"),
        (50450, "Large gain"),
        (50350, "Pullback"),
        (50500, "New high"),
        (50400, "Another pullback")
    ]
    
    for price, description in price_scenarios:
        decision = trailing_system.update_position(
            position_id=position_id,
            current_price=price,
            market_condition=market_condition,
            mode=TrailingMode.MOMENTUM_ADAPTIVE
        )
        
        print(f"\n💰 {description}: ${price:.2f}")
        print(f"   Profit: {decision.profit_percentage:.4f} ({decision.profit_percentage*100:.2f}%)")
        print(f"   Action: {decision.action}")
        print(f"   New Target: ${decision.target_price:.2f}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Reasoning: {decision.reasoning}")
    
    # Show final statistics
    status = trailing_system.get_system_status()
    print(f"\n📈 System Status:")
    print(f"   Active Positions: {status['active_positions']}")
    print(f"   Total Unrealized: {status['total_unrealized_profit']:.4f}")
    print(f"   Momentum Strength: {status['momentum_strength']:.2f}")
    
    print(f"\n🎯 Trailing Take Profit System - IMPLEMENTATION COMPLETE")
    print("🚀 Ready for integration with autonomous scalping bot")