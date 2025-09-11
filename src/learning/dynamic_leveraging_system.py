"""
Dynamic Leveraging System for Autonomous Crypto Scalping
=======================================================

This module implements an intelligent dynamic leveraging system that automatically
adjusts leverage based on market conditions, volatility, performance metrics, and
tick-level market microstructure analysis for high-frequency crypto scalping.

Key Features:
- Adaptive leverage based on market regime and volatility
- Performance-based leverage scaling
- Tick-level microstructure analysis
- Risk-adjusted leverage calculations
- Real-time leverage optimization
- Emergency leverage reduction mechanisms
- Integration with existing risk management

Author: Autonomous Systems Team
Date: 2025-01-22
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time
import statistics
from datetime import datetime, timedelta
import math

# Import existing risk management components
try:
    from .adaptive_risk_management import (
        AdaptiveRiskManager, MarketCondition, MarketRegime, RiskLevel,
        PortfolioRiskMetrics
    )
    from .performance_based_risk_adjustment import PerformanceBasedRiskAdjuster
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class LeverageMode(Enum):
    """Different leverage calculation modes"""
    CONSERVATIVE = "conservative"    # Lower leverage, emphasis on safety
    BALANCED = "balanced"           # Moderate leverage based on conditions
    AGGRESSIVE = "aggressive"       # Higher leverage for optimal returns
    SCALPING = "scalping"          # Ultra-high frequency with moderate leverage
    ADAPTIVE = "adaptive"          # Fully adaptive based on all factors


class MarketMicrostructure(Enum):
    """Market microstructure conditions"""
    TIGHT_SPREAD = "tight_spread"      # Low spread, high liquidity
    WIDE_SPREAD = "wide_spread"        # High spread, lower liquidity
    HIGH_VOLUME = "high_volume"        # High trading volume
    LOW_VOLUME = "low_volume"          # Low trading volume
    IMBALANCED = "imbalanced"          # Order book imbalance
    BALANCED = "balanced"              # Balanced order book


@dataclass
class LeverageConfig:
    """Configuration for dynamic leveraging system"""
    # Base leverage settings
    min_leverage: float = 1.0
    max_leverage: float = 50.0
    base_leverage: float = 10.0
    
    # Scalping-specific settings
    scalping_max_leverage: float = 25.0
    tick_sensitivity: float = 0.1
    spread_threshold: float = 0.0005  # 0.05%
    
    # Performance-based adjustments
    performance_lookback: int = 100   # trades
    win_rate_threshold: float = 0.6
    profit_factor_threshold: float = 1.2
    
    # Risk-based constraints
    max_drawdown_leverage_reduction: float = 0.5
    volatility_scaling_factor: float = 2.0
    correlation_penalty: float = 0.8
    
    # Emergency settings
    emergency_leverage_cap: float = 5.0
    emergency_drawdown_threshold: float = 0.05
    
    # Microstructure settings
    spread_leverage_penalty: float = 0.7
    volume_leverage_bonus: float = 1.2
    imbalance_leverage_penalty: float = 0.8


@dataclass
class LeverageDecision:
    """Result of leverage calculation"""
    recommended_leverage: float
    base_leverage: float
    regime_adjustment: float
    volatility_adjustment: float
    performance_adjustment: float
    microstructure_adjustment: float
    emergency_adjustment: float
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TickData:
    """Tick-level market data for microstructure analysis"""
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    spread: float
    mid_price: float
    
    def __post_init__(self):
        if self.spread == 0:
            self.spread = self.ask_price - self.bid_price
        if self.mid_price == 0:
            self.mid_price = (self.bid_price + self.ask_price) / 2


class MicrostructureAnalyzer:
    """Analyzes tick-level market microstructure for leverage decisions"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.tick_history = deque(maxlen=window_size)
        self.spread_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        
    def update_tick(self, tick: TickData) -> None:
        """Update with new tick data"""
        self.tick_history.append(tick)
        self.spread_history.append(tick.spread)
        self.volume_history.append(tick.volume)
        
    def analyze_microstructure(self) -> Dict[str, Any]:
        """Analyze current market microstructure conditions"""
        
        if len(self.tick_history) < 10:
            return {
                'spread_condition': MarketMicrostructure.BALANCED,
                'volume_condition': MarketMicrostructure.BALANCED,
                'imbalance_condition': MarketMicrostructure.BALANCED,
                'liquidity_score': 0.5,
                'microstructure_quality': 0.5
            }
        
        # Analyze spread conditions
        recent_spreads = list(self.spread_history)[-20:]
        avg_spread = statistics.mean(recent_spreads)
        spread_volatility = statistics.stdev(recent_spreads) if len(recent_spreads) > 1 else 0
        
        if avg_spread < 0.001:  # Very tight spread
            spread_condition = MarketMicrostructure.TIGHT_SPREAD
        elif avg_spread > 0.005:  # Wide spread
            spread_condition = MarketMicrostructure.WIDE_SPREAD
        else:
            spread_condition = MarketMicrostructure.BALANCED
        
        # Analyze volume conditions
        recent_volumes = list(self.volume_history)[-20:]
        avg_volume = statistics.mean(recent_volumes)
        volume_volatility = statistics.stdev(recent_volumes) if len(recent_volumes) > 1 else 0
        
        volume_condition = (MarketMicrostructure.HIGH_VOLUME if avg_volume > 1000
                          else MarketMicrostructure.LOW_VOLUME)
        
        # Analyze order book imbalance
        recent_ticks = list(self.tick_history)[-10:]
        bid_sizes = [tick.bid_size for tick in recent_ticks]
        ask_sizes = [tick.ask_size for tick in recent_ticks]
        
        if bid_sizes and ask_sizes:
            avg_bid_size = statistics.mean(bid_sizes)
            avg_ask_size = statistics.mean(ask_sizes)
            imbalance_ratio = abs(avg_bid_size - avg_ask_size) / (avg_bid_size + avg_ask_size)
            
            imbalance_condition = (MarketMicrostructure.IMBALANCED if imbalance_ratio > 0.3
                                 else MarketMicrostructure.BALANCED)
        else:
            imbalance_condition = MarketMicrostructure.BALANCED
        
        # Calculate overall liquidity score
        liquidity_score = self._calculate_liquidity_score(avg_spread, avg_volume, spread_volatility)
        
        # Calculate microstructure quality score
        microstructure_quality = self._calculate_microstructure_quality(
            spread_condition, volume_condition, imbalance_condition
        )
        
        return {
            'spread_condition': spread_condition,
            'volume_condition': volume_condition,
            'imbalance_condition': imbalance_condition,
            'liquidity_score': liquidity_score,
            'microstructure_quality': microstructure_quality,
            'avg_spread': avg_spread,
            'avg_volume': avg_volume,
            'spread_volatility': spread_volatility
        }
    
    def _calculate_liquidity_score(self, spread: float, volume: float, spread_vol: float) -> float:
        """Calculate liquidity score based on spread and volume"""
        
        # Lower spread = higher liquidity
        spread_score = max(0, 1 - spread * 1000)  # Normalize spread
        
        # Higher volume = higher liquidity
        volume_score = min(1, volume / 10000)  # Normalize volume
        
        # Lower spread volatility = higher liquidity
        volatility_score = max(0, 1 - spread_vol * 10000)
        
        # Weighted combination
        liquidity_score = (0.4 * spread_score + 0.4 * volume_score + 0.2 * volatility_score)
        
        return max(0, min(1, liquidity_score))
    
    def _calculate_microstructure_quality(self, spread_cond, volume_cond, imbalance_cond) -> float:
        """Calculate overall microstructure quality score"""
        
        quality_scores = {
            MarketMicrostructure.TIGHT_SPREAD: 1.0,
            MarketMicrostructure.BALANCED: 0.7,
            MarketMicrostructure.WIDE_SPREAD: 0.3,
            MarketMicrostructure.HIGH_VOLUME: 1.0,
            MarketMicrostructure.LOW_VOLUME: 0.5,
            MarketMicrostructure.IMBALANCED: 0.4
        }
        
        spread_score = quality_scores.get(spread_cond, 0.5)
        volume_score = quality_scores.get(volume_cond, 0.5)
        imbalance_score = quality_scores.get(imbalance_cond, 0.5)
        
        # Weighted average
        overall_quality = (0.4 * spread_score + 0.3 * volume_score + 0.3 * imbalance_score)
        
        return max(0, min(1, overall_quality))


class PerformanceLeverageOptimizer:
    """Optimizes leverage based on trading performance metrics"""
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.trade_results = deque(maxlen=lookback_period)
        self.leverage_history = deque(maxlen=lookback_period)
        
    def add_trade_result(self, pnl: float, leverage_used: float, duration: float) -> None:
        """Add a trade result for performance analysis"""
        
        self.trade_results.append({
            'pnl': pnl,
            'leverage': leverage_used,
            'duration': duration,
            'timestamp': datetime.now()
        })
        
        self.leverage_history.append(leverage_used)
    
    def calculate_performance_adjustment(self) -> Tuple[float, str]:
        """Calculate leverage adjustment based on performance"""
        
        if len(self.trade_results) < 10:
            return 1.0, "Insufficient trade history"
        
        recent_trades = list(self.trade_results)[-50:]  # Last 50 trades
        
        # Calculate performance metrics
        win_rate = sum(1 for trade in recent_trades if trade['pnl'] > 0) / len(recent_trades)
        
        winning_trades = [trade['pnl'] for trade in recent_trades if trade['pnl'] > 0]
        losing_trades = [abs(trade['pnl']) for trade in recent_trades if trade['pnl'] < 0]
        
        avg_win = statistics.mean(winning_trades) if winning_trades else 0
        avg_loss = statistics.mean(losing_trades) if losing_trades else 1
        
        profit_factor = (avg_win * len(winning_trades)) / (avg_loss * len(losing_trades)) if losing_trades else 2.0
        
        # Calculate leverage efficiency
        leverage_efficiency = self._calculate_leverage_efficiency(recent_trades)
        
        # Determine adjustment based on performance
        adjustment = 1.0
        reasoning = "Baseline performance"
        
        if win_rate > 0.7 and profit_factor > 1.5:
            adjustment = 1.2  # Increase leverage for strong performance
            reasoning = f"Strong performance: WR={win_rate:.2%}, PF={profit_factor:.2f}"
        elif win_rate > 0.6 and profit_factor > 1.2:
            adjustment = 1.1  # Slight increase
            reasoning = f"Good performance: WR={win_rate:.2%}, PF={profit_factor:.2f}"
        elif win_rate < 0.4 or profit_factor < 0.8:
            adjustment = 0.7  # Reduce leverage for poor performance
            reasoning = f"Poor performance: WR={win_rate:.2%}, PF={profit_factor:.2f}"
        elif win_rate < 0.5 or profit_factor < 1.0:
            adjustment = 0.85  # Slight reduction
            reasoning = f"Below average performance: WR={win_rate:.2%}, PF={profit_factor:.2f}"
        
        # Apply leverage efficiency factor
        adjustment *= leverage_efficiency
        
        return max(0.3, min(2.0, adjustment)), reasoning
    
    def _calculate_leverage_efficiency(self, trades: List[Dict]) -> float:
        """Calculate how efficiently leverage is being used"""
        
        if not trades:
            return 1.0
        
        # Calculate risk-adjusted returns for different leverage levels
        leverage_groups = defaultdict(list)
        
        for trade in trades:
            leverage_bucket = round(trade['leverage'])
            risk_adjusted_return = trade['pnl'] / trade['leverage'] if trade['leverage'] > 0 else 0
            leverage_groups[leverage_bucket].append(risk_adjusted_return)
        
        # Find optimal leverage level
        best_efficiency = 0
        optimal_leverage = 1
        
        for leverage, returns in leverage_groups.items():
            if len(returns) >= 5:  # Need sufficient samples
                avg_return = statistics.mean(returns)
                efficiency = avg_return / leverage if leverage > 0 else 0
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    optimal_leverage = leverage
        
        # Current average leverage
        current_avg_leverage = statistics.mean([trade['leverage'] for trade in trades])
        
        # Efficiency factor
        if current_avg_leverage > 0:
            efficiency_factor = optimal_leverage / current_avg_leverage
            return max(0.5, min(1.5, efficiency_factor))
        
        return 1.0


class DynamicLeverageManager:
    """Main dynamic leverage management system"""
    
    def __init__(self, 
                 config: Optional[LeverageConfig] = None,
                 risk_manager: Optional[Any] = None):
        
        self.config = config or LeverageConfig()
        self.risk_manager = risk_manager
        
        # Initialize components
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.performance_optimizer = PerformanceLeverageOptimizer()
        
        # State tracking
        self.current_leverage = self.config.base_leverage
        self.leverage_history = deque(maxlen=1000)
        self.decision_history = deque(maxlen=500)
        
        # Emergency state
        self.emergency_mode = False
        self.emergency_triggered_time = None
        
        logger.info("Dynamic Leverage Manager initialized")
    
    def calculate_optimal_leverage(self,
                                 market_condition: MarketCondition,
                                 portfolio_metrics: PortfolioRiskMetrics,
                                 tick_data: Optional[TickData] = None,
                                 mode: LeverageMode = LeverageMode.ADAPTIVE) -> LeverageDecision:
        """Calculate optimal leverage based on all available factors"""
        
        # Start with base leverage
        base_leverage = self.config.base_leverage
        
        # 1. Market regime adjustment
        regime_adjustment = self._calculate_regime_adjustment(market_condition.regime)
        
        # 2. Volatility adjustment
        volatility_adjustment = self._calculate_volatility_adjustment(market_condition.volatility)
        
        # 3. Performance-based adjustment
        performance_adjustment, perf_reasoning = self.performance_optimizer.calculate_performance_adjustment()
        
        # 4. Microstructure adjustment
        microstructure_adjustment = 1.0
        micro_reasoning = "No tick data available"
        
        if tick_data:
            self.microstructure_analyzer.update_tick(tick_data)
            microstructure_analysis = self.microstructure_analyzer.analyze_microstructure()
            microstructure_adjustment = self._calculate_microstructure_adjustment(microstructure_analysis)
            micro_reasoning = f"Microstructure quality: {microstructure_analysis['microstructure_quality']:.2f}"
        
        # 5. Emergency adjustment
        emergency_adjustment = self._calculate_emergency_adjustment(portfolio_metrics)
        
        # 6. Mode-specific adjustment
        mode_adjustment = self._calculate_mode_adjustment(mode)
        
        # Calculate final leverage
        raw_leverage = (base_leverage * 
                       regime_adjustment * 
                       volatility_adjustment * 
                       performance_adjustment * 
                       microstructure_adjustment * 
                       emergency_adjustment * 
                       mode_adjustment)
        
        # Apply constraints
        final_leverage = self._apply_leverage_constraints(raw_leverage, portfolio_metrics)
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(
            market_condition, portfolio_metrics, tick_data
        )
        
        # Create reasoning
        reasoning = (f"Base: {base_leverage:.1f}, "
                    f"Regime: {regime_adjustment:.2f}, "
                    f"Volatility: {volatility_adjustment:.2f}, "
                    f"Performance: {performance_adjustment:.2f}, "
                    f"Microstructure: {microstructure_adjustment:.2f}, "
                    f"Emergency: {emergency_adjustment:.2f}")
        
        decision = LeverageDecision(
            recommended_leverage=final_leverage,
            base_leverage=base_leverage,
            regime_adjustment=regime_adjustment,
            volatility_adjustment=volatility_adjustment,
            performance_adjustment=performance_adjustment,
            microstructure_adjustment=microstructure_adjustment,
            emergency_adjustment=emergency_adjustment,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # Store decision
        self.decision_history.append(decision)
        self.leverage_history.append(final_leverage)
        self.current_leverage = final_leverage
        
        logger.info(f"Leverage decision: {final_leverage:.2f}x (confidence: {confidence:.2f})")
        
        return decision
    
    def _calculate_regime_adjustment(self, regime: MarketRegime) -> float:
        """Calculate leverage adjustment based on market regime"""
        
        regime_multipliers = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.TRENDING: 1.3,      # Higher leverage in trending markets
            MarketRegime.RANGE_BOUND: 1.1,   # Slightly higher for scalping
            MarketRegime.VOLATILE: 0.6,      # Lower leverage in volatile markets
            MarketRegime.BULL_RUN: 1.4,      # Higher leverage in bull runs
            MarketRegime.CRASH: 0.3,         # Very low leverage in crashes
            MarketRegime.RECOVERY: 0.8       # Conservative in recovery
        }
        
        return regime_multipliers.get(regime, 1.0)
    
    def _calculate_volatility_adjustment(self, volatility: float) -> float:
        """Calculate leverage adjustment based on volatility"""
        
        # Inverse relationship: higher volatility = lower leverage
        if volatility < 0.01:
            return 1.2  # Low volatility, can use higher leverage
        elif volatility < 0.02:
            return 1.0  # Normal volatility
        elif volatility < 0.05:
            return 0.7  # High volatility, reduce leverage
        else:
            return 0.4  # Very high volatility, significantly reduce
    
    def _calculate_microstructure_adjustment(self, microstructure: Dict[str, Any]) -> float:
        """Calculate leverage adjustment based on market microstructure"""
        
        quality_score = microstructure['microstructure_quality']
        liquidity_score = microstructure['liquidity_score']
        
        # High quality microstructure allows higher leverage
        base_adjustment = 0.7 + (quality_score * 0.6)  # Range: 0.7 to 1.3
        
        # Apply specific condition penalties
        spread_condition = microstructure['spread_condition']
        volume_condition = microstructure['volume_condition']
        imbalance_condition = microstructure['imbalance_condition']
        
        if spread_condition == MarketMicrostructure.WIDE_SPREAD:
            base_adjustment *= self.config.spread_leverage_penalty
        
        if volume_condition == MarketMicrostructure.HIGH_VOLUME:
            base_adjustment *= self.config.volume_leverage_bonus
        elif volume_condition == MarketMicrostructure.LOW_VOLUME:
            base_adjustment *= 0.9
        
        if imbalance_condition == MarketMicrostructure.IMBALANCED:
            base_adjustment *= self.config.imbalance_leverage_penalty
        
        return max(0.3, min(1.5, base_adjustment))
    
    def _calculate_emergency_adjustment(self, portfolio_metrics: PortfolioRiskMetrics) -> float:
        """Calculate emergency leverage reduction if needed"""
        
        emergency_factors = []
        
        # Check drawdown
        if portfolio_metrics.current_drawdown > self.config.emergency_drawdown_threshold:
            emergency_factors.append("High drawdown")
            self.emergency_mode = True
            self.emergency_triggered_time = datetime.now()
        
        # Check if still in emergency mode
        if self.emergency_mode:
            if self.emergency_triggered_time:
                time_since_emergency = datetime.now() - self.emergency_triggered_time
                if time_since_emergency.total_seconds() > 3600:  # 1 hour recovery period
                    if portfolio_metrics.current_drawdown < self.config.emergency_drawdown_threshold * 0.5:
                        self.emergency_mode = False
                        logger.info("Emergency mode deactivated")
        
        if self.emergency_mode:
            return 0.2  # Severely limit leverage in emergency
        
        return 1.0
    
    def _calculate_mode_adjustment(self, mode: LeverageMode) -> float:
        """Calculate adjustment based on leverage mode"""
        
        mode_multipliers = {
            LeverageMode.CONSERVATIVE: 0.7,
            LeverageMode.BALANCED: 1.0,
            LeverageMode.AGGRESSIVE: 1.4,
            LeverageMode.SCALPING: 1.2,  # Moderate leverage for scalping
            LeverageMode.ADAPTIVE: 1.0   # Base, will be modified by other factors
        }
        
        return mode_multipliers.get(mode, 1.0)
    
    def _apply_leverage_constraints(self, leverage: float, portfolio_metrics: PortfolioRiskMetrics) -> float:
        """Apply final leverage constraints"""
        
        # Apply basic min/max constraints
        constrained_leverage = max(self.config.min_leverage, 
                                 min(self.config.max_leverage, leverage))
        
        # Apply emergency cap if needed
        if self.emergency_mode:
            constrained_leverage = min(constrained_leverage, self.config.emergency_leverage_cap)
        
        # Apply scalping-specific cap
        if constrained_leverage > self.config.scalping_max_leverage:
            constrained_leverage = self.config.scalping_max_leverage
        
        return constrained_leverage
    
    def _calculate_decision_confidence(self, 
                                     market_condition: MarketCondition,
                                     portfolio_metrics: PortfolioRiskMetrics,
                                     tick_data: Optional[TickData]) -> float:
        """Calculate confidence in the leverage decision"""
        
        confidence_factors = []
        
        # Market condition confidence
        confidence_factors.append(market_condition.confidence)
        
        # Data availability
        if tick_data:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Historical performance consistency
        if len(self.leverage_history) > 10:
            recent_leverage = list(self.leverage_history)[-10:]
            leverage_stability = 1.0 - (statistics.stdev(recent_leverage) / statistics.mean(recent_leverage))
            confidence_factors.append(max(0.3, leverage_stability))
        else:
            confidence_factors.append(0.5)
        
        # Emergency mode reduces confidence
        if self.emergency_mode:
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(0.8)
        
        return statistics.mean(confidence_factors)
    
    def update_trade_result(self, pnl: float, leverage_used: float, duration: float) -> None:
        """Update system with trade result"""
        self.performance_optimizer.add_trade_result(pnl, leverage_used, duration)
    
    def get_leverage_statistics(self) -> Dict[str, Any]:
        """Get leverage usage statistics"""
        
        if not self.leverage_history:
            return {}
        
        recent_leverage = list(self.leverage_history)[-100:]
        
        return {
            'current_leverage': self.current_leverage,
            'average_leverage': statistics.mean(recent_leverage),
            'max_leverage_used': max(recent_leverage),
            'min_leverage_used': min(recent_leverage),
            'leverage_volatility': statistics.stdev(recent_leverage) if len(recent_leverage) > 1 else 0,
            'emergency_mode': self.emergency_mode,
            'decisions_made': len(self.decision_history),
            'last_decision_time': self.decision_history[-1].timestamp if self.decision_history else None
        }


# Factory function
def create_dynamic_leverage_manager(custom_config: Optional[Dict[str, Any]] = None,
                                  risk_manager: Optional[Any] = None) -> DynamicLeverageManager:
    """Create dynamic leverage manager with custom configuration"""
    
    config = LeverageConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return DynamicLeverageManager(config=config, risk_manager=risk_manager)


# Demo and testing
if __name__ == "__main__":
    print("🎯 Dynamic Leveraging System Demo")
    print("=" * 50)
    
    # Create leverage manager
    leverage_manager = create_dynamic_leverage_manager({
        'max_leverage': 25.0,
        'scalping_max_leverage': 20.0
    })
    
    # Simulate different market conditions
    from datetime import datetime
    
    # Mock classes for demo
    class MockMarketCondition:
        def __init__(self, regime, volatility, confidence):
            self.regime = regime
            self.volatility = volatility
            self.confidence = confidence
    
    class MockPortfolioMetrics:
        def __init__(self, drawdown=0.02):
            self.current_drawdown = drawdown
            self.daily_var = 0.03
            self.leverage_ratio = 2.0
    
    class MockMarketRegime:
        NORMAL = "normal"
        TRENDING = "trending"
        VOLATILE = "volatile"
        SCALPING = "scalping"
    
    scenarios = [
        ("Normal Market", MockMarketRegime.NORMAL, 0.02, 0.8),
        ("Trending Market", MockMarketRegime.TRENDING, 0.015, 0.9),
        ("Volatile Market", MockMarketRegime.VOLATILE, 0.08, 0.7),
    ]
    
    for name, regime, volatility, confidence in scenarios:
        print(f"\n📊 Testing {name}:")
        
        market_condition = MockMarketCondition(regime, volatility, confidence)
        portfolio_metrics = MockPortfolioMetrics()
        
        # Create mock tick data
        tick_data = TickData(
            timestamp=datetime.now(),
            bid_price=50000.0,
            ask_price=50001.0,
            bid_size=10.0,
            ask_size=12.0,
            last_price=50000.5,
            volume=100.0,
            spread=1.0,
            mid_price=50000.5
        )
        
        decision = leverage_manager.calculate_optimal_leverage(
            market_condition, portfolio_metrics, tick_data
        )
        
        print(f"   📈 Recommended Leverage: {decision.recommended_leverage:.2f}x")
        print(f"   🎯 Confidence: {decision.confidence:.2%}")
        print(f"   🔍 Reasoning: {decision.reasoning}")
        
        # Simulate trade result
        leverage_manager.update_trade_result(
            pnl=np.random.normal(10, 20),
            leverage_used=decision.recommended_leverage,
            duration=30.0
        )
    
    # Show statistics
    stats = leverage_manager.get_leverage_statistics()
    print(f"\n📊 Leverage Statistics:")
    print(f"   Average Leverage: {stats.get('average_leverage', 0):.2f}x")
    print(f"   Max Leverage Used: {stats.get('max_leverage_used', 0):.2f}x")
    print(f"   Emergency Mode: {stats.get('emergency_mode', False)}")
    
    print(f"\n🎯 Dynamic Leveraging System - IMPLEMENTATION COMPLETE")
    print("🚀 Ready for integration with autonomous scalping bot")