"""
Adaptive Market Regime Integration
=================================

This module bridges the market adaptation system with the Mixture of Experts (MoE) 
engine to provide seamless autonomous regime detection and strategy adaptation.

Key Features:
- Real-time regime detection and classification
- Automatic MoE expert selection based on market conditions
- Dynamic strategy parameter adaptation
- Performance feedback integration
- Autonomous regime transition handling
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch

# Import existing components
from src.learning.self_adaptation.market_adaptation import (
    MarketAnalyzer, MarketCondition, MarketRegime, AdaptationConfig
)
from src.models.mixture_of_experts import (
    MixtureOfExperts, MarketRegimeDetector, RegimeClassification, MoESignal
)
from src.learning.continuous_learning_pipeline import ContinuousLearningPipeline

logger = logging.getLogger(__name__)


class RegimeTransition(Enum):
    """Types of regime transitions"""
    STABLE = "stable"              # No change in regime
    GRADUAL = "gradual"           # Slow transition
    SUDDEN = "sudden"             # Rapid transition
    REVERSAL = "reversal"         # Complete reversal (bull to bear, etc.)


@dataclass
class AdaptiveStrategy:
    """Adaptive strategy configuration for different regimes"""
    regime: MarketRegime
    strategy_name: str
    parameters: Dict[str, float]
    confidence_threshold: float
    position_sizing_multiplier: float
    risk_multiplier: float
    update_frequency: float  # seconds


@dataclass
class RegimeTransitionEvent:
    """Event representing a regime transition"""
    timestamp: datetime
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_type: RegimeTransition
    confidence: float
    duration: float  # seconds
    market_condition: MarketCondition


class AdaptiveStrategyManager:
    """Manages adaptive strategies for different market regimes"""
    
    def __init__(self):
        self.strategies: Dict[MarketRegime, AdaptiveStrategy] = {}
        self.current_strategy: Optional[AdaptiveStrategy] = None
        self.strategy_performance: Dict[MarketRegime, List[float]] = {}
        self._initialize_default_strategies()
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default adaptive strategies for each regime"""
        
        # Normal market conditions
        self.strategies[MarketRegime.NORMAL] = AdaptiveStrategy(
            regime=MarketRegime.NORMAL,
            strategy_name="balanced_scalping",
            parameters={
                "position_size": 0.02,
                "stop_loss": 0.005,
                "take_profit": 0.01,
                "confidence_threshold": 0.7,
                "update_frequency": 1.0
            },
            confidence_threshold=0.7,
            position_sizing_multiplier=1.0,
            risk_multiplier=1.0,
            update_frequency=1.0
        )
        
        # High volatility conditions
        self.strategies[MarketRegime.VOLATILE] = AdaptiveStrategy(
            regime=MarketRegime.VOLATILE,
            strategy_name="volatility_scalping",
            parameters={
                "position_size": 0.015,  # Reduced size
                "stop_loss": 0.003,     # Tighter stops
                "take_profit": 0.008,   # Smaller targets
                "confidence_threshold": 0.8,  # Higher confidence needed
                "update_frequency": 0.5  # More frequent updates
            },
            confidence_threshold=0.8,
            position_sizing_multiplier=0.75,
            risk_multiplier=1.5,
            update_frequency=0.5
        )
        
        # Trending market conditions
        self.strategies[MarketRegime.TRENDING] = AdaptiveStrategy(
            regime=MarketRegime.TRENDING,
            strategy_name="trend_following",
            parameters={
                "position_size": 0.025,  # Larger positions
                "stop_loss": 0.008,     # Wider stops
                "take_profit": 0.015,   # Larger targets
                "confidence_threshold": 0.65,  # Lower threshold for trends
                "update_frequency": 2.0  # Less frequent updates
            },
            confidence_threshold=0.65,
            position_sizing_multiplier=1.25,
            risk_multiplier=0.8,
            update_frequency=2.0
        )
        
        # Range-bound conditions
        self.strategies[MarketRegime.RANGE_BOUND] = AdaptiveStrategy(
            regime=MarketRegime.RANGE_BOUND,
            strategy_name="mean_reversion",
            parameters={
                "position_size": 0.02,
                "stop_loss": 0.004,
                "take_profit": 0.008,
                "confidence_threshold": 0.75,
                "update_frequency": 1.5
            },
            confidence_threshold=0.75,
            position_sizing_multiplier=1.0,
            risk_multiplier=0.9,
            update_frequency=1.5
        )
        
        # Crisis/crash conditions
        self.strategies[MarketRegime.CRASH] = AdaptiveStrategy(
            regime=MarketRegime.CRASH,
            strategy_name="defensive",
            parameters={
                "position_size": 0.005,  # Very small positions
                "stop_loss": 0.002,     # Very tight stops
                "take_profit": 0.004,   # Small targets
                "confidence_threshold": 0.9,  # Very high confidence needed
                "update_frequency": 0.25  # Very frequent updates
            },
            confidence_threshold=0.9,
            position_sizing_multiplier=0.25,
            risk_multiplier=2.0,
            update_frequency=0.25
        )
        
        # Bull run conditions
        self.strategies[MarketRegime.BULL_RUN] = AdaptiveStrategy(
            regime=MarketRegime.BULL_RUN,
            strategy_name="momentum_long",
            parameters={
                "position_size": 0.03,   # Larger positions
                "stop_loss": 0.01,      # Wider stops
                "take_profit": 0.02,    # Larger targets
                "confidence_threshold": 0.6,  # Lower threshold
                "update_frequency": 3.0  # Less frequent updates
            },
            confidence_threshold=0.6,
            position_sizing_multiplier=1.5,
            risk_multiplier=0.7,
            update_frequency=3.0
        )
        
        logger.info("Initialized adaptive strategies for all market regimes")
    
    def get_strategy_for_regime(self, regime: MarketRegime) -> Optional[AdaptiveStrategy]:
        """Get the adaptive strategy for a specific regime"""
        return self.strategies.get(regime)
    
    def update_strategy_performance(self, regime: MarketRegime, performance: float) -> None:
        """Update performance metrics for a strategy"""
        if regime not in self.strategy_performance:
            self.strategy_performance[regime] = []
        
        self.strategy_performance[regime].append(performance)
        
        # Keep only recent performance history
        if len(self.strategy_performance[regime]) > 100:
            self.strategy_performance[regime] = self.strategy_performance[regime][-100:]
    
    def get_strategy_performance(self, regime: MarketRegime) -> List[float]:
        """Get performance history for a strategy"""
        return self.strategy_performance.get(regime, [])


class RegimeTransitionDetector:
    """Detects and classifies regime transitions"""
    
    def __init__(self, stability_window: int = 60):
        self.stability_window = stability_window
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self.transition_history: List[RegimeTransitionEvent] = []
    
    def add_regime_observation(self, regime: MarketRegime, timestamp: datetime = None) -> None:
        """Add a regime observation"""
        timestamp = timestamp or datetime.now()
        self.regime_history.append((timestamp, regime))
        
        # Keep only recent history
        cutoff_time = timestamp - timedelta(seconds=self.stability_window * 2)
        self.regime_history = [
            (t, r) for t, r in self.regime_history if t > cutoff_time
        ]
    
    def detect_transition(self, current_regime: MarketRegime, 
                         market_condition: MarketCondition) -> Optional[RegimeTransitionEvent]:
        """Detect if a regime transition has occurred"""
        
        if len(self.regime_history) < 2:
            return None
        
        # Get recent regime
        recent_regimes = [r for _, r in self.regime_history[-10:]]
        
        if not recent_regimes:
            return None
        
        previous_regime = recent_regimes[-2] if len(recent_regimes) >= 2 else recent_regimes[-1]
        
        if previous_regime == current_regime:
            return None  # No transition
        
        # Classify transition type
        transition_type = self._classify_transition(previous_regime, current_regime, recent_regimes)
        
        # Calculate transition duration
        duration = 0.0
        if len(self.regime_history) >= 2:
            duration = (self.regime_history[-1][0] - self.regime_history[-2][0]).total_seconds()
        
        transition_event = RegimeTransitionEvent(
            timestamp=datetime.now(),
            from_regime=previous_regime,
            to_regime=current_regime,
            transition_type=transition_type,
            confidence=market_condition.confidence,
            duration=duration,
            market_condition=market_condition
        )
        
        self.transition_history.append(transition_event)
        
        # Keep only recent transition history
        if len(self.transition_history) > 1000:
            self.transition_history = self.transition_history[-1000:]
        
        return transition_event
    
    def _classify_transition(self, from_regime: MarketRegime, to_regime: MarketRegime,
                           recent_regimes: List[MarketRegime]) -> RegimeTransition:
        """Classify the type of regime transition"""
        
        # Check for reversal patterns
        if ((from_regime == MarketRegime.BULL_RUN and to_regime == MarketRegime.CRASH) or
            (from_regime == MarketRegime.CRASH and to_regime == MarketRegime.BULL_RUN)):
            return RegimeTransition.REVERSAL
        
        # Check for sudden transitions (less than 3 observations of intermediate regimes)
        intermediate_count = len([r for r in recent_regimes if r not in [from_regime, to_regime]])
        if intermediate_count <= 2:
            return RegimeTransition.SUDDEN
        
        # Check for gradual transitions
        if intermediate_count > 5:
            return RegimeTransition.GRADUAL
        
        return RegimeTransition.STABLE


class AdaptiveRegimeIntegration:
    """
    Main integration system that coordinates regime detection, strategy adaptation,
    and MoE expert selection for autonomous operation.
    """
    
    def __init__(self, moe_engine: MixtureOfExperts, 
                 learning_pipeline: Optional[ContinuousLearningPipeline] = None):
        
        self.moe_engine = moe_engine
        self.learning_pipeline = learning_pipeline
        
        # Core components
        self.market_analyzer = MarketAnalyzer(AdaptationConfig())
        self.strategy_manager = AdaptiveStrategyManager()
        self.transition_detector = RegimeTransitionDetector()
        
        # State tracking
        self.current_regime: Optional[MarketRegime] = None
        self.current_strategy: Optional[AdaptiveStrategy] = None
        self.current_market_condition: Optional[MarketCondition] = None
        
        # Performance tracking
        self.regime_performance: Dict[MarketRegime, float] = {}
        self.transition_performance: Dict[RegimeTransition, float] = {}
        
        # Threading
        self.is_running = False
        self.adaptation_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.regime_change_callbacks: List[Callable] = []
        self.strategy_change_callbacks: List[Callable] = []
        
        logger.info("Adaptive regime integration system initialized")
    
    def start(self) -> None:
        """Start the adaptive regime integration system"""
        if self.is_running:
            logger.warning("Adaptive regime integration already running")
            return
        
        logger.info("Starting adaptive regime integration")
        
        self.is_running = True
        self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
        self.adaptation_thread.start()
        
        logger.info("Adaptive regime integration started")
    
    def stop(self) -> None:
        """Stop the adaptive regime integration system"""
        logger.info("Stopping adaptive regime integration")
        
        self.is_running = False
        if self.adaptation_thread:
            self.adaptation_thread.join(timeout=5.0)
        
        logger.info("Adaptive regime integration stopped")
    
    def _adaptation_loop(self) -> None:
        """Main adaptation loop"""
        logger.info("Starting adaptive regime integration loop")
        
        while self.is_running:
            try:
                # Get current market condition
                market_condition = self.market_analyzer.get_current_condition()
                self.current_market_condition = market_condition
                
                # Detect regime transitions
                transition = self.transition_detector.detect_transition(
                    market_condition.regime, market_condition
                )
                
                # Handle regime changes
                if market_condition.regime != self.current_regime:
                    self._handle_regime_change(market_condition.regime, transition)
                
                # Update MoE engine with current regime
                self._update_moe_regime(market_condition)
                
                # Adapt strategy parameters
                self._adapt_strategy_parameters(market_condition)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                time.sleep(1.0)  # 1 second adaptation cycle
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                time.sleep(5.0)
    
    def _handle_regime_change(self, new_regime: MarketRegime, 
                            transition: Optional[RegimeTransitionEvent]) -> None:
        """Handle a regime change event"""
        
        old_regime = self.current_regime
        self.current_regime = new_regime
        
        logger.info(f"Regime change detected: {old_regime} -> {new_regime}")
        
        # Get new strategy for the regime
        new_strategy = self.strategy_manager.get_strategy_for_regime(new_regime)
        if new_strategy:
            old_strategy = self.current_strategy
            self.current_strategy = new_strategy
            
            logger.info(f"Strategy changed: {old_strategy.strategy_name if old_strategy else 'None'} -> {new_strategy.strategy_name}")
            
            # Notify callbacks
            for callback in self.strategy_change_callbacks:
                try:
                    callback(old_strategy, new_strategy, transition)
                except Exception as e:
                    logger.error(f"Error in strategy change callback: {e}")
        
        # Update transition detector
        self.transition_detector.add_regime_observation(new_regime)
        
        # Notify regime change callbacks
        for callback in self.regime_change_callbacks:
            try:
                callback(old_regime, new_regime, transition)
            except Exception as e:
                logger.error(f"Error in regime change callback: {e}")
        
        # Log transition details
        if transition:
            logger.info(f"Transition type: {transition.transition_type}, "
                       f"confidence: {transition.confidence:.3f}, "
                       f"duration: {transition.duration:.1f}s")
    
    def _update_moe_regime(self, market_condition: MarketCondition) -> None:
        """Update MoE engine with current regime information"""
        
        # Map MarketRegime to MoE regime names
        regime_mapping = {
            MarketRegime.NORMAL: "ranging",
            MarketRegime.VOLATILE: "high_volatility",
            MarketRegime.TRENDING: "trending",
            MarketRegime.RANGE_BOUND: "ranging",
            MarketRegime.CRASH: "high_volatility",
            MarketRegime.BULL_RUN: "trending"
        }
        
        moe_regime = regime_mapping.get(market_condition.regime, "ranging")
        
        # Update MoE engine (this would require extending the MoE interface)
        # For now, we'll log the regime update
        logger.debug(f"Updating MoE with regime: {moe_regime}, confidence: {market_condition.confidence:.3f}")
    
    def _adapt_strategy_parameters(self, market_condition: MarketCondition) -> None:
        """Adapt strategy parameters based on market conditions"""
        
        if not self.current_strategy:
            return
        
        # Dynamic parameter adaptation based on market conditions
        adaptation_factor = market_condition.confidence
        
        # Adjust position sizing based on volatility
        volatility_adjustment = 1.0 - (market_condition.volatility * 0.5)
        self.current_strategy.position_sizing_multiplier *= volatility_adjustment
        
        # Adjust risk based on liquidity
        liquidity_adjustment = market_condition.liquidity
        self.current_strategy.risk_multiplier *= (2.0 - liquidity_adjustment)
        
        # Adjust confidence threshold based on trend strength
        trend_adjustment = abs(market_condition.trend_strength) * 0.1
        self.current_strategy.confidence_threshold += trend_adjustment
        
        logger.debug(f"Adapted strategy parameters: "
                    f"pos_mult={self.current_strategy.position_sizing_multiplier:.3f}, "
                    f"risk_mult={self.current_strategy.risk_multiplier:.3f}, "
                    f"conf_thresh={self.current_strategy.confidence_threshold:.3f}")
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics for current regime and strategies"""
        
        if not self.current_regime or not self.current_strategy:
            return
        
        # Placeholder for performance calculation
        # In real implementation, this would calculate actual trading performance
        performance = 0.75  # Dummy performance metric
        
        self.regime_performance[self.current_regime] = performance
        self.strategy_manager.update_strategy_performance(self.current_regime, performance)
    
    def update_market_data(self, price: float, volume: float, 
                          correlated_assets: Optional[List[float]] = None) -> None:
        """Update market data for analysis"""
        self.market_analyzer.update_market_data(
            price=price,
            volume=volume,
            timestamp=datetime.now(),
            correlated_assets=correlated_assets
        )
    
    def get_current_regime_signal(self, market_features: torch.Tensor) -> Optional[MoESignal]:
        """Get trading signal adapted to current market regime"""
        
        if not self.current_strategy or not self.current_market_condition:
            return None
        
        # Generate signal using MoE engine
        moe_signal = self.moe_engine.generate_signal(market_features)
        
        if not moe_signal:
            return None
        
        # Adapt signal based on current regime strategy
        adapted_signal = MoESignal(
            direction=moe_signal.direction,
            confidence=moe_signal.confidence * self.current_market_condition.confidence,
            size=moe_signal.size * self.current_strategy.position_sizing_multiplier,
            regime=self.current_market_condition.regime.value,
            regime_confidence=self.current_market_condition.confidence,
            expert_contributions=moe_signal.expert_contributions
        )
        
        return adapted_signal
    
    def register_regime_change_callback(self, callback: Callable) -> None:
        """Register callback for regime change events"""
        self.regime_change_callbacks.append(callback)
    
    def register_strategy_change_callback(self, callback: Callable) -> None:
        """Register callback for strategy change events"""
        self.strategy_change_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "current_regime": self.current_regime.value if self.current_regime else None,
            "current_strategy": self.current_strategy.strategy_name if self.current_strategy else None,
            "market_condition": {
                "volatility": self.current_market_condition.volatility if self.current_market_condition else 0.0,
                "trend_strength": self.current_market_condition.trend_strength if self.current_market_condition else 0.0,
                "confidence": self.current_market_condition.confidence if self.current_market_condition else 0.0
            } if self.current_market_condition else {},
            "regime_performance": self.regime_performance,
            "is_running": self.is_running,
            "recent_transitions": len(self.transition_detector.transition_history[-10:])
        }


# Factory function for easy integration
def create_adaptive_regime_integration(moe_engine: MixtureOfExperts,
                                     learning_pipeline: Optional[ContinuousLearningPipeline] = None) -> AdaptiveRegimeIntegration:
    """Create and return configured adaptive regime integration system"""
    return AdaptiveRegimeIntegration(moe_engine, learning_pipeline)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create MoE engine (placeholder)
        moe_engine = MixtureOfExperts()
        
        # Create integration system
        integration = create_adaptive_regime_integration(moe_engine)
        
        # Register callbacks
        def on_regime_change(old_regime, new_regime, transition):
            print(f"Regime changed: {old_regime} -> {new_regime}")
        
        integration.register_regime_change_callback(on_regime_change)
        
        try:
            # Start system
            integration.start()
            
            # Simulate market data updates
            for i in range(100):
                price = 50000 + np.random.randn() * 100
                volume = 1000 + np.random.randn() * 200
                integration.update_market_data(price, volume)
                await asyncio.sleep(0.1)
            
            # Show status
            status = integration.get_system_status()
            print(f"System Status: {status}")
            
        finally:
            integration.stop()
    
    asyncio.run(main())