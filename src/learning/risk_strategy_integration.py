"""
Risk-Strategy Integration Layer
==============================

This module implements the integration layer between the adaptive risk management
system and the dynamic strategy switching system, enabling coordinated risk-strategy
management that responds to market conditions.

Key Features:
- Coordinated risk-strategy decision making
- Unified market condition assessment
- Risk-aware strategy selection
- Strategy-aware risk parameter adjustment
- Performance feedback integration

Implements Task 15.1.3.2: Integration layer with existing dynamic strategy switching
Author: Autonomous Systems Team  
Date: 2025-01-22
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# Type imports only for type hints
if TYPE_CHECKING:
    from .adaptive_risk_management import (
        AdaptiveRiskManager, MarketCondition, MarketRegime as RiskMarketRegime,
        RiskLevel, PortfolioRiskMetrics
    )
    from .dynamic_strategy_switching import (
        DynamicStrategyManager, StrategyType, MoESignal
    )
else:
    # Runtime fallbacks for when modules might not be available
    AdaptiveRiskManager = Any
    DynamicStrategyManager = Any
    StrategyType = Any
    MoESignal = Any
    MarketCondition = Any
    PortfolioRiskMetrics = Any
    RiskLevel = Any
    RiskMarketRegime = Any

# Try to import at runtime for actual usage
SYSTEMS_AVAILABLE = True
try:
    from .adaptive_risk_management import MarketRegime as RiskMarketRegime
    from .dynamic_strategy_switching import StrategyType as RuntimeStrategyType
except ImportError as e:
    SYSTEMS_AVAILABLE = False
    logger.warning(f"Could not import required systems: {e}")
    
    # Define fallback enums
    class RiskMarketRegime:
        NORMAL = "normal"
        VOLATILE = "volatile"
        TRENDING = "trending"
        RANGE_BOUND = "range_bound"
        BULL_RUN = "bull_run"
        CRASH = "crash"
    
    class RuntimeStrategyType:
        MOMENTUM = "momentum"
        MEAN_REVERSION = "mean_reversion"
        MARKET_MAKING = "market_making"


class CoordinationMode(Enum):
    """Coordination modes between risk and strategy systems"""
    RISK_DRIVEN = "risk_driven"         # Risk management drives strategy selection
    STRATEGY_DRIVEN = "strategy_driven" # Strategy drives risk parameters
    COORDINATED = "coordinated"         # Full coordination between systems


@dataclass
class IntegrationConfig:
    """Configuration for risk-strategy integration"""
    coordination_mode: CoordinationMode = CoordinationMode.COORDINATED
    risk_weight: float = 0.6            # Weight for risk considerations (0-1)
    strategy_weight: float = 0.4        # Weight for strategy considerations (0-1)
    max_position_adjustment: float = 0.5 # Maximum position size adjustment
    coordination_interval: float = 5.0   # Seconds between coordination cycles


@dataclass
class CoordinatedDecision:
    """Result of coordinated risk-strategy decision making"""
    strategy_type: Optional[Any]
    risk_parameters: Dict[str, float]
    position_adjustments: Dict[str, float]
    reasoning: str
    confidence: float
    risk_score: float
    expected_performance: float
    timestamp: datetime = field(default_factory=datetime.now)


class MarketConditionUnifier:
    """Unifies market condition assessment between systems"""
    
    def __init__(self):
        self.regime_mapping = {
            "normal": {"risk_regime": RiskMarketRegime.NORMAL, "risk_multiplier": 1.0},
            "volatile": {"risk_regime": RiskMarketRegime.VOLATILE, "risk_multiplier": 0.5},
            "trending": {"risk_regime": RiskMarketRegime.TRENDING, "risk_multiplier": 1.2},
            "range_bound": {"risk_regime": RiskMarketRegime.RANGE_BOUND, "risk_multiplier": 0.8},
            "bull_run": {"risk_regime": RiskMarketRegime.BULL_RUN, "risk_multiplier": 1.5},
            "crash": {"risk_regime": RiskMarketRegime.CRASH, "risk_multiplier": 0.2}
        }
    
    def unify_market_condition(self, regime: str, volatility: float, 
                             confidence: float = 0.8) -> Dict[str, Any]:
        """Create unified market condition for both systems"""
        
        regime_info = self.regime_mapping.get(regime.lower(), 
                                            self.regime_mapping["normal"])
        
        return {
            "regime_str": regime.lower(),
            "risk_regime": regime_info["risk_regime"],
            "volatility": volatility,
            "confidence": confidence,
            "risk_multiplier": regime_info["risk_multiplier"],
            "risk_market_condition": MarketCondition(
                regime=regime_info["risk_regime"],
                volatility=volatility,
                trend_strength=0.5,
                correlation_level=0.5,
                liquidity_score=0.8,
                confidence=confidence
            )
        }


class CoordinatedDecisionEngine:
    """Engine for making coordinated risk-strategy decisions"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.decision_history = deque(maxlen=500)
        
    def make_coordinated_decision(self,
                                unified_condition: Dict[str, Any],
                                risk_assessment: Dict[str, Any],
                                strategy_signal: Optional[MoESignal],
                                portfolio_metrics: PortfolioRiskMetrics) -> CoordinatedDecision:
        """Make a coordinated decision considering both risk and strategy factors"""
        
        # Determine strategy
        target_strategy = None
        expected_performance = 0.0
        
        if strategy_signal:
            target_strategy = self._infer_strategy_from_signal(strategy_signal)
            expected_performance = strategy_signal.confidence * strategy_signal.size
        
        # Calculate risk parameters
        risk_params = self._calculate_risk_parameters(unified_condition)
        
        # Calculate position adjustments
        position_adjustments = self._calculate_position_adjustments(
            risk_assessment, strategy_signal
        )
        
        # Calculate overall confidence
        risk_confidence = 1.0 - (risk_assessment.get('risk_score', 0) / 100.0)
        strategy_confidence = strategy_signal.confidence if strategy_signal else 0.5
        overall_confidence = (self.config.risk_weight * risk_confidence + 
                            self.config.strategy_weight * strategy_confidence)
        
        decision = CoordinatedDecision(
            strategy_type=target_strategy,
            risk_parameters=risk_params,
            position_adjustments=position_adjustments,
            reasoning=f"Coordinated decision: risk_weight={self.config.risk_weight:.2f}",
            confidence=overall_confidence,
            risk_score=risk_assessment.get('risk_score', 0),
            expected_performance=expected_performance
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _infer_strategy_from_signal(self, signal: MoESignal) -> Optional[StrategyType]:
        """Infer strategy type from MoE signal characteristics"""
        if not signal:
            return None
        
        regime = signal.regime.lower() if signal.regime else ""
        direction = abs(signal.direction)
        
        if regime in ["trending", "bull_run"] and direction > 0.5:
            return StrategyType.MOMENTUM
        elif regime in ["range_bound", "volatile"] and direction < 0.3:
            return StrategyType.MEAN_REVERSION
        elif regime in ["normal", "range_bound"] and signal.size < 0.4:
            return StrategyType.MARKET_MAKING
        
        return StrategyType.MEAN_REVERSION  # Default conservative choice
    
    def _calculate_risk_parameters(self, unified_condition: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk parameters based on conditions"""
        base_risk = 0.02
        risk_multiplier = unified_condition.get('risk_multiplier', 1.0)
        volatility = unified_condition.get('volatility', 0.02)
        
        return {
            'risk_per_trade': base_risk * risk_multiplier,
            'max_position_size': 0.1 * risk_multiplier,
            'stop_loss_multiplier': 1.0 / risk_multiplier,
            'volatility_adjustment': max(0.5, min(2.0, volatility * 50))
        }
    
    def _calculate_position_adjustments(self,
                                      risk_assessment: Dict[str, Any],
                                      strategy_signal: Optional[MoESignal]) -> Dict[str, float]:
        """Calculate position adjustments balancing risk and strategy needs"""
        adjustments = {}
        
        # Risk-based adjustments
        risk_adjustments = risk_assessment.get('adjustments', [])
        risk_multiplier = 1.0
        
        for adjustment in risk_adjustments:
            if adjustment.urgency in [RiskLevel.HIGH, RiskLevel.EXTREME]:
                risk_multiplier = min(risk_multiplier, adjustment.adjustment_ratio)
        
        # Strategy-based adjustments
        strategy_multiplier = 1.0
        if strategy_signal:
            strategy_multiplier = min(2.0, max(0.5, strategy_signal.confidence))
        
        # Blend adjustments
        final_multiplier = (self.config.risk_weight * risk_multiplier + 
                          self.config.strategy_weight * strategy_multiplier)
        
        # Apply maximum adjustment limit
        final_multiplier = max(1.0 - self.config.max_position_adjustment,
                             min(1.0 + self.config.max_position_adjustment, final_multiplier))
        
        adjustments['global_position_multiplier'] = final_multiplier
        return adjustments


class RiskStrategyIntegrator:
    """Main integration system coordinating risk management and strategy switching"""
    
    def __init__(self, 
                 risk_manager: Optional[AdaptiveRiskManager] = None,
                 strategy_manager: Optional[DynamicStrategyManager] = None,
                 config: Optional[IntegrationConfig] = None):
        
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.config = config or IntegrationConfig()
        
        # Integration components
        self.condition_unifier = MarketConditionUnifier()
        self.decision_engine = CoordinatedDecisionEngine(self.config)
        
        # State tracking
        self.current_unified_condition = None
        self.last_decision = None
        self.integration_active = False
        
        # Performance tracking
        self.coordination_metrics = {
            'decisions_made': 0,
            'conflicts_resolved': 0,
            'performance_improvements': 0
        }
        
        logger.info("Risk-Strategy Integration system initialized")
    
    def start_integration(self) -> bool:
        """Start the integrated coordination system"""
        if not SYSTEMS_AVAILABLE:
            logger.error("Required systems not available for integration")
            return False
        
        if not self.risk_manager or not self.strategy_manager:
            logger.error("Risk manager or strategy manager not provided")
            return False
        
        self.integration_active = True
        logger.info("Risk-Strategy integration started")
        return True
    
    def stop_integration(self) -> None:
        """Stop the integration system"""
        self.integration_active = False
        logger.info("Risk-Strategy integration stopped")
    
    def update_market_conditions(self, regime: str, volatility: float, 
                               confidence: float = 0.8) -> Dict[str, Any]:
        """Update market conditions for both systems"""
        
        # Create unified market condition
        self.current_unified_condition = self.condition_unifier.unify_market_condition(
            regime, volatility, confidence
        )
        
        # Update individual systems
        if self.risk_manager:
            self.risk_manager.update_market_condition(
                regime=self.current_unified_condition['risk_regime'],
                volatility=volatility,
                confidence=confidence
            )
        
        if self.strategy_manager:
            self.strategy_manager.update_regime(
                regime, confidence, self.current_unified_condition
            )
        
        return self.current_unified_condition
    
    def generate_coordinated_signal(self,
                                  market_data: torch.Tensor,
                                  portfolio_value: float,
                                  portfolio_metrics: PortfolioRiskMetrics) -> Optional[Dict[str, Any]]:
        """Generate coordinated trading signal considering both risk and strategy"""
        
        if not self.current_unified_condition:
            logger.warning("No market condition available for coordinated signal")
            return None
        
        # Get strategy signal
        strategy_signal = None
        if self.strategy_manager:
            strategy_signal = self.strategy_manager.generate_signal(
                market_data,
                self.current_unified_condition['regime_str'],
                self.current_unified_condition['confidence']
            )
        
        # Get risk assessment
        risk_assessment = {}
        if self.risk_manager:
            risk_assessment = self.risk_manager.assess_portfolio_risk(portfolio_metrics)
        
        # Make coordinated decision
        decision = self.decision_engine.make_coordinated_decision(
            self.current_unified_condition,
            risk_assessment,
            strategy_signal,
            portfolio_metrics
        )
        
        self.last_decision = decision
        self.coordination_metrics['decisions_made'] += 1
        
        # Create coordinated signal if strategy signal exists
        if strategy_signal and decision.strategy_type:
            
            # Calculate risk-adjusted position size
            position_info = self.risk_manager.calculate_position_size(
                portfolio_value=portfolio_value,
                entry_price=1.0,
                stop_loss_price=None,
                market_condition=self.current_unified_condition['risk_market_condition'],
                asset_symbol="DEFAULT"
            )
            
            # Apply coordinated adjustments
            adjusted_position_size = position_info['position_size']
            for adj_key, adj_value in decision.position_adjustments.items():
                if adj_key == 'global_position_multiplier':
                    adjusted_position_size *= adj_value
            
            return {
                'direction': strategy_signal.direction,
                'confidence': decision.confidence,
                'size': min(strategy_signal.size, adjusted_position_size),
                'strategy_type': decision.strategy_type.value,
                'risk_parameters': decision.risk_parameters,
                'position_adjustments': decision.position_adjustments,
                'reasoning': decision.reasoning,
                'risk_score': decision.risk_score,
                'expected_performance': decision.expected_performance
            }
        
        return None
    
    def update_trade_performance(self, trade_result: Dict[str, Any]) -> None:
        """Update performance feedback for both systems"""
        
        # Update strategy manager
        if self.strategy_manager:
            self.strategy_manager.update_trade_result(trade_result)
        
        # Track performance improvement
        if self.last_decision and trade_result.get('pnl', 0) > 0:
            self.coordination_metrics['performance_improvements'] += 1
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'integration_active': self.integration_active,
            'has_risk_manager': self.risk_manager is not None,
            'has_strategy_manager': self.strategy_manager is not None,
            'coordination_metrics': self.coordination_metrics,
            'last_decision_time': self.last_decision.timestamp if self.last_decision else None,
            'current_regime': self.current_unified_condition['regime_str'] if self.current_unified_condition else None
        }


# Convenience functions
def create_integrated_system(custom_risk_limits: Optional[Dict[str, Any]] = None,
                           custom_integration_config: Optional[Dict[str, Any]] = None) -> RiskStrategyIntegrator:
    """Create fully integrated risk-strategy system"""
    
    # Create risk manager
    from .adaptive_risk_management import create_adaptive_risk_manager
    risk_manager = create_adaptive_risk_manager(custom_risk_limits)
    
    # Create strategy manager
    from .dynamic_strategy_switching import create_dynamic_strategy_manager
    strategy_manager = create_dynamic_strategy_manager()
    
    # Create integration config
    integration_config = IntegrationConfig()
    if custom_integration_config:
        for key, value in custom_integration_config.items():
            if hasattr(integration_config, key):
                setattr(integration_config, key, value)
    
    # Create integrated system
    integrator = RiskStrategyIntegrator(
        risk_manager=risk_manager,
        strategy_manager=strategy_manager,
        config=integration_config
    )
    
    return integrator


# Demo function
if __name__ == "__main__":
    print("🎯 Risk-Strategy Integration System Demo")
    print("=" * 50)
    
    # Create integrated system
    integrator = create_integrated_system()
    
    # Start integration
    if integrator.start_integration():
        print("✅ Integration started successfully")
        
        # Simulate market conditions
        test_regimes = ["normal", "volatile", "trending", "range_bound"]
        
        for regime in test_regimes:
            print(f"\n📊 Testing regime: {regime}")
            
            # Update market conditions
            integrator.update_market_conditions(regime, 0.02, 0.8)
            
            # Create mock portfolio metrics
            portfolio_metrics = PortfolioRiskMetrics(
                total_exposure=0.5,
                daily_var=0.02,
                current_drawdown=0.05,
                volatility=0.02
            )
            
            # Generate coordinated signal
            market_data = torch.randn(100)
            signal = integrator.generate_coordinated_signal(
                market_data, 100000.0, portfolio_metrics
            )
            
            if signal:
                print(f"   📈 Signal: {signal['direction']:.3f} direction, "
                      f"{signal['confidence']:.3f} confidence")
                print(f"   🎯 Strategy: {signal['strategy_type']}")
                print(f"   ⚠️ Risk Score: {signal['risk_score']:.1f}")
            else:
                print("   ❌ No coordinated signal generated")
        
        # Show integration status
        status = integrator.get_integration_status()
        print(f"\n🎯 Integration Status:")
        print(f"   Decisions Made: {status['coordination_metrics']['decisions_made']}")
        print(f"   Performance Improvements: {status['coordination_metrics']['performance_improvements']}")
        
        integrator.stop_integration()
        print("✅ Integration stopped")
        
    print(f"\n🎯 Task 15.1.3.2: Risk-Strategy Integration - IMPLEMENTATION COMPLETE")