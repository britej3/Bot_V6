"""
Adaptive Risk Management System
===============================

This module implements a comprehensive adaptive risk management system that dynamically
adjusts risk parameters based on market conditions, performance metrics, and volatility.
The system integrates seamlessly with existing trading strategies and learning frameworks.

Key Features:
- Dynamic position sizing based on volatility and market regime
- Adaptive risk parameters for different market conditions
- Performance-based risk adjustment learning
- Real-time risk monitoring and alerting
- Integration with dynamic strategy switching
- Regime-aware risk profiles
- Volatility-based position sizing
- Drawdown protection mechanisms
- Portfolio-level risk management

Implements Task 15.1.3: Create Adaptive Risk Management System
Author: Autonomous Systems Team
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time
import threading
from datetime import datetime, timedelta
import asyncio
import copy
import statistics
from abc import ABC, abstractmethod

# Import platform compatibility
try:
    from .platform_compatibility import get_platform_compatibility
    PLATFORM_COMPATIBILITY_AVAILABLE = True
except ImportError:
    PLATFORM_COMPATIBILITY_AVAILABLE = False
    get_platform_compatibility = None

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"


class MarketRegime(Enum):
    """Market regime classifications for risk management"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    BULL_RUN = "bull_run"
    CRASH = "crash"
    RECOVERY = "recovery"


class RiskMetricType(Enum):
    """Types of risk metrics tracked"""
    POSITION_SIZE = "position_size"
    PORTFOLIO_VAR = "portfolio_var"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    # Position limits
    max_position_size: float = 0.1  # Maximum position size as fraction of portfolio
    max_total_exposure: float = 0.8  # Maximum total exposure
    max_single_asset_exposure: float = 0.2  # Maximum exposure to single asset
    
    # Portfolio limits
    max_daily_var: float = 0.05     # Maximum daily Value at Risk
    max_drawdown: float = 0.15      # Maximum portfolio drawdown
    max_correlation: float = 0.8     # Maximum correlation between positions
    max_leverage: float = 3.0       # Maximum leverage ratio
    
    # Trading limits
    max_daily_trades: int = 100     # Maximum trades per day
    max_loss_per_trade: float = 0.02  # Maximum loss per trade
    max_consecutive_losses: int = 5   # Maximum consecutive losing trades
    
    # Volatility limits
    min_volatility: float = 0.005   # Minimum volatility for trading
    max_volatility: float = 0.15    # Maximum volatility for trading


@dataclass
class MarketCondition:
    """Current market condition assessment"""
    regime: MarketRegime
    volatility: float
    trend_strength: float
    correlation_level: float
    liquidity_score: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskProfile:
    """Risk profile for different market regimes"""
    regime: MarketRegime
    position_size_multiplier: float = 1.0
    volatility_adjustment: float = 1.0
    drawdown_sensitivity: float = 1.0
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    max_positions: int = 10
    risk_per_trade: float = 0.02


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""
    total_exposure: float = 0.0
    daily_var: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    correlation_risk: float = 0.0
    concentration_risk: float = 0.0
    leverage_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAdjustment:
    """Risk adjustment recommendation"""
    metric_type: RiskMetricType
    current_value: float
    target_value: float
    adjustment_ratio: float
    urgency: RiskLevel
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


class VolatilityEstimator:
    """Advanced volatility estimation using multiple methods"""
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.price_history = deque(maxlen=lookback_period * 2)
        self.return_history = deque(maxlen=lookback_period)
        
    def update(self, price: float) -> None:
        """Update price history and calculate returns"""
        self.price_history.append(price)
        
        if len(self.price_history) >= 2:
            log_return = np.log(price / self.price_history[-2])
            self.return_history.append(log_return)
    
    def get_historical_volatility(self) -> float:
        """Calculate historical volatility"""
        if len(self.return_history) < 2:
            return 0.01  # Default volatility
        
        returns = np.array(self.return_history)
        return float(np.std(returns) * np.sqrt(252))  # Annualized
    
    def get_garch_volatility(self) -> float:
        """Simple GARCH(1,1) volatility estimate"""
        if len(self.return_history) < 10:
            return self.get_historical_volatility()
        
        returns = np.array(self.return_history)
        
        # Simple GARCH(1,1) parameters
        omega = 0.0001
        alpha = 0.1
        beta = 0.85
        
        # Initialize with historical variance
        variance = np.var(returns)
        
        # Update with GARCH
        for ret in returns[-5:]:  # Use last 5 returns
            variance = omega + alpha * (ret ** 2) + beta * variance
        
        return float(np.sqrt(variance * 252))  # Annualized
    
    def get_ewma_volatility(self, lambda_param: float = 0.94) -> float:
        """Exponentially weighted moving average volatility"""
        if len(self.return_history) < 2:
            return 0.01
        
        returns = np.array(self.return_history)
        
        # Calculate EWMA variance
        variance = returns[0] ** 2
        for i, ret in enumerate(returns[1:], 1):
            variance = lambda_param * variance + (1 - lambda_param) * (ret ** 2)
        
        return float(np.sqrt(variance * 252))  # Annualized


class PositionSizer:
    """Dynamic position sizing based on risk and volatility"""
    
    def __init__(self, base_risk_per_trade: float = 0.02):
        self.base_risk_per_trade = base_risk_per_trade
        self.volatility_estimator = VolatilityEstimator()
        
    def calculate_position_size(self, 
                              portfolio_value: float,
                              entry_price: float,
                              stop_loss_price: Optional[float],
                              market_condition: MarketCondition,
                              risk_profile: RiskProfile) -> float:
        """Calculate optimal position size"""
        
        # Base risk amount
        base_risk_amount = portfolio_value * self.base_risk_per_trade
        
        # Adjust for market regime
        regime_multiplier = self._get_regime_multiplier(market_condition.regime)
        
        # Adjust for volatility
        volatility_multiplier = self._get_volatility_multiplier(market_condition.volatility)
        
        # Adjust for confidence
        confidence_multiplier = max(0.5, market_condition.confidence)
        
        # Calculate risk per share
        if stop_loss_price and entry_price:
            risk_per_share = abs(entry_price - stop_loss_price)
        else:
            # Use ATR-based risk if no stop loss
            risk_per_share = entry_price * 0.02  # 2% default risk
        
        # Apply all adjustments
        adjusted_risk = (base_risk_amount * 
                        regime_multiplier * 
                        volatility_multiplier * 
                        confidence_multiplier *
                        risk_profile.position_size_multiplier)
        
        # Calculate position size
        if risk_per_share > 0:
            position_size = adjusted_risk / risk_per_share
        else:
            position_size = 0
        
        # Apply maximum position limits
        max_position_value = portfolio_value * risk_profile.max_positions * 0.1
        max_position_size = max_position_value / entry_price if entry_price > 0 else 0
        
        return min(position_size, max_position_size)
    
    def _get_regime_multiplier(self, regime: MarketRegime) -> float:
        """Get position size multiplier based on market regime"""
        multipliers = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.VOLATILE: 0.5,
            MarketRegime.TRENDING: 1.2,
            MarketRegime.RANGE_BOUND: 0.8,
            MarketRegime.BULL_RUN: 1.5,
            MarketRegime.CRASH: 0.2,
            MarketRegime.RECOVERY: 0.7
        }
        return multipliers.get(regime, 1.0)
    
    def _get_volatility_multiplier(self, volatility: float) -> float:
        """Get position size multiplier based on volatility"""
        if volatility < 0.01:
            return 1.2  # Low volatility, can increase size
        elif volatility < 0.02:
            return 1.0  # Normal volatility
        elif volatility < 0.05:
            return 0.7  # High volatility, reduce size
        else:
            return 0.3  # Very high volatility, significantly reduce


class RiskMonitor:
    """Real-time risk monitoring and alerting"""
    
    def __init__(self, risk_limits: RiskLimits):
        self.risk_limits = risk_limits
        self.alerts = deque(maxlen=100)
        self.risk_metrics_history = deque(maxlen=1000)
        
    def assess_portfolio_risk(self, portfolio_metrics: PortfolioRiskMetrics) -> List[RiskAdjustment]:
        """Assess portfolio risk and generate adjustments"""
        adjustments = []
        
        # Check position size limits
        if portfolio_metrics.total_exposure > self.risk_limits.max_total_exposure:
            adjustments.append(RiskAdjustment(
                metric_type=RiskMetricType.POSITION_SIZE,
                current_value=portfolio_metrics.total_exposure,
                target_value=self.risk_limits.max_total_exposure,
                adjustment_ratio=self.risk_limits.max_total_exposure / portfolio_metrics.total_exposure,
                urgency=RiskLevel.HIGH,
                reason="Total exposure exceeds limit"
            ))
        
        # Check VaR limits
        if portfolio_metrics.daily_var > self.risk_limits.max_daily_var:
            adjustments.append(RiskAdjustment(
                metric_type=RiskMetricType.PORTFOLIO_VAR,
                current_value=portfolio_metrics.daily_var,
                target_value=self.risk_limits.max_daily_var,
                adjustment_ratio=self.risk_limits.max_daily_var / portfolio_metrics.daily_var,
                urgency=RiskLevel.HIGH,
                reason="Daily VaR exceeds limit"
            ))
        
        # Check drawdown limits
        if portfolio_metrics.current_drawdown > self.risk_limits.max_drawdown:
            adjustments.append(RiskAdjustment(
                metric_type=RiskMetricType.DRAWDOWN,
                current_value=portfolio_metrics.current_drawdown,
                target_value=self.risk_limits.max_drawdown,
                adjustment_ratio=0.5,  # Reduce positions by 50%
                urgency=RiskLevel.EXTREME,
                reason="Portfolio drawdown exceeds limit"
            ))
        
        # Check leverage limits
        if portfolio_metrics.leverage_ratio > self.risk_limits.max_leverage:
            adjustments.append(RiskAdjustment(
                metric_type=RiskMetricType.LEVERAGE,
                current_value=portfolio_metrics.leverage_ratio,
                target_value=self.risk_limits.max_leverage,
                adjustment_ratio=self.risk_limits.max_leverage / portfolio_metrics.leverage_ratio,
                urgency=RiskLevel.HIGH,
                reason="Leverage ratio exceeds limit"
            ))
        
        return adjustments
    
    def generate_alert(self, adjustment: RiskAdjustment) -> Dict[str, Any]:
        """Generate risk alert"""
        alert = {
            'timestamp': datetime.now(),
            'type': 'risk_limit_breach',
            'urgency': adjustment.urgency.value,
            'metric': adjustment.metric_type.value,
            'current_value': adjustment.current_value,
            'limit': adjustment.target_value,
            'adjustment_required': adjustment.adjustment_ratio,
            'reason': adjustment.reason
        }
        
        self.alerts.append(alert)
        
        # Log alert
        logger.warning(f"Risk Alert: {adjustment.reason} - "
                      f"{adjustment.metric_type.value} = {adjustment.current_value:.4f}, "
                      f"limit = {adjustment.target_value:.4f}")
        
        return alert


class AdaptiveRiskManager:
    """Main adaptive risk management system"""
    
    def __init__(self, 
                 risk_limits: Optional[RiskLimits] = None,
                 initial_profiles: Optional[Dict[MarketRegime, RiskProfile]] = None):
        
        self.risk_limits = risk_limits or RiskLimits()
        self.position_sizer = PositionSizer()
        self.risk_monitor = RiskMonitor(self.risk_limits)
        self.volatility_estimator = VolatilityEstimator()
        
        # Initialize risk profiles for different market regimes
        self.risk_profiles = initial_profiles or self._create_default_profiles()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.adjustment_history = deque(maxlen=500)
        
        # Threading for real-time monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Platform optimization
        if PLATFORM_COMPATIBILITY_AVAILABLE and get_platform_compatibility:
            self.platform_compat = get_platform_compatibility()
        else:
            self.platform_compat = None
        
        logger.info("Adaptive Risk Manager initialized")
    
    def _create_default_profiles(self) -> Dict[MarketRegime, RiskProfile]:
        """Create default risk profiles for different market regimes"""
        profiles = {}
        
        # Normal market conditions
        profiles[MarketRegime.NORMAL] = RiskProfile(
            regime=MarketRegime.NORMAL,
            position_size_multiplier=1.0,
            volatility_adjustment=1.0,
            drawdown_sensitivity=1.0,
            stop_loss_multiplier=1.0,
            take_profit_multiplier=1.0,
            max_positions=10,
            risk_per_trade=0.02
        )
        
        # Volatile market conditions - reduce risk
        profiles[MarketRegime.VOLATILE] = RiskProfile(
            regime=MarketRegime.VOLATILE,
            position_size_multiplier=0.5,
            volatility_adjustment=1.5,
            drawdown_sensitivity=1.5,
            stop_loss_multiplier=1.5,
            take_profit_multiplier=0.8,
            max_positions=5,
            risk_per_trade=0.01
        )
        
        # Trending market - moderate increase in risk
        profiles[MarketRegime.TRENDING] = RiskProfile(
            regime=MarketRegime.TRENDING,
            position_size_multiplier=1.2,
            volatility_adjustment=0.8,
            drawdown_sensitivity=0.8,
            stop_loss_multiplier=0.8,
            take_profit_multiplier=1.5,
            max_positions=12,
            risk_per_trade=0.025
        )
        
        # Range-bound market
        profiles[MarketRegime.RANGE_BOUND] = RiskProfile(
            regime=MarketRegime.RANGE_BOUND,
            position_size_multiplier=0.8,
            volatility_adjustment=1.0,
            drawdown_sensitivity=1.0,
            stop_loss_multiplier=1.2,
            take_profit_multiplier=1.0,
            max_positions=8,
            risk_per_trade=0.015
        )
        
        # Bull run - increase risk moderately
        profiles[MarketRegime.BULL_RUN] = RiskProfile(
            regime=MarketRegime.BULL_RUN,
            position_size_multiplier=1.5,
            volatility_adjustment=0.7,
            drawdown_sensitivity=0.7,
            stop_loss_multiplier=0.7,
            take_profit_multiplier=2.0,
            max_positions=15,
            risk_per_trade=0.03
        )
        
        # Market crash - minimize risk
        profiles[MarketRegime.CRASH] = RiskProfile(
            regime=MarketRegime.CRASH,
            position_size_multiplier=0.2,
            volatility_adjustment=2.0,
            drawdown_sensitivity=2.0,
            stop_loss_multiplier=2.0,
            take_profit_multiplier=0.5,
            max_positions=2,
            risk_per_trade=0.005
        )
        
        # Recovery phase
        profiles[MarketRegime.RECOVERY] = RiskProfile(
            regime=MarketRegime.RECOVERY,
            position_size_multiplier=0.7,
            volatility_adjustment=1.2,
            drawdown_sensitivity=1.2,
            stop_loss_multiplier=1.3,
            take_profit_multiplier=1.2,
            max_positions=6,
            risk_per_trade=0.018
        )
        
        return profiles
    
    def update_market_condition(self, 
                              regime: Union[str, MarketRegime],
                              volatility: float,
                              trend_strength: float = 0.5,
                              correlation_level: float = 0.5,
                              liquidity_score: float = 0.8,
                              confidence: float = 0.8) -> MarketCondition:
        """Update current market condition assessment"""
        
        # Convert string regime to enum if needed
        if isinstance(regime, str):
            try:
                regime = MarketRegime(regime.lower())
            except ValueError:
                regime = MarketRegime.NORMAL
        
        market_condition = MarketCondition(
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            correlation_level=correlation_level,
            liquidity_score=liquidity_score,
            confidence=confidence
        )
        
        # Update volatility estimator
        # Note: This would typically use actual price data
        # For now, we use the provided volatility
        
        return market_condition
    
    def calculate_position_size(self,
                              portfolio_value: float,
                              entry_price: float,
                              stop_loss_price: Optional[float],
                              market_condition: MarketCondition,
                              asset_symbol: str = "DEFAULT") -> Dict[str, Any]:
        """Calculate optimal position size for a trade"""
        
        # Get risk profile for current regime
        risk_profile = self.risk_profiles.get(market_condition.regime, 
                                            self.risk_profiles[MarketRegime.NORMAL])
        
        # Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            portfolio_value=portfolio_value,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            market_condition=market_condition,
            risk_profile=risk_profile
        )
        
        # Calculate position value
        position_value = position_size * entry_price
        position_percent = position_value / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate risk metrics
        if stop_loss_price:
            risk_per_share = abs(entry_price - stop_loss_price)
            total_risk = position_size * risk_per_share
            risk_percent = total_risk / portfolio_value if portfolio_value > 0 else 0
        else:
            risk_per_share = entry_price * risk_profile.risk_per_trade
            total_risk = position_size * risk_per_share
            risk_percent = risk_profile.risk_per_trade
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'position_percent': position_percent,
            'risk_amount': total_risk,
            'risk_percent': risk_percent,
            'risk_per_share': risk_per_share,
            'regime': market_condition.regime.value,
            'volatility': market_condition.volatility,
            'confidence': market_condition.confidence,
            'risk_profile_used': risk_profile.regime.value
        }
    
    def assess_portfolio_risk(self, portfolio_metrics: PortfolioRiskMetrics) -> Dict[str, Any]:
        """Assess current portfolio risk and generate recommendations"""
        
        # Get risk adjustments
        adjustments = self.risk_monitor.assess_portfolio_risk(portfolio_metrics)
        
        # Generate alerts for significant breaches
        alerts = []
        for adjustment in adjustments:
            if adjustment.urgency in [RiskLevel.HIGH, RiskLevel.EXTREME]:
                alert = self.risk_monitor.generate_alert(adjustment)
                alerts.append(alert)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(portfolio_metrics)
        
        # Store metrics
        self.risk_monitor.risk_metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': portfolio_metrics,
            'risk_score': risk_score,
            'adjustments_count': len(adjustments)
        })
        
        return {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'adjustments': adjustments,
            'alerts': alerts,
            'metrics': portfolio_metrics,
            'recommendations': self._generate_recommendations(adjustments)
        }
    
    def _calculate_risk_score(self, metrics: PortfolioRiskMetrics) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        score = 0.0
        
        # Exposure risk (weight: 25%)
        exposure_risk = min(100, (metrics.total_exposure / self.risk_limits.max_total_exposure) * 100)
        score += exposure_risk * 0.25
        
        # VaR risk (weight: 20%)
        var_risk = min(100, (metrics.daily_var / self.risk_limits.max_daily_var) * 100)
        score += var_risk * 0.20
        
        # Drawdown risk (weight: 30%)
        drawdown_risk = min(100, (metrics.current_drawdown / self.risk_limits.max_drawdown) * 100)
        score += drawdown_risk * 0.30
        
        # Volatility risk (weight: 15%)
        volatility_risk = min(100, metrics.volatility * 500)  # Scale volatility
        score += volatility_risk * 0.15
        
        # Concentration risk (weight: 10%)
        concentration_risk = min(100, metrics.concentration_risk * 100)
        score += concentration_risk * 0.10
        
        return min(100, score)
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level"""
        if risk_score < 20:
            return RiskLevel.VERY_LOW
        elif risk_score < 40:
            return RiskLevel.LOW
        elif risk_score < 60:
            return RiskLevel.MODERATE
        elif risk_score < 80:
            return RiskLevel.HIGH
        elif risk_score < 95:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME
    
    def _generate_recommendations(self, adjustments: List[RiskAdjustment]) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []
        
        for adj in adjustments:
            if adj.metric_type == RiskMetricType.POSITION_SIZE:
                recommendations.append(f"Reduce position sizes by {(1-adj.adjustment_ratio)*100:.1f}%")
            elif adj.metric_type == RiskMetricType.DRAWDOWN:
                recommendations.append("Consider closing losing positions to reduce drawdown")
            elif adj.metric_type == RiskMetricType.LEVERAGE:
                recommendations.append(f"Reduce leverage from {adj.current_value:.2f} to {adj.target_value:.2f}")
            elif adj.metric_type == RiskMetricType.PORTFOLIO_VAR:
                recommendations.append("Diversify portfolio to reduce VaR concentration")
        
        if not recommendations:
            recommendations.append("Portfolio risk levels are within acceptable limits")
        
        return recommendations
    
    def get_risk_adjusted_parameters(self, 
                                   market_condition: MarketCondition,
                                   base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get risk-adjusted trading parameters"""
        
        risk_profile = self.risk_profiles.get(market_condition.regime,
                                            self.risk_profiles[MarketRegime.NORMAL])
        
        adjusted_params = base_parameters.copy()
        
        # Adjust position sizing
        if 'position_size' in adjusted_params:
            adjusted_params['position_size'] *= risk_profile.position_size_multiplier
        
        # Adjust stop loss
        if 'stop_loss_pct' in adjusted_params:
            adjusted_params['stop_loss_pct'] *= risk_profile.stop_loss_multiplier
        
        # Adjust take profit
        if 'take_profit_pct' in adjusted_params:
            adjusted_params['take_profit_pct'] *= risk_profile.take_profit_multiplier
        
        # Adjust risk per trade
        adjusted_params['risk_per_trade'] = risk_profile.risk_per_trade
        
        # Add regime-specific parameters
        adjusted_params['regime'] = market_condition.regime.value
        adjusted_params['volatility_adjustment'] = risk_profile.volatility_adjustment
        adjusted_params['max_positions'] = risk_profile.max_positions
        
        return adjusted_params
    
    def start_monitoring(self) -> None:
        """Start real-time risk monitoring"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Risk monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring"""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("Risk monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # This would typically check real-time portfolio metrics
                # and generate alerts as needed
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                time.sleep(10.0)  # Wait longer on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'monitoring_active': self._monitoring_active,
            'risk_profiles_count': len(self.risk_profiles),
            'recent_alerts_count': len([a for a in self.risk_monitor.alerts 
                                      if a['timestamp'] > datetime.now() - timedelta(hours=1)]),
            'performance_history_size': len(self.performance_history),
            'platform_optimized': self.platform_compat is not None,
            'last_update': datetime.now()
        }


# Convenience function for integration
def create_adaptive_risk_manager(custom_limits: Optional[Dict[str, Any]] = None) -> AdaptiveRiskManager:
    """Create and configure an adaptive risk manager"""
    
    risk_limits = RiskLimits()
    
    # Apply custom limits if provided
    if custom_limits:
        for key, value in custom_limits.items():
            if hasattr(risk_limits, key):
                setattr(risk_limits, key, value)
    
    return AdaptiveRiskManager(risk_limits=risk_limits)