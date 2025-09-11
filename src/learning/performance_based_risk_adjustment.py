"""
Performance-Based Risk Adjustment System
========================================

This module implements a learning system that adjusts risk parameters based on
trading performance outcomes and market conditions.

Key Features:
- Performance-based risk parameter learning
- Outcome-driven risk adjustments
- Market condition adaptive learning
- Risk-return optimization

Implements Task 15.1.3.3: Performance-based risk adjustment mechanisms
Author: Autonomous Systems Team
Date: 2025-01-22
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Individual trade outcome record"""
    trade_id: str
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    position_size: float
    market_regime: str
    volatility: float
    risk_parameters: Dict[str, float]
    strategy_type: str


@dataclass
class PerformanceMetrics:
    """Performance statistics for analysis"""
    total_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float


@dataclass
class RiskAdjustmentConfig:
    """Configuration for performance-based risk adjustment"""
    learning_rate: float = 0.01
    min_risk_multiplier: float = 0.1
    max_risk_multiplier: float = 3.0
    max_adjustment_step: float = 0.1
    min_trades_for_learning: int = 20
    performance_window: int = 100


class PerformanceAnalyzer:
    """Analyzes trading performance for risk adjustment learning"""
    
    def __init__(self):
        self.trade_outcomes = deque(maxlen=1000)
        
    def add_trade_outcome(self, outcome: TradeOutcome) -> None:
        """Add a new trade outcome"""
        self.trade_outcomes.append(outcome)
        
    def calculate_performance_metrics(self, window: int = 100) -> PerformanceMetrics:
        """Calculate performance metrics for recent trades"""
        
        recent_trades = list(self.trade_outcomes)[-window:] if window else list(self.trade_outcomes)
        
        if not recent_trades:
            return PerformanceMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Basic metrics
        total_trades = len(recent_trades)
        winning_trades = sum(1 for t in recent_trades if t.pnl > 0)
        win_rate = winning_trades / total_trades
        
        # Returns and risk metrics
        returns = [t.pnl_percent for t in recent_trades]
        total_return = sum(returns)
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
        
        # Drawdown calculation
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Sharpe ratio
        mean_return = statistics.mean(returns) if returns else 0.0
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        
        return PerformanceMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio
        )
    
    def analyze_risk_parameter_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by risk parameter levels"""
        
        if len(self.trade_outcomes) < 20:
            return {}
        
        # Group trades by risk levels
        risk_groups = {'low': [], 'medium': [], 'high': []}
        
        for trade in self.trade_outcomes:
            risk_level = trade.risk_parameters.get('risk_per_trade', 0.02)
            
            if risk_level < 0.015:
                risk_groups['low'].append(trade)
            elif risk_level < 0.025:
                risk_groups['medium'].append(trade)
            else:
                risk_groups['high'].append(trade)
        
        # Calculate performance for each group
        results = {}
        for group_name, trades in risk_groups.items():
            if len(trades) >= 5:
                avg_return = statistics.mean([t.pnl_percent for t in trades])
                win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
                
                results[group_name] = {
                    'avg_return': avg_return,
                    'win_rate': win_rate,
                    'trade_count': len(trades)
                }
        
        return results


class RiskParameterOptimizer:
    """Optimizes risk parameters based on performance feedback"""
    
    def __init__(self, config: RiskAdjustmentConfig):
        self.config = config
        self.current_parameters = {
            'risk_per_trade': 0.02,
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'volatility_adjustment': 1.0
        }
        self.parameter_history = deque(maxlen=200)
        self.performance_history = deque(maxlen=200)
        
    def update_parameters(self, metrics: PerformanceMetrics, 
                         market_condition: Dict[str, Any]) -> Dict[str, float]:
        """Update risk parameters based on performance"""
        
        # Calculate performance score (0-1 scale)
        performance_score = self._calculate_performance_score(metrics)
        
        # Store history
        self.performance_history.append(performance_score)
        self.parameter_history.append(self.current_parameters.copy())
        
        # Need sufficient history for learning
        if len(self.performance_history) < self.config.min_trades_for_learning:
            return self.current_parameters.copy()
        
        # Calculate updates for each parameter
        updates = {}
        for param_name in self.current_parameters.keys():
            gradient = self._calculate_gradient(param_name)
            update = self._apply_learning_rule(param_name, gradient, market_condition)
            updates[param_name] = update
        
        # Apply updates with bounds
        for param_name, update in updates.items():
            old_value = self.current_parameters[param_name]
            new_value = old_value + update
            
            # Apply parameter-specific bounds
            if param_name == 'risk_per_trade':
                new_value = max(0.005, min(0.05, new_value))
            else:
                new_value = max(self.config.min_risk_multiplier, 
                              min(self.config.max_risk_multiplier, new_value))
            
            self.current_parameters[param_name] = new_value
        
        return self.current_parameters.copy()
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate single performance score from metrics"""
        
        # Weighted combination of key metrics
        return_score = max(0, min(1, (metrics.total_return + 0.1) / 0.2))
        risk_score = max(0, 1 - metrics.max_drawdown / 0.15)
        sharpe_score = max(0, min(1, (metrics.sharpe_ratio + 1) / 3))
        
        # Combine with equal weights
        return (return_score + risk_score + sharpe_score) / 3
    
    def _calculate_gradient(self, param_name: str) -> float:
        """Calculate gradient using finite differences"""
        
        if len(self.performance_history) < 10:
            return 0.0
        
        # Get recent parameter values and performance scores
        recent_params = list(self.parameter_history)[-10:]
        recent_scores = list(self.performance_history)[-10:]
        
        # Calculate correlation between parameter changes and performance
        param_values = [p[param_name] for p in recent_params]
        
        if len(set(param_values)) < 2:
            return 0.0
        
        # Simple finite difference
        param_changes = []
        score_changes = []
        
        for i in range(1, len(recent_params)):
            param_change = param_values[i] - param_values[i-1]
            score_change = recent_scores[i] - recent_scores[i-1]
            
            if abs(param_change) > 1e-6:
                param_changes.append(param_change)
                score_changes.append(score_change)
        
        if not param_changes:
            return 0.0
        
        # Average gradient
        gradients = [sc / pc for pc, sc in zip(param_changes, score_changes)]
        return statistics.mean(gradients)
    
    def _apply_learning_rule(self, param_name: str, gradient: float, 
                           market_condition: Dict[str, Any]) -> float:
        """Apply learning rule with market condition adjustment"""
        
        # Base learning rate
        learning_rate = self.config.learning_rate
        
        # Adjust for market volatility
        volatility = market_condition.get('volatility', 0.02)
        volatility_factor = 1.0 / (1.0 + volatility * 5)
        learning_rate *= volatility_factor
        
        # Calculate update
        update = learning_rate * gradient
        
        # Clip to maximum step
        return max(-self.config.max_adjustment_step,
                  min(self.config.max_adjustment_step, update))


class PerformanceBasedRiskAdjuster:
    """Main system for performance-based risk adjustment"""
    
    def __init__(self, config: Optional[RiskAdjustmentConfig] = None):
        self.config = config or RiskAdjustmentConfig()
        self.analyzer = PerformanceAnalyzer()
        self.optimizer = RiskParameterOptimizer(self.config)
        
        self.last_adjustment_time = datetime.now()
        self.adjustment_count = 0
        
        logger.info("Performance-based risk adjuster initialized")
    
    def add_trade_outcome(self, 
                         trade_id: str,
                         entry_time: datetime,
                         exit_time: datetime,
                         pnl: float,
                         pnl_percent: float,
                         position_size: float,
                         market_regime: str,
                         volatility: float,
                         risk_parameters: Dict[str, float],
                         strategy_type: str = "default") -> None:
        """Add trade outcome for learning"""
        
        outcome = TradeOutcome(
            trade_id=trade_id,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl=pnl,
            pnl_percent=pnl_percent,
            position_size=position_size,
            market_regime=market_regime,
            volatility=volatility,
            risk_parameters=risk_parameters.copy(),
            strategy_type=strategy_type
        )
        
        self.analyzer.add_trade_outcome(outcome)
        logger.debug(f"Added trade outcome: {trade_id}, PnL: {pnl:.4f}")
    
    def update_risk_parameters(self, market_condition: Dict[str, Any],
                             force_update: bool = False) -> Dict[str, float]:
        """Update risk parameters based on performance"""
        
        # Check update frequency (minimum 1 hour)
        if not force_update:
            time_since_last = datetime.now() - self.last_adjustment_time
            if time_since_last.total_seconds() < 3600:
                return self.optimizer.current_parameters.copy()
        
        # Need sufficient trades
        if len(self.analyzer.trade_outcomes) < self.config.min_trades_for_learning:
            return self.optimizer.current_parameters.copy()
        
        # Calculate recent performance
        metrics = self.analyzer.calculate_performance_metrics(self.config.performance_window)
        
        # Update parameters
        old_params = self.optimizer.current_parameters.copy()
        new_params = self.optimizer.update_parameters(metrics, market_condition)
        
        self.last_adjustment_time = datetime.now()
        self.adjustment_count += 1
        
        # Log significant changes
        for param_name, new_value in new_params.items():
            old_value = old_params[param_name]
            if abs(new_value - old_value) / old_value > 0.05:  # >5% change
                logger.info(f"Risk parameter updated: {param_name} "
                           f"{old_value:.4f} -> {new_value:.4f}")
        
        return new_params
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get comprehensive performance analysis"""
        
        metrics = self.analyzer.calculate_performance_metrics()
        risk_analysis = self.analyzer.analyze_risk_parameter_performance()
        
        return {
            'current_parameters': self.optimizer.current_parameters.copy(),
            'performance_metrics': {
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'total_return': metrics.total_return,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio
            },
            'risk_analysis': risk_analysis,
            'adjustment_count': self.adjustment_count,
            'trade_count': len(self.analyzer.trade_outcomes)
        }
    
    def get_recommendations(self) -> List[str]:
        """Get specific risk management recommendations"""
        
        recommendations = []
        
        if len(self.analyzer.trade_outcomes) < 20:
            recommendations.append("Insufficient trade data for meaningful analysis")
            return recommendations
        
        metrics = self.analyzer.calculate_performance_metrics()
        
        # Win rate recommendations
        if metrics.win_rate < 0.4:
            recommendations.append("Low win rate - consider reducing position sizes")
        elif metrics.win_rate > 0.7:
            recommendations.append("High win rate - potential for increased position sizes")
        
        # Drawdown recommendations
        if metrics.max_drawdown > 0.1:
            recommendations.append("High drawdown detected - implement stricter risk controls")
        
        # Volatility recommendations
        if metrics.volatility > 0.05:
            recommendations.append("High volatility - consider reducing risk exposure")
        
        # Sharpe ratio recommendations
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Low risk-adjusted returns - review strategy and risk parameters")
        
        return recommendations


# Factory function
def create_performance_risk_adjuster(custom_config: Optional[Dict[str, Any]] = None) -> PerformanceBasedRiskAdjuster:
    """Create performance-based risk adjuster with custom configuration"""
    
    config = RiskAdjustmentConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return PerformanceBasedRiskAdjuster(config)


# Demo
if __name__ == "__main__":
    print("🎯 Performance-Based Risk Adjustment System Demo")
    print("=" * 50)
    
    # Create adjuster
    adjuster = create_performance_risk_adjuster()
    
    # Simulate trades
    import random
    from datetime import timedelta
    
    start_time = datetime.now() - timedelta(days=30)
    
    for i in range(100):
        trade_time = start_time + timedelta(hours=i*6)
        
        # Simulate trade outcome
        pnl_pct = random.gauss(0.01, 0.03)  # 1% average return, 3% volatility
        
        adjuster.add_trade_outcome(
            trade_id=f"trade_{i}",
            entry_time=trade_time,
            exit_time=trade_time + timedelta(hours=4),
            pnl=pnl_pct * 1000,  # $1000 position
            pnl_percent=pnl_pct,
            position_size=1000,
            market_regime="normal",
            volatility=0.02,
            risk_parameters={'risk_per_trade': 0.02},
            strategy_type="test"
        )
    
    # Test parameter updates
    market_condition = {'volatility': 0.02, 'regime': 'normal'}
    updated_params = adjuster.update_risk_parameters(market_condition, force_update=True)
    
    print(f"Updated parameters: {updated_params}")
    
    # Get analysis
    analysis = adjuster.get_performance_analysis()
    print(f"Performance metrics: {analysis['performance_metrics']}")
    
    # Get recommendations
    recommendations = adjuster.get_recommendations()
    print(f"Recommendations: {recommendations}")
    
    print("\n🎯 Task 15.1.3.3: Performance-Based Risk Adjustment - IMPLEMENTATION COMPLETE")