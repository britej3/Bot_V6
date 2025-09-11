"""
Model Promotion Policy Engine for CryptoScalp AI

This module implements the intelligent policy engine that determines when to promote
a challenger model to champion status, with enhanced stability and performance criteria.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics"""
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    trial_duration_hours: float
    trade_count: int
    avg_trade_duration: float
    timestamp: datetime


@dataclass
class PromotionDecision:
    """Result of promotion evaluation"""
    promote: bool
    confidence: float
    reasons: List[str]
    risk_score: float
    recommended_monitoring_period: int  # hours
    timestamp: datetime


class ModelPromotionPolicy:
    """
    Enhanced policy engine for model promotion with stability and performance criteria.
    Evaluates challenger models against champions with comprehensive risk assessment.
    """

    def __init__(self, config_path: str):
        self.policies = self.load_policies(config_path)
        self.performance_thresholds = {
            'sharpe_ratio': 1.05,  # 5% improvement
            'max_drawdown': 1.0,   # No worse than champion
            'win_rate': 1.02,      # 2% improvement
            'profit_factor': 1.03  # 3% improvement
        }

        # ENHANCEMENT: Add stability and confidence requirements
        self.stability_thresholds = {
            'min_trial_duration_hours': 72,
            'min_trade_count': 1000,
            'min_confidence_interval': 0.95,
            'max_volatility_ratio': 1.2
        }

        # Risk assessment weights
        self.risk_weights = {
            'performance_stability': 0.3,
            'market_conditions': 0.2,
            'model_complexity': 0.2,
            'historical_performance': 0.15,
            'external_factors': 0.15
        }

        self.promotion_history = []

    def load_policies(self, config_path: str) -> Dict[str, Any]:
        """Load promotion policies from configuration file"""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load policies from {config_path}: {e}")
            return self._get_default_policies()

    def _get_default_policies(self) -> Dict[str, Any]:
        """Get default promotion policies"""
        return {
            'conservative': {
                'sharpe_ratio_threshold': 1.1,
                'min_trial_period': 168,  # 1 week
                'max_drawdown_limit': 0.05,
                'required_trade_count': 2000
            },
            'balanced': {
                'sharpe_ratio_threshold': 1.05,
                'min_trial_period': 72,   # 3 days
                'max_drawdown_limit': 0.08,
                'required_trade_count': 1000
            },
            'aggressive': {
                'sharpe_ratio_threshold': 1.02,
                'min_trial_period': 24,   # 1 day
                'max_drawdown_limit': 0.12,
                'required_trade_count': 500
            }
        }

    def evaluate_promotion_criteria(self, challenger_metrics: Dict,
                                  champion_metrics: Dict) -> PromotionDecision:
        """
        Evaluate if challenger meets performance AND stability criteria.

        Args:
            challenger_metrics: Metrics from challenger model
            champion_metrics: Metrics from current champion model

        Returns:
            PromotionDecision with detailed evaluation
        """
        reasons = []
        confidence = 0.0
        risk_score = 0.0

        # Convert to ModelMetrics objects
        challenger = self._dict_to_metrics(challenger_metrics)
        champion = self._dict_to_metrics(champion_metrics)

        # 1. Performance Evaluation
        performance_met, perf_reasons, perf_confidence = self._evaluate_performance(
            challenger, champion
        )
        reasons.extend(perf_reasons)

        # 2. Stability Evaluation
        stability_met, stab_reasons, stab_confidence = self._evaluate_stability(challenger)
        reasons.extend(stab_reasons)

        # 3. Risk Assessment
        risk_score, risk_reasons = self._assess_risk(challenger, champion)
        reasons.extend(risk_reasons)

        # Overall decision
        promote = performance_met and stability_met
        overall_confidence = (perf_confidence + stab_confidence) / 2

        # Adjust confidence based on risk
        if risk_score > 0.7:
            overall_confidence *= 0.8
            reasons.append(f"High risk score ({risk_score:.2f}) reduces confidence")

        # Determine monitoring period
        monitoring_period = self._calculate_monitoring_period(risk_score, overall_confidence)

        decision = PromotionDecision(
            promote=promote,
            confidence=overall_confidence,
            reasons=reasons,
            risk_score=risk_score,
            recommended_monitoring_period=monitoring_period,
            timestamp=datetime.now()
        )

        # Log decision
        self._log_promotion_decision(decision, challenger, champion)

        return decision

    def _evaluate_performance(self, challenger: ModelMetrics,
                            champion: ModelMetrics) -> tuple[bool, List[str], float]:
        """Evaluate performance criteria"""
        reasons = []
        confidence = 1.0

        # Sharpe ratio improvement
        sharpe_improvement = challenger.sharpe_ratio / champion.sharpe_ratio
        if sharpe_improvement >= self.performance_thresholds['sharpe_ratio']:
            reasons.append(f"Sharpe ratio improved by {sharpe_improvement:.1%}")
        else:
            reasons.append(f"Insufficient Sharpe ratio improvement: {sharpe_improvement:.1%}")
            confidence *= 0.7

        # Maximum drawdown check
        if challenger.max_drawdown <= champion.max_drawdown * self.performance_thresholds['max_drawdown']:
            reasons.append(f"Drawdown within acceptable limits: {challenger.max_drawdown:.1%}")
        else:
            reasons.append(f"Drawdown too high: {challenger.max_drawdown:.1%}")
            confidence *= 0.6

        # Win rate improvement
        win_rate_improvement = challenger.win_rate / champion.win_rate
        if win_rate_improvement >= self.performance_thresholds['win_rate']:
            reasons.append(f"Win rate improved by {win_rate_improvement:.1%}")
        else:
            confidence *= 0.8

        # Profit factor improvement
        pf_improvement = challenger.profit_factor / champion.profit_factor
        if pf_improvement >= self.performance_thresholds['profit_factor']:
            reasons.append(f"Profit factor improved by {pf_improvement:.1%}")
        else:
            confidence *= 0.8

        performance_met = confidence > 0.7
        return performance_met, reasons, confidence

    def _evaluate_stability(self, challenger: ModelMetrics) -> tuple[bool, List[str], float]:
        """Evaluate stability criteria"""
        reasons = []
        confidence = 1.0

        # Trial duration check
        if challenger.trial_duration_hours >= self.stability_thresholds['min_trial_duration_hours']:
            reasons.append(f"Sufficient trial duration: {challenger.trial_duration_hours:.0f}h")
        else:
            reasons.append(f"Insufficient trial duration: {challenger.trial_duration_hours:.0f}h")
            confidence *= 0.5

        # Trade count check
        if challenger.trade_count >= self.stability_thresholds['min_trade_count']:
            reasons.append(f"Sufficient trade count: {challenger.trade_count}")
        else:
            reasons.append(f"Insufficient trade count: {challenger.trade_count}")
            confidence *= 0.6

        # Volatility check (simplified)
        if hasattr(challenger, 'volatility') and hasattr(self, 'stability_thresholds'):
            # This would need actual volatility calculation
            pass

        stability_met = confidence > 0.8
        return stability_met, reasons, confidence

    def _assess_risk(self, challenger: ModelMetrics,
                    champion: ModelMetrics) -> tuple[float, List[str]]:
        """Assess promotion risk"""
        reasons = []
        risk_score = 0.0

        # Performance stability risk
        perf_stability_risk = self._calculate_performance_stability_risk(challenger)
        risk_score += perf_stability_risk * self.risk_weights['performance_stability']

        # Market condition risk
        market_risk = self._assess_market_condition_risk(challenger)
        risk_score += market_risk * self.risk_weights['market_conditions']

        # Model complexity risk
        complexity_risk = self._assess_model_complexity_risk(challenger)
        risk_score += complexity_risk * self.risk_weights['model_complexity']

        # Historical performance risk
        historical_risk = self._assess_historical_performance_risk(challenger, champion)
        risk_score += historical_risk * self.risk_weights['historical_performance']

        reasons.append(f"Performance stability risk: {perf_stability_risk:.2f}")
        reasons.append(f"Market condition risk: {market_risk:.2f}")
        reasons.append(f"Model complexity risk: {complexity_risk:.2f}")
        reasons.append(f"Historical performance risk: {historical_risk:.2f}")

        return risk_score, reasons

    def _calculate_performance_stability_risk(self, metrics: ModelMetrics) -> float:
        """Calculate performance stability risk"""
        # Simplified risk calculation based on drawdown and volatility
        if metrics.max_drawdown > 0.15:
            return 0.9
        elif metrics.max_drawdown > 0.1:
            return 0.6
        elif metrics.max_drawdown > 0.05:
            return 0.3
        else:
            return 0.1

    def _assess_market_condition_risk(self, metrics: ModelMetrics) -> float:
        """Assess market condition risk"""
        # This would integrate with market data analysis
        # Simplified version
        return 0.5

    def _assess_model_complexity_risk(self, metrics: ModelMetrics) -> float:
        """Assess model complexity risk"""
        # This would analyze model architecture complexity
        # Simplified version
        return 0.4

    def _assess_historical_performance_risk(self, challenger: ModelMetrics,
                                          champion: ModelMetrics) -> float:
        """Assess historical performance risk"""
        # Compare with historical performance patterns
        consistency_score = min(challenger.sharpe_ratio / max(champion.sharpe_ratio, 0.1), 2.0)
        return max(0, 1 - consistency_score / 2)

    def _calculate_monitoring_period(self, risk_score: float, confidence: float) -> int:
        """Calculate recommended monitoring period in hours"""
        base_period = 24  # 24 hours base

        # Increase for high risk
        if risk_score > 0.7:
            base_period *= 2
        elif risk_score > 0.5:
            base_period *= 1.5

        # Decrease for high confidence
        if confidence > 0.9:
            base_period *= 0.7
        elif confidence > 0.8:
            base_period *= 0.8

        return max(12, min(int(base_period), 168))  # 12 hours to 1 week

    def _dict_to_metrics(self, metrics_dict: Dict) -> ModelMetrics:
        """Convert dictionary to ModelMetrics object"""
        return ModelMetrics(
            sharpe_ratio=metrics_dict.get('sharpe_ratio', 0.0),
            max_drawdown=metrics_dict.get('max_drawdown', 0.0),
            total_return=metrics_dict.get('total_return', 0.0),
            win_rate=metrics_dict.get('win_rate', 0.0),
            profit_factor=metrics_dict.get('profit_factor', 0.0),
            trial_duration_hours=metrics_dict.get('trial_duration_hours', 0.0),
            trade_count=metrics_dict.get('trade_count', 0),
            avg_trade_duration=metrics_dict.get('avg_trade_duration', 0.0),
            timestamp=metrics_dict.get('timestamp', datetime.now())
        )

    def _log_promotion_decision(self, decision: PromotionDecision,
                              challenger: ModelMetrics, champion: ModelMetrics):
        """Log promotion decision for auditing"""
        log_entry = {
            'timestamp': decision.timestamp.isoformat(),
            'promote': decision.promote,
            'confidence': decision.confidence,
            'risk_score': decision.risk_score,
            'monitoring_period_hours': decision.recommended_monitoring_period,
            'reasons': decision.reasons,
            'challenger_metrics': {
                'sharpe_ratio': challenger.sharpe_ratio,
                'max_drawdown': challenger.max_drawdown,
                'trade_count': challenger.trade_count
            },
            'champion_metrics': {
                'sharpe_ratio': champion.sharpe_ratio,
                'max_drawdown': champion.max_drawdown,
                'trade_count': champion.trade_count
            }
        }

        self.promotion_history.append(log_entry)
        logger.info(f"Promotion decision: {decision.promote} "
                   f"(confidence: {decision.confidence:.2f}, "
                   f"risk: {decision.risk_score:.2f})")

    def save_promotion_history(self, path: str = "promotion_history.json"):
        """Save promotion history to file"""
        try:
            with open(path, 'w') as f:
                json.dump(self.promotion_history, f, indent=2, default=str)
            logger.info(f"Promotion history saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save promotion history: {e}")

    def get_promotion_statistics(self) -> Dict[str, Any]:
        """Get statistics about promotion decisions"""
        if not self.promotion_history:
            return {'total_decisions': 0}

        total_decisions = len(self.promotion_history)
        promotions = sum(1 for entry in self.promotion_history if entry['promote'])
        avg_confidence = sum(entry['confidence'] for entry in self.promotion_history) / total_decisions
        avg_risk = sum(entry['risk_score'] for entry in self.promotion_history) / total_decisions

        return {
            'total_decisions': total_decisions,
            'promotion_rate': promotions / total_decisions,
            'average_confidence': avg_confidence,
            'average_risk_score': avg_risk,
            'recent_decisions': self.promotion_history[-10:]  # Last 10 decisions
        }


# Example usage and testing
if __name__ == "__main__":
    # Create test policy engine
    policy = ModelPromotionPolicy("config/promotion_policies.yaml")

    # Example metrics
    challenger_metrics = {
        'sharpe_ratio': 1.8,
        'max_drawdown': 0.08,
        'total_return': 0.25,
        'win_rate': 0.65,
        'profit_factor': 1.4,
        'trial_duration_hours': 100,
        'trade_count': 1500,
        'avg_trade_duration': 45.0,
        'timestamp': datetime.now()
    }

    champion_metrics = {
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.10,
        'total_return': 0.18,
        'win_rate': 0.58,
        'profit_factor': 1.2,
        'trial_duration_hours': 200,
        'trade_count': 3000,
        'avg_trade_duration': 50.0,
        'timestamp': datetime.now() - timedelta(hours=200)
    }

    # Evaluate promotion
    decision = policy.evaluate_promotion_criteria(challenger_metrics, champion_metrics)

    print(f"Promotion Decision: {decision.promote}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Risk Score: {decision.risk_score:.2f}")
    print(f"Monitoring Period: {decision.recommended_monitoring_period} hours")
    print("Reasons:")
    for reason in decision.reasons:
        print(f"  - {reason}")

    # Save history
    policy.save_promotion_history()
    print(f"Promotion statistics: {policy.get_promotion_statistics()}")