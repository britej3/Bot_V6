"""
Self-Awareness System for CryptoScalp AI

This module implements self-awareness features that allow the trading bot to:
1. Track its own execution performance and slippage
2. Learn from market impact and execution quality
3. Adapt trading behavior based on feedback loops
4. Maintain state awareness for intelligent decision making
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class ExecutionEvent:
    """Record of a single execution event"""
    timestamp: float
    symbol: str
    side: str  # 'buy' or 'sell'
    intended_quantity: float
    executed_quantity: float
    intended_price: float
    executed_price: float
    slippage: float
    latency_ms: float
    market_impact_score: float
    order_type: str


@dataclass
class PositionState:
    """Current position state"""
    symbol: str
    size: float
    direction: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_time: float


@dataclass
class ExecutionMetrics:
    """Aggregated execution metrics"""
    avg_slippage_last_10: float
    avg_latency_last_10: float
    time_since_last_trade: float
    market_impact_score: float
    execution_quality_score: float
    recent_fill_rate: float


@dataclass
class AdaptiveParameters:
    """Adaptive trading parameters"""
    confidence_multiplier: float
    risk_multiplier: float
    should_reduce_activity: bool
    suggested_position_size: float
    suggested_timeout_ms: int


class ExecutionStateTracker:
    """
    Tracks and analyzes bot's execution performance and market impact
    """

    def __init__(self, max_history: int = 1000):
        self.execution_history = deque(maxlen=max_history)
        self.position_states = {}  # symbol -> PositionState
        self.market_impact_history = deque(maxlen=100)

    def record_execution(self, execution: ExecutionEvent):
        """Record a new execution event"""
        self.execution_history.append(execution)
        self._update_position_state(execution)
        self._update_market_impact(execution)

        logger.info(f"Recorded execution: {execution.symbol} {execution.side} "
                   f"qty={execution.executed_quantity} "
                   f"slippage={execution.slippage:.4f}")

    def _update_position_state(self, execution: ExecutionEvent):
        """Update position state based on execution"""
        symbol = execution.symbol

        if symbol not in self.position_states:
            self.position_states[symbol] = PositionState(
                symbol=symbol,
                size=0,
                direction='flat',
                entry_price=execution.executed_price,
                current_price=execution.executed_price,
                unrealized_pnl=0,
                stop_loss=None,
                take_profit=None,
                entry_time=execution.timestamp
            )

        position = self.position_states[symbol]

        if execution.side == 'buy':
            if position.direction == 'short':
                # Closing short position
                position.size -= execution.executed_quantity
                if position.size <= 0:
                    position.direction = 'flat'
                    position.size = 0
            else:
                # Opening or adding to long position
                if position.size == 0:
                    position.entry_price = execution.executed_price
                    position.entry_time = execution.timestamp
                else:
                    # Update average entry price
                    total_value = (position.size * position.entry_price +
                                 execution.executed_quantity * execution.executed_price)
                    position.entry_price = total_value / (position.size + execution.executed_quantity)

                position.size += execution.executed_quantity
                position.direction = 'long'
        else:  # sell
            if position.direction == 'long':
                # Closing long position
                position.size -= execution.executed_quantity
                if position.size <= 0:
                    position.direction = 'flat'
                    position.size = 0
            else:
                # Opening or adding to short position
                if position.size == 0:
                    position.entry_price = execution.executed_price
                    position.entry_time = execution.timestamp
                else:
                    # Update average entry price
                    total_value = (position.size * position.entry_price +
                                 execution.executed_quantity * execution.executed_price)
                    position.entry_price = total_value / (position.size + execution.executed_quantity)

                position.size += execution.executed_quantity
                position.direction = 'short'

    def _update_market_impact(self, execution: ExecutionEvent):
        """Update market impact analysis"""
        # Calculate price movement relative to order size
        price_impact = abs(execution.executed_price - execution.intended_price) / execution.intended_price
        size_impact = execution.executed_quantity / 1000  # Normalized by typical order size

        market_impact = price_impact * size_impact
        self.market_impact_history.append(market_impact)

    def get_execution_metrics(self) -> ExecutionMetrics:
        """Calculate current execution metrics"""
        if not self.execution_history:
            return ExecutionMetrics(
                avg_slippage_last_10=0,
                avg_latency_last_10=0,
                time_since_last_trade=float('inf'),
                market_impact_score=0,
                execution_quality_score=0.5,
                recent_fill_rate=1.0
            )

        # Get recent executions
        recent_executions = list(self.execution_history)[-10:]

        # Calculate average slippage
        slippages = [abs(ex.slippage) for ex in recent_executions]
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0

        # Calculate average latency
        latencies = [ex.latency_ms for ex in recent_executions]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        # Time since last trade
        time_since_last = time.time() - self.execution_history[-1].timestamp

        # Market impact score
        market_impact = sum(self.market_impact_history) / len(self.market_impact_history) if self.market_impact_history else 0

        # Execution quality score (0-1, higher is better)
        slippage_score = max(0, 1 - (avg_slippage / 0.01))  # Assume 1% max slippage
        latency_score = max(0, 1 - (avg_latency / 100))     # Assume 100ms max latency
        execution_quality = (slippage_score + latency_score) / 2

        # Fill rate
        intended_total = sum(ex.intended_quantity for ex in recent_executions)
        executed_total = sum(ex.executed_quantity for ex in recent_executions)
        fill_rate = executed_total / intended_total if intended_total > 0 else 1.0

        return ExecutionMetrics(
            avg_slippage_last_10=avg_slippage,
            avg_latency_last_10=avg_latency,
            time_since_last_trade=time_since_last,
            market_impact_score=market_impact,
            execution_quality_score=execution_quality,
            recent_fill_rate=fill_rate
        )

    def get_position_state(self, symbol: str) -> Optional[PositionState]:
        """Get current position state for symbol"""
        return self.position_states.get(symbol)

    def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all current positions"""
        return dict(self.position_states)

    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        total_exposure = 0
        total_unrealized_pnl = 0
        position_count = 0

        for position in self.position_states.values():
            if position.direction != 'flat':
                total_exposure += abs(position.size * position.current_price)
                total_unrealized_pnl += position.unrealized_pnl
                position_count += 1

        return {
            'total_exposure': total_exposure,
            'total_unrealized_pnl': total_unrealized_pnl,
            'position_count': position_count,
            'avg_position_size': total_exposure / position_count if position_count > 0 else 0
        }


class AdaptiveBehaviorSystem:
    """
    Adaptive behavior system that adjusts trading parameters based on execution feedback
    """

    def __init__(self):
        self.confidence_adjustment = 1.0
        self.risk_multiplier = 1.0
        self.activity_level = 1.0
        self.last_adaptation = time.time()

    def adapt_to_execution_quality(self, execution_metrics: ExecutionMetrics):
        """Adapt behavior based on execution quality"""
        current_time = time.time()

        # Don't adapt too frequently
        if current_time - self.last_adaptation < 60:  # Minimum 1 minute between adaptations
            return

        # Adjust confidence based on slippage
        if execution_metrics.avg_slippage_last_10 > 0.001:  # High slippage
            self.confidence_adjustment *= 0.9  # Reduce confidence
            self.risk_multiplier *= 0.8       # Reduce risk
            logger.info(f"High slippage detected, reducing confidence to {self.confidence_adjustment:.2f}")
        elif execution_metrics.avg_slippage_last_10 < 0.0001:  # Very low slippage
            self.confidence_adjustment = min(1.5, self.confidence_adjustment * 1.1)
            self.risk_multiplier = min(2.0, self.risk_multiplier * 1.2)
            logger.info(f"Low slippage detected, increasing confidence to {self.confidence_adjustment:.2f}")

        # Adjust for latency
        if execution_metrics.avg_latency_last_10 > 50:  # High latency
            self.confidence_adjustment *= 0.95
            self.activity_level *= 0.9
            logger.info(f"High latency detected, reducing activity to {self.activity_level:.2f}")

        # Adjust for market impact
        if execution_metrics.market_impact_score > 0.001:
            self.risk_multiplier *= 0.7
            self.activity_level *= 0.8
            logger.info(f"High market impact detected, reducing risk to {self.risk_multiplier:.2f}")

        # Adjust for time since last trade
        if execution_metrics.time_since_last_trade > 3600:  # 1 hour
            self.activity_level = min(1.5, self.activity_level * 1.1)
        elif execution_metrics.time_since_last_trade < 60:  # Less than 1 minute
            self.activity_level = max(0.5, self.activity_level * 0.9)

        # Apply bounds
        self.confidence_adjustment = max(0.1, min(2.0, self.confidence_adjustment))
        self.risk_multiplier = max(0.1, min(3.0, self.risk_multiplier))
        self.activity_level = max(0.1, min(2.0, self.activity_level))

        self.last_adaptation = current_time

    def get_adaptive_parameters(self, base_position_size: float = 1.0) -> AdaptiveParameters:
        """Get current adaptive parameters"""
        should_reduce_activity = (
            self.activity_level < 0.5 or
            self.risk_multiplier < 0.5 or
            self.confidence_adjustment < 0.5
        )

        suggested_position_size = base_position_size * self.risk_multiplier * self.activity_level

        # Adjust timeout based on execution quality
        if self.activity_level > 1.2:
            suggested_timeout = 1000  # Faster execution
        elif self.activity_level < 0.8:
            suggested_timeout = 5000  # Slower, more patient execution
        else:
            suggested_timeout = 2000  # Normal execution

        return AdaptiveParameters(
            confidence_multiplier=self.confidence_adjustment,
            risk_multiplier=self.risk_multiplier,
            should_reduce_activity=should_reduce_activity,
            suggested_position_size=suggested_position_size,
            suggested_timeout_ms=suggested_timeout
        )

    def reset_adaptations(self):
        """Reset adaptations to baseline"""
        self.confidence_adjustment = 1.0
        self.risk_multiplier = 1.0
        self.activity_level = 1.0
        logger.info("Reset adaptations to baseline")


class MarketImpactAnalyzer:
    """
    Analyzes and predicts market impact of bot's trading activities
    """

    def __init__(self):
        self.impact_history = []

    def analyze_impact(self, order_size: float, market_volatility: float,
                      order_flow_imbalance: float) -> float:
        """Analyze expected market impact of an order"""
        # Simple impact model (could be more sophisticated)
        base_impact = order_size * 0.001  # 0.1% base impact per unit size
        volatility_adjustment = market_volatility * 0.5
        imbalance_adjustment = order_flow_imbalance * 0.3

        impact = base_impact * (1 + volatility_adjustment + imbalance_adjustment)
        return min(impact, 0.05)  # Cap at 5%

    def predict_optimal_order_size(self, target_impact: float = 0.001,
                                 market_conditions: Dict[str, float]) -> float:
        """Predict optimal order size to minimize market impact"""
        # This would use the trained impact model
        # For now, return a simple calculation
        volatility = market_conditions.get('volatility', 0.05)
        liquidity = market_conditions.get('liquidity', 1.0)

        optimal_size = (target_impact / volatility) * liquidity
        return max(optimal_size, 0.1)  # Minimum order size


class MarketImpactAnalyzer:
    """
    Analyzes and predicts market impact of bot's trading activities
    """

    def __init__(self):
        self.impact_history = []

    def analyze_impact(self, order_size: float, market_volatility: float,
                      order_flow_imbalance: float) -> float:
        """
        Analyze expected market impact of an order
        """
        # Simple impact model (could be more sophisticated)
        base_impact = order_size * 0.001  # 0.1% base impact per unit size
        volatility_adjustment = market_volatility * 0.5
        imbalance_adjustment = order_flow_imbalance * 0.3

        impact = base_impact * (1 + volatility_adjustment + imbalance_adjustment)
        return min(impact, 0.05)  # Cap at 5%

    def predict_optimal_order_size(self, target_impact: float = 0.001,
                                 market_conditions: Dict[str, float]) -> float:
        """
        Predict optimal order size to minimize market impact
        """
        # This would use the trained impact model
        # For now, return a simple calculation
        volatility = market_conditions.get('volatility', 0.05)
        liquidity = market_conditions.get('liquidity', 1.0)

        optimal_size = (target_impact / volatility) * liquidity
        return max(optimal_size, 0.1)  # Minimum order size


class FailurePredictor(nn.Module):
    """
    Predicts potential system failures based on self-awareness features.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1), # Output a single failure probability/score
            nn.Sigmoid() # For probability output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def predict_failure(self, features: Dict[str, float]) -> float:
        """
        Predicts the likelihood of a failure given a set of features.
        """
        # Convert features dict to tensor
        # This requires a consistent order of features, which needs to be managed.
        # For now, we'll assume a fixed order or a mapping.
        # A more robust solution would involve a feature preprocessor.
        feature_tensor = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = self.forward(feature_tensor)
        return prediction.item()


class SelfAwarenessEngine:
    """
    Main self-awareness engine that integrates all components
    """

    def __init__(self):
        self.execution_tracker = ExecutionStateTracker()
        self.adaptive_system = AdaptiveBehaviorSystem()
        self.impact_analyzer = MarketImpactAnalyzer()
        # Initialize FailurePredictor
        # The input_dim should match the number of features generated by generate_self_awareness_features
        # For now, using a placeholder. This needs to be dynamically determined or explicitly defined.
        self.failure_predictor = FailurePredictor(input_dim=16) # Placeholder input_dim

    async def process_execution_feedback(self, execution: ExecutionEvent):
        """Process execution feedback and update internal state"""
        # Record execution
        self.execution_tracker.record_execution(execution)

        # Get current metrics
        execution_metrics = self.execution_tracker.get_execution_metrics()

        # Adapt behavior
        self.adaptive_system.adapt_to_execution_quality(execution_metrics)

        # Log adaptation
        adaptive_params = self.adaptive_system.get_adaptive_parameters()
        logger.info(f"Adapted parameters: confidence={adaptive_params.confidence_multiplier:.2f}, "
                   f"risk={adaptive_params.risk_multiplier:.2f}, "
                   f"activity={adaptive_params.suggested_position_size:.2f}")

    def generate_self_awareness_features(self) -> Dict[str, float]:
        """Generate self-awareness features for model input"""
        execution_metrics = self.execution_tracker.get_execution_metrics()
        portfolio_metrics = self.execution_tracker.calculate_portfolio_metrics()
        adaptive_params = self.adaptive_system.get_adaptive_parameters()

        return {
            # Execution metrics
            'avg_slippage_last_10': execution_metrics.avg_slippage_last_10,
            'avg_latency_last_10': execution_metrics.avg_latency_last_10,
            'time_since_last_trade': execution_metrics.time_since_last_trade,
            'market_impact_score': execution_metrics.market_impact_score,
            'execution_quality_score': execution_metrics.execution_quality_score,
            'recent_fill_rate': execution_metrics.recent_fill_rate,

            # Portfolio metrics
            'total_exposure': portfolio_metrics['total_exposure'],
            'total_unrealized_pnl': portfolio_metrics['total_unrealized_pnl'],
            'position_count': portfolio_metrics['position_count'],
            'avg_position_size': portfolio_metrics['avg_position_size'],

            # Adaptive parameters
            'confidence_multiplier': adaptive_params.confidence_multiplier,
            'risk_multiplier': adaptive_params.risk_multiplier,
            'activity_level': adaptive_params.suggested_position_size / adaptive_params.suggested_position_size,  # Normalized

            # Position states (for major positions)
            'has_open_positions': portfolio_metrics['position_count'] > 0,
            'largest_position_size': max(
                [pos.size for pos in self.execution_tracker.get_all_positions().values()],
                default=0
            )
        }

    def should_adjust_behavior(self) -> Dict[str, Any]:
        """Determine if behavior adjustment is needed"""
        execution_metrics = self.execution_tracker.get_execution_metrics()
        adaptive_params = self.adaptive_system.get_adaptive_parameters()

        adjustments = {
            'reduce_activity': adaptive_params.should_reduce_activity,
            'increase_caution': execution_metrics.market_impact_score > 0.002,
            'speed_up_execution': execution_metrics.avg_latency_last_10 > 100,
            'improve_slippage': execution_metrics.avg_slippage_last_10 > 0.002,
            'rebalance_portfolio': self.execution_tracker.calculate_portfolio_metrics()['position_count'] > 5
        }

        return adjustments

    def get_system_insights(self) -> Dict[str, Any]:
        """Get comprehensive system insights"""
        return {
            'execution_metrics': asdict(self.execution_tracker.get_execution_metrics()),
            'portfolio_metrics': self.execution_tracker.calculate_portfolio_metrics(),
            'adaptive_parameters': asdict(self.adaptive_system.get_adaptive_parameters()),
            'behavior_adjustments': self.should_adjust_behavior(),
            'position_states': {
                symbol: asdict(state)
                for symbol, state in self.execution_tracker.get_all_positions().items()
            }
        }

    async def continuous_monitoring(self, update_interval: float = 5.0):
        """Continuous monitoring and adaptation loop"""
        logger.info("Starting continuous self-awareness monitoring")

        while True:
            try:
                # Get current state
                system_insights = self.get_system_insights()

                # Log insights periodically
                if int(time.time()) % 60 == 0:  # Log every minute
                    logger.info(f"System insights: {system_insights}")

                # Check for critical conditions
                adjustments = self.should_adjust_behavior()
                if any(adjustments.values()):
                    logger.warning(f"Behavior adjustments needed: {adjustments}")

                # --- Predictive Healing System Integration ---
                # Generate features for failure prediction
                features_for_prediction = self.generate_self_awareness_features()
                failure_probability = self.failure_predictor.predict_failure(features_for_prediction)

                if failure_probability > 0.7: # Threshold for high probability of failure
                    logger.critical(f"High probability of failure detected: {failure_probability:.2f}. Initiating proactive healing actions.")
                    # Placeholder for proactive healing actions
                    await self._initiate_proactive_healing(failure_probability, features_for_prediction)
                elif failure_probability > 0.4: # Moderate probability
                    logger.warning(f"Moderate probability of failure detected: {failure_probability:.2f}. Suggesting cautious adjustments.")
                    # Placeholder for cautious adjustments or alerts
                    await self._suggest_cautious_adjustments(failure_probability, features_for_prediction)

                await asyncio.sleep(update_interval)

            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(update_interval)

    async def _initiate_proactive_healing(self, failure_probability: float, features: Dict[str, float]):
        """
        Placeholder for initiating proactive healing actions.
        These actions could include:
        - Reducing position sizes
        - Temporarily pausing trading
        - Switching to a more conservative strategy
        - Triggering model retraining
        - Alerting human operators
        """
        logger.info(f"Proactive healing initiated. Failure probability: {failure_probability:.2f}")
        # Example: Reduce activity level significantly
        self.adaptive_system.activity_level *= 0.5
        self.adaptive_system.risk_multiplier *= 0.5
        # In a real system, this would call specific healing modules.

    async def _suggest_cautious_adjustments(self, failure_probability: float, features: Dict[str, float]):
        """
        Placeholder for suggesting cautious adjustments or alerts.
        These could include:
        - Logging detailed warnings
        - Sending non-critical alerts
        - Slightly reducing activity
        """
        logger.info(f"Cautious adjustments suggested. Failure probability: {failure_probability:.2f}")
        # Example: Slightly reduce activity level
        self.adaptive_system.activity_level *= 0.9


# Example usage and testing
if __name__ == "__main__":
    # Initialize self-awareness engine
    sa_engine = SelfAwarenessEngine()

    # Example execution event
    execution = ExecutionEvent(
        timestamp=time.time(),
        symbol="BTCUSDT",
        side="buy",
        intended_quantity=1.0,
        executed_quantity=0.95,
        intended_price=50000,
        executed_price=50050,
        slippage=50,
        latency_ms=15,
        market_impact_score=0.001,
        order_type="market"
    )

    # Process execution feedback
    import asyncio
    asyncio.run(sa_engine.process_execution_feedback(execution))

    # Generate self-awareness features
    features = sa_engine.generate_self_awareness_features()
    print(f"Self-awareness features: {features}")

    # Get system insights
    insights = sa_engine.get_system_insights()
    print(f"System insights: {insights}")

    print("Self-awareness system initialized successfully!")