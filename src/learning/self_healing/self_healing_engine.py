"""
EnhancedSelfHealingEngine: Proactive and reactive healing with ML predictions
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import subprocess
from datetime import datetime, timedelta
from collections import deque
import asyncio
import traceback

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of system failures"""
    NETWORK_ERROR = "network_error"
    API_TIMEOUT = "api_timeout"
    DATA_QUALITY = "data_quality"
    MODEL_ERROR = "model_error"
    MEMORY_ERROR = "memory_error"
    COMPUTATION_ERROR = "computation_error"
    EXCHANGE_ERROR = "exchange_error"
    DATABASE_ERROR = "database_error"
    UNKNOWN = "unknown"

class HealingAction(Enum):
    """Types of healing actions"""
    RETRY_OPERATION = "retry_operation"
    SWITCH_EXCHANGE = "switch_exchange"
    REDUCE_FREQUENCY = "reduce_frequency"
    INCREASE_TIMEOUT = "increase_timeout"
    CLEAR_CACHE = "clear_cache"
    RESTART_COMPONENT = "restart_component"
    FALLBACK_STRATEGY = "fallback_strategy"
    ALERT_OPERATOR = "alert_operator"
    SHUTDOWN_GRACEFULLY = "shutdown_gracefully"

@dataclass
class SystemHealth:
    """Represents current system health status"""
    timestamp: datetime
    cpu_usage: float  # 0-100%
    memory_usage: float  # 0-100%
    disk_usage: float  # 0-100%
    network_latency: float  # ms
    api_response_time: float  # ms
    error_rate: float  # 0-1 errors per second
    data_quality_score: float  # 0-1 quality score
    model_accuracy: float  # 0-1 accuracy score
    overall_health: float  # 0-1 overall health score

@dataclass
class FailureEvent:
    """Represents a system failure event"""
    timestamp: datetime
    failure_type: FailureType
    component: str
    error_message: str
    severity: int  # 1-10 severity level
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealingEvent:
    """Represents a healing action taken"""
    timestamp: datetime
    action: HealingAction
    component: str
    success: bool
    duration: float  # seconds
    outcome: str
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealingConfig:
    """Configuration for self-healing system"""
    # Monitoring parameters
    health_check_interval: float = 5.0  # seconds
    failure_detection_window: int = 60  # seconds
    
    # Prediction parameters
    prediction_window: int = 300  # seconds to predict failures
    prediction_threshold: float = 0.7  # confidence threshold for predictions
    
    # Healing parameters
    max_retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    healing_timeout: float = 30.0  # seconds
    
    # Alerting parameters
    alert_threshold: float = 0.3  # health score threshold for alerts
    critical_alert_threshold: float = 0.1  # critical health threshold

class HealthMonitor:
    """Monitors system health across multiple dimensions"""
    
    def __init__(self, config: HealingConfig):
        self.config = config
        self.health_history = deque(maxlen=1000)
        self.failure_history = deque(maxlen=1000)
        self.component_status: Dict[str, Dict[str, Any]] = {}
        
    def update_component_status(self, component: str, status: Dict[str, Any]) -> None:
        """Update status for a specific component"""
        self.component_status[component] = {
            "timestamp": datetime.now(),
            "status": status
        }
    
    def get_system_health(self) -> SystemHealth:
        """Calculate overall system health"""
        timestamp = datetime.now()
        
        # Aggregate metrics from components
        metrics = self._aggregate_metrics()
        
        # Calculate individual scores
        cpu_score = max(0, 1 - (metrics.get("cpu_usage", 50) / 100))
        memory_score = max(0, 1 - (metrics.get("memory_usage", 50) / 100))
        disk_score = max(0, 1 - (metrics.get("disk_usage", 50) / 100))
        latency_score = max(0, 1 - (metrics.get("network_latency", 100) / 1000))
        error_score = max(0, 1 - metrics.get("error_rate", 0.1) * 10)
        data_quality_score = metrics.get("data_quality_score", 0.8)
        model_accuracy_score = metrics.get("model_accuracy", 0.9)
        
        # Weighted overall score
        overall_health = (
            cpu_score * 0.15 +
            memory_score * 0.15 +
            disk_score * 0.1 +
            latency_score * 0.2 +
            error_score * 0.15 +
            data_quality_score * 0.15 +
            model_accuracy_score * 0.1
        )
        
        health = SystemHealth(
            timestamp=timestamp,
            cpu_usage=metrics.get("cpu_usage", 50),
            memory_usage=metrics.get("memory_usage", 50),
            disk_usage=metrics.get("disk_usage", 50),
            network_latency=metrics.get("network_latency", 100),
            api_response_time=metrics.get("api_response_time", 200),
            error_rate=metrics.get("error_rate", 0.1),
            data_quality_score=data_quality_score,
            model_accuracy=model_accuracy_score,
            overall_health=overall_health
        )
        
        # Store in history
        self.health_history.append(health)
        
        return health
    
    def _aggregate_metrics(self) -> Dict[str, float]:
        """Aggregate metrics from all components"""
        aggregated = {}
        component_count = len(self.component_status)
        
        if component_count == 0:
            return {
                "cpu_usage": 50,
                "memory_usage": 50,
                "disk_usage": 50,
                "network_latency": 100,
                "api_response_time": 200,
                "error_rate": 0.1,
                "data_quality_score": 0.8,
                "model_accuracy": 0.9
            }
        
        # Sum up metrics from all components
        sums = {}
        counts = {}
        
        for component, data in self.component_status.items():
            status = data.get("status", {})
            for key, value in status.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0) + value
                    counts[key] = counts.get(key, 0) + 1
        
        # Calculate averages
        for key, sum_value in sums.items():
            aggregated[key] = sum_value / counts[key]
            
        return aggregated
    
    def record_failure(self, failure: FailureEvent) -> None:
        """Record a failure event"""
        self.failure_history.append(failure)
        logger.warning(f"Failure recorded: {failure.failure_type.value} in {failure.component}")

class FailurePredictor:
    """Predicts potential system failures using ML"""
    
    def __init__(self, config: HealingConfig):
        self.config = config
        self.prediction_model = self._create_prediction_model()
        self.prediction_history: List[Dict] = []
        
    def _create_prediction_model(self) -> torch.nn.Module:
        """Create a simple neural network for failure prediction"""
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 64),  # 8 input features
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, len(FailureType))
        )
        return model
    
    def predict_failures(self, health: SystemHealth) -> List[Tuple[FailureType, float]]:
        """Predict potential failures based on current health"""
        # Convert health to feature vector
        features = torch.tensor([
            health.cpu_usage / 100,
            health.memory_usage / 100,
            health.disk_usage / 100,
            health.network_latency / 1000,
            health.api_response_time / 1000,
            health.error_rate * 10,
            health.data_quality_score,
            health.model_accuracy
        ], dtype=torch.float32).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            logits = self.prediction_model(features)
            probabilities = torch.softmax(logits, dim=1)
            
        # Convert to failure predictions
        predictions = []
        for i, failure_type in enumerate(FailureType):
            prob = probabilities[0, i].item()
            if prob > self.config.prediction_threshold:
                predictions.append((failure_type, prob))
        
        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    
    def update_model(self, health_data: List[SystemHealth], failure_data: List[FailureEvent]) -> None:
        """Update prediction model with new data"""
        # In a real implementation, this would retrain the model
        # For now, we'll just log that an update is needed
        if health_data and failure_data:
            logger.info("Prediction model update triggered with new data")

class HealingStrategySelector:
    """Selects optimal healing strategies based on failure type and context"""
    
    def __init__(self, config: HealingConfig):
        self.config = config
        self.healing_history: List[HealingEvent] = []
        self.strategy_success_rates: Dict[Tuple[FailureType, HealingAction], float] = {}
        
    def select_healing_action(
        self, 
        failure: FailureEvent, 
        health: SystemHealth
    ) -> List[HealingAction]:
        """Select appropriate healing actions for a failure"""
        actions = []
        
        # Select actions based on failure type
        if failure.failure_type == FailureType.NETWORK_ERROR:
            actions = [
                HealingAction.RETRY_OPERATION,
                HealingAction.INCREASE_TIMEOUT,
                HealingAction.SWITCH_EXCHANGE
            ]
        elif failure.failure_type == FailureType.API_TIMEOUT:
            actions = [
                HealingAction.INCREASE_TIMEOUT,
                HealingAction.REDUCE_FREQUENCY,
                HealingAction.SWITCH_EXCHANGE
            ]
        elif failure.failure_type == FailureType.DATA_QUALITY:
            actions = [
                HealingAction.CLEAR_CACHE,
                HealingAction.RETRY_OPERATION,
                HealingAction.FALLBACK_STRATEGY
            ]
        elif failure.failure_type == FailureType.MODEL_ERROR:
            actions = [
                HealingAction.RESTART_COMPONENT,
                HealingAction.FALLBACK_STRATEGY,
                HealingAction.ALERT_OPERATOR
            ]
        elif failure.failure_type == FailureType.MEMORY_ERROR:
            actions = [
                HealingAction.CLEAR_CACHE,
                HealingAction.RESTART_COMPONENT,
                HealingAction.REDUCE_FREQUENCY
            ]
        elif failure.failure_type == FailureType.EXCHANGE_ERROR:
            actions = [
                HealingAction.SWITCH_EXCHANGE,
                HealingAction.RETRY_OPERATION,
                HealingAction.FALLBACK_STRATEGY
            ]
        elif failure.failure_type == FailureType.DATABASE_ERROR:
            actions = [
                HealingAction.RETRY_OPERATION,
                HealingAction.RESTART_COMPONENT,
                HealingAction.ALERT_OPERATOR
            ]
        else:
            # Unknown or computation errors
            actions = [
                HealingAction.RETRY_OPERATION,
                HealingAction.RESTART_COMPONENT,
                HealingAction.ALERT_OPERATOR
            ]
        
        # Adjust based on severity
        if failure.severity > 7:
            actions.append(HealingAction.ALERT_OPERATOR)
        if failure.severity >= 9: # Changed from > 9 to >= 9
            actions.append(HealingAction.SHUTDOWN_GRACEFULLY)
            
        # Adjust based on system health
        if health.overall_health < 0.3:
            actions.insert(0, HealingAction.ALERT_OPERATOR)
            
        return actions
    
    def record_healing_outcome(self, event: HealingEvent) -> None:
        """Record the outcome of a healing action"""
        self.healing_history.append(event)
        
        # Update success rates (simplified)
        if hasattr(event, 'failure_type'):
            key = (event.failure_type, event.action)
            current_rate = self.strategy_success_rates.get(key, 0.5)
            new_rate = current_rate * 0.9 + (1.0 if event.success else 0.0) * 0.1
            self.strategy_success_rates[key] = new_rate
            
        logger.info(f"Healing action {event.action.value} {'succeeded' if event.success else 'failed'} "
                   f"in {event.duration:.2f}s: {event.outcome}")

class HealingExecutor:
    """Executes healing actions"""
    
    def __init__(self, config: HealingConfig):
        self.config = config
        self.healing_actions: Dict[HealingAction, Callable] = {}
        self.register_default_actions()
        
    def register_default_actions(self) -> None:
        """Register default healing actions"""
        self.healing_actions[HealingAction.RETRY_OPERATION] = self._retry_operation
        self.healing_actions[HealingAction.SWITCH_EXCHANGE] = self._switch_exchange
        self.healing_actions[HealingAction.REDUCE_FREQUENCY] = self._reduce_frequency
        self.healing_actions[HealingAction.INCREASE_TIMEOUT] = self._increase_timeout
        self.healing_actions[HealingAction.CLEAR_CACHE] = self._clear_cache
        self.healing_actions[HealingAction.RESTART_COMPONENT] = self._restart_component
        self.healing_actions[HealingAction.FALLBACK_STRATEGY] = self._fallback_strategy
        self.healing_actions[HealingAction.ALERT_OPERATOR] = self._alert_operator
        self.healing_actions[HealingAction.SHUTDOWN_GRACEFULLY] = self._shutdown_gracefully
    
    async def execute_healing_action(
        self, 
        action: HealingAction, 
        component: str, 
        context: Dict[str, Any]
    ) -> HealingEvent:
        """Execute a healing action"""
        start_time = datetime.now()
        
        try:
            # Get action function
            action_func = self.healing_actions.get(action)
            if not action_func:
                raise ValueError(f"Unknown healing action: {action}")
            
            # Execute action
            outcome = await action_func(component, context)
            success = True
            
        except Exception as e:
            outcome = f"Action failed: {str(e)}"
            success = False
            logger.error(f"Healing action {action.value} failed: {str(e)}")
            logger.error(traceback.format_exc())
        
        duration = (datetime.now() - start_time).total_seconds()
        
        healing_event = HealingEvent(
            timestamp=datetime.now(),
            action=action,
            component=component,
            success=success,
            duration=duration,
            outcome=outcome,
            context=context
        )
        
        return healing_event
    
    async def _retry_operation(self, component: str, context: Dict[str, Any]) -> str:
        """Retry a failed operation"""
        max_attempts = context.get("max_attempts", self.config.max_retry_attempts)
        delay = context.get("delay", self.config.retry_delay)
        
        for attempt in range(max_attempts):
            try:
                # Simulate operation retry
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                return f"Operation succeeded on attempt {attempt + 1}"
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                continue
                
        return "Retry attempts exhausted"
    
    async def _switch_exchange(self, component: str, context: Dict[str, Any]) -> str:
        """Switch to a different exchange"""
        current_exchange = context.get("current_exchange", "unknown")
        available_exchanges = context.get("available_exchanges", [])
        
        if available_exchanges:
            new_exchange = available_exchanges[0]  # Simplified selection
            return f"Switched from {current_exchange} to {new_exchange}"
        else:
            raise Exception("No alternative exchanges available")
    
    async def _reduce_frequency(self, component: str, context: Dict[str, Any]) -> str:
        """Reduce operation frequency"""
        current_frequency = context.get("frequency", 1.0)
        new_frequency = current_frequency * 0.5
        return f"Reduced frequency from {current_frequency} to {new_frequency}"
    
    async def _increase_timeout(self, component: str, context: Dict[str, Any]) -> str:
        """Increase operation timeout"""
        current_timeout = context.get("timeout", 10.0)
        new_timeout = current_timeout * 2.0
        return f"Increased timeout from {current_timeout}s to {new_timeout}s"
    
    async def _clear_cache(self, component: str, context: Dict[str, Any]) -> str:
        """Clear system cache"""
        cache_size = context.get("cache_size", "unknown")
        return f"Cleared cache of size {cache_size}"
    
    async def _restart_component(self, component: str, context: Dict[str, Any]) -> str:
        """Restart a component"""
        return f"Restarted component {component}"
    
    async def _fallback_strategy(self, component: str, context: Dict[str, Any]) -> str:
        """Switch to fallback strategy"""
        current_strategy = context.get("strategy", "unknown")
        return f"Switched from {current_strategy} to fallback strategy"
    
    async def _alert_operator(self, component: str, context: Dict[str, Any]) -> str:
        """Alert human operator"""
        alert_message = context.get("alert_message", "System requires attention")
        return f"Alert sent: {alert_message}"
    
    async def _shutdown_gracefully(self, component: str, context: Dict[str, Any]) -> str:
        """Shutdown system gracefully"""
        reason = context.get("shutdown_reason", "Emergency shutdown")
        return f"System shutdown initiated: {reason}"

class EnhancedSelfHealingEngine:
    """Main self-healing engine"""
    
    def __init__(self, config: Optional[HealingConfig] = None):
        self.config = config or HealingConfig()
        self.monitor = HealthMonitor(self.config)
        self.predictor = FailurePredictor(self.config)
        self.strategy_selector = HealingStrategySelector(self.config)
        self.executor = HealingExecutor(self.config)
        
        self.is_active = True
        self.last_health_check = datetime.min
        self.healing_events: List[HealingEvent] = []
        
        logger.info("EnhancedSelfHealingEngine initialized")
    
    def update_component_status(self, component: str, status: Dict[str, Any]) -> None:
        """Update status for a component"""
        self.monitor.update_component_status(component, status)
    
    async def check_system_health(self) -> SystemHealth:
        """Check current system health"""
        health = self.monitor.get_system_health()
        
        # Check if we should alert based on health
        if health.overall_health < self.config.alert_threshold:
            await self._trigger_alert(health)
            
        return health
    
    async def predict_and_prevent_failures(self, health: SystemHealth) -> None:
        """Predict and prevent potential failures"""
        predictions = self.predictor.predict_failures(health)
        
        for failure_type, probability in predictions:
            logger.warning(f"Predicted failure: {failure_type.value} "
                          f"(probability: {probability:.2f})")
            
            # Trigger preventive healing actions
            await self._trigger_preventive_healing(failure_type, probability, health)
    
    async def handle_failure(self, failure: FailureEvent) -> List[HealingEvent]:
        """Handle a system failure"""
        # Record the failure
        self.monitor.record_failure(failure)
        
        # Get current system health
        health = await self.check_system_health()
        
        # Select healing actions
        actions = self.strategy_selector.select_healing_action(failure, health)
        
        # Execute healing actions
        healing_events = []
        for action in actions:
            context = {
                "failure": failure,
                "health": health,
                "timestamp": datetime.now()
            }
            
            healing_event = await self.executor.execute_healing_action(
                action, failure.component, context
            )
            
            # Add failure type to healing event for tracking
            healing_event.failure_type = failure.failure_type
            
            # Record outcome
            self.strategy_selector.record_healing_outcome(healing_event)
            healing_events.append(healing_event)
            
            # Stop if critical action succeeded
            if action == HealingAction.SHUTDOWN_GRACEFULLY and healing_event.success:
                break
                
        self.healing_events.extend(healing_events)
        return healing_events
    
    async def _trigger_preventive_healing(
        self, 
        failure_type: FailureType, 
        probability: float, 
        health: SystemHealth
    ) -> None:
        """Trigger preventive healing actions"""
        # Create a mock failure event for preventive action
        preventive_failure = FailureEvent(
            timestamp=datetime.now(),
            failure_type=failure_type,
            component="system",
            error_message=f"Predicted failure with {probability:.2f} probability",
            severity=int(probability * 10),
            context={"predicted": True, "probability": probability}
        )
        
        # Handle as regular failure (preventive)
        await self.handle_failure(preventive_failure)
    
    async def _trigger_alert(self, health: SystemHealth) -> None:
        """Trigger alert based on health status"""
        if health.overall_health < self.config.critical_alert_threshold:
            alert_level = "CRITICAL"
        else:
            alert_level = "WARNING"
            
        alert_message = f"System health alert ({alert_level}): {health.overall_health:.2f}"
        
        # Create healing event for alert
        context = {
            "health": health,
            "alert_level": alert_level,
            "message": alert_message
        }
        
        healing_event = await self.executor.execute_healing_action(
            HealingAction.ALERT_OPERATOR, "health_monitor", context
        )
        
        self.healing_events.append(healing_event)
        logger.warning(alert_message)
    
    def get_health_history(self) -> List[SystemHealth]:
        """Get system health history"""
        return list(self.monitor.health_history)
    
    def get_failure_history(self) -> List[FailureEvent]:
        """Get failure history"""
        return list(self.monitor.failure_history)
    
    def get_healing_history(self) -> List[HealingEvent]:
        """Get healing history"""
        return self.healing_events.copy()

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize healing engine
    healing_engine = EnhancedSelfHealingEngine()
    
    # Simulate component status updates
    healing_engine.update_component_status("trading_engine", {
        "cpu_usage": 45.2,
        "memory_usage": 67.8,
        "error_rate": 0.05,
        "response_time": 150.0
    })
    
    # Simulate a failure
    failure = FailureEvent(
        timestamp=datetime.now(),
        failure_type=FailureType.NETWORK_ERROR,
        component="exchange_connector",
        error_message="Connection timeout to exchange API",
        severity=6,
        context={"exchange": "binance", "timeout": 30}
    )
    
    # Handle failure asynchronously
    async def test_healing():
        healing_events = await healing_engine.handle_failure(failure)
        for event in healing_events:
            print(f"Healing event: {event.action.value} - {'Success' if event.success else 'Failed'}")
    
    # Run the test
    asyncio.run(test_healing())
    
    logger.info("EnhancedSelfHealingEngine test completed successfully")