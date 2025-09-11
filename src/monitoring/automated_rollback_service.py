"""
Automated Rollback Service for CryptoScalp AI

This module implements an intelligent rollback system that monitors promoted models
and automatically reverts to previous versions if performance degrades beyond
acceptable thresholds.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class RollbackTrigger(Enum):
    """Types of events that can trigger a rollback"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    SYSTEM_ANOMALY = "system_anomaly"
    MANUAL_OVERRIDE = "manual_override"
    SCHEDULED_MAINTENANCE = "scheduled_maintenance"


@dataclass
class RollbackConfiguration:
    """Configuration for rollback monitoring"""
    model_id: str
    monitoring_window_hours: int = 24
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    risk_limits: Dict[str, float] = field(default_factory=dict)
    min_observation_period_hours: int = 4
    confidence_level: float = 0.95
    max_consecutive_failures: int = 3
    enable_automatic_rollback: bool = True

    def __post_init__(self):
        if not self.performance_thresholds:
            self.performance_thresholds = {
                'sharpe_ratio_drop': 0.3,  # 30% drop
                'max_drawdown_increase': 0.02,  # 2% increase
                'win_rate_drop': 0.05  # 5% drop
            }

        if not self.risk_limits:
            self.risk_limits = {
                'max_drawdown': 0.15,
                'max_consecutive_losses': 5,
                'max_daily_loss': 0.05
            }


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison"""
    model_id: str
    baseline_metrics: Dict[str, float]
    established_at: datetime
    sample_size: int
    confidence_interval: Dict[str, float]


@dataclass
class RollbackEvent:
    """Record of a rollback event"""
    event_id: str
    model_id: str
    trigger: RollbackTrigger
    timestamp: datetime
    reason: str
    performance_before: Dict[str, float]
    performance_after: Optional[Dict[str, float]] = None
    rollback_successful: bool = False
    recovery_actions: List[str] = field(default_factory=list)


class AutomatedRollbackService:
    """
    Service that monitors model performance and automatically rolls back
    to previous versions if performance degrades significantly.
    """

    def __init__(self, config_path: str = "config/rollback_config.json"):
        self.config = self._load_config(config_path)
        self.active_models: Dict[str, RollbackConfiguration] = {}
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.rollback_history: List[RollbackEvent] = []
        self.model_versions: Dict[str, List[str]] = {}  # model_id -> [version1, version2, ...]

        # Monitoring components
        self.performance_monitor = PerformanceMonitor()
        self.risk_monitor = RiskMonitor()
        self.anomaly_detector = AnomalyDetector()

        # Thread safety
        self._lock = threading.Lock()
        self._monitoring_task = None
        self._stop_monitoring = False

        # Callbacks
        self.rollback_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load rollback configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'monitoring_interval_seconds': 300,  # 5 minutes
            'baseline_establishment_period_hours': 24,
            'alert_threshold_warning': 0.7,
            'alert_threshold_critical': 0.9,
            'max_rollback_history_days': 30
        }

    def register_model_for_monitoring(self, model_id: str,
                                    config: RollbackConfiguration):
        """Register a model for automated rollback monitoring"""
        with self._lock:
            self.active_models[model_id] = config

            # Initialize version history
            if model_id not in self.model_versions:
                self.model_versions[model_id] = []

            logger.info(f"Registered model {model_id} for rollback monitoring")

    def establish_performance_baseline(self, model_id: str,
                                     metrics: Dict[str, float],
                                     sample_size: int = 100):
        """Establish performance baseline for a model"""
        baseline = PerformanceBaseline(
            model_id=model_id,
            baseline_metrics=metrics.copy(),
            established_at=datetime.now(),
            sample_size=sample_size,
            confidence_interval=self._calculate_confidence_intervals(metrics, sample_size)
        )

        self.performance_baselines[model_id] = baseline
        logger.info(f"Established performance baseline for {model_id}")

    def start_monitoring(self):
        """Start the automated monitoring service"""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started automated rollback monitoring")

    def stop_monitoring(self):
        """Stop the automated monitoring service"""
        self._stop_monitoring = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            logger.info("Stopped automated rollback monitoring")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Starting rollback monitoring loop")

        while not self._stop_monitoring:
            try:
                await self._perform_monitoring_cycle()
                await asyncio.sleep(self.config['monitoring_interval_seconds'])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(60)

        logger.info("Rollback monitoring loop stopped")

    async def _perform_monitoring_cycle(self):
        """Perform one monitoring cycle for all active models"""
        for model_id, config in list(self.active_models.items()):
            try:
                await self._monitor_single_model(model_id, config)
            except Exception as e:
                logger.error(f"Failed to monitor model {model_id}: {e}")

    async def _monitor_single_model(self, model_id: str, config: RollbackConfiguration):
        """Monitor a single model for rollback conditions"""
        # Get current performance
        current_metrics = await self.performance_monitor.get_current_metrics(model_id)
        if not current_metrics:
            return

        # Check if we have a baseline to compare against
        baseline = self.performance_baselines.get(model_id)
        if not baseline:
            # Establish baseline if enough time has passed
            if await self._should_establish_baseline(model_id):
                await self._establish_baseline(model_id)
            return

        # Check for rollback triggers
        rollback_trigger = await self._check_rollback_conditions(
            model_id, config, baseline, current_metrics
        )

        if rollback_trigger:
            await self._initiate_rollback(model_id, rollback_trigger, current_metrics)

        # Check for alerts even if no rollback
        await self._check_alert_conditions(model_id, config, baseline, current_metrics)

    async def _check_rollback_conditions(self, model_id: str,
                                      config: RollbackConfiguration,
                                      baseline: PerformanceBaseline,
                                      current_metrics: Dict[str, float]) -> Optional[RollbackTrigger]:
        """Check if any rollback conditions are met"""

        # 1. Performance degradation
        if self._check_performance_degradation(baseline.baseline_metrics, current_metrics, config):
            return RollbackTrigger.PERFORMANCE_DEGRADATION

        # 2. Risk limit breach
        if self._check_risk_limit_breach(current_metrics, config):
            return RollbackTrigger.RISK_LIMIT_BREACH

        # 3. System anomalies
        if await self._check_system_anomalies(model_id):
            return RollbackTrigger.SYSTEM_ANOMALY

        return None

    def _check_performance_degradation(self, baseline: Dict[str, float],
                                    current: Dict[str, float],
                                    config: RollbackConfiguration) -> bool:
        """Check for significant performance degradation"""

        # Check each performance metric
        for metric, threshold in config.performance_thresholds.items():
            if metric == 'sharpe_ratio_drop':
                baseline_sharpe = baseline.get('sharpe_ratio', 1.0)
                current_sharpe = current.get('sharpe_ratio', 1.0)
                if baseline_sharpe > 0 and (baseline_sharpe - current_sharpe) / baseline_sharpe > threshold:
                    return True

            elif metric == 'max_drawdown_increase':
                baseline_dd = baseline.get('max_drawdown', 0)
                current_dd = current.get('max_drawdown', 0)
                if current_dd - baseline_dd > threshold:
                    return True

            elif metric == 'win_rate_drop':
                baseline_win = baseline.get('win_rate', 0.5)
                current_win = current.get('win_rate', 0.5)
                if baseline_win > 0 and (baseline_win - current_win) > threshold:
                    return True

        return False

    def _check_risk_limit_breach(self, current_metrics: Dict[str, float],
                               config: RollbackConfiguration) -> bool:
        """Check if any risk limits have been breached"""

        # Check drawdown limit
        if current_metrics.get('max_drawdown', 0) > config.risk_limits.get('max_drawdown', 0.15):
            return True

        # Check consecutive losses
        consecutive_losses = current_metrics.get('consecutive_losses', 0)
        if consecutive_losses > config.risk_limits.get('max_consecutive_losses', 5):
            return True

        # Check daily loss limit
        daily_loss = abs(current_metrics.get('daily_pnl', 0))
        if daily_loss > config.risk_limits.get('max_daily_loss', 0.05):
            return True

        return False

    async def _check_system_anomalies(self, model_id: str) -> bool:
        """Check for system-level anomalies"""
        # This would integrate with system monitoring tools
        # For now, return False (no anomalies)
        return False

    async def _initiate_rollback(self, model_id: str, trigger: RollbackTrigger,
                               current_metrics: Dict[str, float]):
        """Initiate a rollback for the specified model"""
        logger.warning(f"Initiating rollback for {model_id} due to {trigger.value}")

        # Get previous version
        versions = self.model_versions.get(model_id, [])
        if len(versions) < 2:
            logger.error(f"No previous version available for {model_id}")
            return

        previous_version = versions[-2]  # Second to last version
        current_version = versions[-1]

        # Create rollback event
        event = RollbackEvent(
            event_id=f"rollback_{model_id}_{int(time.time())}",
            model_id=model_id,
            trigger=trigger,
            timestamp=datetime.now(),
            reason=self._get_rollback_reason(trigger, current_metrics),
            performance_before=current_metrics
        )

        try:
            # Execute rollback
            await self._execute_rollback(model_id, previous_version)

            # Record success
            event.rollback_successful = True
            event.recovery_actions = [f"Rolled back from {current_version} to {previous_version}"]

            # Trigger callbacks
            for callback in self.rollback_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Rollback callback failed: {e}")

            logger.info(f"Successfully rolled back {model_id} to {previous_version}")

        except Exception as e:
            logger.error(f"Rollback failed for {model_id}: {e}")
            event.recovery_actions = [f"Rollback failed: {str(e)}"]

        # Store event
        self.rollback_history.append(event)

        # Clean old history
        self._cleanup_old_history()

    async def _execute_rollback(self, model_id: str, target_version: str):
        """Execute the actual rollback"""
        # This would integrate with your model deployment system
        # For example: load model weights, update serving endpoints, etc.

        # Simulate rollback delay
        await asyncio.sleep(1)

        # Update version history
        versions = self.model_versions.get(model_id, [])
        versions.append(f"rollback_to_{target_version}")
        self.model_versions[model_id] = versions[-10:]  # Keep last 10 versions

    def _get_rollback_reason(self, trigger: RollbackTrigger,
                           current_metrics: Dict[str, float]) -> str:
        """Generate human-readable rollback reason"""
        if trigger == RollbackTrigger.PERFORMANCE_DEGRADATION:
            return f"Performance degraded: Sharpe={current_metrics.get('sharpe_ratio', 0):.3f}"
        elif trigger == RollbackTrigger.RISK_LIMIT_BREACH:
            return f"Risk limit breached: Drawdown={current_metrics.get('max_drawdown', 0):.1%}"
        elif trigger == RollbackTrigger.SYSTEM_ANOMALY:
            return "System anomaly detected"
        else:
            return f"Rollback triggered by {trigger.value}"

    async def _check_alert_conditions(self, model_id: str,
                                   config: RollbackConfiguration,
                                   baseline: PerformanceBaseline,
                                   current_metrics: Dict[str, float]):
        """Check for alert conditions (warnings before rollback)"""
        degradation_score = self._calculate_degradation_score(
            baseline.baseline_metrics, current_metrics
        )

        if degradation_score > self.config['alert_threshold_critical']:
            await self._trigger_alert(model_id, "CRITICAL", degradation_score)
        elif degradation_score > self.config['alert_threshold_warning']:
            await self._trigger_alert(model_id, "WARNING", degradation_score)

    def _calculate_degradation_score(self, baseline: Dict[str, float],
                                   current: Dict[str, float]) -> float:
        """Calculate overall degradation score"""
        score = 0.0
        metrics = ['sharpe_ratio', 'win_rate', 'profit_factor']

        for metric in metrics:
            baseline_val = baseline.get(metric, 1.0)
            current_val = current.get(metric, 1.0)

            if baseline_val > 0:
                degradation = (baseline_val - current_val) / baseline_val
                score += max(0, degradation)

        return min(score / len(metrics), 1.0)

    async def _trigger_alert(self, model_id: str, level: str, score: float):
        """Trigger alert for model performance issues"""
        message = f"{level} alert for {model_id}: degradation score {score:.2f}"

        for callback in self.alert_callbacks:
            try:
                callback(model_id, level, score, message)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(message)

    def _calculate_confidence_intervals(self, metrics: Dict[str, float],
                                      sample_size: int) -> Dict[str, float]:
        """Calculate confidence intervals for baseline metrics"""
        # Simplified calculation
        intervals = {}
        for metric, value in metrics.items():
            # Assume 5% standard deviation for confidence interval
            std = abs(value * 0.05)
            confidence_interval = 1.96 * std / np.sqrt(sample_size)  # 95% CI
            intervals[metric] = confidence_interval
        return intervals

    async def _should_establish_baseline(self, model_id: str) -> bool:
        """Check if baseline should be established"""
        # This would check if enough time has passed since last promotion
        # For now, always return False (would be implemented based on your deployment cycle)
        return False

    async def _establish_baseline(self, model_id: str):
        """Establish new baseline for model"""
        # This would collect performance data over time and establish baseline
        # For now, create a mock baseline
        mock_metrics = {
            'sharpe_ratio': np.random.normal(1.5, 0.1),
            'max_drawdown': np.random.uniform(0.01, 0.05),
            'win_rate': np.random.uniform(0.55, 0.75),
            'profit_factor': np.random.uniform(1.2, 1.8)
        }

        self.establish_performance_baseline(model_id, mock_metrics, sample_size=1000)

    def add_rollback_callback(self, callback: Callable):
        """Add callback for rollback events"""
        self.rollback_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable):
        """Add callback for alert events"""
        self.alert_callbacks.append(callback)

    def get_rollback_history(self, model_id: Optional[str] = None) -> List[RollbackEvent]:
        """Get rollback history for a model or all models"""
        if model_id:
            return [event for event in self.rollback_history if event.model_id == model_id]
        return self.rollback_history.copy()

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get current status of a model"""
        config = self.active_models.get(model_id)
        baseline = self.performance_baselines.get(model_id)

        return {
            'model_id': model_id,
            'is_monitored': model_id in self.active_models,
            'has_baseline': baseline is not None,
            'baseline_age_hours': ((datetime.now() - baseline.established_at).total_seconds() / 3600
                                 if baseline else None),
            'version_history': self.model_versions.get(model_id, []),
            'recent_rollback_count': len([
                event for event in self.rollback_history
                if event.model_id == model_id and
                event.timestamp > datetime.now() - timedelta(days=7)
            ])
        }

    def _cleanup_old_history(self):
        """Clean up old rollback history"""
        cutoff_date = datetime.now() - timedelta(days=self.config['max_rollback_history_days'])
        self.rollback_history = [
            event for event in self.rollback_history
            if event.timestamp > cutoff_date
        ]

    def save_rollback_history(self, path: str):
        """Save rollback history to file"""
        history_dict = [
            {
                'event_id': event.event_id,
                'model_id': event.model_id,
                'trigger': event.trigger.value,
                'timestamp': event.timestamp.isoformat(),
                'reason': event.reason,
                'performance_before': event.performance_before,
                'performance_after': event.performance_after,
                'rollback_successful': event.rollback_successful,
                'recovery_actions': event.recovery_actions
            }
            for event in self.rollback_history
        ]

        with open(path, 'w') as f:
            json.dump(history_dict, f, indent=2, default=str)

        logger.info(f"Saved rollback history to {path}")


class PerformanceMonitor:
    """Monitor model performance metrics"""

    async def get_current_metrics(self, model_id: str) -> Dict[str, float]:
        """Get current performance metrics for a model"""
        # In practice, this would fetch real metrics from your trading system
        # For now, return mock metrics
        await asyncio.sleep(0.1)  # Simulate async operation

        return {
            'sharpe_ratio': np.random.normal(1.5, 0.2),
            'max_drawdown': np.random.uniform(0.01, 0.10),
            'win_rate': np.random.uniform(0.45, 0.75),
            'profit_factor': np.random.uniform(1.0, 2.0),
            'consecutive_losses': np.random.randint(0, 8),
            'daily_pnl': np.random.normal(0.01, 0.02)
        }


class RiskMonitor:
    """Monitor risk metrics"""

    def check_risk_limits(self, metrics: Dict[str, float],
                         limits: Dict[str, float]) -> bool:
        """Check if risk limits are breached"""
        # Implementation would check various risk metrics
        return False


class AnomalyDetector:
    """Detect system anomalies"""

    async def detect_anomalies(self, model_id: str) -> bool:
        """Detect anomalies for a model"""
        # Implementation would use statistical methods to detect anomalies
        return False


# Example usage and testing
if __name__ == "__main__":
    async def test_rollback_service():
        # Create rollback service
        service = AutomatedRollbackService()

        # Create configuration
        config = RollbackConfiguration(
            model_id="test_model",
            monitoring_window_hours=24,
            min_observation_period_hours=1
        )

        # Register model
        service.register_model_for_monitoring("test_model", config)

        # Establish baseline
        baseline_metrics = {
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.03,
            'win_rate': 0.65,
            'profit_factor': 1.4
        }
        service.establish_performance_baseline("test_model", baseline_metrics)

        # Add test callback
        def test_callback(event):
            print(f"Rollback event: {event.event_id} for {event.model_id}")

        service.add_rollback_callback(test_callback)

        # Start monitoring
        service.start_monitoring()

        # Wait a bit
        await asyncio.sleep(10)

        # Stop monitoring
        service.stop_monitoring()

        # Get status
        status = service.get_model_status("test_model")
        print(f"Model status: {status}")

        # Save history
        service.save_rollback_history("rollback_history.json")

    # Run test
    asyncio.run(test_rollback_service())