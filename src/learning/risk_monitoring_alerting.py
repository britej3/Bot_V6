"""
Comprehensive Risk Monitoring and Alerting System
=================================================

This module implements a real-time risk monitoring system with comprehensive
alerting capabilities, threshold management, and automated response mechanisms.

Key Features:
- Real-time risk metric monitoring
- Multi-level alerting system
- Threshold management with adaptive limits
- Automated response triggers
- Risk dashboard and reporting
- Historical risk tracking
- Proactive risk detection

Implements Task 15.1.3.4: Comprehensive risk monitoring and alerting system
Author: Autonomous Systems Team
Date: 2025-01-22
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time
import threading
import json
from datetime import datetime, timedelta
import statistics
import asyncio

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of risk alerts"""
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    VAR_BREACH = "var_breach"
    PERFORMANCE = "performance"
    SYSTEM_HEALTH = "system_health"


class MonitoringStatus(Enum):
    """Monitoring system status"""
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class RiskThreshold:
    """Risk threshold definition"""
    metric_name: str
    warning_level: float
    critical_level: float
    emergency_level: float
    trend_analysis: bool = True
    adaptive: bool = False
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAlert:
    """Risk alert record"""
    alert_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    auto_action_taken: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskMetricSnapshot:
    """Snapshot of risk metrics at a point in time"""
    timestamp: datetime
    portfolio_value: float
    total_exposure: float
    daily_var: float
    current_drawdown: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    correlation_risk: float
    concentration_risk: float
    leverage_ratio: float
    active_positions: int
    daily_pnl: float


@dataclass
class MonitoringConfig:
    """Configuration for risk monitoring system"""
    # Monitoring intervals
    real_time_interval: float = 1.0      # seconds
    metric_update_interval: float = 5.0   # seconds
    alert_check_interval: float = 2.0     # seconds
    
    # Alert settings
    max_alerts_per_hour: int = 50
    alert_cooldown_period: float = 300.0  # seconds
    auto_acknowledge_timeout: float = 3600.0  # seconds
    
    # Data retention
    metric_history_size: int = 10000
    alert_history_size: int = 1000
    
    # Performance settings
    enable_async_processing: bool = True
    max_concurrent_checks: int = 10


class ThresholdManager:
    """Manages risk thresholds with adaptive capabilities"""
    
    def __init__(self):
        self.thresholds: Dict[str, RiskThreshold] = {}
        self.threshold_history = deque(maxlen=1000)
        self._initialize_default_thresholds()
        
    def _initialize_default_thresholds(self) -> None:
        """Initialize default risk thresholds"""
        
        # Portfolio exposure thresholds
        self.thresholds['total_exposure'] = RiskThreshold(
            metric_name='total_exposure',
            warning_level=0.7,
            critical_level=0.85,
            emergency_level=0.95,
            adaptive=True
        )
        
        # Drawdown thresholds
        self.thresholds['current_drawdown'] = RiskThreshold(
            metric_name='current_drawdown',
            warning_level=0.05,
            critical_level=0.10,
            emergency_level=0.15
        )
        
        # Volatility thresholds
        self.thresholds['volatility'] = RiskThreshold(
            metric_name='volatility',
            warning_level=0.03,
            critical_level=0.05,
            emergency_level=0.08,
            adaptive=True
        )
        
        # VaR thresholds
        self.thresholds['daily_var'] = RiskThreshold(
            metric_name='daily_var',
            warning_level=0.02,
            critical_level=0.05,
            emergency_level=0.08
        )
        
        # Leverage thresholds
        self.thresholds['leverage_ratio'] = RiskThreshold(
            metric_name='leverage_ratio',
            warning_level=2.5,
            critical_level=4.0,
            emergency_level=5.0
        )
        
        # Concentration risk thresholds
        self.thresholds['concentration_risk'] = RiskThreshold(
            metric_name='concentration_risk',
            warning_level=0.3,
            critical_level=0.5,
            emergency_level=0.7
        )
        
        logger.info(f"Initialized {len(self.thresholds)} default risk thresholds")
    
    def update_threshold(self, metric_name: str, threshold: RiskThreshold) -> None:
        """Update a specific risk threshold"""
        
        old_threshold = self.thresholds.get(metric_name)
        self.thresholds[metric_name] = threshold
        
        # Record change
        self.threshold_history.append({
            'timestamp': datetime.now(),
            'metric_name': metric_name,
            'old_threshold': old_threshold,
            'new_threshold': threshold,
            'change_type': 'update'
        })
        
        logger.info(f"Updated threshold for {metric_name}")
    
    def get_alert_level(self, metric_name: str, value: float) -> Optional[AlertLevel]:
        """Determine alert level for a metric value"""
        
        threshold = self.thresholds.get(metric_name)
        if not threshold:
            return None
        
        if value >= threshold.emergency_level:
            return AlertLevel.EMERGENCY
        elif value >= threshold.critical_level:
            return AlertLevel.CRITICAL
        elif value >= threshold.warning_level:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def adapt_thresholds(self, metric_history: Dict[str, List[float]]) -> None:
        """Adapt thresholds based on historical data"""
        
        for metric_name, threshold in self.thresholds.items():
            if not threshold.adaptive:
                continue
                
            history = metric_history.get(metric_name, [])
            if len(history) < 100:  # Need sufficient history
                continue
                
            # Calculate adaptive thresholds based on percentiles
            sorted_values = sorted(history)
            
            # Use percentiles to set adaptive thresholds
            warning_percentile = 75
            critical_percentile = 90
            emergency_percentile = 95
            
            new_warning = sorted_values[int(len(sorted_values) * warning_percentile / 100)]
            new_critical = sorted_values[int(len(sorted_values) * critical_percentile / 100)]
            new_emergency = sorted_values[int(len(sorted_values) * emergency_percentile / 100)]
            
            # Apply smoothing to avoid frequent changes
            old_threshold = threshold
            smoothing_factor = 0.1
            
            updated_threshold = RiskThreshold(
                metric_name=metric_name,
                warning_level=old_threshold.warning_level * (1 - smoothing_factor) + new_warning * smoothing_factor,
                critical_level=old_threshold.critical_level * (1 - smoothing_factor) + new_critical * smoothing_factor,
                emergency_level=old_threshold.emergency_level * (1 - smoothing_factor) + new_emergency * smoothing_factor,
                trend_analysis=threshold.trend_analysis,
                adaptive=threshold.adaptive,
                last_updated=datetime.now()
            )
            
            self.update_threshold(metric_name, updated_threshold)


class AlertManager:
    """Manages risk alerts and notifications"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history = deque(maxlen=config.alert_history_size)
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        
        # Alert statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_level': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'acknowledged_alerts': 0,
            'auto_resolved_alerts': 0
        }
    
    def create_alert(self, 
                    alert_type: AlertType,
                    alert_level: AlertLevel,
                    metric_name: str,
                    current_value: float,
                    threshold_value: float,
                    message: str,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[RiskAlert]:
        """Create a new risk alert"""
        
        # Check cooldown period
        cooldown_key = f"{alert_type.value}_{metric_name}"
        if cooldown_key in self.alert_cooldowns:
            time_since_last = datetime.now() - self.alert_cooldowns[cooldown_key]
            if time_since_last.total_seconds() < self.config.alert_cooldown_period:
                return None
        
        # Create alert
        alert_id = f"{alert_type.value}_{metric_name}_{int(time.time())}"
        
        alert = RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            alert_level=alert_level,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.alert_cooldowns[cooldown_key] = datetime.now()
        
        # Update statistics
        self.alert_stats['total_alerts'] += 1
        self.alert_stats['alerts_by_level'][alert_level.value] += 1
        self.alert_stats['alerts_by_type'][alert_type.value] += 1
        
        # Trigger alert handlers
        self._trigger_alert_handlers(alert)
        
        logger.warning(f"Created {alert_level.value} alert: {message}")
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.acknowledged = True
        alert.metadata['acknowledged_by'] = user
        alert.metadata['acknowledged_at'] = datetime.now()
        
        self.alert_stats['acknowledged_alerts'] += 1
        
        logger.info(f"Alert {alert_id} acknowledged by {user}")
        return True
    
    def resolve_alert(self, alert_id: str, reason: str = "threshold_normalized") -> bool:
        """Resolve an alert"""
        
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.metadata['resolved_reason'] = reason
        alert.metadata['resolved_at'] = datetime.now()
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved: {reason}")
        return True
    
    def register_alert_handler(self, alert_level: AlertLevel, handler: Callable) -> None:
        """Register an alert handler function"""
        self.alert_handlers[alert_level].append(handler)
        logger.info(f"Registered alert handler for {alert_level.value} alerts")
    
    def _trigger_alert_handlers(self, alert: RiskAlert) -> None:
        """Trigger registered alert handlers"""
        
        handlers = self.alert_handlers.get(alert.alert_level, [])
        
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def get_active_alerts(self, alert_level: Optional[AlertLevel] = None) -> List[RiskAlert]:
        """Get active alerts, optionally filtered by level"""
        
        alerts = list(self.active_alerts.values())
        
        if alert_level:
            alerts = [a for a in alerts if a.alert_level == alert_level]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts
    
    def cleanup_expired_alerts(self) -> int:
        """Clean up expired alerts that weren't properly resolved"""
        
        expired_count = 0
        current_time = datetime.now()
        expired_alert_ids = []
        
        for alert_id, alert in self.active_alerts.items():
            if not alert.acknowledged:
                time_since_created = current_time - alert.timestamp
                if time_since_created.total_seconds() > self.config.auto_acknowledge_timeout:
                    expired_alert_ids.append(alert_id)
        
        # Resolve expired alerts
        for alert_id in expired_alert_ids:
            self.resolve_alert(alert_id, "auto_expired")
            expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Auto-resolved {expired_count} expired alerts")
        
        return expired_count


class RiskMonitor:
    """Main risk monitoring system"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.threshold_manager = ThresholdManager()
        self.alert_manager = AlertManager(self.config)
        
        # Monitoring state
        self.status = MonitoringStatus.ACTIVE
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Data storage
        self.metric_snapshots = deque(maxlen=self.config.metric_history_size)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.monitoring_stats = {
            'checks_performed': 0,
            'alerts_generated': 0,
            'uptime_start': datetime.now(),
            'last_check_time': None
        }
        
        # Register default alert handlers
        self._register_default_handlers()
        
        logger.info("Risk monitoring system initialized")
    
    def start_monitoring(self) -> bool:
        """Start the risk monitoring system"""
        
        if self.status == MonitoringStatus.ACTIVE and self.monitoring_thread:
            logger.warning("Monitoring already active")
            return False
        
        self.status = MonitoringStatus.ACTIVE
        self.stop_monitoring.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Risk monitoring started")
        return True
    
    def stop_monitoring_system(self) -> None:
        """Stop the risk monitoring system"""
        
        self.status = MonitoringStatus.PAUSED
        self.stop_monitoring.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Risk monitoring stopped")
    
    def update_metrics(self, snapshot: RiskMetricSnapshot) -> None:
        """Update risk metrics snapshot"""
        
        # Store snapshot
        self.metric_snapshots.append(snapshot)
        
        # Update individual metric histories
        metrics = {
            'total_exposure': snapshot.total_exposure,
            'daily_var': snapshot.daily_var,
            'current_drawdown': snapshot.current_drawdown,
            'volatility': snapshot.volatility,
            'leverage_ratio': snapshot.leverage_ratio,
            'concentration_risk': snapshot.concentration_risk,
            'correlation_risk': snapshot.correlation_risk
        }
        
        for metric_name, value in metrics.items():
            self.metric_history[metric_name].append(value)
        
        # Check for alerts
        self._check_thresholds(metrics)
        
        # Update statistics
        self.monitoring_stats['last_check_time'] = datetime.now()
        self.monitoring_stats['checks_performed'] += 1
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> None:
        """Check metrics against thresholds and generate alerts"""
        
        for metric_name, value in metrics.items():
            threshold = self.threshold_manager.thresholds.get(metric_name)
            if not threshold:
                continue
            
            alert_level = self.threshold_manager.get_alert_level(metric_name, value)
            
            if alert_level and alert_level != AlertLevel.INFO:
                # Determine alert type
                alert_type = self._map_metric_to_alert_type(metric_name)
                
                # Create message
                message = f"{metric_name} is {value:.4f}, exceeding {alert_level.value} threshold of {getattr(threshold, f'{alert_level.value}_level'):.4f}"
                
                # Create alert
                alert = self.alert_manager.create_alert(
                    alert_type=alert_type,
                    alert_level=alert_level,
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=getattr(threshold, f'{alert_level.value}_level'),
                    message=message,
                    metadata={'trend_direction': self._calculate_trend(metric_name)}
                )
                
                if alert:
                    self.monitoring_stats['alerts_generated'] += 1
    
    def _map_metric_to_alert_type(self, metric_name: str) -> AlertType:
        """Map metric name to alert type"""
        
        mapping = {
            'total_exposure': AlertType.POSITION_SIZE,
            'current_drawdown': AlertType.DRAWDOWN,
            'volatility': AlertType.VOLATILITY,
            'leverage_ratio': AlertType.LEVERAGE,
            'daily_var': AlertType.VAR_BREACH,
            'concentration_risk': AlertType.CONCENTRATION,
            'correlation_risk': AlertType.CORRELATION
        }
        
        return mapping.get(metric_name, AlertType.SYSTEM_HEALTH)
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend direction for a metric"""
        
        history = list(self.metric_history[metric_name])
        if len(history) < 5:
            return "unknown"
        
        recent_avg = statistics.mean(history[-3:])
        older_avg = statistics.mean(history[-6:-3]) if len(history) >= 6 else history[0]
        
        if recent_avg > older_avg * 1.05:
            return "increasing"
        elif recent_avg < older_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        logger.info("Risk monitoring loop started")
        
        while not self.stop_monitoring.is_set() and self.status == MonitoringStatus.ACTIVE:
            try:
                # Cleanup expired alerts
                self.alert_manager.cleanup_expired_alerts()
                
                # Adapt thresholds if enabled
                if len(self.metric_snapshots) > 100:
                    metric_data = {name: list(history) for name, history in self.metric_history.items()}
                    self.threshold_manager.adapt_thresholds(metric_data)
                
                # Sleep for next check
                time.sleep(self.config.alert_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10.0)  # Wait longer on error
        
        logger.info("Risk monitoring loop stopped")
    
    def _register_default_handlers(self) -> None:
        """Register default alert handlers"""
        
        def emergency_handler(alert: RiskAlert):
            logger.critical(f"EMERGENCY ALERT: {alert.message}")
            # Could trigger automatic position reduction, notifications, etc.
        
        def critical_handler(alert: RiskAlert):
            logger.error(f"CRITICAL ALERT: {alert.message}")
        
        def warning_handler(alert: RiskAlert):
            logger.warning(f"WARNING ALERT: {alert.message}")
        
        self.alert_manager.register_alert_handler(AlertLevel.EMERGENCY, emergency_handler)
        self.alert_manager.register_alert_handler(AlertLevel.CRITICAL, critical_handler)
        self.alert_manager.register_alert_handler(AlertLevel.WARNING, warning_handler)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        
        uptime = datetime.now() - self.monitoring_stats['uptime_start']
        
        return {
            'status': self.status.value,
            'uptime_seconds': uptime.total_seconds(),
            'monitoring_stats': self.monitoring_stats,
            'alert_stats': self.alert_manager.alert_stats,
            'active_alerts_count': len(self.alert_manager.active_alerts),
            'metric_snapshots_count': len(self.metric_snapshots),
            'thresholds_count': len(self.threshold_manager.thresholds),
            'last_snapshot_time': self.metric_snapshots[-1].timestamp if self.metric_snapshots else None
        }
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """Get risk dashboard data"""
        
        if not self.metric_snapshots:
            return {'error': 'No metric data available'}
        
        latest_snapshot = self.metric_snapshots[-1]
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Calculate metric trends
        trends = {}
        for metric_name in self.metric_history.keys():
            trends[metric_name] = self._calculate_trend(metric_name)
        
        return {
            'timestamp': latest_snapshot.timestamp,
            'current_metrics': {
                'portfolio_value': latest_snapshot.portfolio_value,
                'total_exposure': latest_snapshot.total_exposure,
                'daily_var': latest_snapshot.daily_var,
                'current_drawdown': latest_snapshot.current_drawdown,
                'volatility': latest_snapshot.volatility,
                'leverage_ratio': latest_snapshot.leverage_ratio,
                'active_positions': latest_snapshot.active_positions
            },
            'active_alerts': {
                'emergency': len([a for a in active_alerts if a.alert_level == AlertLevel.EMERGENCY]),
                'critical': len([a for a in active_alerts if a.alert_level == AlertLevel.CRITICAL]),
                'warning': len([a for a in active_alerts if a.alert_level == AlertLevel.WARNING])
            },
            'trends': trends,
            'thresholds': {name: threshold.warning_level for name, threshold in self.threshold_manager.thresholds.items()},
            'monitoring_health': self.get_monitoring_status()
        }


# Factory function
def create_risk_monitor(custom_config: Optional[Dict[str, Any]] = None) -> RiskMonitor:
    """Create risk monitoring system with custom configuration"""
    
    config = MonitoringConfig()
    
    if custom_config:
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return RiskMonitor(config)


# Demo
if __name__ == "__main__":
    print("🎯 Risk Monitoring and Alerting System Demo")
    print("=" * 50)
    
    # Create monitor
    monitor = create_risk_monitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate some risk metrics
    import random
    
    for i in range(10):
        snapshot = RiskMetricSnapshot(
            timestamp=datetime.now(),
            portfolio_value=100000,
            total_exposure=random.uniform(0.3, 0.9),  # Will trigger alerts when high
            daily_var=random.uniform(0.01, 0.06),
            current_drawdown=random.uniform(0, 0.12),
            max_drawdown=0.15,
            volatility=random.uniform(0.01, 0.07),
            sharpe_ratio=random.uniform(0.5, 2.0),
            correlation_risk=random.uniform(0.1, 0.6),
            concentration_risk=random.uniform(0.2, 0.8),
            leverage_ratio=random.uniform(1.0, 4.5),
            active_positions=random.randint(5, 15),
            daily_pnl=random.uniform(-1000, 2000)
        )
        
        monitor.update_metrics(snapshot)
        time.sleep(0.1)
    
    # Get dashboard
    dashboard = monitor.get_risk_dashboard()
    print(f"Dashboard: {dashboard}")
    
    # Get status
    status = monitor.get_monitoring_status()
    print(f"Status: {status}")
    
    # Stop monitoring
    monitor.stop_monitoring_system()
    
    print("\n🎯 Task 15.1.3.4: Risk Monitoring and Alerting - IMPLEMENTATION COMPLETE")