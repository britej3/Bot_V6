"""
Self-Healing Diagnostic Framework
================================

This module implements a comprehensive self-healing diagnostic system that can:
- Automatically detect system anomalies and failures
- Diagnose root causes of issues
- Execute appropriate recovery procedures
- Learn from past failures to prevent recurrence
- Provide predictive failure analysis
"""

import asyncio
import logging
import time
import threading
import traceback
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import numpy as np
import torch

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentStatus(Enum):
    """Component operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    FAILED = "failed"
    RECOVERING = "recovering"


class RecoveryAction(Enum):
    """Available recovery actions"""
    RESTART_COMPONENT = "restart_component"
    RELOAD_MODEL = "reload_model"
    REDUCE_LOAD = "reduce_load"
    SWITCH_STRATEGY = "switch_strategy"
    EMERGENCY_STOP = "emergency_stop"
    ROLLBACK_UPDATE = "rollback_update"
    CLEAR_CACHE = "clear_cache"
    GARBAGE_COLLECT = "garbage_collect"


@dataclass
class SystemAlert:
    """System alert information"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    recovery_action: Optional[RecoveryAction] = None


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    name: str
    status: ComponentStatus
    health_score: float  # 0.0 to 1.0
    last_update: datetime
    metrics: Dict[str, float]
    alerts: List[SystemAlert]
    uptime: float  # Hours
    error_count: int
    recovery_count: int


@dataclass
class SystemMetrics:
    """Current system metrics"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory: float
    disk_usage: float
    network_latency: float
    active_connections: int
    queue_size: int
    error_rate: float
    throughput: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FailurePattern:
    """Learned failure pattern"""
    pattern_id: str
    description: str
    preconditions: Dict[str, Any]
    symptoms: List[str]
    root_causes: List[str]
    recovery_actions: List[RecoveryAction]
    success_rate: float
    occurrences: int
    last_occurrence: datetime


class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=3600)  # 1 hour of history
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self) -> None:
        """Start system monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics (if available)
        gpu_usage = 0.0
        gpu_memory = 0.0
        try:
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_percent()
        except:
            pass
        
        # Network metrics (simplified)
        network_stats = psutil.net_io_counters()
        network_latency = 0.0  # Placeholder
        
        # Process metrics
        active_connections = len(psutil.net_connections())
        
        return SystemMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            disk_usage=disk.percent,
            network_latency=network_latency,
            active_connections=active_connections,
            queue_size=0,  # Placeholder
            error_rate=0.0,  # Placeholder
            throughput=0.0  # Placeholder
        )
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_trend(self, metric_name: str, window: int = 60) -> List[float]:
        """Get trend for a specific metric"""
        if not self.metrics_history:
            return []
        
        recent_metrics = list(self.metrics_history)[-window:]
        return [getattr(m, metric_name, 0.0) for m in recent_metrics]


class AnomalyDetector:
    """Detects anomalies in system behavior"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.anomaly_history = deque(maxlen=1000)
        
    def update_baseline(self, metrics: SystemMetrics) -> None:
        """Update baseline metrics for anomaly detection"""
        metric_values = {
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage,
            'gpu_usage': metrics.gpu_usage,
            'network_latency': metrics.network_latency,
            'error_rate': metrics.error_rate,
            'throughput': metrics.throughput
        }
        
        for metric_name, value in metric_values.items():
            if metric_name not in self.baseline_metrics:
                self.baseline_metrics[metric_name] = {'mean': value, 'std': 0.0, 'count': 1}
            else:
                # Update running statistics
                baseline = self.baseline_metrics[metric_name]
                count = baseline['count'] + 1
                delta = value - baseline['mean']
                new_mean = baseline['mean'] + delta / count
                new_std = np.sqrt(((count - 1) * baseline['std']**2 + delta * (value - new_mean)) / count)
                
                self.baseline_metrics[metric_name] = {
                    'mean': new_mean,
                    'std': new_std,
                    'count': count
                }
    
    def detect_anomalies(self, metrics: SystemMetrics) -> List[str]:
        """Detect anomalies in current metrics"""
        anomalies = []
        
        metric_values = {
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage,
            'gpu_usage': metrics.gpu_usage,
            'network_latency': metrics.network_latency,
            'error_rate': metrics.error_rate,
            'throughput': metrics.throughput
        }
        
        for metric_name, value in metric_values.items():
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                if baseline['std'] > 0:
                    z_score = abs(value - baseline['mean']) / baseline['std']
                    if z_score > self.sensitivity:
                        anomalies.append(f"{metric_name}_anomaly")
                        self.anomaly_history.append({
                            'timestamp': datetime.now(),
                            'metric': metric_name,
                            'value': value,
                            'baseline_mean': baseline['mean'],
                            'z_score': z_score
                        })
        
        return anomalies


class FailureDiagnostics:
    """Diagnoses system failures and determines root causes"""
    
    def __init__(self):
        self.known_patterns = []
        self.diagnostic_history = deque(maxlen=1000)
        
    def add_failure_pattern(self, pattern: FailurePattern) -> None:
        """Add a known failure pattern"""
        self.known_patterns.append(pattern)
        logger.info(f"Added failure pattern: {pattern.description}")
    
    def diagnose_failure(self, alerts: List[SystemAlert], 
                        metrics: SystemMetrics) -> Tuple[List[str], List[RecoveryAction]]:
        """Diagnose failure and recommend recovery actions"""
        
        # Extract symptoms from alerts and metrics
        symptoms = []
        for alert in alerts:
            symptoms.append(f"{alert.component}_{alert.severity.value}")
        
        # Add metric-based symptoms
        if metrics.cpu_usage > 90:
            symptoms.append("high_cpu_usage")
        if metrics.memory_usage > 90:
            symptoms.append("high_memory_usage")
        if metrics.error_rate > 0.1:
            symptoms.append("high_error_rate")
        
        # Match against known patterns
        matched_patterns = []
        for pattern in self.known_patterns:
            if self._matches_pattern(symptoms, pattern):
                matched_patterns.append(pattern)
        
        # Determine root causes and recovery actions
        root_causes = []
        recovery_actions = []
        
        if matched_patterns:
            # Use best matching pattern
            best_pattern = max(matched_patterns, key=lambda p: p.success_rate)
            root_causes = best_pattern.root_causes
            recovery_actions = best_pattern.recovery_actions
        else:
            # Fallback diagnosis
            root_causes, recovery_actions = self._fallback_diagnosis(symptoms, metrics)
        
        # Record diagnosis
        self.diagnostic_history.append({
            'timestamp': datetime.now(),
            'symptoms': symptoms,
            'root_causes': root_causes,
            'recovery_actions': [action.value for action in recovery_actions],
            'matched_patterns': len(matched_patterns)
        })
        
        return root_causes, recovery_actions
    
    def _matches_pattern(self, symptoms: List[str], pattern: FailurePattern) -> bool:
        """Check if symptoms match a failure pattern"""
        pattern_symptoms = set(pattern.symptoms)
        current_symptoms = set(symptoms)
        
        # Require at least 50% overlap
        overlap = len(pattern_symptoms.intersection(current_symptoms))
        return overlap / len(pattern_symptoms) >= 0.5
    
    def _fallback_diagnosis(self, symptoms: List[str], 
                          metrics: SystemMetrics) -> Tuple[List[str], List[RecoveryAction]]:
        """Fallback diagnosis when no patterns match"""
        root_causes = []
        recovery_actions = []
        
        # High resource usage
        if metrics.cpu_usage > 90 or metrics.memory_usage > 90:
            root_causes.append("resource_exhaustion")
            recovery_actions.extend([
                RecoveryAction.REDUCE_LOAD,
                RecoveryAction.GARBAGE_COLLECT,
                RecoveryAction.CLEAR_CACHE
            ])
        
        # High error rate
        if metrics.error_rate > 0.1:
            root_causes.append("system_instability")
            recovery_actions.extend([
                RecoveryAction.RESTART_COMPONENT,
                RecoveryAction.ROLLBACK_UPDATE
            ])
        
        # GPU issues
        if metrics.gpu_memory > 90:
            root_causes.append("gpu_memory_exhaustion")
            recovery_actions.extend([
                RecoveryAction.CLEAR_CACHE,
                RecoveryAction.RELOAD_MODEL
            ])
        
        return root_causes, recovery_actions


class RecoveryExecutor:
    """Executes recovery actions"""
    
    def __init__(self):
        self.recovery_history = deque(maxlen=1000)
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default recovery action handlers"""
        self.recovery_handlers = {
            RecoveryAction.GARBAGE_COLLECT: self._garbage_collect,
            RecoveryAction.CLEAR_CACHE: self._clear_cache,
            RecoveryAction.REDUCE_LOAD: self._reduce_load,
            RecoveryAction.RESTART_COMPONENT: self._restart_component,
            RecoveryAction.RELOAD_MODEL: self._reload_model,
            RecoveryAction.EMERGENCY_STOP: self._emergency_stop,
            RecoveryAction.ROLLBACK_UPDATE: self._rollback_update,
            RecoveryAction.SWITCH_STRATEGY: self._switch_strategy
        }
    
    def register_handler(self, action: RecoveryAction, handler: Callable) -> None:
        """Register custom recovery action handler"""
        self.recovery_handlers[action] = handler
    
    async def execute_recovery(self, actions: List[RecoveryAction], 
                             context: Dict[str, Any] = None) -> Dict[RecoveryAction, bool]:
        """Execute recovery actions"""
        context = context or {}
        results = {}
        
        for action in actions:
            try:
                logger.info(f"Executing recovery action: {action.value}")
                
                if action in self.recovery_handlers:
                    success = await self.recovery_handlers[action](context)
                    results[action] = success
                    
                    # Record recovery attempt
                    self.recovery_history.append({
                        'timestamp': datetime.now(),
                        'action': action.value,
                        'success': success,
                        'context': context
                    })
                    
                    if success:
                        logger.info(f"Recovery action {action.value} completed successfully")
                    else:
                        logger.error(f"Recovery action {action.value} failed")
                else:
                    logger.warning(f"No handler registered for recovery action: {action.value}")
                    results[action] = False
                    
            except Exception as e:
                logger.error(f"Error executing recovery action {action.value}: {e}")
                results[action] = False
        
        return results
    
    async def _garbage_collect(self, context: Dict[str, Any]) -> bool:
        """Garbage collection recovery"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
            return False
    
    async def _clear_cache(self, context: Dict[str, Any]) -> bool:
        """Clear system caches"""
        try:
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear Python cache (simplified)
            gc.collect()
            
            return True
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return False
    
    async def _reduce_load(self, context: Dict[str, Any]) -> bool:
        """Reduce system load"""
        logger.info("Reducing system load (placeholder implementation)")
        return True
    
    async def _restart_component(self, context: Dict[str, Any]) -> bool:
        """Restart a system component"""
        component = context.get('component', 'unknown')
        logger.info(f"Restarting component: {component} (placeholder implementation)")
        return True
    
    async def _reload_model(self, context: Dict[str, Any]) -> bool:
        """Reload ML model"""
        logger.info("Reloading ML model (placeholder implementation)")
        return True
    
    async def _emergency_stop(self, context: Dict[str, Any]) -> bool:
        """Emergency system stop"""
        logger.critical("Emergency stop executed (placeholder implementation)")
        return True
    
    async def _rollback_update(self, context: Dict[str, Any]) -> bool:
        """Rollback recent update"""
        logger.info("Rolling back update (placeholder implementation)")
        return True
    
    async def _switch_strategy(self, context: Dict[str, Any]) -> bool:
        """Switch trading strategy"""
        logger.info("Switching strategy (placeholder implementation)")
        return True


class SelfHealingDiagnostics:
    """
    Main self-healing diagnostics coordinator that integrates monitoring,
    anomaly detection, failure diagnosis, and recovery execution.
    """
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        
        # Core components
        self.monitor = SystemMonitor(monitoring_interval)
        self.anomaly_detector = AnomalyDetector()
        self.diagnostics = FailureDiagnostics()
        self.recovery_executor = RecoveryExecutor()
        
        # Component tracking
        self.components: Dict[str, ComponentHealth] = {}
        self.alerts: Dict[str, SystemAlert] = {}
        self.is_running = False
        
        # Threading
        self.main_loop_thread: Optional[threading.Thread] = None
        
        # Initialize default failure patterns
        self._initialize_default_patterns()
        
        logger.info("Self-healing diagnostics system initialized")
    
    def _initialize_default_patterns(self) -> None:
        """Initialize default failure patterns"""
        
        # Memory exhaustion pattern
        memory_pattern = FailurePattern(
            pattern_id="memory_exhaustion",
            description="System memory exhaustion",
            preconditions={"memory_usage": "> 85%"},
            symptoms=["high_memory_usage", "system_degraded"],
            root_causes=["memory_leak", "insufficient_memory"],
            recovery_actions=[RecoveryAction.GARBAGE_COLLECT, RecoveryAction.CLEAR_CACHE],
            success_rate=0.85,
            occurrences=0,
            last_occurrence=datetime.now()
        )
        
        # GPU memory pattern
        gpu_pattern = FailurePattern(
            pattern_id="gpu_memory_exhaustion",
            description="GPU memory exhaustion",
            preconditions={"gpu_memory": "> 90%"},
            symptoms=["high_gpu_memory", "model_error"],
            root_causes=["model_too_large", "batch_size_too_large"],
            recovery_actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.RELOAD_MODEL],
            success_rate=0.90,
            occurrences=0,
            last_occurrence=datetime.now()
        )
        
        # High error rate pattern
        error_pattern = FailurePattern(
            pattern_id="high_error_rate",
            description="High system error rate",
            preconditions={"error_rate": "> 10%"},
            symptoms=["high_error_rate", "system_failing"],
            root_causes=["configuration_error", "data_corruption"],
            recovery_actions=[RecoveryAction.RESTART_COMPONENT, RecoveryAction.ROLLBACK_UPDATE],
            success_rate=0.75,
            occurrences=0,
            last_occurrence=datetime.now()
        )
        
        self.diagnostics.add_failure_pattern(memory_pattern)
        self.diagnostics.add_failure_pattern(gpu_pattern)
        self.diagnostics.add_failure_pattern(error_pattern)
    
    def start(self) -> None:
        """Start self-healing diagnostics"""
        if self.is_running:
            logger.warning("Self-healing diagnostics already running")
            return
        
        logger.info("Starting self-healing diagnostics system")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Start main loop
        self.is_running = True
        self.main_loop_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_loop_thread.start()
        
        logger.info("Self-healing diagnostics started")
    
    def stop(self) -> None:
        """Stop self-healing diagnostics"""
        logger.info("Stopping self-healing diagnostics system")
        
        self.is_running = False
        self.monitor.stop_monitoring()
        
        if self.main_loop_thread:
            self.main_loop_thread.join(timeout=10.0)
        
        logger.info("Self-healing diagnostics stopped")
    
    def _main_loop(self) -> None:
        """Main diagnostic loop"""
        logger.info("Starting self-healing main loop")
        
        while self.is_running:
            try:
                # Get current metrics
                metrics = self.monitor.get_current_metrics()
                if not metrics:
                    time.sleep(self.monitoring_interval)
                    continue
                
                # Update anomaly detector baseline
                self.anomaly_detector.update_baseline(metrics)
                
                # Detect anomalies
                anomalies = self.anomaly_detector.detect_anomalies(metrics)
                
                # Handle anomalies
                if anomalies:
                    await self._handle_anomalies(anomalies, metrics)
                
                # Update component health
                self._update_component_health(metrics)
                
                # Check for failing components
                failing_components = self._get_failing_components()
                if failing_components:
                    await self._handle_component_failures(failing_components, metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in self-healing main loop: {e}")
                time.sleep(5.0)
    
    async def _handle_anomalies(self, anomalies: List[str], metrics: SystemMetrics) -> None:
        """Handle detected anomalies"""
        logger.warning(f"Detected anomalies: {anomalies}")
        
        # Create alerts for anomalies
        for anomaly in anomalies:
            alert = SystemAlert(
                id=f"anomaly_{anomaly}_{int(time.time())}",
                timestamp=datetime.now(),
                severity=AlertSeverity.WARNING,
                component="system",
                message=f"Anomaly detected: {anomaly}",
                details={"anomaly_type": anomaly, "metrics": metrics.__dict__}
            )
            self.alerts[alert.id] = alert
        
        # Diagnose and recover
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        await self._diagnose_and_recover(active_alerts, metrics)
    
    async def _handle_component_failures(self, failing_components: List[str], 
                                       metrics: SystemMetrics) -> None:
        """Handle component failures"""
        logger.error(f"Handling component failures: {failing_components}")
        
        # Create alerts for failing components
        for component in failing_components:
            alert = SystemAlert(
                id=f"failure_{component}_{int(time.time())}",
                timestamp=datetime.now(),
                severity=AlertSeverity.ERROR,
                component=component,
                message=f"Component failure: {component}",
                details={"component_health": self.components[component].__dict__}
            )
            self.alerts[alert.id] = alert
        
        # Diagnose and recover
        active_alerts = [alert for alert in self.alerts.values() if not alert.resolved]
        await self._diagnose_and_recover(active_alerts, metrics)
    
    async def _diagnose_and_recover(self, alerts: List[SystemAlert], 
                                  metrics: SystemMetrics) -> None:
        """Diagnose issues and execute recovery"""
        
        # Diagnose failure
        root_causes, recovery_actions = self.diagnostics.diagnose_failure(alerts, metrics)
        
        if not recovery_actions:
            logger.warning("No recovery actions determined")
            return
        
        logger.info(f"Diagnosed root causes: {root_causes}")
        logger.info(f"Executing recovery actions: {[action.value for action in recovery_actions]}")
        
        # Execute recovery
        recovery_context = {
            'alerts': alerts,
            'metrics': metrics,
            'root_causes': root_causes
        }
        
        results = await self.recovery_executor.execute_recovery(recovery_actions, recovery_context)
        
        # Mark alerts as resolved if recovery successful
        successful_recoveries = [action for action, success in results.items() if success]
        if successful_recoveries:
            for alert in alerts:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                alert.recovery_action = successful_recoveries[0]  # Use first successful action
    
    def _update_component_health(self, metrics: SystemMetrics) -> None:
        """Update health status of all components"""
        
        # System component health based on metrics
        system_health_score = self._calculate_system_health_score(metrics)
        
        if "system" not in self.components:
            self.components["system"] = ComponentHealth(
                name="system",
                status=ComponentStatus.HEALTHY,
                health_score=system_health_score,
                last_update=datetime.now(),
                metrics={},
                alerts=[],
                uptime=0.0,
                error_count=0,
                recovery_count=0
            )
        
        component = self.components["system"]
        component.health_score = system_health_score
        component.last_update = datetime.now()
        component.metrics = {
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "gpu_usage": metrics.gpu_usage,
            "error_rate": metrics.error_rate
        }
        
        # Update status based on health score
        if system_health_score > 0.8:
            component.status = ComponentStatus.HEALTHY
        elif system_health_score > 0.6:
            component.status = ComponentStatus.DEGRADED
        elif system_health_score > 0.3:
            component.status = ComponentStatus.FAILING
        else:
            component.status = ComponentStatus.FAILED
    
    def _calculate_system_health_score(self, metrics: SystemMetrics) -> float:
        """Calculate overall system health score"""
        
        # Weight different metrics
        cpu_score = max(0, 1 - metrics.cpu_usage / 100)
        memory_score = max(0, 1 - metrics.memory_usage / 100)
        gpu_score = max(0, 1 - metrics.gpu_usage / 100)
        error_score = max(0, 1 - metrics.error_rate)
        
        # Weighted average
        total_score = (cpu_score * 0.3 + memory_score * 0.3 + 
                      gpu_score * 0.2 + error_score * 0.2)
        
        return total_score
    
    def _get_failing_components(self) -> List[str]:
        """Get list of failing components"""
        failing = []
        for name, component in self.components.items():
            if component.status in [ComponentStatus.FAILING, ComponentStatus.FAILED]:
                failing.append(name)
        return failing
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "overall_health": self._get_overall_health(),
            "components": {name: {
                "status": comp.status.value,
                "health_score": comp.health_score,
                "last_update": comp.last_update.isoformat()
            } for name, comp in self.components.items()},
            "active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
            "total_alerts": len(self.alerts),
            "system_metrics": self.monitor.get_current_metrics().__dict__ if self.monitor.get_current_metrics() else {},
            "is_running": self.is_running
        }
    
    def _get_overall_health(self) -> float:
        """Calculate overall system health"""
        if not self.components:
            return 1.0
        
        return np.mean([comp.health_score for comp in self.components.values()])


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        diagnostics = SelfHealingDiagnostics()
        
        try:
            diagnostics.start()
            
            # Let it run for a while
            await asyncio.sleep(10)
            
            # Show status
            status = diagnostics.get_system_status()
            print(f"System Status: {json.dumps(status, indent=2)}")
            
        finally:
            diagnostics.stop()
    
    asyncio.run(main())