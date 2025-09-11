"""
Performance Monitoring for XGBoost Enhanced Crypto Futures Scalping Platform
Comprehensive monitoring, alerting, and performance analytics
"""

import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque
import json
import asyncio

try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from src.config.trading_config import AdvancedTradingConfig

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.start_time = time.time()

        # Prediction metrics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.average_prediction_time = 0.0
        self.prediction_times = deque(maxlen=1000)

        # Feature engineering metrics
        self.total_features_processed = 0
        self.feature_processing_times = deque(maxlen=1000)
        self.average_feature_time = 0.0

        # Trading metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.win_rate = 0.0

        # System metrics
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.disk_usage = deque(maxlen=100)

        # Error metrics
        self.total_errors = 0
        self.validation_errors = 0
        self.risk_errors = 0
        self.system_errors = 0

        # Model metrics
        self.model_accuracy = 0.0
        self.model_precision = 0.0
        self.model_recall = 0.0
        self.model_f1_score = 0.0

        # Data quality metrics
        self.data_quality_score = 1.0
        self.data_latency = 0.0
        self.missing_data_points = 0


class AlertManager:
    """Manage alerts and notifications"""

    def __init__(self, config: AdvancedTradingConfig):
        self.config = config
        self.alerts = []
        self.alert_callbacks = []

        # Alert thresholds
        self.thresholds = {
            'max_drawdown': config.max_drawdown_pct,
            'max_cpu_usage': 90.0,
            'max_memory_usage': 90.0,
            'max_prediction_time': 1.0,  # seconds
            'max_feature_time': 0.5,     # seconds
            'min_win_rate': 0.4,
            'max_error_rate': 0.1,
            'max_data_latency': 5.0      # seconds
        }

    def add_alert_callback(self, callback: Callable):
        """Add alert callback"""
        self.alert_callbacks.append(callback)

    def trigger_alert(self, alert_type: str, message: str, severity: str = "WARNING"):
        """Trigger an alert"""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }

        self.alerts.append(alert)

        # Keep only recent alerts
        if len(self.alerts) > 1000:
            self.alerts.pop(0)

        logger.warning(f"🚨 {severity}: {alert_type} - {message}")

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                asyncio.create_task(callback(alert))
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def check_thresholds(self, metrics: PerformanceMetrics):
        """Check if any thresholds are exceeded"""
        # Check drawdown
        if metrics.max_drawdown >= self.thresholds['max_drawdown']:
            self.trigger_alert(
                "DRAWDOWN_LIMIT",
                ".2%",
                "CRITICAL"
            )

        # Check win rate
        if metrics.total_trades > 10 and metrics.win_rate < self.thresholds['min_win_rate']:
            self.trigger_alert(
                "LOW_WIN_RATE",
                ".2%",
                "WARNING"
            )

        # Check error rate
        error_rate = metrics.total_errors / max(metrics.total_predictions, 1)
        if error_rate > self.thresholds['max_error_rate']:
            self.trigger_alert(
                "HIGH_ERROR_RATE",
                ".2%",
                "WARNING"
            )

        # Check performance degradation
        if metrics.average_prediction_time > self.thresholds['max_prediction_time']:
            self.trigger_alert(
                "SLOW_PREDICTIONS",
                ".3f",
                "WARNING"
            )

        # Check system resources
        if metrics.cpu_usage and list(metrics.cpu_usage)[-1] > self.thresholds['max_cpu_usage']:
            self.trigger_alert(
                "HIGH_CPU_USAGE",
                ".1f",
                "WARNING"
            )

        if metrics.memory_usage and list(metrics.memory_usage)[-1] > self.thresholds['max_memory_usage']:
            self.trigger_alert(
                "HIGH_MEMORY_USAGE",
                ".1f",
                "WARNING"
            )


class XGBoostPerformanceMonitor:
    """
    Comprehensive performance monitoring for XGBoost trading system

    Features:
    - Real-time metrics collection
    - Prometheus integration
    - Alert management
    - Performance analytics
    - System health monitoring
    - Resource usage tracking
    """

    def __init__(self, config: AdvancedTradingConfig):
        self.config = config
        self.metrics = PerformanceMetrics()
        self.alert_manager = AlertManager(config)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.collection_interval = 5  # seconds

        # Prometheus metrics (if available)
        self.prometheus_metrics = {}
        if PROMETHEUS_AVAILABLE:
            self._initialize_prometheus_metrics()

        # Performance history
        self.performance_history = deque(maxlen=1000)

        logger.info("📊 XGBoost Performance Monitor initialized")

    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            self.prometheus_metrics = {
                'predictions_total': prom.Counter(
                    'xgboost_predictions_total',
                    'Total number of predictions made'
                ),
                'prediction_latency': prom.Histogram(
                    'xgboost_prediction_latency_seconds',
                    'Prediction latency in seconds'
                ),
                'trades_total': prom.Counter(
                    'xgboost_trades_total',
                    'Total number of trades executed'
                ),
                'pnl_total': prom.Gauge(
                    'xgboost_pnl_total',
                    'Total profit and loss'
                ),
                'cpu_usage': prom.Gauge(
                    'xgboost_cpu_usage_percent',
                    'CPU usage percentage'
                ),
                'memory_usage': prom.Gauge(
                    'xgboost_memory_usage_percent',
                    'Memory usage percentage'
                ),
                'win_rate': prom.Gauge(
                    'xgboost_win_rate',
                    'Current win rate'
                ),
                'errors_total': prom.Counter(
                    'xgboost_errors_total',
                    'Total number of errors'
                )
            }

            logger.info("✅ Prometheus metrics initialized")

        except Exception as e:
            logger.error(f"❌ Prometheus initialization failed: {e}")

    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("📊 Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)

        logger.info("📊 Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                self.alert_manager.check_thresholds(self.metrics)
                self._update_prometheus_metrics()

                # Store performance snapshot
                snapshot = self.get_performance_snapshot()
                self.performance_history.append(snapshot)

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1)

    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.cpu_usage.append(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics.memory_usage.append(memory_percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self.metrics.disk_usage.append(disk_percent)

        except Exception as e:
            logger.error(f"System metrics collection error: {e}")

    def _update_prometheus_metrics(self):
        """Update Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE or not self.prometheus_metrics:
            return

        try:
            # Update gauges
            self.prometheus_metrics['cpu_usage'].set(list(self.metrics.cpu_usage)[-1] if self.metrics.cpu_usage else 0)
            self.prometheus_metrics['memory_usage'].set(list(self.metrics.memory_usage)[-1] if self.metrics.memory_usage else 0)
            self.prometheus_metrics['pnl_total'].set(self.metrics.total_pnl)
            self.prometheus_metrics['win_rate'].set(self.metrics.win_rate)

        except Exception as e:
            logger.error(f"Prometheus update error: {e}")

    def record_prediction(self, prediction_time: float, success: bool = True):
        """Record a prediction"""
        try:
            self.metrics.total_predictions += 1

            if success:
                self.metrics.successful_predictions += 1
            else:
                self.metrics.failed_predictions += 1

            # Update prediction time metrics
            self.metrics.prediction_times.append(prediction_time)
            self.metrics.average_prediction_time = sum(self.metrics.prediction_times) / len(self.metrics.prediction_times)

            # Update Prometheus
            if 'predictions_total' in self.prometheus_metrics:
                self.prometheus_metrics['predictions_total'].inc()

            if 'prediction_latency' in self.prometheus_metrics:
                self.prometheus_metrics['prediction_latency'].observe(prediction_time)

        except Exception as e:
            logger.error(f"Prediction recording error: {e}")

    def record_feature_processing(self, processing_time: float):
        """Record feature processing"""
        try:
            self.metrics.total_features_processed += 1
            self.metrics.feature_processing_times.append(processing_time)
            self.metrics.average_feature_time = sum(self.metrics.feature_processing_times) / len(self.metrics.feature_processing_times)

        except Exception as e:
            logger.error(f"Feature processing recording error: {e}")

    def record_trade(self, pnl: float, win: bool = False):
        """Record a trade"""
        try:
            self.metrics.total_trades += 1

            if win:
                self.metrics.winning_trades += 1

            self.metrics.total_pnl += pnl

            # Update win rate
            if self.metrics.total_trades > 0:
                self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

            # Update drawdown
            peak_equity = 10000 + max(0, self.metrics.total_pnl)  # Simplified
            current_equity = 10000 + self.metrics.total_pnl
            drawdown = (peak_equity - current_equity) / peak_equity
            self.metrics.max_drawdown = max(self.metrics.max_drawdown, drawdown)

            # Update Prometheus
            if 'trades_total' in self.prometheus_metrics:
                self.prometheus_metrics['trades_total'].inc()

        except Exception as e:
            logger.error(f"Trade recording error: {e}")

    def record_error(self, error_type: str = "general"):
        """Record an error"""
        try:
            self.metrics.total_errors += 1

            if error_type == "validation":
                self.metrics.validation_errors += 1
            elif error_type == "risk":
                self.metrics.risk_errors += 1
            elif error_type == "system":
                self.metrics.system_errors += 1

            # Update Prometheus
            if 'errors_total' in self.prometheus_metrics:
                self.prometheus_metrics['errors_total'].inc()

        except Exception as e:
            logger.error(f"Error recording error: {e}")

    def update_model_metrics(self, accuracy: float, precision: float, recall: float, f1_score: float):
        """Update model performance metrics"""
        try:
            self.metrics.model_accuracy = accuracy
            self.metrics.model_precision = precision
            self.metrics.model_recall = recall
            self.metrics.model_f1_score = f1_score

        except Exception as e:
            logger.error(f"Model metrics update error: {e}")

    def get_performance_snapshot(self) -> Dict[str, Any]:
        """Get current performance snapshot"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': time.time() - self.metrics.start_time,
            'predictions': {
                'total': self.metrics.total_predictions,
                'successful': self.metrics.successful_predictions,
                'failed': self.metrics.failed_predictions,
                'average_time': self.metrics.average_prediction_time
            },
            'features': {
                'total_processed': self.metrics.total_features_processed,
                'average_time': self.metrics.average_feature_time
            },
            'trading': {
                'total_trades': self.metrics.total_trades,
                'winning_trades': self.metrics.winning_trades,
                'win_rate': self.metrics.win_rate,
                'total_pnl': self.metrics.total_pnl,
                'max_drawdown': self.metrics.max_drawdown
            },
            'system': {
                'cpu_usage': list(self.metrics.cpu_usage)[-1] if self.metrics.cpu_usage else 0,
                'memory_usage': list(self.metrics.memory_usage)[-1] if self.metrics.memory_usage else 0,
                'disk_usage': list(self.metrics.disk_usage)[-1] if self.metrics.disk_usage else 0
            },
            'errors': {
                'total': self.metrics.total_errors,
                'validation': self.metrics.validation_errors,
                'risk': self.metrics.risk_errors,
                'system': self.metrics.system_errors
            },
            'model': {
                'accuracy': self.metrics.model_accuracy,
                'precision': self.metrics.model_precision,
                'recall': self.metrics.model_recall,
                'f1_score': self.metrics.model_f1_score
            }
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        snapshot = self.get_performance_snapshot()

        # Calculate additional metrics
        uptime_hours = snapshot['uptime_seconds'] / 3600

        report = {
            **snapshot,
            'performance_score': self._calculate_performance_score(),
            'efficiency_metrics': self._calculate_efficiency_metrics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'alert_summary': self._get_alert_summary(),
            'recommendations': self._generate_recommendations()
        }

        return report

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            score = 100.0

            # Win rate component (40% weight)
            if self.metrics.total_trades > 10:
                win_rate_score = self.metrics.win_rate * 100
                score = score * 0.4 + win_rate_score * 0.6

            # Efficiency component (30% weight)
            if self.metrics.average_prediction_time > 0:
                efficiency_penalty = min(self.metrics.average_prediction_time * 10, 30)
                score -= efficiency_penalty

            # Error rate component (20% weight)
            if self.metrics.total_predictions > 0:
                error_rate = self.metrics.failed_predictions / self.metrics.total_predictions
                error_penalty = error_rate * 20
                score -= error_penalty

            # Drawdown component (10% weight)
            drawdown_penalty = self.metrics.max_drawdown * 10
            score -= drawdown_penalty

            return max(0, min(100, score))

        except Exception as e:
            logger.error(f"Performance score calculation error: {e}")
            return 50.0

    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        try:
            return {
                'predictions_per_second': self.metrics.total_predictions / max(self.metrics.start_time - time.time(), 1),
                'features_per_second': self.metrics.total_features_processed / max(self.metrics.start_time - time.time(), 1),
                'trades_per_hour': self.metrics.total_trades / max((time.time() - self.metrics.start_time) / 3600, 1),
                'cpu_efficiency': 100 - (sum(self.metrics.cpu_usage) / len(self.metrics.cpu_usage) if self.metrics.cpu_usage else 0),
                'memory_efficiency': 100 - (sum(self.metrics.memory_usage) / len(self.metrics.memory_usage) if self.metrics.memory_usage else 0)
            }

        except Exception as e:
            logger.error(f"Efficiency metrics calculation error: {e}")
            return {}

    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics"""
        try:
            return {
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'sortino_ratio': self._calculate_sortino_ratio(),
                'calmar_ratio': self._calculate_calmar_ratio(),
                'max_drawdown_pct': self.metrics.max_drawdown * 100,
                'recovery_factor': abs(self.metrics.total_pnl) / max(self.metrics.max_drawdown, 0.001),
                'profit_factor': self._calculate_profit_factor()
            }

        except Exception as e:
            logger.error(f"Risk metrics calculation error: {e}")
            return {}

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(self.performance_history) < 2:
                return 0.0

            # Simplified calculation - in production would use daily returns
            returns = [h['trading']['total_pnl'] for h in self.performance_history]
            if len(returns) < 2:
                return 0.0

            return_std = np.std(returns)
            if return_std == 0:
                return 0.0

            average_return = np.mean(returns)
            return average_return / return_std

        except Exception as e:
            logger.error(f"Sharpe ratio calculation error: {e}")
            return 0.0

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio"""
        try:
            if len(self.performance_history) < 2:
                return 0.0

            returns = [h['trading']['total_pnl'] for h in self.performance_history]
            negative_returns = [r for r in returns if r < 0]

            if not negative_returns:
                return float('inf')

            downside_std = np.std(negative_returns)
            if downside_std == 0:
                return float('inf')

            average_return = np.mean(returns)
            return average_return / downside_std

        except Exception as e:
            logger.error(f"Sortino ratio calculation error: {e}")
            return 0.0

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        try:
            if self.metrics.max_drawdown == 0:
                return float('inf')

            # Simplified - in production would use annualized return
            total_return = self.metrics.total_pnl
            return total_return / self.metrics.max_drawdown

        except Exception as e:
            logger.error(f"Calmar ratio calculation error: {e}")
            return 0.0

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        try:
            # This would require detailed trade data in production
            if self.metrics.total_trades == 0:
                return 1.0

            # Simplified calculation
            return (self.metrics.total_pnl + 10000) / 10000

        except Exception as e:
            logger.error(f"Profit factor calculation error: {e}")
            return 1.0

    def _get_alert_summary(self) -> Dict[str, int]:
        """Get alert summary"""
        try:
            alerts = self.alert_manager.alerts[-100:]  # Last 100 alerts

            summary = {
                'total': len(alerts),
                'critical': len([a for a in alerts if a['severity'] == 'CRITICAL']),
                'warning': len([a for a in alerts if a['severity'] == 'WARNING']),
                'info': len([a for a in alerts if a['severity'] == 'INFO'])
            }

            return summary

        except Exception as e:
            logger.error(f"Alert summary error: {e}")
            return {'total': 0, 'critical': 0, 'warning': 0, 'info': 0}

    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        try:
            # Performance recommendations
            if self.metrics.average_prediction_time > 0.5:
                recommendations.append("Consider optimizing model inference for faster predictions")

            if self.metrics.win_rate < 0.5 and self.metrics.total_trades > 20:
                recommendations.append("Consider reviewing trading strategy parameters")

            if self.metrics.max_drawdown > 0.1:
                recommendations.append("High drawdown detected - consider reducing position sizes")

            if len(self.metrics.cpu_usage) > 0 and list(self.metrics.cpu_usage)[-1] > 80:
                recommendations.append("High CPU usage detected - monitor system resources")

            if self.metrics.total_errors > self.metrics.total_predictions * 0.05:
                recommendations.append("High error rate detected - check system stability")

            # Default recommendation
            if not recommendations:
                recommendations.append("System performance is within normal parameters")

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            recommendations.append("Unable to generate recommendations due to error")

        return recommendations

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.reset()
        logger.info("📊 Performance metrics reset")

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            health_status = {
                'overall_status': 'healthy',
                'components': {
                    'cpu': 'healthy' if (self.metrics.cpu_usage and list(self.metrics.cpu_usage)[-1] < 80) else 'warning',
                    'memory': 'healthy' if (self.metrics.memory_usage and list(self.metrics.memory_usage)[-1] < 80) else 'warning',
                    'disk': 'healthy' if (self.metrics.disk_usage and list(self.metrics.disk_usage)[-1] < 80) else 'warning',
                    'predictions': 'healthy' if self.metrics.average_prediction_time < 1.0 else 'warning',
                    'trading': 'healthy' if self.metrics.win_rate > 0.4 else 'warning'
                },
                'issues': []
            }

            # Check for issues
            if health_status['components']['cpu'] == 'warning':
                health_status['issues'].append('High CPU usage')

            if health_status['components']['memory'] == 'warning':
                health_status['issues'].append('High memory usage')

            if health_status['components']['predictions'] == 'warning':
                health_status['issues'].append('Slow predictions')

            if health_status['components']['trading'] == 'warning':
                health_status['issues'].append('Low win rate')

            # Determine overall status
            if any(component == 'warning' for component in health_status['components'].values()):
                health_status['overall_status'] = 'warning'

            if len(health_status['issues']) > 2:
                health_status['overall_status'] = 'critical'

            return health_status

        except Exception as e:
            logger.error(f"System health check error: {e}")
            return {
                'overall_status': 'error',
                'components': {},
                'issues': ['Health check failed']
            }