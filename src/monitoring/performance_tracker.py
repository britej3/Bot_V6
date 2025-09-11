"""
Performance Tracker Stub
=======================

Stub implementation of the Performance Tracker.
Provides basic functionality for performance monitoring.
"""

import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Stub Performance Tracker

    This is a placeholder implementation that provides the basic interface
    expected by the Nautilus integration system.
    """

    def __init__(self):
        self.is_active = True
        self.metrics = {}
        self.tracker_name = "Performance Tracker (Stub)"

        logger.info(f"📊 {self.tracker_name} initialized")

    def record_metric(self, metric_name: str, value: Any, timestamp: datetime = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

        self.metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })

        # Keep only last 1000 records per metric
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]

        logger.debug(f"📊 Metric recorded: {metric_name} = {value}")

    def get_metrics(self, metric_name: str = None, limit: int = 100) -> Dict[str, Any]:
        """Get recorded metrics"""
        if metric_name:
            if metric_name in self.metrics:
                return {
                    metric_name: self.metrics[metric_name][-limit:]
                }
            else:
                return {metric_name: []}
        else:
            # Return summary of all metrics
            summary = {}
            for name, records in self.metrics.items():
                if records:
                    values = [r['value'] for r in records[-limit:]]
                    summary[name] = {
                        'count': len(values),
                        'latest': values[-1] if values else None,
                        'average': sum(values) / len(values) if values else 0,
                        'min': min(values) if values else 0,
                        'max': max(values) if values else 0
                    }
            return summary

    def reset(self, metric_name: str = None):
        """Reset metrics"""
        if metric_name:
            if metric_name in self.metrics:
                self.metrics[metric_name] = []
                logger.info(f"🔄 Reset metric: {metric_name}")
        else:
            self.metrics = {}
            logger.info("🔄 Reset all metrics")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_metrics = len(self.metrics)
        total_records = sum(len(records) for records in self.metrics.values())

        return {
            'tracker_name': self.tracker_name,
            'is_active': self.is_active,
            'total_metrics': total_metrics,
            'total_records': total_records,
            'metrics_summary': self.get_metrics(),
            'timestamp': datetime.utcnow()
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'tracker_name': self.tracker_name,
            'status': 'healthy' if self.is_active else 'inactive',
            'is_active': self.is_active,
            'performance': self.get_performance_summary(),
            'timestamp': datetime.utcnow()
        }


# Global instance
performance_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get performance tracker instance"""
    return performance_tracker


def initialize_performance_tracker():
    """Initialize performance tracker"""
    return performance_tracker


def shutdown_performance_tracker():
    """Shutdown performance tracker"""
    performance_tracker.is_active = False
    logger.info("⏹️ Performance tracker shutdown")


# Export key classes and functions
__all__ = [
    'PerformanceTracker',
    'get_performance_tracker',
    'initialize_performance_tracker',
    'shutdown_performance_tracker'
]