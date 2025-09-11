"""
Enhanced Health Check Endpoints for CryptoScalp AI
==================================================

This module implements comprehensive health check endpoints for all system components
with detailed metrics and status reporting.

Task: INFRA_DEPLOY_002 - Production Infrastructure & Deployment Readiness
Author: Infrastructure Team
Date: 2025-08-24
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import logging
import time
import asyncio
from datetime import datetime, timezone
import psutil
import redis.asyncio as redis

from src.database.enhanced_pool_manager import EnhancedDatabaseManager, DatabaseConfig, DatabaseType
from src.database.redis_manager import RedisManager, RedisConfig
from src.monitoring.comprehensive_monitoring import ComprehensiveMonitoringSystem
from src.config import get_settings

logger = logging.getLogger(__name__)

# Create router
health_router = APIRouter(prefix="/health", tags=["health"])

# Global references to system components
db_manager: Optional[EnhancedDatabaseManager] = None
redis_manager: Optional[RedisManager] = None
monitoring_system: Optional[ComprehensiveMonitoringSystem] = None


def set_health_components(
    database_manager: EnhancedDatabaseManager,
    redis_manager_instance: RedisManager,
    monitoring_system_instance: ComprehensiveMonitoringSystem
):
    """Set global references to system components for health checks"""
    global db_manager, redis_manager, monitoring_system
    db_manager = database_manager
    redis_manager = redis_manager_instance
    monitoring_system = monitoring_system_instance


@health_router.get("/liveness")
async def liveness_check():
    """
    Liveness probe - indicates if the application is running
    
    Returns:
        Health status
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "cryptoscalp-ai"
    }


@health_router.get("/readiness")
async def readiness_check():
    """
    Readiness probe - indicates if the application is ready to serve requests
    
    Returns:
        Readiness status with component checks
    """
    settings = get_settings()
    
    component_statuses = {}
    overall_ready = True
    
    # Check database connectivity
    try:
        if db_manager and db_manager.is_connected:
            db_health = await db_manager.health_check()
            component_statuses["database"] = {
                "status": "healthy" if db_health else "unhealthy",
                "details": await db_manager.get_metrics() if db_health else "Connection failed"
            }
            if not db_health:
                overall_ready = False
        else:
            component_statuses["database"] = {
                "status": "unhealthy",
                "details": "Database manager not initialized"
            }
            overall_ready = False
    except Exception as e:
        component_statuses["database"] = {
            "status": "unhealthy",
            "details": str(e)
        }
        overall_ready = False
    
    # Check Redis connectivity
    try:
        if redis_manager and redis_manager.is_connected:
            redis_health = await redis_manager.health_check()
            component_statuses["redis"] = {
                "status": "healthy" if redis_health else "unhealthy",
                "details": await redis_manager.get_metrics() if redis_health else "Connection failed"
            }
            if not redis_health:
                overall_ready = False
        else:
            component_statuses["redis"] = {
                "status": "unhealthy",
                "details": "Redis manager not initialized"
            }
            overall_ready = False
    except Exception as e:
        component_statuses["redis"] = {
            "status": "unhealthy",
            "details": str(e)
        }
        overall_ready = False
    
    # Check system resources
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        component_statuses["system"] = {
            "status": "healthy" if cpu_percent < 90 and memory.percent < 90 else "degraded",
            "details": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "available_memory_gb": memory.available / (1024**3),
                "free_disk_gb": disk.free / (1024**3)
            }
        }
        
        if cpu_percent > 90 or memory.percent > 90:
            overall_ready = False
            
    except Exception as e:
        component_statuses["system"] = {
            "status": "unhealthy",
            "details": str(e)
        }
        overall_ready = False
    
    return {
        "status": "ready" if overall_ready else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "cryptoscalp-ai",
        "components": component_statuses
    }


@health_router.get("/metrics")
async def get_metrics():
    """
    Get detailed system metrics in Prometheus format
    
    Returns:
        Prometheus-formatted metrics
    """
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
        
    try:
        prometheus_metrics = await monitoring_system.get_prometheus_metrics()
        return prometheus_metrics
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@health_router.get("/status")
async def get_detailed_status():
    """
    Get detailed system status and health information
    
    Returns:
        Detailed system status
    """
    status_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "cryptoscalp-ai",
        "version": "1.0.0"
    }
    
    # Add system health if monitoring system is available
    if monitoring_system:
        try:
            status_info["health"] = monitoring_system.get_health_status()
        except Exception as e:
            status_info["health"] = {"error": str(e)}
    
    # Add database metrics if available
    if db_manager:
        try:
            status_info["database"] = {
                "connected": db_manager.is_connected,
                "metrics": await db_manager.get_metrics() if db_manager.is_connected else None
            }
        except Exception as e:
            status_info["database"] = {"error": str(e)}
    
    # Add Redis metrics if available
    if redis_manager:
        try:
            status_info["redis"] = {
                "connected": redis_manager.is_connected,
                "metrics": await redis_manager.get_metrics() if redis_manager.is_connected else None
            }
        except Exception as e:
            status_info["redis"] = {"error": str(e)}
    
    # Add system resource metrics
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        status_info["system_resources"] = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "disk_usage_percent": (disk.used / disk.total) * 100,
            "network_bytes_sent": net_io.bytes_sent,
            "network_bytes_recv": net_io.bytes_recv,
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "process_count": len(psutil.pids())
        }
    except Exception as e:
        status_info["system_resources"] = {"error": str(e)}
        
    return status_info


@health_router.get("/components")
async def get_component_status():
    """
    Get status of all system components
    
    Returns:
        Component status information
    """
    component_status = {}
    
    # Database status
    if db_manager:
        try:
            component_status["database"] = {
                "type": db_manager.config.db_type.value,
                "host": db_manager.config.host,
                "port": db_manager.config.port,
                "connected": db_manager.is_connected,
                "uptime": time.time() - db_manager.connected_since if db_manager.connected_since else 0,
                "metrics": await db_manager.get_metrics() if db_manager.is_connected else None
            }
        except Exception as e:
            component_status["database"] = {"error": str(e)}
    
    # Redis status
    if redis_manager:
        try:
            component_status["redis"] = {
                "host": redis_manager.config.host,
                "port": redis_manager.config.port,
                "connected": redis_manager.is_connected,
                "metrics": await redis_manager.get_metrics() if redis_manager.is_connected else None
            }
        except Exception as e:
            component_status["redis"] = {"error": str(e)}
    
    # Monitoring system status
    if monitoring_system:
        try:
            component_status["monitoring"] = {
                "running": monitoring_system.is_running,
                "metrics_count": len(monitoring_system.metrics),
                "alerts_count": len(monitoring_system.alert_events),
                "notification_channels": list(monitoring_system.notification_channels.keys())
            }
        except Exception as e:
            component_status["monitoring"] = {"error": str(e)}
            
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": component_status
    }


@health_router.get("/alerts")
async def get_alerts(limit: int = 50):
    """
    Get recent alert events
    
    Args:
        limit: Maximum number of alerts to return
        
    Returns:
        Recent alert events
    """
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
        
    try:
        alert_events = monitoring_system.get_alert_events(limit=limit)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alerts": [
                {
                    "name": event.alert_name,
                    "severity": event.severity.value,
                    "message": event.message,
                    "timestamp": datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat(),
                    "resolved": event.resolved,
                    "resolved_timestamp": datetime.fromtimestamp(event.resolved_timestamp, tz=timezone.utc).isoformat() 
                                    if event.resolved_timestamp else None,
                    "triggered_value": event.triggered_value
                }
                for event in alert_events
            ]
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@health_router.post("/alert/{alert_name}/trigger")
async def trigger_alert(alert_name: str, message: str, triggered_value: float = None):
    """
    Manually trigger an alert (for testing)
    
    Args:
        alert_name: Name of alert to trigger
        message: Alert message
        triggered_value: Value that triggered alert
        
    Returns:
        Confirmation
    """
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
        
    try:
        await monitoring_system.trigger_alert(alert_name, message, triggered_value)
        return {"message": f"Alert {alert_name} triggered"}
    except Exception as e:
        logger.error(f"Error triggering alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger alert")


@health_router.post("/alert/{alert_name}/resolve")
async def resolve_alert(alert_name: str, message: str = ""):
    """
    Manually resolve an alert (for testing)
    
    Args:
        alert_name: Name of alert to resolve
        message: Resolution message
        
    Returns:
        Confirmation
    """
    if not monitoring_system:
        raise HTTPException(status_code=503, detail="Monitoring system not available")
        
    try:
        await monitoring_system.resolve_alert(alert_name, message)
        return {"message": f"Alert {alert_name} resolved"}
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to resolve alert")


@health_router.get("/benchmark")
async def run_benchmark():
    """
    Run system benchmark to test performance
    
    Returns:
        Benchmark results
    """
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {}
    }
    
    # Database benchmark
    if db_manager:
        try:
            db_benchmark = await db_manager.benchmark(iterations=100)
            results["benchmarks"]["database"] = db_benchmark
        except Exception as e:
            results["benchmarks"]["database"] = {"error": str(e)}
    
    # Redis benchmark
    if redis_manager:
        try:
            redis_benchmark = await redis_manager.benchmark(iterations=1000)
            results["benchmarks"]["redis"] = redis_benchmark
        except Exception as e:
            results["benchmarks"]["redis"] = {"error": str(e)}
            
    return results