"""
Unit tests for EnhancedSelfHealingEngine
"""
import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.learning.self_healing.self_healing_engine import (
    EnhancedSelfHealingEngine,
    HealingConfig,
    HealthMonitor,
    FailurePredictor,
    HealingStrategySelector,
    HealingExecutor,
    SystemHealth,
    FailureEvent,
    HealingEvent,
    FailureType,
    HealingAction
)

class TestHealingConfig:
    """Test HealingConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = HealingConfig()
        
        assert config.health_check_interval == 5.0
        assert config.failure_detection_window == 60
        assert config.prediction_window == 300
        assert config.prediction_threshold == 0.7
        assert config.max_retry_attempts == 3
        assert config.retry_delay == 1.0
        assert config.healing_timeout == 30.0
        assert config.alert_threshold == 0.3
        assert config.critical_alert_threshold == 0.1

class TestHealthMonitor:
    """Test HealthMonitor functionality"""
    
    @pytest.fixture
    def monitor(self):
        """Create HealthMonitor instance for testing"""
        config = HealingConfig()
        return HealthMonitor(config)
    
    def test_initialization(self, monitor):
        """Test HealthMonitor initialization"""
        assert isinstance(monitor, HealthMonitor)
        assert isinstance(monitor.config, HealingConfig)
        assert isinstance(monitor.health_history, type(monitor.health_history))
        assert isinstance(monitor.failure_history, type(monitor.failure_history))
        assert isinstance(monitor.component_status, dict)
    
    def test_update_component_status(self, monitor):
        """Test component status updates"""
        status = {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "error_rate": 0.05
        }
        
        monitor.update_component_status("test_component", status)
        
        assert "test_component" in monitor.component_status
        assert monitor.component_status["test_component"]["status"] == status
        assert isinstance(monitor.component_status["test_component"]["timestamp"], datetime)
    
    def test_get_system_health(self, monitor):
        """Test system health calculation"""
        # Add some component statuses
        monitor.update_component_status("component1", {
            "cpu_usage": 40.0,
            "memory_usage": 50.0,
            "error_rate": 0.02
        })
        
        monitor.update_component_status("component2", {
            "cpu_usage": 50.0,
            "memory_usage": 70.0,
            "error_rate": 0.08
        })
        
        health = monitor.get_system_health()
        
        assert isinstance(health, SystemHealth)
        assert isinstance(health.timestamp, datetime)
        assert 0.0 <= health.overall_health <= 1.0
        assert 0.0 <= health.cpu_usage <= 100.0
        assert 0.0 <= health.memory_usage <= 100.0
        assert health.error_rate >= 0.0
    
    def test_record_failure(self, monitor):
        """Test failure recording"""
        failure = FailureEvent(
            timestamp=datetime.now(),
            failure_type=FailureType.NETWORK_ERROR,
            component="test_component",
            error_message="Connection failed",
            severity=5
        )
        
        monitor.record_failure(failure)
        
        assert len(monitor.failure_history) == 1
        assert monitor.failure_history[0] == failure

class TestFailurePredictor:
    """Test FailurePredictor functionality"""
    
    @pytest.fixture
    def predictor(self):
        """Create FailurePredictor instance for testing"""
        config = HealingConfig()
        return FailurePredictor(config)
    
    def test_initialization(self, predictor):
        """Test FailurePredictor initialization"""
        assert isinstance(predictor, FailurePredictor)
        assert isinstance(predictor.config, HealingConfig)
        assert isinstance(predictor.prediction_model, torch.nn.Module)
        assert isinstance(predictor.prediction_history, list)
    
    def test_predict_failures(self, predictor):
        """Test failure prediction"""
        health = SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=80.0,
            memory_usage=90.0,
            disk_usage=70.0,
            network_latency=200.0,
            api_response_time=500.0,
            error_rate=0.5,
            data_quality_score=0.6,
            model_accuracy=0.7,
            overall_health=0.4
        )
        
        predictions = predictor.predict_failures(health)
        
        assert isinstance(predictions, list)
        # Check that predictions are tuples of (FailureType, probability)
        for pred in predictions:
            assert isinstance(pred, tuple)
            assert len(pred) == 2
            assert isinstance(pred[0], FailureType)
            assert isinstance(pred[1], float)
            assert 0.0 <= pred[1] <= 1.0

class TestHealingStrategySelector:
    """Test HealingStrategySelector functionality"""
    
    @pytest.fixture
    def selector(self):
        """Create HealingStrategySelector instance for testing"""
        config = HealingConfig()
        return HealingStrategySelector(config)
    
    def test_initialization(self, selector):
        """Test HealingStrategySelector initialization"""
        assert isinstance(selector, HealingStrategySelector)
        assert isinstance(selector.config, HealingConfig)
        assert isinstance(selector.healing_history, list)
        assert isinstance(selector.strategy_success_rates, dict)
    
    def test_select_healing_action_network_error(self, selector):
        """Test healing action selection for network errors"""
        failure = FailureEvent(
            timestamp=datetime.now(),
            failure_type=FailureType.NETWORK_ERROR,
            component="test_component",
            error_message="Connection timeout",
            severity=6
        )
        
        health = SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=50.0,
            disk_usage=50.0,
            network_latency=100.0,
            api_response_time=200.0,
            error_rate=0.1,
            data_quality_score=0.8,
            model_accuracy=0.9,
            overall_health=0.7
        )
        
        actions = selector.select_healing_action(failure, health)
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert HealingAction.RETRY_OPERATION in actions
        assert HealingAction.INCREASE_TIMEOUT in actions
        assert HealingAction.SWITCH_EXCHANGE in actions
    
    def test_select_healing_action_critical_severity(self, selector):
        """Test healing action selection for critical severity"""
        failure = FailureEvent(
            timestamp=datetime.now(),
            failure_type=FailureType.MODEL_ERROR,
            component="test_component",
            error_message="Model failed",
            severity=9  # Critical severity
        )
        
        health = SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=50.0,
            memory_usage=50.0,
            disk_usage=50.0,
            network_latency=100.0,
            api_response_time=200.0,
            error_rate=0.1,
            data_quality_score=0.8,
            model_accuracy=0.9,
            overall_health=0.7
        )
        
        actions = selector.select_healing_action(failure, health)
        
        assert HealingAction.ALERT_OPERATOR in actions
        assert HealingAction.SHUTDOWN_GRACEFULLY in actions

class TestHealingExecutor:
    """Test HealingExecutor functionality"""
    
    @pytest.fixture
    def executor(self):
        """Create HealingExecutor instance for testing"""
        config = HealingConfig()
        return HealingExecutor(config)
    
    def test_initialization(self, executor):
        """Test HealingExecutor initialization"""
        assert isinstance(executor, HealingExecutor)
        assert isinstance(executor.config, HealingConfig)
        assert isinstance(executor.healing_actions, dict)
        # Check that default actions are registered
        assert HealingAction.RETRY_OPERATION in executor.healing_actions
        assert HealingAction.SWITCH_EXCHANGE in executor.healing_actions
    
    @pytest.mark.asyncio
    async def test_execute_healing_action_retry(self, executor):
        """Test executing retry operation healing action"""
        context = {
            "max_attempts": 2,
            "delay": 0.1  # Short delay for testing
        }
        
        healing_event = await executor.execute_healing_action(
            HealingAction.RETRY_OPERATION,
            "test_component",
            context
        )
        
        assert isinstance(healing_event, HealingEvent)
        assert healing_event.action == HealingAction.RETRY_OPERATION
        assert healing_event.component == "test_component"
        assert healing_event.success == True
        assert isinstance(healing_event.duration, float)
        assert healing_event.duration >= 0

class TestEnhancedSelfHealingEngine:
    """Test EnhancedSelfHealingEngine main class"""
    
    @pytest.fixture
    def healing_engine(self):
        """Create EnhancedSelfHealingEngine instance for testing"""
        return EnhancedSelfHealingEngine()
    
    def test_initialization(self, healing_engine):
        """Test EnhancedSelfHealingEngine initialization"""
        assert isinstance(healing_engine, EnhancedSelfHealingEngine)
        assert isinstance(healing_engine.config, HealingConfig)
        assert isinstance(healing_engine.monitor, HealthMonitor)
        assert isinstance(healing_engine.predictor, FailurePredictor)
        assert isinstance(healing_engine.strategy_selector, HealingStrategySelector)
        assert isinstance(healing_engine.executor, HealingExecutor)
        assert healing_engine.is_active == True
    
    def test_update_component_status(self, healing_engine):
        """Test component status update"""
        status = {
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "error_rate": 0.05
        }
        
        healing_engine.update_component_status("test_component", status)
        
        # Check that monitor was updated
        assert "test_component" in healing_engine.monitor.component_status
    
    @pytest.mark.asyncio
    async def test_check_system_health(self, healing_engine):
        """Test system health check"""
        # Add some component status
        healing_engine.update_component_status("component1", {
            "cpu_usage": 40.0,
            "memory_usage": 50.0,
            "error_rate": 0.02
        })
        
        health = await healing_engine.check_system_health()
        
        assert isinstance(health, SystemHealth)
        assert isinstance(health.timestamp, datetime)
        assert 0.0 <= health.overall_health <= 1.0
    
    @pytest.mark.asyncio
    async def test_handle_failure(self, healing_engine):
        """Test failure handling"""
        failure = FailureEvent(
            timestamp=datetime.now(),
            failure_type=FailureType.NETWORK_ERROR,
            component="test_component",
            error_message="Connection timeout",
            severity=6
        )
        
        # Mock the executor to avoid actual execution delays
        healing_engine.executor.execute_healing_action = AsyncMock(
            return_value=HealingEvent(
                timestamp=datetime.now(),
                action=HealingAction.RETRY_OPERATION,
                component="test_component",
                success=True,
                duration=0.1,
                outcome="Retried successfully"
            )
        )
        
        healing_events = await healing_engine.handle_failure(failure)
        
        assert isinstance(healing_events, list)
        assert len(healing_events) > 0
        # Check that failure was recorded
        assert len(healing_engine.get_failure_history()) == 1
    
    def test_get_history_methods(self, healing_engine):
        """Test history retrieval methods"""
        # Add some dummy data
        healing_engine.healing_events.append(HealingEvent(
            timestamp=datetime.now(),
            action=HealingAction.RETRY_OPERATION,
            component="test",
            success=True,
            duration=0.1,
            outcome="test"
        ))
        
        health_history = healing_engine.get_health_history()
        failure_history = healing_engine.get_failure_history()
        healing_history = healing_engine.get_healing_history()
        
        assert isinstance(health_history, list)
        assert isinstance(failure_history, list)
        assert isinstance(healing_history, list)

if __name__ == "__main__":
    pytest.main([__file__])