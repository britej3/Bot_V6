"""
Integration tests for LearningManager and related components
"""
import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import time

from src.learning.learning_manager import (
    LearningManager,
    LearningConfig,
    SystemMode
)
from src.learning.meta_learning.meta_learning_engine import MetaLearningConfig
from src.learning.self_adaptation.market_adaptation import AdaptationConfig
from src.learning.self_healing.self_healing_engine import HealingConfig
from src.learning.neural_networks.enhanced_neural_network import NetworkConfig
from src.trading.hft_engine.ultra_low_latency_engine import ExecutionConfig, Order, OrderSide, OrderType

class TestLearningManagerIntegration:
    """Integration tests for LearningManager"""
    
    @pytest.fixture
    def learning_manager(self):
        """Create LearningManager instance for testing"""
        config = LearningConfig(
            enable_self_learning=True,
            enable_self_adaptation=True,
            enable_self_healing=True,
            learning_interval=1.0,  # Short interval for testing
            adaptation_interval=1.0,
            healing_check_interval=0.5
        )
        return LearningManager(config)
    
    def test_initialization(self, learning_manager):
        """Test LearningManager initialization with all components"""
        assert isinstance(learning_manager, LearningManager)
        assert isinstance(learning_manager.config, LearningConfig)
        assert learning_manager.mode == SystemMode.NORMAL
        assert learning_manager.is_running == False
        
        # Check that all components are initialized
        assert learning_manager.meta_learning_engine is not None
        assert learning_manager.market_adaptation is not None
        assert learning_manager.self_healing_engine is not None
        assert learning_manager.neural_network is not None
        assert learning_manager.trading_engine is not None
    
    def test_start_stop(self, learning_manager):
        """Test starting and stopping the learning manager"""
        # Start the manager
        learning_manager.start()
        assert learning_manager.is_running == True
        
        # Let it run for a moment
        time.sleep(0.1)
        
        # Stop the manager
        learning_manager.stop()
        assert learning_manager.is_running == False
    
    def test_update_market_data_integration(self, learning_manager):
        """Test market data update integration across components"""
        # Start the manager
        learning_manager.start()
        
        # Update market data
        learning_manager.update_market_data(
            symbol="BTC/USDT",
            price=45000.0,
            volume=1000.0
        )
        
        # Check that data was propagated to components
        # Market adaptation should have data
        condition = learning_manager.market_adaptation.get_current_condition()
        assert condition is not None
        
        # Trading engine should have market data
        market_data = learning_manager.trading_engine.get_market_data("BTC/USDT")
        assert market_data is not None
        assert market_data.bid_price < market_data.ask_price
        
        # Stop the manager
        learning_manager.stop()
    
    def test_add_trading_experience_integration(self, learning_manager):
        """Test trading experience integration"""
        # Start the manager
        learning_manager.start()
        
        # Add trading experience
        experience = {
            "symbol": "BTC/USDT",
            "pnl": 10.5,
            "holding_time": 120.0,
            "slippage": 0.001,
            "timestamp": datetime.now()
        }
        
        learning_manager.add_trading_experience(experience)
        
        # Check that experience was added to buffer
        buffer_size = learning_manager.experience_buffer.get_size()
        assert buffer_size == 1
        
        # Check that performance metrics were updated
        metrics = learning_manager.performance_tracker.get_current_metrics()
        assert "total_trades" in metrics
        assert "total_pnl" in metrics
        assert metrics["total_trades"] == 1
        assert metrics["total_pnl"] == 10.5
        
        # Stop the manager
        learning_manager.stop()
    
    def test_execute_order_integration(self, learning_manager):
        """Test order execution integration"""
        # Start the manager
        learning_manager.start()
        
        # Add market data first
        learning_manager.update_market_data(
            symbol="BTC/USDT",
            price=45000.0,
            volume=1000.0
        )
        
        # Execute an order
        order_dict = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "quantity": 0.001,
            "order_type": "MARKET"
        }
        
        result = learning_manager.execute_order(order_dict)
        
        # Check result
        assert isinstance(result, dict)
        assert "order_id" in result
        assert "status" in result
        assert "latency_ms" in result
        assert result["status"] in ["filled", "rejected"]
        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0
        
        # Stop the manager
        learning_manager.stop()
    
    def test_get_system_status_integration(self, learning_manager):
        """Test system status integration"""
        # Start the manager
        learning_manager.start()
        
        # Get system status
        status = learning_manager.get_system_status()
        
        # Check status structure
        assert isinstance(status, dict)
        assert "mode" in status
        assert "is_running" in status
        assert "performance_metrics" in status
        assert "component_status" in status
        assert "experience_buffer_size" in status
        assert "timestamp" in status
        
        assert status["is_running"] == True
        assert isinstance(status["performance_metrics"], dict)
        assert isinstance(status["component_status"], dict)
        assert isinstance(status["experience_buffer_size"], int)
        
        # Stop the manager
        learning_manager.stop()
    
    def test_get_performance_report_integration(self, learning_manager):
        """Test performance report integration"""
        # Start the manager
        learning_manager.start()
        
        # Add some data
        learning_manager.update_market_data(
            symbol="BTC/USDT",
            price=45000.0,
            volume=1000.0
        )
        
        # Get performance report
        report = learning_manager.get_performance_report()
        
        # Check report structure
        assert isinstance(report, dict)
        assert "system_status" in report
        assert "recent_performance" in report
        assert "healing_report" in report
        assert "trading_report" in report
        assert "timestamp" in report
        
        # Check sub-reports
        assert isinstance(report["system_status"], dict)
        assert isinstance(report["recent_performance"], dict)
        assert isinstance(report["healing_report"], dict)
        assert isinstance(report["trading_report"], dict)
        
        # Stop the manager
        learning_manager.stop()
    
    def test_component_status_updates(self, learning_manager):
        """Test that component statuses are updated"""
        # Start the manager
        learning_manager.start()
        
        # Let it run for a moment to allow status updates
        time.sleep(1.5)
        
        # Check component statuses
        status = learning_manager.get_system_status()
        component_statuses = status["component_status"]
        
        # Check that all main components have status updates
        expected_components = [
            "meta_learning",
            "market_adaptation", 
            "self_healing",
            "neural_network",
            "trading_engine"
        ]
        
        for component in expected_components:
            assert component in component_statuses
            assert "status" in component_statuses[component]
            assert "last_update" in component_statuses[component]
        
        # Stop the manager
        learning_manager.stop()
    
    def test_experience_buffer_integration(self, learning_manager):
        """Test experience buffer integration"""
        # Start the manager
        learning_manager.start()
        
        # Add multiple experiences
        for i in range(5):
            experience = {
                "symbol": "BTC/USDT",
                "pnl": np.random.normal(0, 5),
                "holding_time": np.random.uniform(30, 300),
                "slippage": np.random.uniform(0, 0.01),
                "timestamp": datetime.now()
            }
            learning_manager.add_trading_experience(experience)
        
        # Check buffer size
        buffer_size = learning_manager.experience_buffer.get_size()
        assert buffer_size == 5
        
        # Stop the manager
        learning_manager.stop()
    
    @pytest.mark.asyncio
    async def test_async_components_integration(self, learning_manager):
        """Test integration with async components"""
        # Start the manager
        learning_manager.start()
        
        # Let it run for a moment to allow async operations
        await asyncio.sleep(1.0)
        
        # Check that async components are working
        status = learning_manager.get_system_status()
        assert status["is_running"] == True
        
        # Stop the manager
        learning_manager.stop()

class TestLearningManagerPerformance:
    """Performance tests for LearningManager"""
    
    @pytest.fixture
    def performance_manager(self):
        """Create LearningManager with performance-focused config"""
        config = LearningConfig(
            enable_self_learning=True,
            enable_self_adaptation=True,
            enable_self_healing=True,
            learning_interval=0.1,  # Very frequent for testing
            adaptation_interval=0.1,
            healing_check_interval=0.05
        )
        return LearningManager(config)
    
    def test_high_frequency_updates(self, performance_manager):
        """Test handling of high-frequency market data updates"""
        performance_manager.start()
        
        # Simulate high-frequency updates
        start_time = time.time()
        update_count = 0
        
        while time.time() - start_time < 1.0:  # 1 second test
            performance_manager.update_market_data(
                symbol="BTC/USDT",
                price=45000.0 + np.random.normal(0, 10),
                volume=np.random.exponential(1000)
            )
            update_count += 1
            time.sleep(0.01)  # 100Hz updates
        
        # Check that we handled a reasonable number of updates
        assert update_count > 50  # Should handle at least 50 updates in 1 second
        
        performance_manager.stop()
    
    def test_concurrent_order_execution(self, performance_manager):
        """Test concurrent order execution"""
        performance_manager.start()
        
        # Add market data
        performance_manager.update_market_data(
            symbol="BTC/USDT",
            price=45000.0,
            volume=1000.0
        )
        
        # Execute multiple orders concurrently
        order_dicts = []
        for i in range(10):
            order_dict = {
                "symbol": "BTC/USDT",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "quantity": 0.001,
                "order_type": "MARKET"
            }
            order_dicts.append(order_dict)
        
        results = []
        start_time = time.time()
        
        for order_dict in order_dicts:
            result = performance_manager.execute_order(order_dict)
            results.append(result)
        
        execution_time = time.time() - start_time
        
        # Check results
        assert len(results) == 10
        for result in results:
            assert isinstance(result, dict)
            assert "order_id" in result
            assert "status" in result
        
        # Check performance (should be fast)
        assert execution_time < 1.0  # All orders should execute in under 1 second
        
        performance_manager.stop()

class TestLearningManagerErrorHandling:
    """Error handling tests for LearningManager"""
    
    @pytest.fixture
    def error_manager(self):
        """Create LearningManager for error handling tests"""
        config = LearningConfig()
        return LearningManager(config)
    
    def test_invalid_order_execution(self, error_manager):
        """Test handling of invalid order execution"""
        error_manager.start()
        
        # Try to execute invalid order
        invalid_order = {
            "symbol": "INVALID/PAIR",
            "side": "INVALID",
            "quantity": -1.0,  # Negative quantity
            "order_type": "INVALID"
        }
        
        result = error_manager.execute_order(invalid_order)
        
        # Should return error result, not crash
        assert isinstance(result, dict)
        assert "timestamp" in result
        
        error_manager.stop()
    
    def test_missing_market_data(self, error_manager):
        """Test handling of missing market data"""
        error_manager.start()
        
        # Try to execute order without market data
        order_dict = {
            "symbol": "NEW/PAIR",
            "side": "BUY",
            "quantity": 0.001,
            "order_type": "MARKET"
        }
        
        result = error_manager.execute_order(order_dict)
        
        # Should still execute (with dummy data)
        assert isinstance(result, dict)
        assert "order_id" in result
        assert "status" in result
        
        error_manager.stop()

if __name__ == "__main__":
    pytest.main([__file__])