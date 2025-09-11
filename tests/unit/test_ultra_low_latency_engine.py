"""
Unit tests for UltraLowLatencyTradingEngine
"""
import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import time
import asyncio

from src.trading.hft_engine.ultra_low_latency_engine import (
    UltraLowLatencyTradingEngine,
    ExecutionConfig,
    LatencyOptimizer,
    IntelligentOrderRouter,
    AdaptiveTimeoutManager,
    ExecutionEngine,
    Order,
    MarketData,
    OrderType,
    OrderSide,
    OrderStatus
)

class TestExecutionConfig:
    """Test ExecutionConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = ExecutionConfig()
        
        assert config.target_latency_ms == 50.0
        assert config.max_acceptable_latency_ms == 100.0
        assert config.routing_algorithm == "ml_optimized"
        assert isinstance(config.exchange_preferences, list)
        assert config.order_timeout_ms == 5000.0
        assert config.retry_attempts == 3
        assert config.executor_threads == 4
        assert config.max_concurrent_orders == 100
        assert config.enable_jit == True
    
    def test_custom_values(self):
        """Test custom configuration values"""
        custom_config = ExecutionConfig(
            target_latency_ms=25.0,
            routing_algorithm="latency",
            exchange_preferences=["kraken", "bitfinex"],
            executor_threads=8
        )
        
        assert custom_config.target_latency_ms == 25.0
        assert custom_config.routing_algorithm == "latency"
        assert custom_config.exchange_preferences == ["kraken", "bitfinex"]
        assert custom_config.executor_threads == 8

class TestLatencyOptimizer:
    """Test LatencyOptimizer functionality"""
    
    @pytest.fixture
    def optimizer(self):
        """Create LatencyOptimizer instance for testing"""
        config = ExecutionConfig()
        return LatencyOptimizer(config)
    
    def test_initialization(self, optimizer):
        """Test LatencyOptimizer initialization"""
        assert isinstance(optimizer, LatencyOptimizer)
        assert isinstance(optimizer.config, ExecutionConfig)
        assert isinstance(optimizer.latency_history, list)
        assert isinstance(optimizer.exchange_latencies, dict)
    
    def test_calculate_spread_jit(self, optimizer):
        """Test JIT-compiled spread calculation"""
        spread = optimizer.calculate_spread(45000.0, 45001.0)
        expected_spread = (45001.0 - 45000.0) / 45000.0
        
        assert abs(spread - expected_spread) < 1e-10
    
    def test_calculate_market_impact_jit(self, optimizer):
        """Test JIT-compiled market impact calculation"""
        impact = optimizer.calculate_market_impact(10.0, 100.0)
        expected_impact = 10.0 / 100.0
        
        assert abs(impact - expected_impact) < 1e-10
    
    def test_optimize_order_parameters(self, optimizer):
        """Test order parameter optimization"""
        order = Order(
            order_id="test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.001
        )
        
        market_data = MarketData(
            symbol="BTC/USDT",
            bid_price=45000.0,
            ask_price=45001.0,
            bid_size=5.0,
            ask_size=5.0,
            timestamp=datetime.now()
        )
        
        optimized_params = optimizer.optimize_order_parameters(order, market_data)
        
        assert isinstance(optimized_params, dict)
        assert "price" in optimized_params
        assert "exchange" in optimized_params
        assert "spread" in optimized_params
        assert "impact" in optimized_params
        # For market buy orders, optimized price should be higher than ask price
        assert optimized_params["price"] >= market_data.ask_price
    
    def test_update_latency_metrics(self, optimizer):
        """Test latency metrics update"""
        optimizer.update_latency_metrics("binance", 25.5)
        
        assert "binance" in optimizer.exchange_latencies
        assert optimizer.exchange_latencies["binance"] == 25.5
        assert len(optimizer.latency_history) == 1
        assert optimizer.latency_history[0] == 25.5

class TestIntelligentOrderRouter:
    """Test IntelligentOrderRouter functionality"""
    
    @pytest.fixture
    def router(self):
        """Create IntelligentOrderRouter instance for testing"""
        config = ExecutionConfig()
        return IntelligentOrderRouter(config)
    
    def test_initialization(self, router):
        """Test IntelligentOrderRouter initialization"""
        assert isinstance(router, IntelligentOrderRouter)
        assert isinstance(router.config, ExecutionConfig)
        assert isinstance(router.router_model, torch.nn.Module)
        assert isinstance(router.routing_history, list)
    
    def test_predict_optimal_route(self, router):
        """Test optimal route prediction"""
        order = Order(
            order_id="test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=45000.0
        )
        
        market_data = MarketData(
            symbol="BTC/USDT",
            bid_price=45000.0,
            ask_price=45001.0,
            bid_size=5.0,
            ask_size=5.0,
            timestamp=datetime.now()
        )
        
        latency_metrics = {"binance": 25.0, "okx": 30.0}
        
        optimal_exchange = router.predict_optimal_route(order, market_data, latency_metrics)
        
        assert isinstance(optimal_exchange, str)
        # Should be one of the preferred exchanges
        assert optimal_exchange in router.config.exchange_preferences or optimal_exchange == "default"
    
    def test_route_order(self, router):
        """Test order routing"""
        order = Order(
            order_id="test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=45000.0
        )
        
        market_data = MarketData(
            symbol="BTC/USDT",
            bid_price=45000.0,
            ask_price=45001.0,
            bid_size=5.0,
            ask_size=5.0,
            timestamp=datetime.now()
        )
        
        # Create a mock latency optimizer
        latency_optimizer = Mock()
        latency_optimizer._select_optimal_exchange.return_value = "binance"
        # Mock the exchange_latencies attribute to be a dictionary
        latency_optimizer.exchange_latencies = {"binance": 20.0, "okx": 30.0} # Provide some dummy latency data

        routed_exchange = router.route_order(order, market_data, latency_optimizer)
        
        assert isinstance(routed_exchange, str)
        assert len(router.routing_history) == 1

class TestAdaptiveTimeoutManager:
    """Test AdaptiveTimeoutManager functionality"""
    
    @pytest.fixture
    def timeout_manager(self):
        """Create AdaptiveTimeoutManager instance for testing"""
        config = ExecutionConfig()
        return AdaptiveTimeoutManager(config)
    
    def test_initialization(self, timeout_manager):
        """Test AdaptiveTimeoutManager initialization"""
        assert isinstance(timeout_manager, AdaptiveTimeoutManager)
        assert isinstance(timeout_manager.config, ExecutionConfig)
        assert isinstance(timeout_manager.timeout_history, list)
        assert isinstance(timeout_manager.adaptive_timeouts, dict)
    
    def test_get_adaptive_timeout(self, timeout_manager):
        """Test adaptive timeout calculation"""
        timeout = timeout_manager.get_adaptive_timeout("binance", 0.01)  # 1% volatility
        
        assert isinstance(timeout, float)
        assert timeout >= 100.0  # Minimum timeout
        assert timeout <= 10000.0  # Maximum timeout
    
    def test_update_timeout_performance(self, timeout_manager):
        """Test timeout performance update"""
        timeout_manager.update_timeout_performance("binance", 1000.0, True)
        
        assert len(timeout_manager.timeout_history) == 1
        assert timeout_manager.timeout_history[0] == ("binance", 1000.0, True)

class TestExecutionEngine:
    """Test ExecutionEngine functionality"""
    
    @pytest.fixture
    def execution_engine(self):
        """Create ExecutionEngine instance for testing"""
        config = ExecutionConfig()
        return ExecutionEngine(config)
    
    def test_initialization(self, execution_engine):
        """Test ExecutionEngine initialization"""
        assert isinstance(execution_engine, ExecutionEngine)
        assert isinstance(execution_engine.config, ExecutionConfig)
        assert isinstance(execution_engine.latency_optimizer, LatencyOptimizer)
        assert isinstance(execution_engine.order_router, IntelligentOrderRouter)
        assert isinstance(execution_engine.timeout_manager, AdaptiveTimeoutManager)
        assert isinstance(execution_engine.active_orders, dict)
        assert execution_engine.successful_executions == 0
        assert execution_engine.failed_executions == 0
    
    def test_submit_order(self, execution_engine):
        """Test order submission"""
        order = Order(
            order_id="test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=45000.0
        )
        
        market_data = MarketData(
            symbol="BTC/USDT",
            bid_price=45000.0,
            ask_price=45001.0,
            bid_size=5.0,
            ask_size=5.0,
            timestamp=datetime.now()
        )
        
        executed_order = execution_engine.submit_order(order, market_data)
        
        assert isinstance(executed_order, Order)
        assert executed_order.order_id == order.order_id
        assert executed_order.status in [OrderStatus.FILLED, OrderStatus.REJECTED]
        assert executed_order.latency >= 0
        assert order.order_id in execution_engine.active_orders
    
    def test_get_performance_metrics(self, execution_engine):
        """Test performance metrics retrieval"""
        # Add some dummy execution times
        execution_engine.execution_times = [25.0, 30.0, 35.0]
        execution_engine.successful_executions = 2
        execution_engine.failed_executions = 1
        
        metrics = execution_engine.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "average_latency_ms" in metrics
        assert "min_latency_ms" in metrics
        assert "max_latency_ms" in metrics
        assert "success_rate" in metrics
        assert "total_executions" in metrics
        assert metrics["total_executions"] == 3

class TestUltraLowLatencyTradingEngine:
    """Test UltraLowLatencyTradingEngine main class"""
    
    @pytest.fixture
    def trading_engine(self):
        """Create UltraLowLatencyTradingEngine instance for testing"""
        config = ExecutionConfig()
        return UltraLowLatencyTradingEngine(config)
    
    def test_initialization(self, trading_engine):
        """Test UltraLowLatencyTradingEngine initialization"""
        assert isinstance(trading_engine, UltraLowLatencyTradingEngine)
        assert isinstance(trading_engine.config, ExecutionConfig)
        assert isinstance(trading_engine.execution_engine, ExecutionEngine)
        assert isinstance(trading_engine.market_data_cache, dict)
        assert isinstance(trading_engine.performance_checks, list)
    
    def test_update_market_data(self, trading_engine):
        """Test market data update"""
        market_data = MarketData(
            symbol="BTC/USDT",
            bid_price=45000.0,
            ask_price=45001.0,
            bid_size=5.0,
            ask_size=5.0,
            timestamp=datetime.now()
        )
        
        trading_engine.update_market_data(market_data)
        
        assert "BTC/USDT" in trading_engine.market_data_cache
        assert trading_engine.market_data_cache["BTC/USDT"] == market_data
    
    def test_execute_order(self, trading_engine):
        """Test order execution"""
        # Add market data first
        market_data = MarketData(
            symbol="BTC/USDT",
            bid_price=45000.0,
            ask_price=45001.0,
            bid_size=5.0,
            ask_size=5.0,
            timestamp=datetime.now()
        )
        trading_engine.update_market_data(market_data)
        
        order = Order(
            order_id="test_order",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=45000.0
        )
        
        executed_order = trading_engine.execute_order(order)
        
        assert isinstance(executed_order, Order)
        assert executed_order.order_id == order.order_id
        assert executed_order.status in [OrderStatus.FILLED, OrderStatus.REJECTED]
    
    def test_get_performance_report(self, trading_engine):
        """Test performance report generation"""
        report = trading_engine.get_performance_report()
        
        assert isinstance(report, dict)
        assert "performance_metrics" in report
        assert "latency_target_compliance" in report
        assert "uptime" in report
        assert "timestamp" in report
        assert isinstance(report["latency_target_compliance"], bool)
    
    def test_get_market_data(self, trading_engine):
        """Test market data retrieval"""
        # Test with no data
        market_data = trading_engine.get_market_data("BTC/USDT")
        assert market_data is None
        
        # Add data and test
        test_data = MarketData(
            symbol="BTC/USDT",
            bid_price=45000.0,
            ask_price=45001.0,
            bid_size=5.0,
            ask_size=5.0,
            timestamp=datetime.now()
        )
        trading_engine.update_market_data(test_data)
        
        market_data = trading_engine.get_market_data("BTC/USDT")
        assert market_data == test_data
    
    def test_get_active_orders(self, trading_engine):
        """Test active orders retrieval"""
        active_orders = trading_engine.get_active_orders()
        
        assert isinstance(active_orders, dict)
        # Should be a copy, not the original dict
        assert active_orders is not trading_engine.execution_engine.active_orders

if __name__ == "__main__":
    pytest.main([__file__])