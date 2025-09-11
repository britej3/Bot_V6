"""
Unit Tests for Production HFT Engine
==================================
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.trading.hft_engine_production import (
    HighFrequencyTradingEngine,
    RiskManager,
    CircuitBreaker,
    TradingOrder,
    OrderStatus,
    OrderType
)


class TestTradingOrder:
    """Test cases for TradingOrder class"""

    def test_order_creation(self):
        """Test basic order creation"""
        order = TradingOrder(
            order_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            order_type=OrderType.LIMIT,
            timestamp=time.time()
        )

        assert order.order_id == "test_123"
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.quantity == 0.1
        assert order.price == 50000.0
        assert order.order_type == OrderType.LIMIT
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0.0
        assert order.fees == 0.0

    def test_remaining_quantity(self):
        """Test remaining quantity calculation"""
        order = TradingOrder(
            order_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time()
        )

        assert order.remaining_quantity == 1.0

        order.filled_quantity = 0.5
        assert order.remaining_quantity == 0.5

        order.filled_quantity = 1.0
        assert order.remaining_quantity == 0.0

    def test_is_complete(self):
        """Test order completion status"""
        order = TradingOrder(
            order_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time()
        )

        # Initially not complete
        assert not order.is_complete

        # Test different completion states
        for status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            order.status = status
            assert order.is_complete

        # Test non-complete states
        for status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILL]:
            order.status = status
            assert not order.is_complete


class TestRiskManager:
    """Test cases for RiskManager class"""

    def test_risk_manager_creation(self):
        """Test risk manager initialization"""
        rm = RiskManager(max_position_size=0.5, max_daily_loss=5000.0)
        assert rm.max_position_size == 0.5
        assert rm.max_daily_loss == 5000.0
        assert rm.daily_loss == 0.0
        assert len(rm.open_positions) == 0
        assert len(rm.open_orders) == 0

    def test_validate_order_approved(self):
        """Test order validation - approved case"""
        rm = RiskManager(max_position_size=1.0, max_daily_loss=10000.0)

        order = TradingOrder(
            order_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time()
        )

        is_valid, reason = rm.validate_order(order)
        assert is_valid
        assert reason == "Order approved"

    def test_validate_order_position_limit_exceeded(self):
        """Test order validation - position limit exceeded"""
        rm = RiskManager(max_position_size=0.1)

        order = TradingOrder(
            order_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.2,
            price=50000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time()
        )

        is_valid, reason = rm.validate_order(order)
        assert not is_valid
        assert "Position size would exceed limit" in reason

    def test_validate_order_daily_loss_limit(self):
        """Test order validation - daily loss limit exceeded"""
        rm = RiskManager(max_daily_loss=1000.0)
        rm.daily_loss = 1000.0

        order = TradingOrder(
            order_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time()
        )

        is_valid, reason = rm.validate_order(order)
        assert not is_valid
        assert reason == "Daily loss limit exceeded"

    def test_validate_order_max_open_orders(self):
        """Test order validation - max open orders exceeded"""
        rm = RiskManager(max_open_orders=1)

        # Add an existing order
        existing_order = TradingOrder(
            order_id="existing_123",
            symbol="ETH/USDT",
            side="buy",
            quantity=0.1,
            price=3000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time()
        )
        rm.register_order(existing_order)

        # Try to add another order
        new_order = TradingOrder(
            order_id="new_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time()
        )

        is_valid, reason = rm.validate_order(new_order)
        assert not is_valid
        assert "Maximum open orders limit reached" in reason

    def test_update_position_buy(self):
        """Test position update for buy orders"""
        rm = RiskManager()

        order = TradingOrder(
            order_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time(),
            status=OrderStatus.FILLED,
            filled_quantity=0.1
        )

        rm.update_position(order)
        assert rm.open_positions["BTC/USDT"] == 0.1

    def test_update_position_sell(self):
        """Test position update for sell orders"""
        rm = RiskManager()
        rm.open_positions["BTC/USDT"] = 0.2

        order = TradingOrder(
            order_id="test_123",
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            price=50000.0,
            order_type=OrderType.MARKET,
            timestamp=time.time(),
            status=OrderStatus.FILLED,
            filled_quantity=0.1
        )

        rm.update_position(order)
        assert rm.open_positions["BTC/USDT"] == 0.1


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class"""

    def test_circuit_breaker_creation(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"

    def test_successful_call(self):
        """Test successful function call"""
        cb = CircuitBreaker()

        def test_func():
            return "success"

        result = cb.call(test_func)
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"

    def test_failure_handling(self):
        """Test failure handling and circuit opening"""
        cb = CircuitBreaker(failure_threshold=2)

        def failing_func():
            raise Exception("Test error")

        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.failure_count == 1
        assert cb.state == "CLOSED"

        # Second failure - should open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.failure_count == 2
        assert cb.state == "OPEN"

    def test_circuit_recovery(self):
        """Test circuit breaker recovery"""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        def failing_func():
            raise Exception("Test error")

        def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == "OPEN"

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should attempt recovery and succeed
        result = cb.call(success_func)
        assert result == "success"
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"


class TestHighFrequencyTradingEngine:
    """Test cases for HighFrequencyTradingEngine class"""

    @pytest.fixture
    def engine(self):
        """Create test engine instance"""
        return HighFrequencyTradingEngine()

    @pytest.fixture
    def risk_manager(self):
        """Create test risk manager"""
        return RiskManager(max_position_size=1.0, max_daily_loss=10000.0)

    def test_engine_creation(self, engine):
        """Test engine initialization"""
        assert not engine.is_running
        assert engine.engine_name == "High Frequency Trading Engine (Production)"
        assert len(engine.orders) == 0
        assert isinstance(engine.risk_manager, RiskManager)
        assert isinstance(engine.circuit_breaker, CircuitBreaker)

    @pytest.mark.asyncio
    async def test_submit_order_valid(self, engine):
        """Test submitting a valid order"""
        order_request = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 0.1,
            'order_type': 'market'
        }

        result = await engine.submit_order(order_request)

        assert 'order_id' in result
        assert result['status'] == 'filled'
        assert result['engine'] == 'hft_engine_production'
        assert 'executed_price' in result
        assert 'executed_quantity' in result

    @pytest.mark.asyncio
    async def test_submit_order_missing_fields(self, engine):
        """Test submitting order with missing required fields"""
        incomplete_request = {
            'symbol': 'BTC/USDT',
            'side': 'buy'
            # Missing quantity and order_type
        }

        result = await engine.submit_order(incomplete_request)

        assert result['status'] == 'rejected'
        assert 'error' in result
        assert 'Missing required field' in result['error']

    @pytest.mark.asyncio
    async def test_submit_order_risk_rejection(self, engine):
        """Test order rejection by risk manager"""
        # Set very restrictive risk limits
        engine.risk_manager.max_position_size = 0.01

        order_request = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 0.1,
            'order_type': 'market'
        }

        result = await engine.submit_order(order_request)

        assert result['status'] == 'rejected'
        assert result['error_code'] == 'risk_rejection'

    @pytest.mark.asyncio
    async def test_cancel_order(self, engine):
        """Test order cancellation"""
        # First submit an order
        order_request = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 0.1,
            'order_type': 'market'
        }

        submit_result = await engine.submit_order(order_request)
        order_id = submit_result['order_id']

        # Then cancel it
        cancel_result = await engine.cancel_order(order_id)

        assert cancel_result['order_id'] == order_id
        assert cancel_result['status'] == 'cancelled'

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, engine):
        """Test cancelling non-existent order"""
        result = await engine.cancel_order("nonexistent_id")

        assert result['order_id'] == "nonexistent_id"
        assert result['status'] == 'not_found'

    @pytest.mark.asyncio
    async def test_engine_start_stop(self, engine):
        """Test engine start and stop"""
        assert not engine.is_running

        await engine.start()
        assert engine.is_running

        await engine.stop()
        assert not engine.is_running

    def test_performance_metrics(self, engine):
        """Test performance metrics retrieval"""
        metrics = engine.get_performance_metrics()

        assert metrics['engine_name'] == "High Frequency Trading Engine (Production)"
        assert 'is_running' in metrics
        assert 'metrics' in metrics
        assert 'risk_manager' in metrics
        assert 'circuit_breaker' in metrics
        assert 'timestamp' in metrics

    def test_health_check(self, engine):
        """Test health check functionality"""
        health = engine.health_check()

        assert 'engine_name' in health
        assert 'status' in health
        assert 'is_running' in health
        assert 'issues' in health
        assert 'performance' in health
        assert 'timestamp' in health

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, engine):
        """Test circuit breaker integration with order processing"""
        # Force circuit breaker to open by simulating failures
        original_execute = engine._execute_order

        async def failing_execute(order):
            raise Exception("Simulated exchange failure")

        engine._execute_order = failing_execute

        # Submit orders until circuit breaker opens
        order_request = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 0.1,
            'order_type': 'market'
        }

        # Submit multiple failing orders
        for _ in range(6):  # More than default threshold of 5
            result = await engine.submit_order(order_request)
            assert result['status'] == 'rejected'

        # Check that circuit breaker is open
        assert engine.circuit_breaker.state == "OPEN"

        # Restore original method
        engine._execute_order = original_execute


if __name__ == "__main__":
    pytest.main([__file__, "-v"])