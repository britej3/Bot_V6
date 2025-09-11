"""
Unit Tests for Tick Data API

Tests for the tick data API endpoints, models, and service functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

from src.api.models import (
    TickDataPoint,
    TickDataResponse,
    TickDataConfig,
    TickDataError
)
from src.api.tick_data_service import TickDataService, TickDataCache
from src.api.routers.tick_data import (
    get_tick_data_service,
    get_tick_data_config,
    _is_valid_symbol
)


class TestTickDataModels:
    """Test tick data models"""

    def test_tick_data_point_creation(self):
        """Test creating a TickDataPoint"""
        tick = TickDataPoint(
            timestamp=time.time(),
            symbol="BTC/USDT",
            price=50000.0,
            volume=1.5,
            side="buy",
            exchange_timestamp=time.time(),
            source_exchange="binance"
        )

        assert tick.symbol == "BTC/USDT"
        assert tick.price == 50000.0
        assert tick.volume == 1.5
        assert tick.side == "buy"
        assert tick.source_exchange == "binance"

    def test_tick_data_response_creation(self):
        """Test creating a TickDataResponse"""
        ticks = [
            TickDataPoint(
                timestamp=time.time(),
                symbol="BTC/USDT",
                price=50000.0,
                volume=1.5,
                source_exchange="binance"
            )
        ]

        response = TickDataResponse(
            symbol="BTC/USDT",
            limit=10,
            data=ticks,
            message="Success",
            total_count=1,
            request_timestamp=time.time()
        )

        assert response.symbol == "BTC/USDT"
        assert len(response.data) == 1
        assert response.total_count == 1
        assert response.message == "Success"

    def test_tick_data_config_defaults(self):
        """Test TickDataConfig default values"""
        config = TickDataConfig()

        assert config.max_limit == 1000
        assert config.default_limit == 100
        assert config.cache_ttl == 30
        assert config.rate_limit_per_minute == 60
        assert "binance" in config.supported_exchanges


class TestTickDataService:
    """Test tick data service functionality"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return TickDataConfig(
            max_limit=100,
            default_limit=10,
            cache_ttl=5,
            rate_limit_per_minute=30
        )

    @pytest.fixture
    def service(self, config):
        """Test service instance"""
        with patch('src.api.tick_data_service.ccxt'):
            service = TickDataService(config)
            return service

    def test_service_initialization(self, service):
        """Test service initialization"""
        assert service.config is not None
        assert isinstance(service.cache, TickDataCache)
        assert service.is_running is False

    def test_cache_functionality(self, service):
        """Test cache operations"""
        symbol = "BTC/USDT"
        tick = TickDataPoint(
            timestamp=time.time(),
            symbol=symbol,
            price=50000.0,
            volume=1.0,
            source_exchange="test"
        )

        # Add tick to cache
        service.cache.add_tick(symbol, tick)

        # Retrieve from cache
        cached_ticks = service.cache.get_ticks(symbol, 10)

        assert len(cached_ticks) == 1
        assert cached_ticks[0].symbol == symbol
        assert cached_ticks[0].price == 50000.0

    def test_rate_limit_check(self, service):
        """Test rate limiting functionality"""
        exchange_name = "binance"

        # Should allow first request
        assert service._check_rate_limit(exchange_name) is True

        # Fill up the rate limit
        for _ in range(59):  # 59 more to reach 60
            service._check_rate_limit(exchange_name)

        # Should deny request when limit reached
        assert service._check_rate_limit(exchange_name) is False

    @pytest.mark.asyncio
    async def test_service_start_stop(self, service):
        """Test service start and stop"""
        await service.start()
        assert service.is_running is True

        await service.stop()
        assert service.is_running is False

    def test_get_best_exchange(self, service):
        """Test exchange selection logic"""
        # Initially no exchanges should be connected
        best_exchange = service._get_best_exchange()
        assert best_exchange is None

        # Mock an exchange connection
        service.exchanges['binance'] = Mock()
        service.exchanges['binance'].is_connected = True
        service.exchanges['binance'].error_count = 0
        service.exchanges['binance'].last_used = time.time()

        best_exchange = service._get_best_exchange()
        assert best_exchange == 'binance'


class TestTickDataValidation:
    """Test data validation functions"""

    def test_valid_symbol_validation(self):
        """Test valid symbol validation"""
        valid_symbols = [
            "BTC/USDT",
            "ETH/BTC",
            "ADA/ETH",
            "SOL/USD"
        ]

        for symbol in valid_symbols:
            assert _is_valid_symbol(symbol) is True

    def test_invalid_symbol_validation(self):
        """Test invalid symbol validation"""
        invalid_symbols = [
            "",  # Empty
            "BTC",  # No quote currency
            "BTC/",  # Empty quote
            "/USDT",  # Empty base
            "BTCUSDT",  # No separator
            "VERY_LONG_BASE_CURRENCY/USDT",  # Too long base
            "BTC/VERY_LONG_QUOTE_CURRENCY",  # Too long quote
            "bt/USDT",  # Too short base
            "BTC/us",  # Too short quote
        ]

        for symbol in invalid_symbols:
            assert _is_valid_symbol(symbol) is False


class TestTickDataCache:
    """Test tick data cache functionality"""

    @pytest.fixture
    def cache(self):
        """Test cache instance"""
        return TickDataCache(max_size=10, ttl=2)

    def test_cache_add_and_retrieve(self, cache):
        """Test adding and retrieving ticks from cache"""
        symbol = "BTC/USDT"
        tick = TickDataPoint(
            timestamp=time.time(),
            symbol=symbol,
            price=50000.0,
            volume=1.0,
            source_exchange="test"
        )

        cache.add_tick(symbol, tick)
        retrieved = cache.get_ticks(symbol, 5)

        assert len(retrieved) == 1
        assert retrieved[0].symbol == symbol
        assert retrieved[0].price == 50000.0

    def test_cache_ttl_expiration(self, cache):
        """Test cache TTL expiration"""
        symbol = "BTC/USDT"
        tick = TickDataPoint(
            timestamp=time.time(),
            symbol=symbol,
            price=50000.0,
            volume=1.0,
            source_exchange="test"
        )

        cache.add_tick(symbol, tick)

        # Should retrieve immediately
        retrieved = cache.get_ticks(symbol, 5)
        assert len(retrieved) == 1

        # Wait for TTL to expire
        time.sleep(3)

        # Should return empty after TTL
        retrieved = cache.get_ticks(symbol, 5)
        assert len(retrieved) == 0

    def test_cache_size_limit(self, cache):
        """Test cache size limiting"""
        symbol = "BTC/USDT"

        # Add more ticks than cache size
        for i in range(15):
            tick = TickDataPoint(
                timestamp=time.time() + i,
                symbol=symbol,
                price=50000.0 + i,
                volume=1.0,
                source_exchange="test"
            )
            cache.add_tick(symbol, tick)

        # Should only keep the most recent 10
        retrieved = cache.get_ticks(symbol, 20)
        assert len(retrieved) <= 10

        # Most recent should be the highest price
        assert retrieved[0].price == 50000.0 + 14


class TestTickDataAPIIntegration:
    """Integration tests for tick data API"""

    @pytest.fixture
    def client(self):
        """Test client fixture"""
        from src.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/tick-data/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "service" in data
        assert data["service"] == "tick_data_api"

    def test_supported_symbols_endpoint(self, client):
        """Test supported symbols endpoint"""
        response = client.get("/api/v1/tick-data/symbols")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "BTC/USDT" in data

    def test_supported_exchanges_endpoint(self, client):
        """Test supported exchanges endpoint"""
        response = client.get("/api/v1/tick-data/exchanges")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "binance" in data

    def test_service_config_endpoint(self, client):
        """Test service configuration endpoint"""
        response = client.get("/api/v1/tick-data/config")
        assert response.status_code == 200

        data = response.json()
        assert "max_limit" in data
        assert "default_limit" in data
        assert "supported_exchanges" in data

    def test_get_tick_data_invalid_symbol(self, client):
        """Test tick data endpoint with invalid symbol"""
        response = client.get("/api/v1/tick-data/INVALID_SYMBOL")
        assert response.status_code == 400

        data = response.json()
        assert "error_code" in data
        assert data["error_code"] == "INVALID_SYMBOL"

    def test_get_tick_data_valid_symbol(self, client):
        """Test tick data endpoint with valid symbol"""
        # This will likely return an error since we don't have real exchange connections
        # but it should not be a 400 error for invalid symbol format
        response = client.get("/api/v1/tick-data/BTC/USDT")
        assert response.status_code != 400  # Should not be bad request

    def test_service_statistics_endpoint(self, client):
        """Test service statistics endpoint"""
        response = client.get("/api/v1/tick-data/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data


# Pytest fixtures for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])