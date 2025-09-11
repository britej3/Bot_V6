"""
Pytest configuration and shared fixtures for CryptoScalp AI
"""
import asyncio
import os
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, Generator

import redis
import psycopg2
import pytest_asyncio
try:
    from clickhouse_driver import Client as ClickHouseClient
except ImportError:
    ClickHouseClient = None
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.main import app
from src.config import Settings, settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Get test settings"""
    test_settings = Settings(
        CRYPTOSCALP_ENV="test",
        DATABASE_URL="postgresql://cryptoscalp_test:testpassword@localhost:5433/cryptoscalp_test",
        REDIS_URL="redis://localhost:6379/1",
        CLICKHOUSE_URL="clickhouse://localhost:9000/test_db",
        RABBITMQ_URL="amqp://guest:guest@localhost:5672/",
        DEBUG=True,
        LOG_LEVEL="DEBUG"
    )
    return test_settings


@pytest.fixture(scope="session")
def test_app(settings: Settings):
    """Create test FastAPI app"""
    return app


@pytest.fixture(scope="session")
def client(test_app) -> TestClient:
    """Create test client"""
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def redis_client(settings: Settings):
    """Create mock Redis client for testing"""
    from unittest.mock import MagicMock
    mock_redis = MagicMock()
    yield mock_redis


@pytest.fixture(scope="session")
def clickhouse_client(settings: Settings):
    """Create mock ClickHouse client for testing"""
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    yield mock_client
    # No cleanup needed for mock


@pytest.fixture(scope="function")
def mock_redis(redis_client):
    """Mock Redis client that cleans up after each test"""
    yield redis_client
    # For mock Redis client, we don't need to do anything


@pytest.fixture(scope="function")
def sample_market_data() -> Dict[str, Any]:
    """Sample market data for testing"""
    return {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "price": 45000.0,
        "quantity": 0.001,
        "timestamp": 1640995200.0,
        "side": "BUY"
    }


@pytest.fixture(scope="function")
def sample_trading_config() -> Dict[str, Any]:
    """Sample trading configuration for testing"""
    return {
        "max_position_size": 1000,
        "stop_loss_percentage": 0.02,
        "take_profit_percentage": 0.04,
        "leverage": 1,
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "exchanges": ["binance", "okx"]
    }


@pytest.fixture(scope="function")
def sample_model_config() -> Dict[str, Any]:
    """Sample model configuration for testing"""
    return {
        "model_type": "ensemble",
        "lookback_window": 100,
        "prediction_horizon": 10,
        "features": [
            "price", "volume", "rsi", "macd", "bollinger_bands"
        ],
        "hyperparameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        }
    }


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Test data directory"""
    test_dir = Path(__file__).parent / "fixtures"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def mock_exchange_data() -> Dict[str, Any]:
    """Mock exchange data for testing"""
    return {
        "binance": {
            "api_key": "test_binance_key",
            "secret": "test_binance_secret",
            "testnet": True
        },
        "okx": {
            "api_key": "test_okx_key",
            "secret": "test_okx_secret",
            "passphrase": "test_passphrase",
            "testnet": True
        }
    }


@pytest.fixture(autouse=True)
def clear_cache(redis_client):
    """Clear Redis cache before each test"""
    # For mock Redis client, we don't need to do anything
    pass


# Custom markers
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests that require database"
    )
    config.addinivalue_line(
        "markers", "redis: marks tests that require Redis"
    )
    config.addinivalue_line(
        "markers", "external_api: marks tests that call external APIs"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Setup test database schema"""
    # This would typically create test database schema
    # For now, we'll just ensure the connection works
    yield
    # Cleanup after all tests


@pytest.fixture(scope="function")
def mock_time():
    """Mock time for testing time-dependent functions"""
    import time
    from unittest.mock import patch

    fixed_time = 1640995200.0  # 2022-01-01 00:00:00 UTC
    with patch('time.time', return_value=fixed_time):
        yield fixed_time


@pytest.fixture(scope="function")
def mock_async_context():
    """Mock async context for testing async functions"""
    return {"test_context": True}


@pytest.fixture(scope="function")
def sample_api_response():
    """Sample API response for testing"""
    return {
        "status": "success",
        "data": {
            "symbol": "BTC/USDT",
            "price": "45000.00",
            "timestamp": 1640995200000
        }
    }