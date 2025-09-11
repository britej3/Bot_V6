"""
Configuration for Tick Data System

This module provides configuration settings for the tick data API system
with safety and performance parameters.
"""

import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class TickDataSettings(BaseSettings):
    """Settings for tick data system"""

    # API Configuration
    api_prefix: str = "/api/v1/tick-data"
    api_version: str = "v1"
    api_title: str = "Tick Data API"
    api_description: str = "Real-time tick data from cryptocurrency exchanges"

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)
    rate_limit_per_hour: int = Field(default=1000, ge=1, le=10000)
    burst_limit: int = Field(default=10, ge=1, le=100)

    # Data Limits
    max_limit: int = Field(default=1000, ge=1, le=5000)
    default_limit: int = Field(default=100, ge=1, le=1000)
    min_limit: int = Field(default=1, ge=1, le=100)

    # Caching
    cache_ttl: int = Field(default=30, ge=1, le=300)  # seconds
    cache_max_size: int = Field(default=10000, ge=100, le=100000)
    cache_cleanup_interval: int = Field(default=60, ge=10, le=300)  # seconds

    # Safety Settings
    read_only_mode: bool = Field(default=True)
    validate_symbols: bool = Field(default=True)
    sanitize_inputs: bool = Field(default=True)
    enable_audit_log: bool = Field(default=True)

    # Exchange Configuration
    supported_exchanges: List[str] = Field(default=[
        "binance", "okx", "bybit", "coinbase", "kraken",
        "bitfinex", "huobi", "kucoin", "gate"
    ])

    fallback_exchanges: List[str] = Field(default=[
        "binance", "okx", "bybit"
    ])

    # Exchange-specific settings
    exchange_settings: Dict[str, Dict[str, Any]] = Field(default={
        "binance": {
            "rate_limit": 1000,  # requests per second
            "timeout": 30000,
            "sandbox": True
        },
        "okx": {
            "rate_limit": 500,
            "timeout": 30000,
            "sandbox": True
        },
        "bybit": {
            "rate_limit": 1000,
            "timeout": 30000,
            "sandbox": True
        }
    })

    # Data Validation
    max_symbol_length: int = Field(default=20, ge=5, le=50)
    allowed_quote_currencies: List[str] = Field(default=[
        "USDT", "USD", "BTC", "ETH", "BNB", "EUR", "GBP", "JPY"
    ])

    # Performance Settings
    max_concurrent_requests: int = Field(default=50, ge=1, le=200)
    request_timeout: int = Field(default=30, ge=5, le=120)  # seconds
    connection_pool_size: int = Field(default=10, ge=1, le=100)

    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # Security
    api_key_required: bool = Field(default=False)
    allowed_ips: List[str] = Field(default=["127.0.0.1", "localhost"])
    cors_origins: List[str] = Field(default=["*"])

    # Historical Data (for backtesting)
    enable_historical_data: bool = Field(default=False)
    historical_data_sources: List[str] = Field(default=["kaggle", "local", "api"])
    max_historical_days: int = Field(default=365, ge=1, le=3650)

    @validator('supported_exchanges')
    def validate_supported_exchanges(cls, v):
        """Validate that supported exchanges are valid"""
        valid_exchanges = [
            "binance", "okx", "bybit", "coinbase", "kraken",
            "bitfinex", "huobi", "kucoin", "gate", "bitstamp",
            "gemini", "bitflyer", "bittrex", "poloniex"
        ]

        invalid_exchanges = [ex for ex in v if ex.lower() not in valid_exchanges]
        if invalid_exchanges:
            raise ValueError(f"Invalid exchanges: {invalid_exchanges}")

        return [ex.lower() for ex in v]

    @validator('fallback_exchanges')
    def validate_fallback_exchanges(cls, v, values):
        """Validate that fallback exchanges are subset of supported exchanges"""
        if 'supported_exchanges' in values:
            supported = values['supported_exchanges']
            invalid_fallbacks = [ex for ex in v if ex.lower() not in supported]
            if invalid_fallbacks:
                raise ValueError(f"Fallback exchanges must be in supported exchanges: {invalid_fallbacks}")

        return [ex.lower() for ex in v]

    @validator('max_limit')
    def validate_max_limit(cls, v, values):
        """Validate that max_limit is greater than default_limit"""
        if 'default_limit' in values and v < values['default_limit']:
            raise ValueError("max_limit must be greater than or equal to default_limit")

        return v

    class Config:
        """Pydantic config"""
        env_prefix = "TICK_DATA_"
        env_file = ".env"
        case_sensitive = False


# Global settings instance
tick_data_settings = TickDataSettings()


def get_tick_data_config() -> TickDataSettings:
    """Get tick data configuration"""
    return tick_data_settings


def reload_config() -> TickDataSettings:
    """Reload configuration from environment"""
    global tick_data_settings
    tick_data_settings = TickDataSettings()
    return tick_data_settings


def get_exchange_config(exchange_name: str) -> Dict[str, Any]:
    """Get configuration for specific exchange"""
    return tick_data_settings.exchange_settings.get(
        exchange_name.lower(),
        {"rate_limit": 1000, "timeout": 30000, "sandbox": True}
    )


def is_exchange_supported(exchange_name: str) -> bool:
    """Check if exchange is supported"""
    return exchange_name.lower() in tick_data_settings.supported_exchanges


def get_supported_symbols_for_exchange(exchange_name: str) -> List[str]:
    """Get supported symbols for exchange (common trading pairs)"""
    common_symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
        "DOT/USDT", "MATIC/USDT", "AVAX/USDT", "LINK/USDT", "UNI/USDT",
        "XRP/USDT", "DOGE/USDT", "SHIB/USDT", "LTC/USDT", "TRX/USDT"
    ]

    # Filter based on exchange-specific availability if needed
    return common_symbols


# Configuration validation
def validate_config() -> List[str]:
    """Validate current configuration and return any warnings"""
    warnings = []

    settings = tick_data_settings

    if settings.max_limit > 5000:
        warnings.append("max_limit is very high, consider reducing for better performance")

    if settings.cache_ttl > 300:
        warnings.append("cache_ttl is high, may consume significant memory")

    if settings.rate_limit_per_minute > 100:
        warnings.append("High rate limit may impact exchange API limits")

    if not settings.read_only_mode:
        warnings.append("WARNING: read_only_mode is disabled, system can make changes")

    return warnings