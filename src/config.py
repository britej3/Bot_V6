"""
Configuration settings for CryptoScalp AI
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict, AnyHttpUrl, RedisDsn, PostgresDsn, validator


class Settings(BaseSettings):
    """Application settings"""

    # Application settings
    app_name: str = Field(default="CryptoScalp AI", env="CRYPTOSCALP_APP_NAME")
    version: str = Field(default="1.0.0", env="CRYPTOSCALP_VERSION")
    environment: str = Field(default="development", env="CRYPTOSCALP_ENV")

    # API settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_debug: bool = Field(default=False, env="API_DEBUG")  # Debug disabled by default for security

    # Database settings - No defaults for security
    database_url: Optional[PostgresDsn] = Field(
        default=None,
        env="DATABASE_URL",
        description="PostgreSQL database URL - must be provided in production"
    )
    redis_url: Optional[RedisDsn] = Field(
        default=None,
        env="REDIS_URL",
        description="Redis URL - must be provided in production"
    )
    clickhouse_url: Optional[str] = Field(
        default=None,
        env="CLICKHOUSE_URL",
        description="ClickHouse URL - must be provided in production"
    )

    # Message queue settings - No defaults for security
    rabbitmq_url: Optional[str] = Field(
        default=None,
        env="RABBITMQ_URL",
        description="RabbitMQ URL - must be provided in production"
    )

    # Trading settings
    supported_exchanges: List[str] = Field(
        default=["binance", "okx", "bybit", "coinbase"],
        env="SUPPORTED_EXCHANGES"
    )
    max_position_size: float = Field(default=0.01, env="MAX_POSITION_SIZE")
    min_order_size: float = Field(default=0.0001, env="MIN_ORDER_SIZE")
    risk_per_trade: float = Field(default=0.02, env="RISK_PER_TRADE")

    # ML settings
    model_update_interval: int = Field(default=3600, env="MODEL_UPDATE_INTERVAL")  # seconds
    feature_lookback_window: int = Field(default=100, env="FEATURE_LOOKBACK_WINDOW")
    prediction_horizon: int = Field(default=5, env="PREDICTION_HORIZON")  # seconds

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default="logs/cryptoscalp.log", env="LOG_FILE")

    # Additional fields for testing compatibility
    CRYPTOSCALP_ENV: Optional[str] = Field(default=None, env="CRYPTOSCALP_ENV")
    DEBUG: Optional[bool] = Field(default=None, env="DEBUG")
    RELOAD_ON_CHANGE: Optional[bool] = Field(default=None, env="RELOAD_ON_CHANGE")
    API_PORT: Optional[int] = Field(default=None, env="API_PORT")
    MAX_WORKERS: Optional[int] = Field(default=4, env="MAX_WORKERS")  # Add default value
    MAX_POSITION_SIZE: Optional[float] = Field(default=None, env="MAX_POSITION_SIZE")

    # Fields used in conftest.py
    DATABASE_URL: Optional[PostgresDsn] = Field(default=None, env="DATABASE_URL")
    REDIS_URL: Optional[RedisDsn] = Field(default=None, env="REDIS_URL")
    CLICKHOUSE_URL: Optional[str] = Field(default=None, env="CLICKHOUSE_URL")
    RABBITMQ_URL: Optional[str] = Field(default=None, env="RABBITMQ_URL")
    LOG_LEVEL: Optional[str] = Field(default=None, env="LOG_LEVEL")

    # AI/ML API settings
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field(default=None, env="OPENAI_API_BASE")
    openai_model: Optional[str] = Field(default=None, env="OPENAI_MODEL")
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._validate_production_config()

    def _validate_production_config(self):
        """Validate configuration for production environment"""
        if self.environment == "production":
            required_fields = {
                'database_url': self.database_url,
                'redis_url': self.redis_url,
                'rabbitmq_url': self.rabbitmq_url
            }

            missing_fields = [field for field, value in required_fields.items() if not value]

            if missing_fields:
                raise ValueError(
                    f"Missing required configuration for production environment: {', '.join(missing_fields)}. "
                    "These must be provided via environment variables."
                )

            # Validate that we're not using default/weak credentials
            if self.database_url and any(weak in str(self.database_url) for weak in ['devpassword', 'password', '123456', 'admin']):
                raise ValueError("Database URL contains weak credentials - use strong passwords in production")

            if self.rabbitmq_url and 'guest:guest' in str(self.rabbitmq_url):
                raise ValueError("RabbitMQ URL uses default guest credentials - configure proper credentials for production")

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level is one of the allowed values"""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v

    @validator('api_port')
    def validate_api_port(cls, v):
        """Validate API port is in valid range"""
        if v is not None and (v < 1 or v > 65535):
            raise ValueError("API port must be between 1 and 65535")
        return v

    @validator('max_position_size')
    def validate_max_position_size(cls, v):
        """Validate max position size is positive"""
        if v is not None and v <= 0:
            raise ValueError("Max position size must be positive")
        return v

    @property
    def redis_url_str(self) -> str:
        """Get Redis URL as string"""
        return str(self.redis_url)

    @property
    def database_url_str(self) -> str:
        """Get database URL as string"""
        return str(self.database_url)


# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance"""
    return settings