"""
Trading Configuration for XGBoost Enhanced Crypto Futures Scalping Platform
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime



class AdvancedTradingConfig(BaseModel):
    """Advanced trading configuration for XGBoost-powered scalping"""

    # Basic trading parameters
    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    mode: str = Field(default="backtest", description="Trading mode: backtest, paper_trade, live_trade")
    risk_per_trade_pct: float = Field(default=0.01, description="Risk per trade as percentage of equity")
    max_position_size_btc: float = Field(default=0.1, description="Maximum position size in BTC")
    max_drawdown_pct: float = Field(default=0.05, description="Maximum drawdown percentage")

    # XGBoost model parameters
    min_confidence_threshold: float = Field(default=0.6, description="Minimum confidence for trade execution")
    n_estimators: int = Field(default=1000, description="Number of XGBoost estimators")
    learning_rate: float = Field(default=0.01, description="XGBoost learning rate")
    max_depth: int = Field(default=6, description="Maximum tree depth")
    min_child_weight: int = Field(default=1, description="Minimum child weight")
    subsample: float = Field(default=0.8, description="Subsample ratio")
    colsample_bytree: float = Field(default=0.8, description="Column subsample ratio")

    # Feature engineering parameters
    lookback_window: int = Field(default=100, description="Lookback window for features")
    feature_horizon: int = Field(default=5, description="Prediction horizon in seconds")
    fft_components: int = Field(default=10, description="Number of FFT components")
    order_book_levels: int = Field(default=20, description="Order book levels to use")

    # MLflow tracking
    mlflow_tracking: bool = Field(default=True, description="Enable MLflow tracking")
    experiment_name: str = Field(default="xgboost_scalping", description="MLflow experiment name")

    # Data parameters
    historical_data_days: int = Field(default=30, description="Days of historical data to download")
    tick_buffer_size: int = Field(default=1000, description="Size of tick buffer")
    update_interval_ms: int = Field(default=100, description="Update interval in milliseconds")

    # Risk management
    max_consecutive_losses: int = Field(default=3, description="Maximum consecutive losses")
    volatility_threshold: float = Field(default=0.02, description="Volatility threshold for trading pause")
    min_trade_interval_ms: int = Field(default=1000, description="Minimum interval between trades")

    # Binance API settings
    binance_api_key: Optional[str] = Field(default=None, description="Binance API key")
    binance_secret_key: Optional[str] = Field(default=None, description="Binance secret key")
    binance_testnet: bool = Field(default=True, description="Use Binance testnet")

    # Redis settings for model serving
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_ml_enabled: bool = Field(default=False, description="Enable Redis ML")

    # Ray Tune settings
    ray_tune_enabled: bool = Field(default=False, description="Enable Ray Tune hyperparameter optimization")
    ray_tune_samples: int = Field(default=50, description="Number of Ray Tune samples")

    # PyTorch Lightning settings
    pytorch_lightning_enabled: bool = Field(default=False, description="Enable PyTorch Lightning")
    batch_size: int = Field(default=32, description="Training batch size")
    epochs: int = Field(default=100, description="Number of training epochs")

    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        use_enum_values = True

    def __init__(self, **data):
        super().__init__(**data)

    @validator('mode')
    def validate_mode(cls, v):
        allowed_modes = ['backtest', 'paper_trade', 'live_trade']
        if v not in allowed_modes:
            raise ValueError(f"Mode must be one of {allowed_modes}")
        return v

    @validator('risk_per_trade_pct')
    def validate_risk_per_trade(cls, v):
        if not 0 < v <= 0.1:
            raise ValueError("Risk per trade must be between 0 and 10%")
        return v

    @validator('max_drawdown_pct')
    def validate_max_drawdown(cls, v):
        if not 0 < v <= 0.5:
            raise ValueError("Max drawdown must be between 0 and 50%")
        return v

    @validator('min_confidence_threshold')
    def validate_confidence(cls, v):
        if not 0.1 <= v <= 0.99:
            raise ValueError("Confidence threshold must be between 0.1 and 0.99")
        return v

    @classmethod
    def from_env(cls) -> "AdvancedTradingConfig":
        """Create config from environment variables"""
        return cls(
            symbol=os.getenv("TRADING_SYMBOL", "BTCUSDT"),
            mode=os.getenv("TRADING_MODE", "backtest"),
            risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.01")),
            max_position_size_btc=float(os.getenv("MAX_POSITION_SIZE_BTC", "0.1")),
            max_drawdown_pct=float(os.getenv("MAX_DRAWDOWN_PCT", "0.05")),
            min_confidence_threshold=float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.6")),
            n_estimators=int(os.getenv("XGB_N_ESTIMATORS", "1000")),
            learning_rate=float(os.getenv("XGB_LEARNING_RATE", "0.01")),
            max_depth=int(os.getenv("XGB_MAX_DEPTH", "6")),
            min_child_weight=int(os.getenv("XGB_MIN_CHILD_WEIGHT", "1")),
            subsample=float(os.getenv("XGB_SUBSAMPLE", "0.8")),
            colsample_bytree=float(os.getenv("XGB_COLSAMPLE_BYTREE", "0.8")),
            lookback_window=int(os.getenv("FEATURE_LOOKBACK_WINDOW", "100")),
            feature_horizon=int(os.getenv("FEATURE_HORIZON", "5")),
            fft_components=int(os.getenv("FFT_COMPONENTS", "10")),
            order_book_levels=int(os.getenv("ORDER_BOOK_LEVELS", "20")),
            mlflow_tracking=os.getenv("MLFLOW_TRACKING", "true").lower() == "true",
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "xgboost_scalping"),
            historical_data_days=int(os.getenv("HISTORICAL_DATA_DAYS", "30")),
            tick_buffer_size=int(os.getenv("TICK_BUFFER_SIZE", "1000")),
            update_interval_ms=int(os.getenv("UPDATE_INTERVAL_MS", "100")),
            max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")),
            volatility_threshold=float(os.getenv("VOLATILITY_THRESHOLD", "0.02")),
            min_trade_interval_ms=int(os.getenv("MIN_TRADE_INTERVAL_MS", "1000")),
            binance_api_key=os.getenv("BINANCE_API_KEY"),
            binance_secret_key=os.getenv("BINANCE_SECRET_KEY"),
            binance_testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_ml_enabled=os.getenv("REDIS_ML_ENABLED", "false").lower() == "true",
            ray_tune_enabled=os.getenv("RAY_TUNE_ENABLED", "false").lower() == "true",
            ray_tune_samples=int(os.getenv("RAY_TUNE_SAMPLES", "50")),
            pytorch_lightning_enabled=os.getenv("PYTORCH_LIGHTNING_ENABLED", "false").lower() == "true",
            batch_size=int(os.getenv("TRAINING_BATCH_SIZE", "32")),
            epochs=int(os.getenv("TRAINING_EPOCHS", "100")),
        )

    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters as dictionary"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }

    def get_binance_config(self) -> Dict[str, Any]:
        """Get Binance configuration"""
        return {
            'api_key': self.binance_api_key,
            'secret_key': self.binance_secret_key,
            'testnet': self.binance_testnet,
            'symbol': self.symbol,
            'recv_window': 10000
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            'host': self.redis_host,
            'port': self.redis_port,
            'db': 0,
            'decode_responses': True
        }


# Default configuration instance
default_trading_config = AdvancedTradingConfig()


def get_trading_config() -> AdvancedTradingConfig:
    """Get the default trading configuration"""
    return default_trading_config