"""
Comprehensive tests for XGBoost Enhanced Crypto Futures Scalping Platform
Tests all components including validation, redundancy, and error handling
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import os

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.config.trading_config import AdvancedTradingConfig, get_trading_config
from src.learning.xgboost_ensemble import XGBoostEnsemble
from src.learning.tick_level_feature_engine import TickLevelFeatureEngine
from src.data_pipeline.binance_data_manager import BinanceDataManager
from src.strategies.xgboost_nautilus_strategy import XGBoostNautilusStrategy, ValidationManager
from src.strategies.base_strategy import BaseTradingStrategy, StrategyState, ValidationError, RiskError


@pytest.fixture
def sample_config():
    """Create a sample trading configuration for testing"""
    return AdvancedTradingConfig(
        symbol="BTCUSDT",
        mode="backtest",
        risk_per_trade_pct=0.01,
        max_position_size_btc=0.1,
        max_drawdown_pct=0.05,
        min_confidence_threshold=0.6,
        n_estimators=10,  # Small number for testing
        learning_rate=0.1,
        max_depth=3,
        lookback_window=50,
        feature_horizon=5,
        fft_components=5,
        mlflow_tracking=False,  # Disable for testing
        redis_ml_enabled=False,  # Disable for testing
        ray_tune_enabled=False,  # Disable for testing
    )


@pytest.fixture
def sample_tick_data():
    """Create sample tick data for testing"""
    return [
        {
            'timestamp': datetime.utcnow(),
            'price': 50000.0,
            'quantity': 0.1,
            'is_buyer_maker': True
        },
        {
            'timestamp': datetime.utcnow() + timedelta(seconds=1),
            'price': 50010.0,
            'quantity': 0.15,
            'is_buyer_maker': False
        },
        {
            'timestamp': datetime.utcnow() + timedelta(seconds=2),
            'price': 49990.0,
            'quantity': 0.2,
            'is_buyer_maker': True
        }
    ]


@pytest.fixture
def sample_historical_data():
    """Create sample historical data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1min')
    prices = 50000 + np.cumsum(np.random.randn(1000) * 10)
    volumes = np.random.uniform(0.1, 1.0, 1000)

    df = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'quantity': volumes,
        'is_buyer_maker': np.random.choice([True, False], 1000)
    })

    return df


class TestAdvancedTradingConfig:
    """Test AdvancedTradingConfig class"""

    def test_config_creation(self, sample_config):
        """Test configuration creation"""
        assert sample_config.symbol == "BTCUSDT"
        assert sample_config.mode == "backtest"
        assert sample_config.risk_per_trade_pct == 0.01
        assert sample_config.min_confidence_threshold == 0.6

    def test_config_validation(self):
        """Test configuration validation"""
        with pytest.raises(ValueError):
            AdvancedTradingConfig(risk_per_trade_pct=0.15)  # Too high

        with pytest.raises(ValueError):
            AdvancedTradingConfig(max_drawdown_pct=0.6)  # Too high

        with pytest.raises(ValueError):
            AdvancedTradingConfig(min_confidence_threshold=1.5)  # Too high

    def test_xgboost_params(self, sample_config):
        """Test XGBoost parameter generation"""
        params = sample_config.get_xgboost_params()
        assert isinstance(params, dict)
        assert 'n_estimators' in params
        assert 'learning_rate' in params
        assert 'max_depth' in params
        assert params['n_estimators'] == 10
        assert params['learning_rate'] == 0.1

    def test_binance_config(self, sample_config):
        """Test Binance configuration generation"""
        binance_config = sample_config.get_binance_config()
        assert isinstance(binance_config, dict)
        assert 'testnet' in binance_config
        assert binance_config['testnet'] == True


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
class TestXGBoostEnsemble:
    """Test XGBoostEnsemble class"""

    def test_ensemble_creation(self, sample_config):
        """Test ensemble creation"""
        ensemble = XGBoostEnsemble(sample_config)
        assert ensemble.config == sample_config
        assert ensemble.n_models == 3
        assert ensemble.ensemble_method == 'weighted'
        assert not ensemble.is_trained

    def test_create_training_data(self, sample_config):
        """Test training data creation"""
        ensemble = XGBoostEnsemble(sample_config)

        # Create sample features and prices
        features = [np.random.randn(10) for _ in range(100)]
        prices = np.random.uniform(49000, 51000, 100)

        X, y = ensemble.create_training_data(features, prices)

        assert X.shape[0] == 95  # 100 - 5 (feature_horizon)
        assert X.shape[1] == 10  # feature dimension
        assert y.shape[0] == 95
        assert len(np.unique(y)) == 2  # Binary classification

    def test_feature_validation(self, sample_config):
        """Test feature validation"""
        ensemble = XGBoostEnsemble(sample_config)

        # Test with empty features
        empty_features = np.array([])
        result = ensemble.predict_with_confidence(empty_features)
        assert result['signal'] == 0
        assert result['confidence'] == 0.0

    def test_model_save_load(self, sample_config, tmp_path):
        """Test model saving and loading"""
        ensemble = XGBoostEnsemble(sample_config)

        # Create minimal training data
        features = [np.random.randn(5) for _ in range(20)]
        prices = np.random.uniform(49000, 51000, 20)

        X, y = ensemble.create_training_data(features, prices)

        # Train model
        ensemble.train_ensemble(X, y, X, y)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        result = ensemble.save_model(str(model_path))
        assert result is None  # No return value

        # Create new ensemble and load model
        new_ensemble = XGBoostEnsemble(sample_config)
        load_result = new_ensemble.load_model(str(model_path))
        assert load_result == True


class TestTickLevelFeatureEngine:
    """Test TickLevelFeatureEngine class"""

    def test_engine_creation(self, sample_config):
        """Test feature engine creation"""
        engine = TickLevelFeatureEngine(sample_config)
        assert engine.config == sample_config
        assert not engine.is_fitted
        assert engine.feature_names is None

    def test_process_tick_data(self, sample_config, sample_tick_data):
        """Test tick data processing"""
        engine = TickLevelFeatureEngine(sample_config)

        # Process tick data
        features = engine.process_tick_data(sample_tick_data[0])

        # Should return empty array for insufficient data
        assert len(features) == 0

        # Add more data to reach buffer size
        for tick in sample_tick_data:
            features = engine.process_tick_data(tick)

        # Should now return features
        assert isinstance(features, np.ndarray)

    def test_feature_validation(self, sample_config):
        """Test feature validation"""
        engine = TickLevelFeatureEngine(sample_config)

        # Test with invalid tick data
        invalid_tick = {
            'timestamp': "invalid",
            'price': -100,  # Invalid price
            'quantity': 0.1
        }

        features = engine.process_tick_data(invalid_tick)

        # Should handle invalid data gracefully
        assert isinstance(features, np.ndarray)

    def test_scaler_fitting(self, sample_config):
        """Test scaler fitting"""
        engine = TickLevelFeatureEngine(sample_config)

        # Create sample features
        sample_features = np.random.randn(10, 5)

        engine.fit_scalers(sample_features)

        assert engine.is_fitted
        assert engine.feature_names is not None
        assert len(engine.feature_names) == 5

    def test_feature_transformation(self, sample_config):
        """Test feature transformation"""
        engine = TickLevelFeatureEngine(sample_config)

        # Fit scalers first
        sample_features = np.random.randn(10, 5)
        engine.fit_scalers(sample_features)

        # Transform features
        test_features = np.random.randn(5)
        transformed = engine.transform_features(test_features)

        assert isinstance(transformed, np.ndarray)


class TestValidationManager:
    """Test ValidationManager class"""

    def test_validation_creation(self, sample_config):
        """Test validation manager creation"""
        validation = ValidationManager(sample_config)
        assert validation.config == sample_config
        assert validation.validation_errors == []
        assert not any(validation.circuit_breakers.values())

    def test_input_validation(self, sample_config):
        """Test input data validation"""
        validation = ValidationManager(sample_config)

        # Valid data
        valid_data = {
            'timestamp': datetime.utcnow(),
            'price': 50000.0,
            'quantity': 0.1
        }
        assert validation.validate_input_data(valid_data)

        # Invalid data - negative price
        invalid_data = {
            'timestamp': datetime.utcnow(),
            'price': -100.0,
            'quantity': 0.1
        }
        assert not validation.validate_input_data(invalid_data)

        # Check that error was added
        assert len(validation.validation_errors) > 0

    def test_prediction_validation(self, sample_config):
        """Test prediction validation"""
        validation = ValidationManager(sample_config)

        # Valid prediction
        valid_prediction = {
            'signal': 1,
            'confidence': 0.8
        }
        assert validation.validate_model_prediction(valid_prediction)

        # Invalid prediction - confidence too high
        invalid_prediction = {
            'signal': 1,
            'confidence': 1.5
        }
        assert not validation.validate_model_prediction(invalid_prediction)

    def test_circuit_breakers(self, sample_config):
        """Test circuit breaker functionality"""
        validation = ValidationManager(sample_config)

        # Initially should be valid
        assert validation.check_circuit_breakers()

        # Trigger consecutive losses
        for _ in range(5):
            validation.update_circuit_breakers(trade_result=False)

        # Should trigger circuit breaker
        assert validation.circuit_breakers['max_consecutive_losses']

        # Should not allow trading
        assert not validation.check_circuit_breakers()

    def test_validation_status(self, sample_config):
        """Test validation status reporting"""
        validation = ValidationManager(sample_config)

        status = validation.get_validation_status()
        assert isinstance(status, dict)
        assert 'is_valid' in status
        assert 'recent_errors' in status
        assert 'circuit_breakers' in status
        assert 'total_errors' in status


class TestBaseTradingStrategy:
    """Test BaseTradingStrategy class"""

    def test_strategy_creation(self, sample_config):
        """Test strategy creation"""
        strategy = BaseTradingStrategy(sample_config)
        assert strategy.config == sample_config
        assert isinstance(strategy.state, StrategyState)
        assert not strategy.state.is_active

    def test_config_validation(self):
        """Test configuration validation"""
        with pytest.raises(ValidationError):
            BaseTradingStrategy(AdvancedTradingConfig(risk_per_trade_pct=0.15))

    def test_position_size_calculation(self, sample_config):
        """Test position size calculation"""
        strategy = BaseTradingStrategy(sample_config)

        position_size = strategy._calculate_position_size(0.8, 50000.0)
        assert position_size >= 0
        assert position_size <= sample_config.max_position_size_btc

    def test_emergency_stop(self, sample_config):
        """Test emergency stop functionality"""
        strategy = BaseTradingStrategy(sample_config)

        strategy.emergency_stop()
        assert strategy.state.emergency_stop

    def test_circuit_breaker_reset(self, sample_config):
        """Test circuit breaker reset"""
        strategy = BaseTradingStrategy(sample_config)

        # Set some circuit breakers
        strategy.state.circuit_breakers['max_consecutive_losses'] = True

        strategy.reset_circuit_breakers()

        assert not any(strategy.state.circuit_breakers.values())


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
class TestXGBoostNautilusStrategy:
    """Test XGBoostNautilusStrategy class"""

    def test_strategy_creation(self, sample_config):
        """Test strategy creation"""
        strategy = XGBoostNautilusStrategy(sample_config)
        assert strategy.config == sample_config
        assert isinstance(strategy.validation_manager, ValidationManager)
        assert not strategy.state.emergency_stop

    @pytest.mark.asyncio
    async def test_initialization(self, sample_config):
        """Test strategy initialization"""
        strategy = XGBoostNautilusStrategy(sample_config)

        # Mock the components to avoid actual initialization
        with patch.object(strategy.data_manager, 'initialize', new_callable=AsyncMock):
            with patch.object(strategy, '_train_models_from_historical_data', new_callable=AsyncMock):
                await strategy.initialize()
                assert strategy.state.is_initialized

    def test_validation_comprehensive(self, sample_config):
        """Test comprehensive validation"""
        strategy = XGBoostNautilusStrategy(sample_config)

        # Test with invalid order execution
        with pytest.raises(ValidationError):
            strategy._validate_order_execution('BUY', -1, 50000)  # Negative position size

        with pytest.raises(ValidationError):
            strategy._validate_order_execution('BUY', 1, -100)  # Negative price

    def test_risk_management(self, sample_config):
        """Test risk management functionality"""
        strategy = XGBoostNautilusStrategy(sample_config)

        # Test drawdown limit
        strategy.state.max_drawdown = 0.1  # 10%
        assert strategy._risk_check()  # Should pass

        strategy.state.max_drawdown = 0.15  # 15%
        assert not strategy._risk_check()  # Should fail

    def test_strategy_status(self, sample_config):
        """Test strategy status reporting"""
        strategy = XGBoostNautilusStrategy(sample_config)

        status = strategy.get_strategy_status()
        assert isinstance(status, dict)
        assert 'is_active' in status
        assert 'is_initialized' in status
        assert 'emergency_stop' in status
        assert 'validation_status' in status


class TestIntegration:
    """Test integration between components"""

    def test_config_integration(self, sample_config):
        """Test configuration integration"""
        from src.learning.xgboost_ensemble import XGBoostEnsemble
        from src.learning.tick_level_feature_engine import TickLevelFeatureEngine

        ensemble = XGBoostEnsemble(sample_config)
        engine = TickLevelFeatureEngine(sample_config)

        assert ensemble.config == sample_config
        assert engine.config == sample_config

    def test_data_flow(self, sample_config, sample_tick_data):
        """Test data flow between components"""
        engine = TickLevelFeatureEngine(sample_config)

        # Process multiple ticks
        for tick in sample_tick_data:
            features = engine.process_tick_data(tick)

        # Should eventually produce features
        assert isinstance(features, np.ndarray)

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_full_pipeline(self, sample_config):
        """Test full pipeline integration"""
        ensemble = XGBoostEnsemble(sample_config)
        engine = TickLevelFeatureEngine(sample_config)

        # Create sample training data
        features = [np.random.randn(10) for _ in range(50)]
        prices = np.random.uniform(49000, 51000, 50)

        X, y = ensemble.create_training_data(features, prices)

        # Fit engine
        engine.fit_scalers(X)

        # Train ensemble
        ensemble.train_ensemble(X, y, X, y)

        assert ensemble.is_trained
        assert engine.is_fitted

        # Test prediction
        test_features = np.random.randn(10)
        prediction = ensemble.predict_with_confidence(test_features)

        assert isinstance(prediction, dict)
        assert 'signal' in prediction
        assert 'confidence' in prediction


# Performance and Benchmark Tests
class TestPerformance:
    """Performance and benchmark tests"""

    def test_feature_engine_performance(self, sample_config):
        """Test feature engine performance"""
        import time

        engine = TickLevelFeatureEngine(sample_config)

        # Create many ticks
        ticks = []
        for i in range(1000):
            ticks.append({
                'timestamp': datetime.utcnow() + timedelta(seconds=i),
                'price': 50000 + np.random.randn() * 100,
                'quantity': np.random.uniform(0.1, 1.0),
                'is_buyer_maker': np.random.choice([True, False])
            })

        start_time = time.time()

        for tick in ticks:
            features = engine.process_tick_data(tick)

        end_time = time.time()

        # Should process 1000 ticks quickly
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Less than 5 seconds

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_ensemble_performance(self, sample_config):
        """Test ensemble performance"""
        import time

        ensemble = XGBoostEnsemble(sample_config)

        # Create training data
        features = [np.random.randn(10) for _ in range(100)]
        prices = np.random.uniform(49000, 51000, 100)

        X, y = ensemble.create_training_data(features, prices)

        start_time = time.time()
        metrics = ensemble.train_ensemble(X, y, X, y)
        end_time = time.time()

        # Should train reasonably quickly
        training_time = end_time - start_time
        assert training_time < 30.0  # Less than 30 seconds

        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics


# Error Handling Tests
class TestErrorHandling:
    """Test error handling and resilience"""

    def test_graceful_degradation(self, sample_config):
        """Test graceful degradation under errors"""
        engine = TickLevelFeatureEngine(sample_config)

        # Test with corrupted data
        corrupted_tick = {
            'timestamp': None,
            'price': "invalid",
            'quantity': None
        }

        features = engine.process_tick_data(corrupted_tick)

        # Should return empty array, not crash
        assert isinstance(features, np.ndarray)

    def test_invalid_config_handling(self):
        """Test handling of invalid configurations"""
        with pytest.raises(ValueError):
            AdvancedTradingConfig(symbol="")  # Empty symbol

    def test_memory_management(self, sample_config):
        """Test memory management with large datasets"""
        import psutil
        import os

        engine = TickLevelFeatureEngine(sample_config)

        # Process many ticks
        for i in range(10000):
            tick = {
                'timestamp': datetime.utcnow(),
                'price': 50000 + np.random.randn(),
                'quantity': np.random.uniform(0.1, 1.0),
                'is_buyer_maker': True
            }
            engine.process_tick_data(tick)

        # Check memory usage (should be reasonable)
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 1000  # Less than 1GB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])