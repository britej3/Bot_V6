"""
Unit tests for AdvancedMarketAdaptation
"""
import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.learning.self_adaptation.market_adaptation import (
    AdvancedMarketAdaptation,
    AdaptationConfig,
    MarketAnalyzer,
    StrategyAdapter,
    MarketCondition,
    MarketRegime
)

class TestAdaptationConfig:
    """Test AdaptationConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = AdaptationConfig()
        
        assert config.volatility_window == 60
        assert config.trend_window == 300
        assert config.correlation_window == 120
        assert config.high_volatility_threshold == 0.02
        assert config.low_volatility_threshold == 0.005
        assert config.strong_trend_threshold == 0.01
        assert config.adaptation_frequency == 30
        assert config.regime_stability_window == 300
        assert config.performance_window == 100

class TestMarketAnalyzer:
    """Test MarketAnalyzer functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create MarketAnalyzer instance for testing"""
        config = AdaptationConfig()
        return MarketAnalyzer(config)
    
    def test_initialization(self, analyzer):
        """Test MarketAnalyzer initialization"""
        assert isinstance(analyzer, MarketAnalyzer)
        assert isinstance(analyzer.config, AdaptationConfig)
        assert len(analyzer.recent_volatility) == 0
        assert len(analyzer.recent_prices) == 0
    
    def test_update_market_data(self, analyzer):
        """Test market data updates"""
        timestamp = datetime.now()
        
        # Test basic update
        analyzer.update_market_data(
            price=45000.0,
            volume=1000.0,
            timestamp=timestamp
        )
        
        assert len(analyzer.recent_prices) == 1
        assert len(analyzer.recent_volume) == 1
        assert len(analyzer.recent_volatility) == 1
        
        # Test update with correlated assets
        analyzer.update_market_data(
            price=45100.0,
            volume=1500.0,
            timestamp=timestamp + timedelta(seconds=1),
            correlated_assets=[45050.0, 45075.0]
        )
        
        assert len(analyzer.recent_correlations) == 1
    
    def test_calculate_volatility(self, analyzer):
        """Test volatility calculation"""
        # Add some price data
        base_time = datetime.now()
        for i in range(10):
            price = 45000 + np.random.normal(0, 100)  # Add noise
            analyzer.recent_volatility.append((base_time + timedelta(seconds=i), price))
        
        volatility = analyzer._calculate_volatility()
        assert isinstance(volatility, float)
        assert 0.0 <= volatility <= 1.0  # Normalized to 0-1
    
    def test_calculate_trend_strength(self, analyzer):
        """Test trend strength calculation"""
        # Add trending price data
        base_time = datetime.now()
        for i in range(20):
            # Create upward trend
            price = 45000 + i * 10 + np.random.normal(0, 50)
            analyzer.recent_prices.append((base_time + timedelta(seconds=i), price))
        
        trend_strength = analyzer._calculate_trend_strength()
        assert isinstance(trend_strength, float)
        assert -1.0 <= trend_strength <= 1.0
    
    def test_calculate_volume_profile(self, analyzer):
        """Test volume profile calculation"""
        # Add volume data
        base_time = datetime.now()
        for i in range(10):
            volume = np.random.exponential(1000)
            analyzer.recent_volume.append((base_time + timedelta(seconds=i), volume))
        
        volume_profile = analyzer._calculate_volume_profile()
        assert isinstance(volume_profile, float)
        assert 0.0 <= volume_profile <= 1.0
    
    def test_classify_regime(self, analyzer):
        """Test market regime classification"""
        # Test normal regime
        regime = analyzer._classify_regime(volatility=0.01, trend_strength=0.005)
        assert regime == MarketRegime.NORMAL
        
        # Test volatile regime
        regime = analyzer._classify_regime(volatility=0.03, trend_strength=0.005)
        assert regime == MarketRegime.VOLATILE
        
        # Test trending regime
        regime = analyzer._classify_regime(volatility=0.003, trend_strength=0.02)
        assert regime == MarketRegime.TRENDING
        
        # Test crash regime
        regime = analyzer._classify_regime(volatility=0.03, trend_strength=-0.02)
        assert regime == MarketRegime.CRASH

class TestStrategyAdapter:
    """Test StrategyAdapter functionality"""
    
    @pytest.fixture
    def adapter(self):
        """Create StrategyAdapter instance for testing"""
        config = AdaptationConfig()
        return StrategyAdapter(config)
    
    def test_initialization(self, adapter):
        """Test StrategyAdapter initialization"""
        assert isinstance(adapter, StrategyAdapter)
        assert isinstance(adapter.config, AdaptationConfig)
        assert adapter.current_strategy_params == {}
        assert isinstance(adapter.strategy_history, list)
    
    def test_get_base_parameters(self, adapter):
        """Test base parameter retrieval"""
        params = adapter._get_base_parameters()
        
        assert isinstance(params, dict)
        assert "position_size_multiplier" in params
        assert "stop_loss_multiplier" in params
        assert "take_profit_multiplier" in params
        assert "frequency_multiplier" in params
        assert "risk_per_trade" in params
        assert params["position_size_multiplier"] == 1.0
        assert params["risk_per_trade"] == 0.02
    
    def test_adapt_strategy_volatile_regime(self, adapter):
        """Test strategy adaptation for volatile regime"""
        condition = MarketCondition(
            timestamp=datetime.now(),
            volatility=0.03,  # High volatility
            trend_strength=0.005,  # Weak trend
            volume_profile=0.7,
            liquidity=0.8,
            correlation=0.5,
            regime=MarketRegime.VOLATILE,
            confidence=0.9
        )
        
        params = adapter.adapt_strategy(condition)
        
        assert isinstance(params, dict)
        # In volatile regime, position size should be reduced
        assert params["position_size_multiplier"] < 1.0
        # Risk per trade should be reduced
        assert params["risk_per_trade"] < 0.02
        assert len(adapter.strategy_history) == 1
    
    def test_adapt_strategy_trending_regime(self, adapter):
        """Test strategy adaptation for trending regime"""
        condition = MarketCondition(
            timestamp=datetime.now(),
            volatility=0.003,  # Low volatility
            trend_strength=0.02,  # Strong uptrend
            volume_profile=0.7,
            liquidity=0.8,
            correlation=0.5,
            regime=MarketRegime.TRENDING,
            confidence=0.9
        )
        
        params = adapter.adapt_strategy(condition)
        
        assert isinstance(params, dict)
        # In trending regime, position size should be increased
        assert params["position_size_multiplier"] > 1.0
        # Risk per trade should be increased
        assert params["risk_per_trade"] > 0.02

class TestAdvancedMarketAdaptation:
    """Test AdvancedMarketAdaptation main class"""
    
    @pytest.fixture
    def adaptation(self):
        """Create AdvancedMarketAdaptation instance for testing"""
        return AdvancedMarketAdaptation()
    
    def test_initialization(self, adaptation):
        """Test AdvancedMarketAdaptation initialization"""
        assert isinstance(adaptation, AdvancedMarketAdaptation)
        assert isinstance(adaptation.config, AdaptationConfig)
        assert isinstance(adaptation.analyzer, MarketAnalyzer)
        assert isinstance(adaptation.adapter, StrategyAdapter)
    
    def test_update_market_data(self, adaptation):
        """Test market data update"""
        adaptation.update_market_data(
            price=45000.0,
            volume=1000.0
        )
        
        # Check that analyzer was updated
        assert len(adaptation.analyzer.recent_prices) == 1
        assert len(adaptation.analyzer.recent_volume) == 1
    
    def test_should_adapt(self, adaptation):
        """Test adaptation timing"""
        # Should adapt immediately after initialization
        assert adaptation.should_adapt()
        
        # Simulate recent adaptation
        adaptation.last_adaptation = datetime.now()
        assert not adaptation.should_adapt()
    
    def test_get_current_condition(self, adaptation):
        """Test current condition retrieval"""
        # Add some data first
        adaptation.update_market_data(price=45000.0, volume=1000.0)
        
        condition = adaptation.get_current_condition()
        
        assert isinstance(condition, MarketCondition)
        assert isinstance(condition.timestamp, datetime)
        assert isinstance(condition.regime, MarketRegime)
    
    def test_get_current_parameters(self, adaptation):
        """Test current parameters retrieval"""
        params = adaptation.get_current_parameters()
        
        assert isinstance(params, dict)
        # Should return base parameters when no adaptation has occurred
        assert params["position_size_multiplier"] == 1.0
        assert params["risk_per_trade"] == 0.02

if __name__ == "__main__":
    pytest.main([__file__])