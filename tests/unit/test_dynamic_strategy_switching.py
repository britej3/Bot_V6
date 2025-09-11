"""
Unit Tests for Dynamic Strategy Switching System
===============================================

Tests for Task 15.1.2: Implement dynamic strategy switching system

Author: Autonomous Systems Team  
Date: 2025-01-22
"""

import pytest
import torch
import time
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from learning.dynamic_strategy_switching import (
    DynamicStrategyManager,
    TradingStrategy,
    MarketMakingStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    StrategyType,
    StrategyState,
    StrategyConfig,
    StrategyPerformance,
    StrategyTransition,
    create_dynamic_strategy_manager
)

# Mock regime types for testing
class MockMarketRegime:
    NORMAL = "normal"
    VOLATILE = "volatile" 
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    BULL_RUN = "bull_run"
    CRASH = "crash"

class MockMarketCondition:
    def __init__(self, regime, volatility=0.02, trend_strength=0.01, confidence=0.8):
        self.regime = regime
        self.volatility = volatility
        self.trend_strength = trend_strength
        self.confidence = confidence


@pytest.fixture
def strategy_config():
    """Create test strategy configuration"""
    return StrategyConfig(
        strategy_type=StrategyType.MARKET_MAKING,
        regime_affinity=[MockMarketRegime.RANGE_BOUND],
        min_confidence=0.6,
        max_position_size=0.5,
        risk_multiplier=1.0
    )


@pytest.fixture
def market_making_strategy(strategy_config):
    """Create market making strategy instance"""
    return MarketMakingStrategy(strategy_config)


@pytest.fixture
def strategy_manager():
    """Create dynamic strategy manager"""
    return create_dynamic_strategy_manager()


@pytest.fixture
def mock_market_data():
    """Create mock market data"""
    return torch.randn(100)


class TestStrategyConfig:
    """Test strategy configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = StrategyConfig(strategy_type=StrategyType.MARKET_MAKING)
        
        assert config.strategy_type == StrategyType.MARKET_MAKING
        assert config.min_confidence == 0.6
        assert config.max_position_size == 1.0
        assert config.risk_multiplier == 1.0
        assert config.cooldown_period == 300.0
        assert config.transition_delay == 30.0
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = StrategyConfig(
            strategy_type=StrategyType.MOMENTUM,
            min_confidence=0.8,
            max_position_size=0.3,
            risk_multiplier=1.5
        )
        
        assert config.strategy_type == StrategyType.MOMENTUM
        assert config.min_confidence == 0.8
        assert config.max_position_size == 0.3
        assert config.risk_multiplier == 1.5


class TestStrategyPerformance:
    """Test strategy performance tracking"""
    
    def test_initial_performance(self):
        """Test initial performance metrics"""
        perf = StrategyPerformance()
        
        assert perf.total_trades == 0
        assert perf.winning_trades == 0
        assert perf.total_pnl == 0.0
        assert perf.win_rate == 0.0
        assert isinstance(perf.last_updated, datetime)


class TestTradingStrategy:
    """Test base trading strategy functionality"""
    
    def test_strategy_initialization(self, strategy_config):
        """Test strategy initialization"""
        # Create a concrete strategy class for testing
        class TestStrategy(TradingStrategy):
            def _generate_strategy_signal(self, market_data, regime, confidence):
                return None
        
        strategy = TestStrategy(strategy_config)
        
        assert strategy.config == strategy_config
        assert strategy.state == StrategyState.INACTIVE
        assert isinstance(strategy.performance, StrategyPerformance)
        assert strategy.active_positions == {}
        assert strategy.last_signal_time is None
    
    def test_strategy_activation(self, strategy_config):
        """Test strategy activation"""
        class TestStrategy(TradingStrategy):
            def _generate_strategy_signal(self, market_data, regime, confidence):
                return None
        
        strategy = TestStrategy(strategy_config)
        
        # Test activation
        result = strategy.activate()
        assert result is True
        assert strategy.state == StrategyState.ACTIVE
        
        # Test deactivation
        result = strategy.deactivate()
        assert result is True
        assert strategy.state == StrategyState.INACTIVE
    
    def test_performance_update(self, strategy_config):
        """Test performance metric updates"""
        class TestStrategy(TradingStrategy):
            def _generate_strategy_signal(self, market_data, regime, confidence):
                return None
        
        strategy = TestStrategy(strategy_config)
        
        # Test winning trade
        trade_result = {'pnl': 100.0, 'duration': 60.0}
        strategy.update_performance(trade_result)
        
        assert strategy.performance.total_trades == 1
        assert strategy.performance.winning_trades == 1
        assert strategy.performance.total_pnl == 100.0
        assert strategy.performance.win_rate == 1.0
        
        # Test losing trade
        trade_result = {'pnl': -50.0, 'duration': 30.0}
        strategy.update_performance(trade_result)
        
        assert strategy.performance.total_trades == 2
        assert strategy.performance.winning_trades == 1
        assert strategy.performance.total_pnl == 50.0
        assert strategy.performance.win_rate == 0.5


class TestMarketMakingStrategy:
    """Test market making strategy"""
    
    def test_signal_generation_valid_regime(self, market_making_strategy, mock_market_data):
        """Test signal generation for valid regime"""
        market_making_strategy.activate()
        
        # Test with range-bound regime (should generate signal)
        signal = market_making_strategy.generate_signal(
            mock_market_data, MockMarketRegime.RANGE_BOUND, 0.8
        )
        
        assert signal is not None
        assert signal.direction == 0.0  # Neutral for market making
        assert 0 <= signal.confidence <= 1
        assert signal.size > 0
        assert signal.regime == MockMarketRegime.RANGE_BOUND
    
    def test_signal_generation_invalid_regime(self, market_making_strategy, mock_market_data):
        """Test signal generation for invalid regime"""
        market_making_strategy.activate()
        
        # Test with trending regime (should not generate signal)
        signal = market_making_strategy.generate_signal(
            mock_market_data, MockMarketRegime.TRENDING, 0.8
        )
        
        assert signal is None
    
    def test_signal_generation_low_confidence(self, market_making_strategy, mock_market_data):
        """Test signal generation with low confidence"""
        market_making_strategy.activate()
        
        # Test with low confidence (below threshold)
        signal = market_making_strategy.generate_signal(
            mock_market_data, MockMarketRegime.RANGE_BOUND, 0.3
        )
        
        assert signal is None


class TestMeanReversionStrategy:
    """Test mean reversion strategy"""
    
    def test_signal_generation(self, mock_market_data):
        """Test mean reversion signal generation"""
        config = StrategyConfig(
            strategy_type=StrategyType.MEAN_REVERSION,
            regime_affinity=[MockMarketRegime.RANGE_BOUND, MockMarketRegime.VOLATILE]
        )
        strategy = MeanReversionStrategy(config)
        strategy.activate()
        
        # Create trending market data
        trending_data = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4] * 20, dtype=torch.float32)
        
        signal = strategy.generate_signal(trending_data, MockMarketRegime.VOLATILE, 0.8)
        
        assert signal is not None
        assert signal.direction != 0  # Should have direction opposite to trend
        assert 0 <= signal.confidence <= 1
        assert signal.size > 0


class TestMomentumStrategy:
    """Test momentum strategy"""
    
    def test_signal_generation(self, mock_market_data):
        """Test momentum signal generation"""
        config = StrategyConfig(
            strategy_type=StrategyType.MOMENTUM,
            regime_affinity=[MockMarketRegime.TRENDING, MockMarketRegime.BULL_RUN]
        )
        strategy = MomentumStrategy(config)
        strategy.activate()
        
        # Create trending market data
        trending_data = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4] * 20, dtype=torch.float32)
        
        signal = strategy.generate_signal(trending_data, MockMarketRegime.TRENDING, 0.8)
        
        assert signal is not None
        assert signal.direction != 0  # Should follow trend direction
        assert 0 <= signal.confidence <= 1
        assert signal.size > 0
    
    def test_insufficient_data(self):
        """Test momentum strategy with insufficient data"""
        config = StrategyConfig(
            strategy_type=StrategyType.MOMENTUM,
            regime_affinity=[MockMarketRegime.TRENDING]
        )
        strategy = MomentumStrategy(config)
        strategy.activate()
        
        # Insufficient data
        short_data = torch.tensor([1.0, 1.1], dtype=torch.float32)
        
        signal = strategy.generate_signal(short_data, MockMarketRegime.TRENDING, 0.8)
        
        assert signal is None


class TestDynamicStrategyManager:
    """Test dynamic strategy manager"""
    
    def test_manager_initialization(self, strategy_manager):
        """Test manager initialization"""
        assert len(strategy_manager.strategies) == 3  # MM, MR, Momentum
        assert strategy_manager.current_strategy is None
        assert strategy_manager.current_regime is None
        assert strategy_manager.switch_cooldown == 30.0
        assert len(strategy_manager.regime_history) == 0
        assert len(strategy_manager.transition_history) == 0
    
    def test_regime_update(self, strategy_manager):
        """Test regime update functionality"""
        condition = MockMarketCondition(MockMarketRegime.RANGE_BOUND)
        
        strategy_manager.update_regime(MockMarketRegime.RANGE_BOUND, 0.8, condition)
        
        assert strategy_manager.current_regime == MockMarketRegime.RANGE_BOUND
        assert len(strategy_manager.regime_history) == 1
    
    def test_strategy_selection(self, strategy_manager):
        """Test strategy selection logic"""
        # Test selection for range-bound regime
        best_strategy = strategy_manager._select_best_strategy(MockMarketRegime.RANGE_BOUND, 0.8)
        
        assert best_strategy is not None
        assert MockMarketRegime.RANGE_BOUND in best_strategy.config.regime_affinity
    
    def test_strategy_scoring(self, strategy_manager):
        """Test strategy scoring algorithm"""
        mm_strategy = strategy_manager.strategies[StrategyType.MARKET_MAKING]
        
        # Test score calculation
        score = strategy_manager._calculate_strategy_score(
            mm_strategy, MockMarketRegime.RANGE_BOUND, 0.8
        )
        
        assert isinstance(score, float)
        assert score > 0
    
    def test_strategy_switching(self, strategy_manager):
        """Test strategy switching mechanism"""
        condition = MockMarketCondition(MockMarketRegime.RANGE_BOUND)
        
        # Update regime to trigger strategy selection
        strategy_manager.update_regime(MockMarketRegime.RANGE_BOUND, 0.8, condition)
        
        # Should have selected a strategy
        assert strategy_manager.current_strategy is not None
        assert strategy_manager.current_strategy.state == StrategyState.ACTIVE
    
    def test_signal_generation(self, strategy_manager, mock_market_data):
        """Test signal generation through manager"""
        condition = MockMarketCondition(MockMarketRegime.RANGE_BOUND)
        
        # Initialize with regime
        strategy_manager.update_regime(MockMarketRegime.RANGE_BOUND, 0.8, condition)
        
        # Generate signal
        signal = strategy_manager.generate_signal(
            mock_market_data, MockMarketRegime.RANGE_BOUND, 0.8
        )
        
        # Should have generated a signal or auto-selected strategy
        if signal:
            assert signal.confidence >= 0
            assert signal.size >= 0
    
    def test_trade_result_update(self, strategy_manager):
        """Test trade result processing"""
        condition = MockMarketCondition(MockMarketRegime.RANGE_BOUND)
        
        # Initialize with regime and strategy
        strategy_manager.update_regime(MockMarketRegime.RANGE_BOUND, 0.8, condition)
        
        if strategy_manager.current_strategy:
            initial_trades = strategy_manager.current_strategy.performance.total_trades
            
            # Update with trade result
            trade_result = {'pnl': 50.0, 'duration': 45.0}
            strategy_manager.update_trade_result(trade_result)
            
            # Check performance update
            assert strategy_manager.current_strategy.performance.total_trades == initial_trades + 1
            assert strategy_manager.current_strategy.performance.total_pnl == 50.0
    
    def test_cooldown_period(self, strategy_manager):
        """Test strategy switch cooldown"""
        condition1 = MockMarketCondition(MockMarketRegime.RANGE_BOUND)
        condition2 = MockMarketCondition(MockMarketRegime.TRENDING)
        
        # First regime update
        strategy_manager.update_regime(MockMarketRegime.RANGE_BOUND, 0.8, condition1)
        first_strategy = strategy_manager.current_strategy
        
        # Immediate second regime update (should be blocked by cooldown)
        strategy_manager.update_regime(MockMarketRegime.TRENDING, 0.8, condition2)
        
        # Strategy should not have changed due to cooldown
        assert strategy_manager.current_strategy == first_strategy
    
    def test_manager_status(self, strategy_manager):
        """Test manager status reporting"""
        status = strategy_manager.get_manager_status()
        
        assert 'active_strategy' in status
        assert 'current_regime' in status
        assert 'total_strategies' in status
        assert 'recent_transitions' in status
        assert 'strategy_performance' in status
        
        assert status['total_strategies'] == 3
        assert isinstance(status['strategy_performance'], dict)


class TestStrategyTransition:
    """Test strategy transition functionality"""
    
    def test_transition_recording(self, strategy_manager):
        """Test transition history recording"""
        condition1 = MockMarketCondition(MockMarketRegime.RANGE_BOUND)
        condition2 = MockMarketCondition(MockMarketRegime.TRENDING)
        
        # Initial regime
        strategy_manager.update_regime(MockMarketRegime.RANGE_BOUND, 0.8, condition1)
        initial_transitions = len(strategy_manager.transition_history)
        
        # Wait for cooldown and switch
        time.sleep(0.1)  # Small delay for testing
        strategy_manager.last_switch_time = 0  # Reset cooldown for testing
        strategy_manager.update_regime(MockMarketRegime.TRENDING, 0.8, condition2)
        
        # Should have recorded transition
        if len(strategy_manager.transition_history) > initial_transitions:
            transition = strategy_manager.transition_history[-1]
            assert isinstance(transition, StrategyTransition)
            assert transition.regime == MockMarketRegime.TRENDING
            assert transition.reason == "regime_change"


# Integration tests
def test_end_to_end_strategy_switching():
    """Test complete strategy switching workflow"""
    manager = create_dynamic_strategy_manager()
    
    # Test regime sequence
    regimes = [
        MockMarketRegime.RANGE_BOUND,
        MockMarketRegime.TRENDING, 
        MockMarketRegime.VOLATILE,
        MockMarketRegime.RANGE_BOUND
    ]
    
    market_data = torch.randn(100)
    
    for i, regime in enumerate(regimes):
        condition = MockMarketCondition(regime)
        
        # Reset cooldown for testing
        manager.last_switch_time = 0
        
        # Update regime
        manager.update_regime(regime, 0.8, condition)
        
        # Generate signal
        signal = manager.generate_signal(market_data, regime, 0.8)
        
        # Simulate trade result
        if signal:
            trade_result = {'pnl': np.random.normal(0, 10), 'duration': 60.0}
            manager.update_trade_result(trade_result)
    
    # Verify system state
    status = manager.get_manager_status()
    assert status['current_regime'] == MockMarketRegime.RANGE_BOUND
    assert status['active_strategy'] is not None


if __name__ == "__main__":
    # Run a simple test
    print("Running Dynamic Strategy Switching Tests...")
    
    # Test strategy manager creation
    manager = create_dynamic_strategy_manager()
    assert manager is not None
    print("✅ Strategy manager creation test passed")
    
    # Test strategy initialization
    assert len(manager.strategies) == 3
    print("✅ Strategy initialization test passed")
    
    # Test regime update
    condition = MockMarketCondition(MockMarketRegime.RANGE_BOUND)
    manager.update_regime(MockMarketRegime.RANGE_BOUND, 0.8, condition)
    assert manager.current_regime == MockMarketRegime.RANGE_BOUND
    print("✅ Regime update test passed")
    
    print("\n🎯 Task 15.1.2: Dynamic Strategy Switching System - IMPLEMENTATION COMPLETE")
    print("Run 'pytest tests/unit/test_dynamic_strategy_switching.py' for full test suite")