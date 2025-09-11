"""
Unit tests for Adaptive Risk Management System

Production-ready unit tests for the core adaptive risk management functionality.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any

from src.learning.adaptive_risk_management import (
    AdaptiveRiskManager,
    RiskLevel,
    MarketRegime,
    RiskLimits,
    MarketCondition,
    RiskProfile,
    PortfolioRiskMetrics,
    VolatilityEstimator,
    PositionSizer,
    RiskMonitor,
    create_adaptive_risk_manager
)


class TestAdaptiveRiskManager:
    """Production-ready unit tests for AdaptiveRiskManager"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing"""
        config = Mock()
        config.max_position_size = 1.0
        config.max_daily_loss = 0.05
        config.max_drawdown = 0.15
        config.max_portfolio_risk = 0.02
        config.max_position_risk = 0.01
        config.volatility_threshold = 0.02
        config.base_position_size = 0.01
        config.volatility_multiplier = 1.0
        config.confidence_threshold = 0.7
        config.max_leverage = 3.0
        config.enable_regime_detection = True
        config.regime_update_interval = 300
        config.regime_confidence_threshold = 0.8
        config.enable_performance_adjustment = True
        config.learning_rate = 0.01
        config.min_trades_for_learning = 50
        config.performance_window = 100
        config.enable_risk_monitoring = True
        config.monitoring_interval = 60
        config.alert_threshold_warning = 0.8
        config.alert_threshold_critical = 0.9
        config.volatility_window = 100
        config.volatility_method = "historical"
        config.garch_p = 1
        config.garch_q = 1
        config.enable_strategy_integration = True
        config.coordination_mode = "risk_aware"
        config.enable_dynamic_leverage = True
        return config
    
    @pytest.fixture
    def risk_manager(self, mock_config):
        """Create test risk manager instance"""
        return create_adaptive_risk_manager(mock_config)
    
    @pytest.mark.asyncio
    async def test_risk_assessment_market_conditions(self, risk_manager):
        """Test risk assessment under different market conditions"""
        # Test high volatility regime
        market_data = {
            'volatility': 0.25,
            'trend_strength': 0.8,
            'liquidity': 0.6,
            'price': 50000,
            'volume': 1000,
            'timestamp': datetime.now()
        }
        
        assessment = await risk_manager.assess_market_risk(market_data)
        
        assert assessment['risk_level'] in [r.value for r in RiskLevel]
        assert assessment['regime'] in [r.value for r in MarketRegime]
        assert 'risk_score' in assessment
        assert 0 <= assessment['risk_score'] <= 1
        assert 'confidence' in assessment
        assert assessment['confidence'] > 0
        
        # Test low volatility regime
        low_vol_data = market_data.copy()
        low_vol_data['volatility'] = 0.05
        
        low_vol_assessment = await risk_manager.assess_market_risk(low_vol_data)
        
        # Low volatility should generally result in lower risk scores
        assert low_vol_assessment['risk_score'] < assessment['risk_score']
    
    @pytest.mark.asyncio
    async def test_position_sizing_limits(self, risk_manager):
        """Test position sizing respects risk limits"""
        # Test with high risk score
        requested_size = 2.0  # Larger than max
        risk_score = 0.9
        
        adjusted_size = risk_manager.calculate_position_size(requested_size, risk_score)
        
        assert adjusted_size <= risk_manager.config.max_position_size
        assert adjusted_size > 0
        assert adjusted_size < requested_size
        
        # Test with low risk score
        low_risk_score = 0.1
        low_risk_adjusted_size = risk_manager.calculate_position_size(requested_size, low_risk_score)
        
        # Lower risk should allow larger position size
        assert low_risk_adjusted_size > adjusted_size
        
        # Test with zero risk score
        zero_risk_size = risk_manager.calculate_position_size(requested_size, 0.0)
        assert zero_risk_size == requested_size
    
    @pytest.mark.asyncio
    async def test_trade_risk_approval(self, risk_manager):
        """Test trade risk approval logic"""
        # Test normal trade
        trade_request = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 0.5,
            'price': 50000,
            'confidence': 0.8
        }
        
        assessment = await risk_manager.assess_trade_risk(trade_request)
        
        assert 'approved' in assessment
        assert 'reason' in assessment
        assert 'risk_score' in assessment
        assert 'position_size' in assessment
        
        # Test oversized trade
        oversized_trade = trade_request.copy()
        oversized_trade['quantity'] = 10.0  # Very large
        
        oversized_assessment = await risk_manager.assess_trade_risk(oversized_trade)
        
        # Should be rejected or heavily adjusted
        if oversized_assessment['approved']:
            assert oversized_assessment['position_size'] < oversized_trade['quantity']
        else:
            assert 'oversized' in oversized_assessment['reason'].lower()
        
        # Test low confidence trade
        low_conf_trade = trade_request.copy()
        low_conf_trade['confidence'] = 0.3
        
        low_conf_assessment = await risk_manager.assess_trade_risk(low_conf_trade)
        
        # Should be rejected due to low confidence
        assert not low_conf_assessment['approved']
        assert 'confidence' in low_conf_assessment['reason'].lower()
    
    def test_health_check(self, risk_manager):
        """Test system health check"""
        health = risk_manager.health_check()
        
        assert 'healthy' in health
        assert 'components' in health
        assert 'timestamp' in health
        assert isinstance(health['healthy'], bool)
        assert isinstance(health['components'], dict)
        assert isinstance(health['timestamp'], datetime)
        
        # Check components status
        components = health['components']
        assert 'volatility_estimator' in components
        assert 'position_sizer' in components
        assert 'risk_monitor' in components
    
    @pytest.mark.asyncio
    async def test_risk_parameter_updates(self, risk_manager):
        """Test dynamic risk parameter updates"""
        original_max_position = risk_manager.config.max_position_size
        original_volatility_threshold = risk_manager.config.volatility_threshold
        
        new_params = {
            'max_position_size': 0.8,
            'volatility_threshold': 0.2
        }
        
        await risk_manager.update_risk_parameters(new_params)
        
        assert risk_manager.config.max_position_size == 0.8
        assert risk_manager.config.volatility_threshold == 0.2
        
        # Test that changes take effect
        trade_request = {
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'quantity': 1.0,
            'price': 50000
        }
        
        assessment = await risk_manager.assess_trade_risk(trade_request)
        
        # Should respect new max position size
        if assessment['approved']:
            assert assessment['position_size'] <= 0.8
    
    @pytest.mark.asyncio
    async def test_market_regime_detection(self, risk_manager):
        """Test market regime detection functionality"""
        # Test trending market
        trending_data = {
            'volatility': 0.15,
            'trend_strength': 0.9,
            'liquidity': 0.7,
            'price_change': 0.05,
            'volume_change': 0.3
        }
        
        regime_result = await risk_manager.detect_market_regime(trending_data)
        
        assert regime_result['regime'] in [r.value for r in MarketRegime]
        assert 'confidence' in regime_result
        assert 0 <= regime_result['confidence'] <= 1
        
        # Test ranging market
        ranging_data = {
            'volatility': 0.08,
            'trend_strength': 0.2,
            'liquidity': 0.6,
            'price_change': 0.01,
            'volume_change': 0.05
        }
        
        ranging_result = await risk_manager.detect_market_regime(ranging_data)
        
        # Should detect different regime
        if regime_result['confidence'] > 0.7 and ranging_result['confidence'] > 0.7:
            assert regime_result['regime'] != ranging_result['regime']
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_calculation(self, risk_manager):
        """Test portfolio risk calculation"""
        positions = [
            {'symbol': 'BTC/USDT', 'size': 0.5, 'price': 50000, 'volatility': 0.2},
            {'symbol': 'ETH/USDT', 'size': 2.0, 'price': 3000, 'volatility': 0.15}
        ]
        
        portfolio_risk = await risk_manager.calculate_portfolio_risk(positions)
        
        assert 'total_risk' in portfolio_risk
        assert 'position_risks' in portfolio_risk
        assert 'correlation_risk' in portfolio_risk
        assert 'concentration_risk' in portfolio_risk
        
        assert portfolio_risk['total_risk'] >= 0
        assert len(portfolio_risk['position_risks']) == len(positions)
        
        # Test with empty portfolio
        empty_risk = await risk_manager.calculate_portfolio_risk([])
        assert empty_risk['total_risk'] == 0
        assert len(empty_risk['position_risks']) == 0
    
    @pytest.mark.asyncio
    async def test_volatility_estimation(self, risk_manager):
        """Test volatility estimation methods"""
        # Generate test price data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)  # 100 days of returns
        
        volatility_result = await risk_manager.estimate_volatility(returns)
        
        assert 'volatility' in volatility_result
        assert 'method' in volatility_result
        assert 'confidence' in volatility_result
        
        assert volatility_result['volatility'] > 0
        assert volatility_result['method'] in ['historical', 'garch', 'ewma']
        assert 0 <= volatility_result['confidence'] <= 1
        
        # Test with different methods
        for method in ['historical', 'ewma']:
            method_result = await risk_manager.estimate_volatility(returns, method=method)
            assert method_result['method'] == method
            assert method_result['volatility'] > 0
    
    @pytest.mark.asyncio
    async def test_risk_limit_enforcement(self, risk_manager):
        """Test risk limit enforcement"""
        # Test portfolio risk limit
        positions = [
            {'symbol': 'BTC/USDT', 'size': 1.0, 'price': 50000, 'volatility': 0.25},
            {'symbol': 'ETH/USDT', 'size': 5.0, 'price': 3000, 'volatility': 0.2}
        ]
        
        limit_check = await risk_manager.check_risk_limits(positions)
        
        assert 'within_limits' in limit_check
        assert 'breached_limits' in limit_check
        assert 'risk_level' in limit_check
        
        # Test with positions that exceed limits
        excessive_positions = [
            {'symbol': 'BTC/USDT', 'size': 5.0, 'price': 50000, 'volatility': 0.3},
            {'symbol': 'ETH/USDT', 'size': 20.0, 'price': 3000, 'volatility': 0.25}
        ]
        
        excessive_check = await risk_manager.check_risk_limits(excessive_positions)
        
        # Should exceed limits
        assert not excessive_check['within_limits']
        assert len(excessive_check['breached_limits']) > 0
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, risk_manager):
        """Test performance tracking for risk adjustment"""
        # Simulate trade outcomes
        trade_outcomes = [
            {'symbol': 'BTC/USDT', 'pnl': 100, 'risk_score': 0.5, 'timestamp': datetime.now()},
            {'symbol': 'ETH/USDT', 'pnl': -50, 'risk_score': 0.7, 'timestamp': datetime.now()},
            {'symbol': 'BTC/USDT', 'pnl': 200, 'risk_score': 0.3, 'timestamp': datetime.now()}
        ]
        
        for outcome in trade_outcomes:
            await risk_manager.record_trade_outcome(outcome)
        
        performance_metrics = await risk_manager.get_performance_metrics()
        
        assert 'total_trades' in performance_metrics
        assert 'win_rate' in performance_metrics
        assert 'avg_risk_score' in performance_metrics
        assert 'risk_adjusted_return' in performance_metrics
        
        assert performance_metrics['total_trades'] == len(trade_outcomes)
        assert 0 <= performance_metrics['win_rate'] <= 1
        assert performance_metrics['avg_risk_score'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, risk_manager):
        """Test error handling in various scenarios"""
        # Test with invalid market data
        invalid_data = {'invalid': 'data'}
        
        with pytest.raises(Exception):
            await risk_manager.assess_market_risk(invalid_data)
        
        # Test with invalid trade request
        invalid_trade = {'invalid': 'request'}
        
        result = await risk_manager.assess_trade_risk(invalid_trade)
        assert not result['approved']
        assert 'error' in result['reason'].lower()
        
        # Test with empty position list
        empty_portfolio_risk = await risk_manager.calculate_portfolio_risk([])
        assert empty_portfolio_risk['total_risk'] == 0
        
        # Test health check during error conditions
        health = risk_manager.health_check()
        assert 'healthy' in health
        # System should remain healthy despite individual errors


class TestVolatilityEstimator:
    """Test VolatilityEstimator component"""
    
    @pytest.fixture
    def estimator(self):
        """Create volatility estimator instance"""
        config = Mock()
        config.volatility_window = 100
        config.volatility_method = "historical"
        config.garch_p = 1
        config.garch_q = 1
        return VolatilityEstimator(config)
    
    @pytest.mark.asyncio
    async def test_historical_volatility(self, estimator):
        """Test historical volatility calculation"""
        # Generate test data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        
        volatility = await estimator.calculate_historical_volatility(returns)
        
        assert volatility > 0
        assert volatility < 1  # Should be reasonable
        
        # Test with different window sizes
        short_window_vol = await estimator.calculate_historical_volatility(returns[:20])
        long_window_vol = await estimator.calculate_historical_volatility(returns)
        
        # Volatility should be more stable with longer window
        assert abs(short_window_vol - 0.02) > abs(long_window_vol - 0.02)
    
    @pytest.mark.asyncio
    async def test_ewma_volatility(self, estimator):
        """Test EWMA volatility calculation"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        
        volatility = await estimator.calculate_ewma_volatility(returns)
        
        assert volatility > 0
        assert volatility < 1
        
        # Test with different lambda values
        high_lambda_vol = await estimator.calculate_ewma_volatility(returns, lambda_param=0.99)
        low_lambda_vol = await estimator.calculate_ewma_volatility(returns, lambda_param=0.94)
        
        # Different lambda should give different results
        assert high_lambda_vol != low_lambda_vol


class TestPositionSizer:
    """Test PositionSizer component"""
    
    @pytest.fixture
    def sizer(self):
        """Create position sizer instance"""
        config = Mock()
        config.base_position_size = 0.01
        config.volatility_multiplier = 1.0
        config.confidence_threshold = 0.7
        config.max_leverage = 3.0
        return PositionSizer(config)
    
    def test_kelly_position_sizing(self, sizer):
        """Test Kelly criterion position sizing"""
        win_rate = 0.6
        avg_win = 1.0
        avg_loss = 0.8
        
        kelly_fraction = sizer.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        assert kelly_fraction > 0
        assert kelly_fraction < 1
        
        # Test with edge cases
        zero_win_rate = sizer.calculate_kelly_fraction(0, avg_win, avg_loss)
        assert zero_win_rate == 0
        
        perfect_win_rate = sizer.calculate_kelly_fraction(1.0, avg_win, avg_loss)
        assert perfect_win_rate > 0
    
    def test_volatility_adjusted_sizing(self, sizer):
        """Test volatility-adjusted position sizing"""
        base_size = 0.01
        volatility = 0.02
        
        adjusted_size = sizer.adjust_for_volatility(base_size, volatility)
        
        assert adjusted_size > 0
        assert adjusted_size <= base_size * sizer.config.max_leverage
        
        # Test with high volatility
        high_vol_size = sizer.adjust_for_volatility(base_size, 0.1)
        low_vol_size = sizer.adjust_for_volatility(base_size, 0.005)
        
        # High volatility should result in smaller position
        assert high_vol_size < low_vol_size


class TestRiskMonitor:
    """Test RiskMonitor component"""
    
    @pytest.fixture
    def monitor(self):
        """Create risk monitor instance"""
        config = Mock()
        config.monitoring_interval = 60
        config.alert_threshold_warning = 0.8
        config.alert_threshold_critical = 0.9
        return RiskMonitor(config)
    
    @pytest.mark.asyncio
    async def test_risk_threshold_monitoring(self, monitor):
        """Test risk threshold monitoring"""
        # Set up risk thresholds
        await monitor.set_risk_threshold('portfolio_risk', 0.02)
        await monitor.set_risk_threshold('position_risk', 0.01)
        
        # Test normal risk level
        normal_metrics = {'portfolio_risk': 0.015, 'position_risk': 0.008}
        alert = await monitor.check_risk_metrics(normal_metrics)
        
        assert alert['level'] == 'normal'
        assert len(alert['breached_thresholds']) == 0
        
        # Test warning level
        warning_metrics = {'portfolio_risk': 0.017, 'position_risk': 0.009}
        warning_alert = await monitor.check_risk_metrics(warning_metrics)
        
        assert warning_alert['level'] == 'warning'
        assert len(warning_alert['breached_thresholds']) > 0
        
        # Test critical level
        critical_metrics = {'portfolio_risk': 0.025, 'position_risk': 0.012}
        critical_alert = await monitor.check_risk_metrics(critical_metrics)
        
        assert critical_alert['level'] == 'critical'
        assert len(critical_alert['breached_thresholds']) > 0
    
    @pytest.mark.asyncio
    async def test_drawdown_monitoring(self, monitor):
        """Test drawdown monitoring"""
        equity_curve = [100000, 99000, 98000, 97000, 96000, 97000, 98000]
        
        drawdown_analysis = await monitor.analyze_drawdown(equity_curve)
        
        assert 'max_drawdown' in drawdown_analysis
        assert 'current_drawdown' in drawdown_analysis
        assert 'drawdown_duration' in drawdown_analysis
        assert 'recovery_time' in drawdown_analysis
        
        assert drawdown_analysis['max_drawdown'] > 0
        assert drawdown_analysis['max_drawdown'] <= 0.04  # 4% max drawdown
        assert drawdown_analysis['current_drawdown'] >= 0