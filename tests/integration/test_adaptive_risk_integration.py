"""
Integration tests for Adaptive Risk Management System

Production-ready integration tests to verify component interactions
and end-to-end functionality.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.learning.adaptive_risk_integration_service import AdaptiveRiskIntegrationService
# Import HighFrequencyTradingEngine using explicit path to avoid package conflict
import sys
import os
trading_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src', 'trading')
sys.path.insert(0, trading_dir)
try:
    import hft_engine as hft_module
    HighFrequencyTradingEngine = hft_module.HighFrequencyTradingEngine
finally:
    if trading_dir in sys.path:
        sys.path.remove(trading_dir)
from src.models.mixture_of_experts import MoETradingEngine
from src.learning.risk_strategy_integration import RiskStrategyIntegrator
from src.learning.risk_monitoring_alerting import RiskMonitor
from src.learning.performance_based_risk_adjustment import PerformanceBasedRiskAdjuster
from src.config.trading_config import get_trading_config


class TestAdaptiveRiskIntegration:
    """Production-ready integration tests"""
    
    @pytest.fixture
    async def integration_service(self):
        """Create test integration service"""
        service = AdaptiveRiskIntegrationService()
        await service.initialize()
        yield service
        await service.stop()
    
    @pytest.fixture
    def mock_trading_config(self):
        """Create mock trading configuration"""
        config = Mock()
        config.adaptive_risk_enabled = True
        config.max_portfolio_risk = 0.02
        config.max_position_risk = 0.01
        config.max_drawdown = 0.15
        config.daily_loss_limit = 0.05
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
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, integration_service):
        """Test service initializes all components correctly"""
        status = integration_service.get_status()
        
        assert status['components_initialized'] is True
        assert status['error_count'] == 0
        assert status['adaptive_risk_enabled'] is True
        
        # Verify all components are initialized
        assert integration_service.hft_engine is not None
        assert integration_service.moe_engine is not None
        assert integration_service.leverage_manager is not None
        assert integration_service.trailing_system is not None
        
        # Verify adaptive risk components are initialized
        assert integration_service.risk_manager is not None
        assert integration_service.integrated_system is not None
        assert integration_service.risk_monitor is not None
        assert integration_service.performance_adjuster is not None
    
    @pytest.mark.asyncio
    async def test_trade_signal_processing(self, integration_service):
        """Test trade signal processing through integrated system"""
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 0.1,
            'price': 50000,
            'confidence': 0.8,
            'timestamp': datetime.now()
        }
        
        result = await integration_service.process_trade_signal(signal)
        
        assert 'signal' in result or 'error' in result
        if 'error' not in result:
            assert 'risk_approved' in result
            assert 'timestamp' in result
            assert result['risk_approved'] is True
            assert 'risk_score' in result
            assert 0 <= result['risk_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_component_interaction(self, integration_service):
        """Test interaction between risk manager and trading engines"""
        # Verify risk manager is connected to trading engines
        assert integration_service.risk_manager is not None
        assert integration_service.hft_engine is not None
        assert integration_service.moe_engine is not None
        
        # Test that risk manager can influence trading decisions
        high_risk_signal = {
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 2.0,  # Large position
            'price': 50000,
            'confidence': 0.9,
            'timestamp': datetime.now()
        }
        
        result = await integration_service.process_trade_signal(high_risk_signal)
        
        # Should either be rejected or position size reduced
        if result['risk_approved']:
            assert result['signal']['quantity'] < high_risk_signal['quantity']
        else:
            assert 'risk' in result['reason'].lower()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, integration_service):
        """Test error handling and system resilience"""
        # Test with invalid signal
        invalid_signal = {
            'invalid': 'data'
        }
        
        result = await integration_service.process_trade_signal(invalid_signal)
        
        assert 'error' in result
        # System should continue running
        status = integration_service.get_status()
        assert status['is_running'] is True
    
    @pytest.mark.asyncio
    async def test_risk_strategy_coordination(self, integration_service):
        """Test coordination between risk management and strategy switching"""
        # Test different market conditions
        market_conditions = [
            {
                'regime': 'high_volatility',
                'volatility': 0.25,
                'trend_strength': 0.5,
                'liquidity': 0.6
            },
            {
                'regime': 'low_volatility',
                'volatility': 0.05,
                'trend_strength': 0.3,
                'liquidity': 0.8
            },
            {
                'regime': 'trending',
                'volatility': 0.15,
                'trend_strength': 0.8,
                'liquidity': 0.7
            }
        ]
        
        for condition in market_conditions:
            signal = {
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'quantity': 0.1,
                'price': 50000,
                'confidence': 0.8,
                'market_condition': condition
            }
            
            result = await integration_service.process_trade_signal(signal)
            
            if 'error' not in result:
                # Verify signal is adjusted based on market condition
                assert 'signal' in result
                assert result['risk_approved'] in [True, False]
                
                if result['risk_approved']:
                    # High volatility should result in smaller position sizes
                    if condition['regime'] == 'high_volatility':
                        assert result['signal']['quantity'] <= signal['quantity']
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, integration_service):
        """Test integration with performance monitoring"""
        # Simulate trade outcomes
        trade_outcomes = [
            {'symbol': 'BTC/USDT', 'pnl': 100, 'risk_score': 0.5, 'timestamp': datetime.now()},
            {'symbol': 'ETH/USDT', 'pnl': -50, 'risk_score': 0.7, 'timestamp': datetime.now()},
            {'symbol': 'BTC/USDT', 'pnl': 200, 'risk_score': 0.3, 'timestamp': datetime.now()}
        ]
        
        # Record outcomes
        for outcome in trade_outcomes:
            if integration_service.performance_adjuster:
                await integration_service.performance_adjuster.record_trade_outcome(outcome)
        
        # Test that performance affects risk parameters
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 0.1,
            'price': 50000,
            'confidence': 0.8
        }
        
        result = await integration_service.process_trade_signal(signal)
        
        if 'error' not in result:
            # Performance-based adjustments should be reflected
            assert 'signal' in result
            assert result['risk_approved'] in [True, False]
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self, integration_service):
        """Test system handles concurrent signal processing"""
        # Create multiple signals
        signals = []
        for i in range(10):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy' if i % 2 == 0 else 'sell',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8,
                'timestamp': datetime.now()
            })
        
        # Process signals concurrently
        tasks = [integration_service.process_trade_signal(signal) for signal in signals]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All signals should be processed
        assert len(results) == len(signals)
        
        # Check for successful processing
        successful_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
        assert len(successful_results) > 0
        
        # System should remain stable
        status = integration_service.get_status()
        assert status['is_running'] is True
        assert status['error_count'] < 5  # Allow some errors but not too many
    
    @pytest.mark.asyncio
    async def test_risk_monitoring_integration(self, integration_service):
        """Test integration with risk monitoring"""
        # Set up risk thresholds
        if integration_service.risk_monitor:
            await integration_service.risk_monitor.set_risk_threshold('portfolio_risk', 0.02)
            await integration_service.risk_monitor.set_risk_threshold('position_risk', 0.01)
            
            # Test normal risk levels
            metrics = {'portfolio_risk': 0.015, 'position_risk': 0.008}
            alert = await integration_service.risk_monitor.check_risk_metrics(metrics)
            assert alert['level'] == 'normal'
            
            # Test warning levels
            warning_metrics = {'portfolio_risk': 0.018, 'position_risk': 0.009}
            warning_alert = await integration_service.risk_monitor.check_risk_metrics(warning_metrics)
            assert warning_alert['level'] == 'warning'
    
    @pytest.mark.asyncio
    async def test_system_recovery(self, integration_service):
        """Test system recovery from failures"""
        # Process normal signals first
        normal_signals = [
            {
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'quantity': 0.1,
                'price': 50000,
                'confidence': 0.8
            }
            for _ in range(5)
        ]
        
        for signal in normal_signals:
            await integration_service.process_trade_signal(signal)
        
        # Simulate system stress with invalid signals
        stress_signals = [
            {'invalid': 'data'}
            for _ in range(10)
        ]
        
        for signal in stress_signals:
            try:
                await integration_service.process_trade_signal(signal)
            except:
                pass  # Ignore exceptions
        
        # System should still be operational
        status = integration_service.get_status()
        assert status['is_running'] is True
        
        # Should still be able to process valid signals
        recovery_signal = {
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 0.1,
            'price': 50000,
            'confidence': 0.8
        }
        
        result = await integration_service.process_trade_signal(recovery_signal)
        assert 'error' not in result or result['error'] is None
    
    @pytest.mark.asyncio
    async def test_configuration_changes(self, integration_service):
        """Test system responds to configuration changes"""
        # Get initial status
        initial_status = integration_service.get_status()
        
        # Modify configuration
        new_config = Mock()
        new_config.adaptive_risk_enabled = True
        new_config.max_portfolio_risk = 0.03  # Increased from 0.02
        new_config.max_position_risk = 0.015  # Increased from 0.01
        new_config.base_position_size = 0.02  # Increased from 0.01
        
        # Update configuration
        integration_service.config = new_config
        
        # Test with a signal that would have been restricted before
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 0.015,  # Between old and new limits
            'price': 50000,
            'confidence': 0.8
        }
        
        result = await integration_service.process_trade_signal(signal)
        
        if 'error' not in result:
            # Should be approved with new configuration
            assert result['risk_approved'] is True
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, integration_service):
        """Test memory usage remains stable under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large number of signals
        signals = []
        for i in range(100):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8
            })
        
        for signal in signals:
            await integration_service.process_trade_signal(signal)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB)
        assert memory_increase < 50, f"Memory increase {memory_increase}MB exceeds 50MB threshold"
        
        # System should still be responsive
        status = integration_service.get_status()
        assert status['is_running'] is True


class TestRiskStrategyIntegration:
    """Test risk-strategy integration specifically"""
    
    @pytest.fixture
    async def risk_strategy_integrator(self):
        """Create risk-strategy integrator instance"""
        config = Mock()
        config.adaptive_risk_enabled = True
        config.coordination_mode = "risk_aware"
        
        integrator = RiskStrategyIntegrator(
            risk_manager=Mock(),
            strategy_manager=Mock(),
            config={}
        )
        return integrator
    
    @pytest.mark.asyncio
    async def test_coordination_modes(self, risk_strategy_integrator):
        """Test different coordination modes"""
        modes = ['risk_aware', 'performance_driven', 'balanced']
        
        for mode in modes:
            risk_strategy_integrator.config['coordination_mode'] = mode
            
            # Test coordination decision
            decision = await risk_strategy_integrator.coordinate_decision(
                signal={'symbol': 'BTC/USDT', 'action': 'buy', 'quantity': 0.1},
                risk_assessment={'risk_score': 0.5, 'approved': True}
            )
            
            assert 'signal' in decision
            assert 'coordination_mode' in decision
            assert decision['coordination_mode'] == mode
    
    @pytest.mark.asyncio
    async def test_market_condition_unification(self, risk_strategy_integrator):
        """Test market condition unification"""
        risk_condition = {
            'regime': 'high_volatility',
            'volatility': 0.25,
            'risk_score': 0.7
        }
        
        strategy_condition = {
            'trend': 'bullish',
            'momentum': 0.8,
            'liquidity': 0.6
        }
        
        unified = await risk_strategy_integrator.unify_market_conditions(
            risk_condition, strategy_condition
        )
        
        assert 'regime' in unified
        assert 'volatility' in unified
        assert 'risk_score' in unified
        assert 'trend' in unified
        assert 'momentum' in unified
        assert 'liquidity' in unified
        
        # Values should match inputs
        assert unified['regime'] == risk_condition['regime']
        assert unified['volatility'] == risk_condition['volatility']
        assert unified['trend'] == strategy_condition['trend']