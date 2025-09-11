"""
Performance tests for Adaptive Risk Management System

Production-ready performance tests to ensure system meets latency
and throughput requirements.
"""

import pytest
import asyncio
import time
import statistics
import psutil
import os
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.learning.adaptive_risk_integration_service import AdaptiveRiskIntegrationService
from src.learning.adaptive_risk_management import create_adaptive_risk_manager
from src.config.trading_config import get_trading_config


class TestAdaptiveRiskPerformance:
    """Production-ready performance tests"""
    
    @pytest.fixture
    async def integration_service(self):
        """Create test integration service"""
        service = AdaptiveRiskIntegrationService()
        await service.initialize()
        await service.start()
        yield service
        await service.stop()
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
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
    async def test_risk_assessment_latency(self, integration_service):
        """Test risk assessment meets latency requirements"""
        signals = []
        for i in range(100):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy' if i % 2 == 0 else 'sell',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8
            })
        
        latencies = []
        for signal in signals:
            start_time = time.time()
            result = await integration_service.process_trade_signal(signal)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        # Production requirements
        assert avg_latency < 50, f"Average latency {avg_latency:.2f}ms exceeds 50ms threshold"
        assert max_latency < 100, f"Max latency {max_latency:.2f}ms exceeds 100ms threshold"
        assert p95_latency < 75, f"95th percentile latency {p95_latency:.2f}ms exceeds 75ms threshold"
        assert p99_latency < 90, f"99th percentile latency {p99_latency:.2f}ms exceeds 90ms threshold"
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self, integration_service):
        """Test system handles concurrent signal processing"""
        signals = []
        for i in range(50):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8
            })
        
        # Process signals concurrently
        tasks = [integration_service.process_trade_signal(signal) for signal in signals]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(signals) / total_time
        
        # Production requirements
        assert throughput > 100, f"Throughput {throughput:.2f} signals/sec below 100 threshold"
        
        # All signals should be processed successfully
        successful_results = [r for r in results if 'error' not in r]
        assert len(successful_results) == len(signals), f"Only {len(successful_results)}/{len(signals)} signals processed successfully"
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, integration_service):
        """Test memory usage remains stable under load"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large number of signals
        signals = []
        for i in range(1000):
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
        
        # Production requirements
        assert memory_increase < 50, f"Memory increase {memory_increase:.2f}MB exceeds 50MB threshold"
        
        # Check for memory leaks by forcing garbage collection
        import gc
        gc.collect()
        post_gc_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should decrease after GC
        assert post_gc_memory <= final_memory, "Memory did not decrease after garbage collection"
    
    @pytest.mark.asyncio
    async def test_volatility_calculation_performance(self, mock_config):
        """Test volatility calculation performance"""
        risk_manager = create_adaptive_risk_manager(mock_config)
        
        # Generate test data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        # Test different volatility methods
        methods = ['historical', 'ewma']
        latencies = {}
        
        for method in methods:
            method_latencies = []
            for _ in range(100):
                start_time = time.time()
                result = await risk_manager.estimate_volatility(returns, method=method)
                end_time = time.time()
                method_latencies.append((end_time - start_time) * 1000)  # ms
            
            latencies[method] = method_latencies
        
        # Check performance for each method
        for method, method_latencies in latencies.items():
            avg_latency = statistics.mean(method_latencies)
            max_latency = max(method_latencies)
            
            assert avg_latency < 10, f"{method} average latency {avg_latency:.2f}ms exceeds 10ms threshold"
            assert max_latency < 50, f"{method} max latency {max_latency:.2f}ms exceeds 50ms threshold"
    
    @pytest.mark.asyncio
    async def test_portfolio_risk_calculation_performance(self, mock_config):
        """Test portfolio risk calculation performance"""
        risk_manager = create_adaptive_risk_manager(mock_config)
        
        # Create test positions
        positions = []
        for i in range(50):
            positions.append({
                'symbol': f'CRYPTO_{i}',
                'size': 0.1 + i * 0.01,
                'price': 50000 + i * 100,
                'volatility': 0.15 + (i % 10) * 0.01
            })
        
        latencies = []
        for _ in range(100):
            start_time = time.time()
            result = await risk_manager.calculate_portfolio_risk(positions)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # ms
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        # Portfolio risk calculation should be fast
        assert avg_latency < 20, f"Portfolio risk avg latency {avg_latency:.2f}ms exceeds 20ms threshold"
        assert max_latency < 100, f"Portfolio risk max latency {max_latency:.2f}ms exceeds 100ms threshold"
    
    @pytest.mark.asyncio
    async def test_market_regime_detection_performance(self, mock_config):
        """Test market regime detection performance"""
        risk_manager = create_adaptive_risk_manager(mock_config)
        
        # Create test market data
        market_conditions = []
        for i in range(100):
            market_conditions.append({
                'volatility': 0.1 + (i % 10) * 0.02,
                'trend_strength': 0.3 + (i % 7) * 0.1,
                'liquidity': 0.5 + (i % 5) * 0.1,
                'price_change': 0.01 + (i % 3) * 0.02,
                'volume_change': 0.05 + (i % 4) * 0.1
            })
        
        latencies = []
        for condition in market_conditions:
            start_time = time.time()
            result = await risk_manager.detect_market_regime(condition)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # ms
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        # Regime detection should be very fast
        assert avg_latency < 5, f"Regime detection avg latency {avg_latency:.2f}ms exceeds 5ms threshold"
        assert max_latency < 20, f"Regime detection max latency {max_latency:.2f}ms exceeds 20ms threshold"
    
    @pytest.mark.asyncio
    async def test_position_sizing_performance(self, mock_config):
        """Test position sizing performance"""
        risk_manager = create_adaptive_risk_manager(mock_config)
        
        # Test various position sizing scenarios
        test_cases = []
        for i in range(100):
            test_cases.append({
                'requested_size': 0.1 + i * 0.01,
                'risk_score': 0.1 + (i % 10) * 0.1
            })
        
        latencies = []
        for case in test_cases:
            start_time = time.time()
            result = risk_manager.calculate_position_size(case['requested_size'], case['risk_score'])
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # ms
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        # Position sizing should be extremely fast
        assert avg_latency < 1, f"Position sizing avg latency {avg_latency:.2f}ms exceeds 1ms threshold"
        assert max_latency < 5, f"Position sizing max latency {max_latency:.2f}ms exceeds 5ms threshold"
    
    @pytest.mark.asyncio
    async def test_system_scalability(self, integration_service):
        """Test system scales with increasing load"""
        # Test with different signal volumes
        signal_volumes = [10, 50, 100, 200]
        throughput_results = {}
        
        for volume in signal_volumes:
            signals = []
            for i in range(volume):
                signals.append({
                    'symbol': 'BTC/USDT',
                    'action': 'buy',
                    'quantity': 0.1,
                    'price': 50000 + i,
                    'confidence': 0.8
                })
            
            # Process signals
            start_time = time.time()
            tasks = [integration_service.process_trade_signal(signal) for signal in signals]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = volume / total_time
            throughput_results[volume] = throughput
        
        # Throughput should remain relatively stable as volume increases
        # Allow for some degradation but not too much
        base_throughput = throughput_results[10]
        high_volume_throughput = throughput_results[200]
        
        # Should not degrade by more than 50%
        degradation_ratio = high_volume_throughput / base_throughput
        assert degradation_ratio > 0.5, f"Throughput degraded by {(1-degradation_ratio)*100:.1f}% under high load"
    
    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(self, integration_service):
        """Test CPU usage remains reasonable under load"""
        process = psutil.Process(os.getpid())
        
        # Get baseline CPU usage
        initial_cpu = process.cpu_percent(interval=1.0)
        
        # Process signals under load
        signals = []
        for i in range(500):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8
            })
        
        start_time = time.time()
        for signal in signals:
            await integration_service.process_trade_signal(signal)
        end_time = time.time()
        
        # Measure CPU usage during processing
        cpu_usage = process.cpu_percent(interval=0.1)
        
        # CPU usage should be reasonable
        assert cpu_usage < 80, f"CPU usage {cpu_usage}% exceeds 80% threshold under load"
        
        # Processing time should be reasonable
        processing_time = end_time - start_time
        assert processing_time < 30, f"Processing time {processing_time:.2f}s exceeds 30s threshold for 500 signals"
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, integration_service):
        """Test system performance under sustained load"""
        # Process signals continuously for 30 seconds
        duration = 30  # seconds
        start_time = time.time()
        signal_count = 0
        error_count = 0
        
        while time.time() - start_time < duration:
            signal = {
                'symbol': 'BTC/USDT',
                'action': 'buy' if signal_count % 2 == 0 else 'sell',
                'quantity': 0.1,
                'price': 50000 + (signal_count % 100),
                'confidence': 0.8
            }
            
            result = await integration_service.process_trade_signal(signal)
            signal_count += 1
            
            if 'error' in result:
                error_count += 1
            
            # Small delay to simulate real trading
            await asyncio.sleep(0.01)
        
        total_time = time.time() - start_time
        throughput = signal_count / total_time
        error_rate = error_count / signal_count
        
        # Performance should remain stable
        assert throughput > 50, f"Sustained throughput {throughput:.2f} signals/sec below 50 threshold"
        assert error_rate < 0.01, f"Error rate {error_rate:.2%} exceeds 1% threshold under sustained load"
        
        # System should remain healthy
        status = integration_service.get_status()
        assert status['is_running'] is True
        assert status['error_count'] < 100  # Allow some errors but not too many
    
    @pytest.mark.asyncio
    async def test_cold_start_performance(self, mock_config):
        """Test system performance from cold start"""
        # Measure cold start time
        start_time = time.time()
        
        # Create new service instance
        service = AdaptiveRiskIntegrationService()
        await service.initialize()
        await service.start()
        
        cold_start_time = time.time() - start_time
        
        # Process first signal
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 0.1,
            'price': 50000,
            'confidence': 0.8
        }
        
        first_signal_start = time.time()
        result = await service.process_trade_signal(signal)
        first_signal_time = time.time() - first_signal_start
        
        # Clean up
        await service.stop()
        
        # Cold start should be reasonable
        assert cold_start_time < 10, f"Cold start time {cold_start_time:.2f}s exceeds 10s threshold"
        assert first_signal_time < 1, f"First signal processing time {first_signal_time:.2f}s exceeds 1s threshold"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_with_large_datasets(self, mock_config):
        """Test memory efficiency with large datasets"""
        risk_manager = create_adaptive_risk_manager(mock_config)
        
        # Create large dataset
        large_returns = np.random.normal(0.001, 0.02, 10000)
        large_positions = []
        for i in range(1000):
            large_positions.append({
                'symbol': f'CRYPTO_{i}',
                'size': 0.1 + i * 0.001,
                'price': 50000 + i * 10,
                'volatility': 0.15 + (i % 100) * 0.001
            })
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process large datasets
        await risk_manager.estimate_volatility(large_returns)
        await risk_manager.calculate_portfolio_risk(large_positions)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable for large datasets
        assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB exceeds 100MB threshold for large datasets"
        
        # Clean up should be effective
        import gc
        gc.collect()
        post_gc_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should decrease significantly after GC
        memory_reduction = peak_memory - post_gc_memory
        assert memory_reduction > memory_increase * 0.5, "Memory cleanup was not effective"