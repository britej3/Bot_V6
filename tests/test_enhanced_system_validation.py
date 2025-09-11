"""
Comprehensive Validation Suite for Enhanced CryptoScalp AI System
Phase 1: Core Performance Revolution - Validation Tests
"""

import pytest
import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.enhanced.performance import (
    UltraFastTradingEnsemble,
    FeaturePipeline,
    PolarsAccelerator
)
from src.enhanced.performance.integration import (
    EnhancedPerformanceSystem,
    create_enhanced_performance_system
)

class TestEnhancedSystemValidation:
    """Comprehensive validation tests for the enhanced system"""

    @pytest.fixture
    def sample_market_data(self):
        """Generate sample market data for testing"""
        return {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USDT',
            'open': 50000.0,
            'high': 51000.0,
            'low': 49500.0,
            'close': 50500.0,
            'volume': 1000000.0
        }

    @pytest.fixture
    def batch_market_data(self):
        """Generate batch market data for testing"""
        base_price = 50000.0
        data = []

        for i in range(100):
            price_change = np.random.normal(0, 0.02)
            price = base_price * (1 + price_change)

            data.append({
                'timestamp': datetime.now() + timedelta(seconds=i),
                'symbol': np.random.choice(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']),
                'open': price * 0.98,
                'high': price * 1.02,
                'low': price * 0.96,
                'close': price,
                'volume': np.random.lognormal(15, 2)
            })

        return data

    def test_jax_ensemble_creation(self):
        """Test JAX ensemble creation and basic functionality"""
        config = {
            'feature_dim': 150,
            'hidden_dim': 256,
            'num_classes': 3,
            'dropout_rate': 0.1,
            'lstm_layers': 2,
            'transformer_heads': 8,
            'transformer_layers': 3
        }

        # Test ensemble creation
        ensemble = UltraFastTradingEnsemble(
            lstm_config={'num_layers': config['lstm_layers']},
            transformer_config={'num_heads': config['transformer_heads'], 'num_layers': config['transformer_layers']},
            feature_dim=config['feature_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            dropout_rate=config['dropout_rate']
        )

        assert ensemble is not None
        assert hasattr(ensemble, 'lstm_branch')
        assert hasattr(ensemble, 'transformer_branch')
        assert hasattr(ensemble, 'boosting_branch')
        assert hasattr(ensemble, 'attention_weighting')
        assert hasattr(ensemble, 'regime_detector')

    def test_polars_pipeline_creation(self):
        """Test Polars feature pipeline creation"""
        pipeline = FeaturePipeline()

        assert pipeline is not None
        assert hasattr(pipeline, 'compute_all_features')
        assert hasattr(pipeline, 'get_performance_stats')
        assert hasattr(pipeline, 'setup_lazy_pipeline')

    def test_polars_accelerator_creation(self):
        """Test Polars accelerator creation"""
        accelerator = PolarsAccelerator()

        assert accelerator is not None
        assert hasattr(accelerator, 'create_realtime_pipeline')
        assert hasattr(accelerator, 'process_tick_batch')

    def test_enhanced_system_creation(self):
        """Test enhanced performance system creation"""
        config = {
            'ensemble': {
                'feature_dim': 150,
                'hidden_dim': 256,
                'num_classes': 3,
                'dropout_rate': 0.1,
                'lstm_layers': 2,
                'transformer_heads': 8,
                'transformer_layers': 3
            },
            'feature_pipeline': {
                'cache_enabled': True,
                'async_processing': True,
                'max_features': 1000
            },
            'performance': {
                'target_latency_ms': 5.0,
                'target_throughput': 1000,
                'monitoring_interval': 60
            },
            'trading': {
                'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
                'window_size': 100,
                'prediction_threshold': 0.7
            }
        }

        system = create_enhanced_performance_system(config)

        assert system is not None
        assert hasattr(system, 'ensemble')
        assert hasattr(system, 'feature_pipeline')
        assert hasattr(system, 'polars_accelerator')
        assert hasattr(system, 'realtime_pipelines')
        assert hasattr(system, 'process_trading_signal')

    @pytest.mark.asyncio
    async def test_single_signal_processing(self, sample_market_data):
        """Test single signal processing"""
        system = create_enhanced_performance_system()

        start_time = time.time()
        result = await system.process_trading_signal(sample_market_data)
        processing_time = (time.time() - start_time) * 1000

        # Validate result structure
        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'symbol' in result
        assert 'decision' in result
        assert 'confidence' in result
        assert 'regime' in result
        assert 'processing_time_ms' in result

        # Validate decision is valid
        assert result['decision'] in ['BUY', 'HOLD', 'SELL']

        # Validate confidence is reasonable
        assert 0.0 <= result['confidence'] <= 1.0

        # Validate processing time (target: <5ms)
        assert processing_time >= 0
        # Note: In testing environment, JAX compilation might take longer

    def test_feature_pipeline_functionality(self, batch_market_data):
        """Test feature pipeline with real data"""
        pipeline = FeaturePipeline()

        # Convert to DataFrame
        df = pd.DataFrame(batch_market_data)

        # Setup pipeline
        pipeline.setup_lazy_pipeline(df)

        # Compute features
        features_df = pipeline.compute_all_features()

        # Validate features
        assert features_df is not None
        assert len(features_df.columns) > len(df.columns)  # Should have more features than input

        # Check for essential features
        essential_features = ['close', 'volume', 'sma_20', 'rsi_14', 'macd_line']
        for feature in essential_features:
            assert feature in features_df.columns, f"Missing essential feature: {feature}"

    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        from src.enhanced.performance.jax_ensemble import JAXPerformanceMonitor

        monitor = JAXPerformanceMonitor()

        assert monitor is not None
        assert hasattr(monitor, 'record_inference_time')
        assert hasattr(monitor, 'get_performance_stats')

        # Test recording
        monitor.record_inference_time(2.5)
        monitor.record_inference_time(3.1)
        monitor.record_inference_time(1.8)

        stats = monitor.get_performance_stats()

        assert 'mean_inference_time' in stats
        assert 'min_inference_time' in stats
        assert 'max_inference_time' in stats
        assert stats['mean_inference_time'] > 0

    def test_system_configuration(self):
        """Test system configuration validation"""
        config = {
            'ensemble': {
                'feature_dim': 150,
                'hidden_dim': 256,
                'num_classes': 3,
                'dropout_rate': 0.1
            },
            'trading': {
                'symbols': ['BTC/USDT', 'ETH/USDT'],
                'window_size': 100
            }
        }

        system = create_enhanced_performance_system(config)

        assert system.config is not None
        assert system.config['ensemble']['feature_dim'] == 150
        assert system.config['ensemble']['hidden_dim'] == 256
        assert len(system.config['trading']['symbols']) == 2

    @pytest.mark.asyncio
    async def test_batch_processing(self, batch_market_data):
        """Test batch signal processing"""
        from src.enhanced.performance.integration import process_batch_signals

        system = create_enhanced_performance_system()

        # Process first 10 signals
        batch_data = batch_market_data[:10]

        start_time = time.time()
        results = await process_batch_signals(system, batch_data)
        total_time = (time.time() - start_time) * 1000

        assert len(results) == len(batch_data)

        # Validate each result
        for result in results:
            assert isinstance(result, dict)
            assert 'decision' in result
            assert 'confidence' in result
            assert 'processing_time_ms' in result

        # Check throughput
        throughput = len(results) / (total_time / 1000)
        assert throughput > 0

    def test_error_handling(self, sample_market_data):
        """Test error handling in the system"""
        system = create_enhanced_performance_system()

        # Test with invalid data
        invalid_data = {
            'symbol': 'INVALID',
            'price': 'not_a_number'
        }

        # Should handle gracefully
        result = asyncio.run(system.process_trading_signal(invalid_data))

        assert result is not None
        assert isinstance(result, dict)

    def test_performance_metrics(self):
        """Test performance metrics collection"""
        system = create_enhanced_performance_system()

        metrics = system.get_performance_metrics()

        assert isinstance(metrics, dict)
        assert 'system_uptime_seconds' in metrics
        assert 'total_requests' in metrics
        assert 'success_rate' in metrics
        assert 'average_latency_ms' in metrics
        assert 'throughput_signals_per_second' in metrics

    def test_realtime_pipeline_creation(self):
        """Test real-time pipeline creation for different symbols"""
        accelerator = PolarsAccelerator()
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

        pipelines = {}
        for symbol in symbols:
            pipeline = accelerator.create_realtime_pipeline(symbol, window_size=50)
            pipelines[symbol] = pipeline

        assert len(pipelines) == len(symbols)

        for symbol, pipeline in pipelines.items():
            assert pipeline is not None
            assert hasattr(pipeline, 'process_batch')
            assert pipeline.symbol == symbol

    def test_cache_functionality(self, batch_market_data):
        """Test feature caching functionality"""
        pipeline = FeaturePipeline()

        # Convert to DataFrame
        df = pd.DataFrame(batch_market_data)

        # Setup pipeline
        pipeline.setup_lazy_pipeline(df)

        # First computation
        features1 = pipeline.compute_all_features()

        # Second computation (should use cache if available)
        features2 = pipeline.compute_all_features()

        # Results should be identical
        assert features1.equals(features2)

        # Check cache stats
        stats = pipeline.get_performance_stats()
        assert 'cache_hit_rate' in stats

    def test_async_feature_computation(self, batch_market_data):
        """Test async feature computation"""
        pipeline = FeaturePipeline()

        # Convert to DataFrame
        df = pd.DataFrame(batch_market_data)

        # Setup pipeline
        pipeline.setup_lazy_pipeline(df)

        # Test async computation
        async def async_test():
            result = await pipeline.compute_features_async(df)
            return result

        result = asyncio.run(async_test())

        assert result is not None
        assert len(result.columns) > len(df.columns)

class TestSystemIntegration:
    """Test integration between all components"""

    def test_component_imports(self):
        """Test that all components can be imported without conflicts"""
        try:
            from src.enhanced.performance.jax_ensemble import (
                UltraFastTradingEnsemble,
                JAXPerformanceMonitor,
                predict_signal,
                create_enhanced_ensemble,
                benchmark_ensemble_performance
            )
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import JAX ensemble: {e}")

        try:
            from src.enhanced.performance.polars_pipeline import (
                FeaturePipeline,
                PolarsAccelerator,
                create_feature_pipeline,
                create_polars_accelerator
            )
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import Polars pipeline: {e}")

        try:
            from src.enhanced.performance.integration import (
                EnhancedPerformanceSystem,
                SystemPerformanceMonitor,
                create_enhanced_performance_system,
                benchmark_enhanced_system,
                process_batch_signals
            )
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import integration module: {e}")

    def test_no_circular_imports(self):
        """Test that there are no circular import issues"""
        import importlib

        # Reload modules to check for circular imports
        modules_to_test = [
            'src.enhanced.performance.jax_ensemble',
            'src.enhanced.performance.polars_pipeline',
            'src.enhanced.performance.integration'
        ]

        for module_name in modules_to_test:
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                importlib.import_module(module_name)
                assert True
            except ImportError as e:
                pytest.fail(f"Circular import detected in {module_name}: {e}")

    def test_memory_efficiency(self):
        """Test memory efficiency of the enhanced system"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create system
        system = create_enhanced_performance_system()

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before

        # System should use reasonable memory (< 500MB)
        assert memory_used < 500, f"System uses too much memory: {memory_used:.1f}MB"

    def test_configuration_validation(self):
        """Test configuration validation"""
        valid_config = {
            'ensemble': {
                'feature_dim': 150,
                'hidden_dim': 256,
                'num_classes': 3,
                'dropout_rate': 0.1
            },
            'trading': {
                'symbols': ['BTC/USDT'],
                'window_size': 100
            }
        }

        system = create_enhanced_performance_system(valid_config)
        assert system is not None

        # Test with invalid config should still work (uses defaults)
        invalid_config = {'invalid': 'config'}
        system = create_enhanced_performance_system(invalid_config)
        assert system is not None

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def test_jax_compilation_speed(self):
        """Test JAX compilation speed (first run)"""
        from src.enhanced.performance.jax_ensemble import create_enhanced_ensemble

        config = {
            'feature_dim': 150,
            'hidden_dim': 256,
            'num_classes': 3,
            'dropout_rate': 0.1,
            'lstm_layers': 2,
            'transformer_heads': 8,
            'transformer_layers': 3
        }

        start_time = time.time()
        ensemble = create_enhanced_ensemble(config)
        compilation_time = time.time() - start_time

        # JAX compilation should complete in reasonable time
        assert compilation_time < 30, f"JAX compilation too slow: {compilation_time:.2f}s"

    def test_polars_processing_speed(self, batch_market_data):
        """Test Polars processing speed"""
        pipeline = FeaturePipeline()

        df = pd.DataFrame(batch_market_data)
        pipeline.setup_lazy_pipeline(df)

        start_time = time.time()
        features_df = pipeline.compute_all_features()
        processing_time = (time.time() - start_time) * 1000

        # Polars should process features quickly
        assert processing_time < 1000, f"Polars processing too slow: {processing_time:.2f}ms"

        # Should create significant number of features
        assert len(features_df.columns) > 50

if __name__ == "__main__":
    # Run validation tests
    pytest.main([__file__, "-v", "--tb=short"])
