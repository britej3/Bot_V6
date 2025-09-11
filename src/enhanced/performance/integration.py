"""
Enhanced Performance Integration Module
Combines all Phase 1 performance components into a unified system
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Enhanced performance components
from .jax_ensemble import (
    UltraFastTradingEnsemble,
    JAXPerformanceMonitor,
    predict_signal,
    create_enhanced_ensemble,
    benchmark_ensemble_performance
)
from .polars_pipeline import (
    FeaturePipeline,
    PolarsAccelerator,
    create_feature_pipeline,
    create_polars_accelerator
)

logger = logging.getLogger(__name__)

class EnhancedPerformanceSystem:
    """
    Integrated high-performance trading system combining all Phase 1 components.
    Target: 90% latency reduction, 10x throughput increase
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced performance system.

        Args:
            config: System configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.performance_stats = {
            'total_requests': 0,
            'successful_predictions': 0,
            'average_latency_ms': 0,
            'throughput_signals_per_second': 0,
            'memory_usage_mb': 0,
            'cache_hit_rate': 0
        }

        # Initialize core components
        self._initialize_components()

        # Performance monitoring
        self.system_monitor = SystemPerformanceMonitor()
        self.jax_monitor = JAXPerformanceMonitor()

        logger.info("Enhanced Performance System initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
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

    def _initialize_components(self):
        """Initialize all performance components"""
        try:
            # 1. JAX/Flax Ensemble (Ultra-fast ML inference)
            ensemble_config = self.config['ensemble']
            self.ensemble = create_enhanced_ensemble(ensemble_config)
            logger.info("✓ JAX/Flax ensemble initialized")

            # 2. Polars Feature Pipeline (Ultra-fast data processing)
            self.feature_pipeline = create_feature_pipeline(self.config['feature_pipeline'])
            logger.info("✓ Polars feature pipeline initialized")

            # 3. Polars Accelerator (Real-time processing)
            self.polars_accelerator = create_polars_accelerator()
            logger.info("✓ Polars accelerator initialized")

            # 4. Real-time pipelines for each symbol
            self.realtime_pipelines = {}
            for symbol in self.config['trading']['symbols']:
                self.realtime_pipelines[symbol] = self.polars_accelerator.create_realtime_pipeline(
                    symbol=symbol,
                    window_size=self.config['trading']['window_size']
                )
            logger.info(f"✓ Real-time pipelines created for {len(self.realtime_pipelines)} symbols")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def process_trading_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trading signal through the enhanced pipeline.

        Args:
            market_data: Raw market data dictionary

        Returns:
            Processed trading signal with prediction
        """
        start_time = time.time()
        symbol = market_data.get('symbol', 'UNKNOWN')

        try:
            # Step 1: Real-time data processing with Polars
            df = self._convert_to_dataframe(market_data)
            processed_data = self.realtime_pipelines[symbol].process_batch(df)

            # Step 2: Feature engineering with Polars
            features_df = await self._compute_features_async(processed_data)

            # Step 3: Convert to JAX array for ML inference
            features_array = self._dataframe_to_jax_array(features_df)

            # Step 4: Ultra-fast ML inference with JAX
            prediction_result = self._predict_signal(features_array)

            # Step 5: Post-process and format result
            result = self._format_trading_signal(
                market_data=market_data,
                prediction=prediction_result,
                features=features_df,
                processing_time=(time.time() - start_time) * 1000
            )

            # Update performance stats
            self._update_performance_stats(
                latency_ms=(time.time() - start_time) * 1000,
                success=True
            )

            return result

        except Exception as e:
            logger.error(f"Error processing trading signal for {symbol}: {e}")
            self._update_performance_stats(
                latency_ms=(time.time() - start_time) * 1000,
                success=False
            )
            return self._create_error_response(market_data, str(e))

    def _convert_to_dataframe(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert market data to DataFrame format"""
        # Convert single tick to DataFrame
        data = {
            'timestamp': [market_data.get('timestamp', datetime.now())],
            'symbol': [market_data.get('symbol', 'UNKNOWN')],
            'open': [market_data.get('open', 0)],
            'high': [market_data.get('high', 0)],
            'low': [market_data.get('low', 0)],
            'close': [market_data.get('close', 0)],
            'volume': [market_data.get('volume', 0)]
        }
        return pd.DataFrame(data)

    async def _compute_features_async(self, data: pd.DataFrame) -> pd.DataFrame:
        """Async feature computation using Polars"""
        if self.config['feature_pipeline'].get('async_processing', False):
            return await self.feature_pipeline.compute_features_async(data)
        else:
            return self.feature_pipeline.compute_all_features(data)

    def _dataframe_to_jax_array(self, df: pd.DataFrame) -> np.ndarray:
        """Convert DataFrame to JAX-compatible array"""
        # Select numerical features only
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        features = df[numerical_cols].fillna(0).values

        # Ensure correct shape for JAX ensemble [batch_size, sequence_length, feature_dim]
        if features.ndim == 1:
            features = features.reshape(1, 1, -1)
        elif features.ndim == 2:
            features = features.reshape(1, features.shape[0], features.shape[1])

        return features.astype(np.float32)

    def _predict_signal(self, features_array: np.ndarray) -> Dict[str, Any]:
        """Make prediction using JAX-compiled ensemble"""
        try:
            # Import JAX array conversion
            import jax.numpy as jnp
            features_jax = jnp.array(features_array)

            # Make prediction
            prediction = predict_signal(self.ensemble, features_jax)

            return {
                'prediction': prediction['prediction'],
                'decision': prediction['decision'],
                'confidence': prediction['confidence'],
                'attention_weights': prediction['attention_weights'],
                'regime_probs': prediction['regime_probs']
            }
        except Exception as e:
            logger.error(f"JAX prediction error: {e}")
            return self._fallback_prediction()

    def _fallback_prediction(self) -> Dict[str, Any]:
        """Fallback prediction when JAX fails"""
        return {
            'prediction': np.array([[0.33, 0.34, 0.33]]),  # Neutral
            'decision': np.array([1]),  # HOLD
            'confidence': np.array([0.34]),
            'attention_weights': None,
            'regime_probs': np.array([0.5, 0.3, 0.2])  # Normal, Volatile, Flash
        }

    def _format_trading_signal(self, market_data: Dict[str, Any],
                              prediction: Dict[str, Any],
                              features: pd.DataFrame,
                              processing_time: float) -> Dict[str, Any]:
        """Format the final trading signal"""
        decision_map = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}

        return {
            'timestamp': market_data.get('timestamp', datetime.now()),
            'symbol': market_data.get('symbol', 'UNKNOWN'),
            'price': market_data.get('close', 0),
            'volume': market_data.get('volume', 0),
            'decision': decision_map.get(int(prediction['decision'][0]), 'HOLD'),
            'confidence': float(prediction['confidence'][0]),
            'regime': self._interpret_regime(prediction['regime_probs']),
            'features_computed': len(features.columns),
            'processing_time_ms': processing_time,
            'model_version': 'enhanced_v1',
            'performance_metrics': self.get_performance_metrics()
        }

    def _interpret_regime(self, regime_probs: np.ndarray) -> str:
        """Interpret market regime from probabilities"""
        regime_names = ['NORMAL', 'VOLATILE', 'FLASH_CRASH']
        return regime_names[np.argmax(regime_probs)]

    def _update_performance_stats(self, latency_ms: float, success: bool):
        """Update system performance statistics"""
        self.performance_stats['total_requests'] += 1
        if success:
            self.performance_stats['successful_predictions'] += 1

        # Rolling average for latency
        current_avg = self.performance_stats['average_latency_ms']
        self.performance_stats['average_latency_ms'] = (current_avg + latency_ms) / 2

        # Throughput calculation
        self.performance_stats['throughput_signals_per_second'] = (
            self.performance_stats['successful_predictions'] /
            max(time.time() - self.system_monitor.start_time, 1)
        )

    def _create_error_response(self, market_data: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Create error response for failed processing"""
        return {
            'timestamp': market_data.get('timestamp', datetime.now()),
            'symbol': market_data.get('symbol', 'UNKNOWN'),
            'decision': 'HOLD',
            'confidence': 0.0,
            'error': error,
            'status': 'error'
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'system_uptime_seconds': time.time() - self.system_monitor.start_time,
            'total_requests': self.performance_stats['total_requests'],
            'success_rate': (self.performance_stats['successful_predictions'] /
                           max(self.performance_stats['total_requests'], 1)),
            'average_latency_ms': self.performance_stats['average_latency_ms'],
            'throughput_signals_per_second': self.performance_stats['throughput_signals_per_second'],
            'jax_performance': self.jax_monitor.get_performance_stats() if hasattr(self, 'jax_monitor') else {},
            'polars_performance': self.feature_pipeline.get_performance_stats()
        }

    def benchmark_system(self, test_data: List[Dict[str, Any]], num_runs: int = 1000) -> Dict[str, Any]:
        """
        Comprehensive system benchmark.

        Args:
            test_data: List of test market data samples
            num_runs: Number of benchmark iterations

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Starting system benchmark with {num_runs} runs...")

        start_time = time.time()
        successful_runs = 0
        latencies = []

        for i in range(num_runs):
            try:
                # Use async version for benchmark
                result = asyncio.run(self.process_trading_signal(test_data[i % len(test_data)]))
                latencies.append(result.get('processing_time_ms', 0))
                successful_runs += 1

                if (i + 1) % 100 == 0:
                    logger.info(f"Benchmark progress: {i + 1}/{num_runs}")

            except Exception as e:
                logger.error(f"Benchmark error at iteration {i}: {e}")

        total_time = time.time() - start_time

        return {
            'total_runs': num_runs,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / num_runs,
            'total_time_seconds': total_time,
            'average_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_signals_per_second': successful_runs / total_time,
            'performance_improvement': self._calculate_improvement(latencies)
        }

    def _calculate_improvement(self, current_latencies: List[float]) -> Dict[str, float]:
        """Calculate performance improvement over baseline"""
        # Baseline performance (from original system)
        baseline_avg_latency = 50.0  # ms
        current_avg_latency = np.mean(current_latencies)

        improvement_pct = ((baseline_avg_latency - current_avg_latency) / baseline_avg_latency) * 100

        return {
            'baseline_latency_ms': baseline_avg_latency,
            'current_latency_ms': current_avg_latency,
            'improvement_percentage': improvement_pct,
            'target_achieved': current_avg_latency <= self.config['performance']['target_latency_ms']
        }

class SystemPerformanceMonitor:
    """System-wide performance monitoring"""

    def __init__(self):
        self.start_time = time.time()
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []

    def record_system_stats(self):
        """Record current system statistics"""
        import psutil

        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)

        # GPU stats if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.gpu_usage.append(gpus[0].load * 100)
        except ImportError:
            pass

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        return {
            'uptime_seconds': time.time() - self.start_time,
            'average_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'average_memory_percent': np.mean(self.memory_usage) if self.memory_usage else 0,
            'average_gpu_percent': np.mean(self.gpu_usage) if self.gpu_usage else 0
        }

# Factory functions
def create_enhanced_performance_system(config: Optional[Dict[str, Any]] = None) -> EnhancedPerformanceSystem:
    """Create an enhanced performance trading system"""
    return EnhancedPerformanceSystem(config)

def benchmark_enhanced_system(system: EnhancedPerformanceSystem,
                             test_data: List[Dict[str, Any]],
                             num_runs: int = 1000) -> Dict[str, Any]:
    """Benchmark the enhanced performance system"""
    return system.benchmark_system(test_data, num_runs)

# Async batch processing for high-throughput scenarios
async def process_batch_signals(system: EnhancedPerformanceSystem,
                               market_data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process multiple signals concurrently"""
    tasks = [
        system.process_trading_signal(data)
        for data in market_data_batch
    ]

    return await asyncio.gather(*tasks)

# Export key classes and functions
__all__ = [
    'EnhancedPerformanceSystem',
    'SystemPerformanceMonitor',
    'create_enhanced_performance_system',
    'benchmark_enhanced_system',
    'process_batch_signals'
]
