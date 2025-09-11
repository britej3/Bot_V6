"""
Performance Validation for Enhanced Inference Pipeline
====================================================

This script validates that the enhanced ML inference pipeline meets the
critical performance targets:
- Inference Latency: <5ms (90th percentile)
- Throughput: >1000 signals/second

It tests the integration between the TickLevelFeatureEngine and the
EnhancedTradingEnsemble, focusing on the JIT-compiled TCN model.
"""

import numpy as np
import jax.numpy as jnp
import time
import sys
import os

# Add src directory to path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from enhanced.ml.ensemble import EnhancedTradingEnsemble
from learning.tick_level_feature_engine import TickLevelFeatureEngine

# Mock config objects for testing
class MockTradingConfig:
    def __init__(self):
        self.tick_buffer_size = 1000
        self.lookback_window = 200
        self.fft_components = 5
        self.order_book_levels = 10
        self.whale_detection_std_dev = 3

def generate_dummy_tick_data(num_ticks: int) -> list:
    """Generates a stream of dummy tick data for simulation."""
    ticks = []
    for i in range(num_ticks):
        tick = {
            'price': 100 + np.sin(i / 10) + np.random.randn() * 0.1,
            'quantity': np.random.rand() * 10,
            'is_buyer_maker': np.random.rand() > 0.5,
            'timestamp': time.time()
        }
        ticks.append(tick)
    return ticks

def test_inference_pipeline_performance():
    """Main validation function to test and benchmark the pipeline."""
    print("--- Setting up Inference Pipeline Validation ---")
    
    # 1. Initialize Feature Engine and Ensemble
    feature_engine_config = MockTradingConfig()
    feature_engine = TickLevelFeatureEngine(config=feature_engine_config)
    
    ensemble_config = {
        "seed": 42,
        "tcn_config": {
            "input_shape": [1, 100, 88], # Batch, Seq Len, Features (adjust based on feature engine output)
            "num_channels": [64, 128],
            "kernel_size": 3,
            "dropout_rate": 0.2
        }
    }
    ensemble = EnhancedTradingEnsemble(config=ensemble_config)

    print("Feature Engine and Ensemble initialized.")

    # 2. Generate sample data and warm up the feature engine
    print("Generating sample data and warming up feature engine...")
    dummy_ticks = generate_dummy_tick_data(500)
    feature_vector_size = 0
    for tick in dummy_ticks:
        features = feature_engine.process_tick_data(tick)
        if features.size > 0:
            feature_vector_size = features.shape[0]
    
    print(f"Feature vector size determined to be: {feature_vector_size}")
    
    # Adjust ensemble input shape based on actual feature vector size
    if feature_vector_size != ensemble_config['tcn_config']['input_shape'][2]:
        print(f"Adjusting TCN input shape to {feature_vector_size}")
        ensemble_config['tcn_config']['input_shape'][2] = feature_vector_size
        ensemble.build_tcn_architecture() # Re-build TCN with correct shape

    # 3. Run the performance benchmark
    print("\n--- Running Official Performance Benchmark ---")
    results = ensemble.benchmark_performance(num_signals=5000)

    # 4. Validate results against targets
    print("\n--- Validating Performance Against Targets ---")
    latency_ok = results['p90_latency_ms'] < 5.0
    throughput_ok = results['throughput_signals_sec'] > 1000.0

    print(f"P90 Latency Check (<5ms): {'✅ PASS' if latency_ok else '❌ FAIL'}")
    print(f"Throughput Check (>1000 signals/s): {'✅ PASS' if throughput_ok else '❌ FAIL'}")

    assert latency_ok, f"Latency target not met: {results['p90_latency_ms']:.4f}ms"
    assert throughput_ok, f"Throughput target not met: {results['throughput_signals_sec']:.2f} signals/s"

    print("\n✅✅✅ Inference pipeline performance validation successful! ✅✅✅")

if __name__ == "__main__":
    test_inference_pipeline_performance()
