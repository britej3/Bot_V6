"""
Enhanced ML Ensemble for Production Trading
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import time
from functools import partial

# Import our enhanced models and components
from .tcn_model import EnhancedTCN, create_enhanced_tcn, TCNPerformanceMonitor
from .tabnet_model import TabNet, create_tabnet_model, TabNetFeatureSelector
from .ppo_trading_agent import PPOTradingAgent, create_ppo_agent, ActionType
from .crypto_feature_engine import CryptoFeatureEngine, create_crypto_feature_engine, MarketData
from .optimized_inference import OptimizedInferencePipeline, create_optimized_pipeline

# Placeholder for legacy models, will be properly imported later
# from src.learning.real_ml_models import RealLSTMModel, RealXGBoostModel

class EnhancedTradingEnsemble:
    """Production-ready ML ensemble for ultra-low latency crypto trading with all advanced models."""
    def __init__(self, config: dict):
        self.config = config
        self.key = jax.random.PRNGKey(config.get("seed", 0))

        # --- Enhanced JAX/Flax Models ---
        self.tcn_model = None
        self.tcn_state = None
        self.tabnet_model = None
        self.tabnet_state = None
        self.tabnet_feature_selector = None

        # --- PPO Trading Agent ---
        self.ppo_agent = None

        # --- Feature Engineering ---
        self.feature_engine = None

        # --- Inference Optimization ---
        self.optimized_inference_pipeline = None

        # --- Legacy Models Integration ---
        self.xgboost_model = None
        self.lstm_model = None

        # --- Dynamic Ensemble Weights ---
        self.base_ensemble_weights = config.get("ensemble_weights", {
            "tcn": 0.35,
            "tabnet": 0.25,
            "ppo": 0.20,
            "xgboost": 0.15,
            "lstm": 0.05
        })
        self.current_ensemble_weights = self.base_ensemble_weights.copy()
        
        # --- Performance Monitoring ---
        self.performance_history = []
        self.tcn_monitor = TCNPerformanceMonitor()
        
        self._build_all_architectures()

    def _build_all_architectures(self):
        """Initialize all model architectures and states."""
        print("Building TCN architecture...")
        self.build_tcn_architecture()
        
        # Placeholders for future integrations
        # print("Building TabNet architecture...")
        # self.build_tabnet_architecture()
        
        # print("Building PPO architecture...")
        # self.build_ppo_architecture()

    def build_tcn_architecture(self):
        """Initializes the Enhanced TCN model with optimized parameters and JIT compilation."""
        tcn_config = self.config.get("tcn_config", {})
        
        # Enhanced configuration for crypto trading
        enhanced_config = {
            "num_channels": tcn_config.get("num_channels", [64, 128, 256]),
            "kernel_size": tcn_config.get("kernel_size", 3),
            "dropout_rate": tcn_config.get("dropout_rate", 0.2),
            "attention_heads": tcn_config.get("attention_heads", 8),
            "use_attention": tcn_config.get("use_attention", True),
            "market_regime_aware": tcn_config.get("market_regime_aware", True),
            "whale_detection": tcn_config.get("whale_detection", True),
            "feature_dims": tcn_config.get("feature_dims", 50),
            "max_sequence_length": tcn_config.get("max_sequence_length", 1000)
        }
        
        self.tcn_model = create_enhanced_tcn(enhanced_config)
        
        # Initialize model state with proper input shape
        input_shape = tcn_config.get("input_shape", [1, 100, 50])  # [batch, seq_len, features]
        dummy_input = jnp.ones(input_shape)
        dummy_market_features = {
            'volume_profile': jnp.ones((input_shape[0], input_shape[1], 10)),
            'order_flow': jnp.ones((input_shape[0], input_shape[1], 5))
        }
        
        # Initialize parameters
        self.key, init_key = jax.random.split(self.key)
        params = self.tcn_model.init(init_key, dummy_input, dummy_market_features, training=False)
        
        # Create optimized training state
        self.tcn_state = train_state.TrainState.create(
            apply_fn=self.tcn_model.apply,
            params=params,
            tx=optax.adamw(learning_rate=1e-3, weight_decay=1e-5)  # AdamW for better generalization
        )

        # JIT-compile the prediction function for ultra-low latency
        @jax.jit
        def predict_tcn_optimized(params, x, market_features=None):
            return self.tcn_model.apply(params, x, market_features, training=False)
            
        self.predict_tcn_jit = predict_tcn_optimized
        
        # Initialize performance monitor
        self.tcn_monitor = TCNPerformanceMonitor()
        
        print(f"Enhanced TCN architecture built with config: {enhanced_config}")

    def predict(self, features: dict) -> dict:
        """Generate unified prediction from enhanced model ensemble with comprehensive outputs."""
        start_time = time.perf_counter()
        
        try:
            # Extract different feature types
            tcn_input = features.get("tcn_input")
            market_features = features.get("market_features", None)
            
            if tcn_input is None:
                return self._fallback_prediction()
            
            # Get Enhanced TCN prediction with comprehensive outputs
            tcn_result = self.predict_tcn_jit(
                {'params': self.tcn_state.params}, 
                tcn_input, 
                market_features
            )
            
            # Extract main trading signal
            price_direction = float(tcn_result['outputs']['price_direction'][0, 0])
            confidence = float(tcn_result['outputs']['confidence'][0, 0])
            volatility_pred = float(tcn_result['outputs']['volatility'][0, 0])
            volume_pred = float(tcn_result['outputs']['volume_prediction'][0, 0])
            
            # Get whale activity if available
            whale_activity = 0.0
            if tcn_result['whale_activity'] is not None:
                whale_activity = float(tcn_result['whale_activity'][0, -1, 0])
            
            # XGBoost prediction (placeholder - will be enhanced in Phase 2)
            xgb_pred = 0.6  # Dummy value for now
            
            # LSTM prediction (placeholder - will be enhanced in Phase 2)
            lstm_pred = 0.4  # Dummy value for now
            
            # Enhanced ensemble combination with market condition weighting
            market_volatility_factor = min(2.0, max(0.5, volatility_pred))
            adjusted_weights = {
                "tcn": self.ensemble_weights["tcn"] * (1.0 + whale_activity * 0.2),
                "xgboost": self.ensemble_weights["xgboost"] * market_volatility_factor,
                "lstm": self.ensemble_weights["lstm"]
            }
            
            # Normalize weights
            total_weight = sum(adjusted_weights.values())
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
            
            # Final ensemble prediction
            final_prediction = (
                price_direction * adjusted_weights["tcn"] +
                (xgb_pred - 0.5) * 2 * adjusted_weights["xgboost"] +  # Convert to -1,1 range
                (lstm_pred - 0.5) * 2 * adjusted_weights["lstm"]
            )
            
            # Generate trading signal
            signal_threshold = 0.1  # Minimum confidence threshold
            if abs(final_prediction) < signal_threshold:
                signal = 0  # Hold
            else:
                signal = 1 if final_prediction > 0 else -1
            
            # Adjust confidence based on ensemble agreement
            ensemble_confidence = confidence * (1.0 + whale_activity * 0.1)
            
            # Record performance metrics
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            self.tcn_monitor.log_inference_time(inference_time)
            
            return {
                "signal": signal,
                "confidence": float(ensemble_confidence),
                "raw_prediction": float(final_prediction),
                "tcn_prediction": float(price_direction),
                "whale_activity": whale_activity,
                "volatility_prediction": volatility_pred,
                "volume_prediction": volume_pred,
                "inference_time_ms": inference_time,
                "ensemble_weights": adjusted_weights,
                "market_features": {
                    "volatility_regime": self._classify_volatility(volatility_pred),
                    "whale_detected": whale_activity > 0.7
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._fallback_prediction()
    
    def _fallback_prediction(self) -> dict:
        """Fallback prediction when main prediction fails"""
        return {
            "signal": 0,
            "confidence": 0.0,
            "raw_prediction": 0.0,
            "error": "Fallback prediction used"
        }
    
    def _classify_volatility(self, volatility_value: float) -> str:
        """Classify volatility level"""
        if volatility_value < 0.5:
            return "low"
        elif volatility_value < 1.5:
            return "normal"
        else:
            return "high"

    def benchmark_performance(self, num_signals=2000):
        """Enhanced performance benchmark with comprehensive metrics."""
        print("\n=== Enhanced Performance Benchmark ===")
        
        # Test input shape
        input_shape = self.config.get("tcn_config", {}).get("input_shape", [1, 100, 50])
        dummy_input = jnp.ones(input_shape)
        dummy_market_features = {
            'volume_profile': jnp.ones((input_shape[0], input_shape[1], 10)),
            'order_flow': jnp.ones((input_shape[0], input_shape[1], 5))
        }
        
        # Warm-up JIT compilation
        print("Warming up JIT compiler...")
        for _ in range(20):  # More warm-up iterations
            result = self.predict_tcn_jit(
                {'params': self.tcn_state.params}, 
                dummy_input, 
                dummy_market_features
            )
            result['outputs']['price_direction'].block_until_ready()
        print("Warm-up complete.")
        
        # Benchmark individual inference latency
        latencies = []
        for _ in range(1000):
            start_time = time.perf_counter()
            result = self.predict_tcn_jit(
                {'params': self.tcn_state.params}, 
                dummy_input, 
                dummy_market_features
            )
            result['outputs']['price_direction'].block_until_ready()
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # milliseconds
        
        # Calculate latency statistics
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[int(len(latencies) * 0.5)]
        p90_latency = sorted(latencies)[int(len(latencies) * 0.9)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"\n--- Latency Results (Enhanced TCN) ---")
        print(f"Average: {avg_latency:.4f} ms")
        print(f"50th Percentile: {p50_latency:.4f} ms")
        print(f"90th Percentile: {p90_latency:.4f} ms")
        print(f"99th Percentile: {p99_latency:.4f} ms")
        
        # Benchmark throughput
        print(f"\n--- Throughput Test ({num_signals} signals) ---")
        start_time = time.time()
        for _ in range(num_signals):
            _ = self.predict_tcn_jit(
                {'params': self.tcn_state.params}, 
                dummy_input, 
                dummy_market_features
            )
        end_time = time.time()
        
        duration = end_time - start_time
        throughput = num_signals / duration
        
        print(f"Duration: {duration:.2f} seconds")
        print(f"Throughput: {throughput:.2f} signals/second")
        
        # Memory usage estimation (simplified)
        print(f"\n--- Resource Utilization ---")
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(self.tcn_state.params))
        estimated_memory_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        print(f"Parameter count: {param_count:,}")
        print(f"Estimated memory usage: {estimated_memory_mb:.2f} MB")
        
        # Check performance targets
        print(f"\n--- Performance Target Validation ---")
        meets_latency = p90_latency < 5.0
        meets_throughput = throughput > 1000
        meets_memory = estimated_memory_mb < 500
        
        print(f"✅ Latency target (<5ms): {meets_latency} ({p90_latency:.4f}ms)")
        print(f"✅ Throughput target (>1000/s): {meets_throughput} ({throughput:.0f}/s)")
        print(f"✅ Memory target (<500MB): {meets_memory} ({estimated_memory_mb:.1f}MB)")
        
        # Overall performance score
        targets_met = sum([meets_latency, meets_throughput, meets_memory])
        print(f"\n🎯 Performance Score: {targets_met}/3 targets met")
        
        if targets_met == 3:
            print("🚀 ALL PERFORMANCE TARGETS MET! Ready for production.")
        else:
            print("⚠️  Some performance targets not met. Further optimization needed.")
        
        return {
            "p90_latency_ms": p90_latency,
            "throughput_signals_sec": throughput,
            "memory_usage_mb": estimated_memory_mb,
            "targets_met": targets_met,
            "performance_details": {
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": p50_latency,
                "p99_latency_ms": p99_latency,
                "parameter_count": param_count
            }
        }

if __name__ == '__main__':
    # Enhanced example usage and testing
    config = {
        "seed": 42,
        "tcn_config": {
            "input_shape": [1, 100, 50],  # Batch, Sequence Length, Features
            "num_channels": [64, 128, 256],
            "kernel_size": 3,
            "dropout_rate": 0.2,
            "attention_heads": 8,
            "use_attention": True,
            "market_regime_aware": True,
            "whale_detection": True,
            "feature_dims": 50
        },
        "ensemble_weights": {
            "tcn": 0.6,  # Higher weight for enhanced TCN
            "xgboost": 0.25,
            "lstm": 0.15
        }
    }
    
    print("Creating Enhanced Trading Ensemble...")
    ensemble = EnhancedTradingEnsemble(config)
    
    # Create enhanced test features
    dummy_features = {
        "tcn_input": jnp.ones(config["tcn_config"]["input_shape"]),
        "market_features": {
            "volume_profile": jnp.random.normal(key=jax.random.PRNGKey(42), 
                                               shape=(1, 100, 10)),
            "order_flow": jnp.random.normal(key=jax.random.PRNGKey(43), 
                                          shape=(1, 100, 5))
        }
    }

    print("\n=== Testing Enhanced Prediction ===")
    prediction = ensemble.predict(dummy_features)
    print(f"Enhanced Prediction Result:")
    for key, value in prediction.items():
        print(f"  {key}: {value}")

    print("\n=== Running Performance Benchmark ===")
    benchmark_results = ensemble.benchmark_performance(num_signals=1000)
    
    print(f"\n=== Benchmark Summary ===")
    print(f"90th Percentile Latency: {benchmark_results['p90_latency_ms']:.4f} ms")
    print(f"Throughput: {benchmark_results['throughput_signals_sec']:.0f} signals/sec")
    print(f"Memory Usage: {benchmark_results['memory_usage_mb']:.1f} MB")
    print(f"Targets Met: {benchmark_results['targets_met']}/3")
