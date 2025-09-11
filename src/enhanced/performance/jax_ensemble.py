"""
JAX/Flax Neural Network Acceleration
Ultra-Fast Trading Ensemble with JAX-compiled models
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class UltraFastTradingEnsemble(nn.Module):
    """
    JAX-compiled ensemble of neural networks for ultra-fast trading inference.
    Target: 2-3ms ensemble time vs current 8-15ms
    """

    # Configuration for different model architectures
    lstm_config: Dict[str, Any]
    transformer_config: Dict[str, Any]
    feature_dim: int = 150  # Number of input features
    hidden_dim: int = 256
    num_classes: int = 3  # BUY, HOLD, SELL
    dropout_rate: float = 0.1

    def setup(self):
        """Initialize ensemble components"""
        # LSTM Branch
        self.lstm_branch = LSTMBranch(
            config=self.lstm_config,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate
        )

        # Transformer Branch
        self.transformer_branch = TransformerBranch(
            config=self.transformer_config,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate
        )

        # Gradient Boosting Branch (XGBoost/LightGBM integration)
        self.boosting_branch = BoostingBranch(
            input_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )

        # Attention-based ensemble weighting
        self.attention_weighting = AttentionWeighting(
            num_models=3,
            feature_dim=self.feature_dim
        )

        # Market regime detector
        self.regime_detector = MarketRegimeDetector(
            feature_dim=self.feature_dim,
            num_regimes=5
        )

    def __call__(self, features: jnp.ndarray, training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through the ensemble with JAX compilation.

        Args:
            features: Input features [batch_size, sequence_length, feature_dim]
            training: Whether in training mode

        Returns:
            Dictionary containing predictions, confidence, and attention weights
        """

        # Detect market regime
        regime_probs = self.regime_detector(features)

        # Get predictions from each model
        lstm_pred = self.lstm_branch(features, training=training)
        transformer_pred = self.transformer_branch(features, training=training)
        boosting_pred = self.boosting_branch(features, training=training)

        # Stack predictions
        all_predictions = jnp.stack([lstm_pred, transformer_pred, boosting_pred], axis=0)

        # Apply attention-based weighting
        ensemble_pred, attention_weights = self.attention_weighting(
            all_predictions, features, regime_probs
        )

        # Calculate confidence and decision
        confidence = jnp.max(jnp.softmax(ensemble_pred, axis=-1), axis=-1)
        decision = jnp.argmax(ensemble_pred, axis=-1)

        return {
            'prediction': ensemble_pred,
            'decision': decision,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'regime_probs': regime_probs,
            'model_predictions': {
                'lstm': lstm_pred,
                'transformer': transformer_pred,
                'boosting': boosting_pred
            }
        }

class LSTMBranch(nn.Module):
    """LSTM branch for temporal pattern recognition"""

    config: Dict[str, Any]
    hidden_dim: int
    num_classes: int
    dropout_rate: float

    def setup(self):
        self.lstm_layers = [
            nn.LSTMCell(features=self.hidden_dim) for _ in range(self.config.get('num_layers', 2))
        ]
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.output_layer = nn.Dense(self.num_classes)

    def __call__(self, features, training=False):
        batch_size, seq_len, feature_dim = features.shape

        # Initialize hidden states
        hidden_states = []
        for _ in range(len(self.lstm_layers)):
            hidden_states.append(jnp.zeros((batch_size, self.hidden_dim)))

        # Process sequence through LSTM layers
        for t in range(seq_len):
            x_t = features[:, t, :]

            for i, lstm_cell in enumerate(self.lstm_layers):
                new_hidden, new_carry = lstm_cell(x_t, hidden_states[i])
                hidden_states[i] = new_hidden
                x_t = self.dropout(new_hidden, deterministic=not training)

        # Final prediction
        output = self.output_layer(hidden_states[-1])
        return output

class TransformerBranch(nn.Module):
    """Transformer branch for attention-based feature learning"""

    config: Dict[str, Any]
    hidden_dim: int
    num_classes: int
    dropout_rate: float

    def setup(self):
        self.embedding = nn.Dense(self.hidden_dim)
        self.transformer_layers = [
            nn.SelfAttention(num_heads=self.config.get('num_heads', 8))
            for _ in range(self.config.get('num_layers', 3))
        ]
        self.layer_norm = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.output_layer = nn.Dense(self.num_classes)

    def __call__(self, features, training=False):
        batch_size, seq_len, feature_dim = features.shape

        # Embed features
        x = self.embedding(features)
        x = self.dropout(x, deterministic=not training)

        # Process through transformer layers
        for attention_layer in self.transformer_layers:
            attention_output = attention_layer(x)
            x = self.layer_norm(x + attention_output)
            x = self.dropout(x, deterministic=not training)

        # Global average pooling
        x = jnp.mean(x, axis=1)

        # Final prediction
        output = self.output_layer(x)
        return output

class BoostingBranch(nn.Module):
    """Gradient boosting branch integration"""

    input_dim: int
    hidden_dim: int
    num_classes: int

    def setup(self):
        self.feature_extraction = nn.Dense(self.hidden_dim)
        self.boosting_layers = [
            nn.Dense(self.hidden_dim) for _ in range(3)
        ]
        self.output_layer = nn.Dense(self.num_classes)

    def __call__(self, features, training=False):
        batch_size, seq_len, feature_dim = features.shape

        # Flatten sequence for boosting-style processing
        x = jnp.reshape(features, (batch_size, -1))

        # Feature extraction
        x = self.feature_extraction(x)
        x = nn.relu(x)

        # Process through boosting-inspired layers
        for layer in self.boosting_layers:
            residual = layer(x)
            x = x + residual  # Residual connection
            x = nn.relu(x)

        # Final prediction
        output = self.output_layer(x)
        return output

class AttentionWeighting(nn.Module):
    """Attention-based ensemble weighting"""

    num_models: int
    feature_dim: int

    def setup(self):
        self.attention_network = nn.Sequential([
            nn.Dense(64),
            nn.relu,
            nn.Dense(self.num_models),
            nn.softmax
        ])

    def __call__(self, model_predictions, features, regime_probs):
        batch_size, seq_len, feature_dim = features.shape

        # Combine features and regime information
        attention_input = jnp.concatenate([
            jnp.mean(features, axis=1),  # Average features
            regime_probs  # Regime probabilities
        ], axis=-1)

        # Calculate attention weights
        attention_weights = self.attention_network(attention_input)

        # Apply weighting to model predictions
        weighted_prediction = jnp.sum(
            model_predictions * attention_weights[:, :, jnp.newaxis],
            axis=0
        )

        return weighted_prediction, attention_weights

class MarketRegimeDetector(nn.Module):
    """Market regime detection using unsupervised learning"""

    feature_dim: int
    num_regimes: int

    def setup(self):
        self.regime_network = nn.Sequential([
            nn.Dense(128),
            nn.relu,
            nn.Dense(64),
            nn.relu,
            nn.Dense(self.num_regimes),
            nn.softmax
        ])

    def __call__(self, features):
        batch_size, seq_len, feature_dim = features.shape

        # Use recent features for regime detection
        recent_features = features[:, -10:, :]  # Last 10 time steps
        regime_input = jnp.mean(recent_features, axis=1)

        regime_probs = self.regime_network(regime_input)
        return regime_probs

class JAXPerformanceMonitor:
    """Performance monitoring for JAX-compiled models"""

    def __init__(self):
        self.inference_times = []
        self.compilation_times = []
        self.memory_usage = []
        self.start_time = None

    def start_timing(self):
        """Start performance timing"""
        self.start_time = time.time()

    def record_inference_time(self, inference_time: float):
        """Record inference time"""
        self.inference_times.append(inference_time)
        logger.info(".4f")

    def record_compilation_time(self, compilation_time: float):
        """Record compilation time"""
        self.compilation_times.append(compilation_time)
        logger.info(".4f")

    def get_performance_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics"""
        if not self.inference_times:
            return {}

        return {
            'mean_inference_time': np.mean(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'p95_inference_time': np.percentile(self.inference_times, 95),
            'p99_inference_time': np.percentile(self.inference_times, 99),
            'mean_compilation_time': np.mean(self.compilation_times) if self.compilation_times else 0,
            'total_inferences': len(self.inference_times),
            'throughput_per_second': len(self.inference_times) / (time.time() - (self.start_time or time.time()))
        }

class TrainState(train_state.TrainState):
    """Enhanced training state with performance tracking"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_monitor = JAXPerformanceMonitor()

# JIT-compiled prediction function for maximum performance
@jax.jit
def predict_signal(model: UltraFastTradingEnsemble, features: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    JIT-compiled prediction function for ultra-fast inference.

    Target: <3ms total ensemble time
    """
    return model(features, training=False)

# Batch prediction for high-throughput scenarios
@jax.jit
def predict_batch(model: UltraFastTradingEnsemble, batch_features: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Batch prediction for processing multiple signals simultaneously.
    """
    def single_prediction(features):
        return model(features, training=False)

    return jax.vmap(single_prediction)(batch_features)

def create_enhanced_ensemble(config: Dict[str, Any]) -> UltraFastTradingEnsemble:
    """
    Factory function to create enhanced trading ensemble with optimal configuration.
    """
    ensemble = UltraFastTradingEnsemble(
        lstm_config=config.get('lstm', {'num_layers': 2}),
        transformer_config=config.get('transformer', {'num_heads': 8, 'num_layers': 3}),
        feature_dim=config.get('feature_dim', 150),
        hidden_dim=config.get('hidden_dim', 256),
        num_classes=config.get('num_classes', 3),
        dropout_rate=config.get('dropout_rate', 0.1)
    )

    return ensemble

def benchmark_ensemble_performance(ensemble: UltraFastTradingEnsemble,
                                 test_data: jnp.ndarray,
                                 num_runs: int = 1000) -> Dict[str, float]:
    """
    Comprehensive benchmark of ensemble performance.
    """
    monitor = JAXPerformanceMonitor()
    monitor.start_timing()

    # Warm-up run (compilation)
    _ = predict_signal(ensemble, test_data[:1])

    # Benchmark runs
    for i in range(num_runs):
        start_time = time.time()
        _ = predict_signal(ensemble, test_data[i % len(test_data):i % len(test_data) + 1])
        inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        monitor.record_inference_time(inference_time)

    return monitor.get_performance_stats()

# Export key functions for integration
__all__ = [
    'UltraFastTradingEnsemble',
    'LSTMBranch',
    'TransformerBranch',
    'BoostingBranch',
    'AttentionWeighting',
    'MarketRegimeDetector',
    'JAXPerformanceMonitor',
    'TrainState',
    'predict_signal',
    'predict_batch',
    'create_enhanced_ensemble',
    'benchmark_ensemble_performance'
]
