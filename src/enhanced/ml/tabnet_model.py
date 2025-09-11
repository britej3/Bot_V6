"""
TabNet Model for Interpretable Feature Selection in Crypto Trading
================================================================

This module provides a JAX/Flax implementation of TabNet optimized for
financial time series data with the following features:

- Sequential attention mechanism for automatic feature selection
- Ghost batch normalization for stable training
- Sparsemax activation for interpretable attention weights
- Feature importance tracking and visualization
- Integration with crypto trading ensemble
- Sub-5ms inference capability

Key Benefits:
- Interpretable feature selection: Understand which features drive predictions
- High performance: Competitive with gradient boosting on tabular data
- Regularization: Built-in sparsity and feature reuse mechanisms
- Flexibility: Handles both categorical and numerical features

Performance Targets:
- Inference Latency: <2ms for TabNet component
- Feature Selection Accuracy: >80% on relevant features
- Memory Usage: <200MB for model parameters
- Integration: Seamless with EnhancedTradingEnsemble
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np
from functools import partial
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TabNetConfig:
    """Configuration for TabNet model"""
    # Model architecture
    feature_dim: int = 50           # Input feature dimension
    output_dim: int = 1             # Output dimension
    n_decision_steps: int = 6       # Number of decision steps
    relaxation_factor: float = 1.3  # Width of each decision step
    sparsity_coefficient: float = 1e-5  # Sparsity regularization
    
    # Attention mechanism
    attention_dim: int = 32         # Attention dimension
    n_shared_layers: int = 2        # Shared layers in feature transformer
    n_decision_layers: int = 2      # Decision layers per step
    
    # Normalization and regularization
    batch_momentum: float = 0.7     # Ghost batch normalization momentum
    virtual_batch_size: int = 256   # Virtual batch size for ghost BN
    dropout_rate: float = 0.1       # Dropout rate
    
    # Feature selection
    mask_type: str = "sparsemax"    # "sparsemax" or "entmax"
    epsilon: float = 1e-10          # Small constant for numerical stability
    
    # Performance optimization
    enable_feature_reuse: bool = True
    enable_gradient_checkpointing: bool = False
    use_mixed_precision: bool = False

def sparsemax(logits: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Sparsemax activation function for sparse attention
    
    Args:
        logits: Input logits
        axis: Axis along which to apply sparsemax
        
    Returns:
        Sparse probability distribution
    """
    # Sort logits in descending order
    sorted_logits = jnp.sort(logits, axis=axis)
    sorted_logits = jnp.flip(sorted_logits, axis=axis)
    
    # Compute cumulative sums
    dim = logits.shape[axis]
    cumsum = jnp.cumsum(sorted_logits, axis=axis)
    
    # Compute k values
    k_values = jnp.arange(1, dim + 1, dtype=jnp.float32)
    if axis != -1:
        # Reshape k_values to broadcast correctly
        shape = [1] * len(logits.shape)
        shape[axis] = dim
        k_values = k_values.reshape(shape)
    
    # Find threshold
    threshold = (cumsum - 1.0) / k_values
    condition = sorted_logits > threshold
    
    # Find k_star (last position where condition is True)
    k_star = jnp.sum(condition.astype(jnp.int32), axis=axis, keepdims=True)
    
    # Compute tau_star
    tau_star = jnp.take_along_axis(threshold, k_star - 1, axis=axis)
    
    # Apply sparsemax
    output = jnp.maximum(logits - tau_star, 0.0)
    return output

class GhostBatchNormalization(nn.Module):
    """Ghost Batch Normalization for TabNet"""
    momentum: float = 0.9
    epsilon: float = 1e-5
    virtual_batch_size: int = 256
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Apply ghost batch normalization"""
        batch_size, *feature_shape = x.shape
        
        if training and batch_size > self.virtual_batch_size:
            # Split into virtual batches
            num_virtual_batches = batch_size // self.virtual_batch_size
            remainder = batch_size % self.virtual_batch_size
            
            # Process virtual batches
            outputs = []
            for i in range(num_virtual_batches):
                start_idx = i * self.virtual_batch_size
                end_idx = start_idx + self.virtual_batch_size
                virtual_batch = x[start_idx:end_idx]
                
                # Apply batch norm to virtual batch
                bn_output = nn.BatchNorm(
                    momentum=self.momentum,
                    epsilon=self.epsilon,
                    use_running_average=not training
                )(virtual_batch)
                outputs.append(bn_output)
            
            # Handle remainder
            if remainder > 0:
                remainder_batch = x[-remainder:]
                bn_output = nn.BatchNorm(
                    momentum=self.momentum,
                    epsilon=self.epsilon,
                    use_running_average=not training
                )(remainder_batch)
                outputs.append(bn_output)
            
            return jnp.concatenate(outputs, axis=0)
        else:
            # Standard batch normalization
            return nn.BatchNorm(
                momentum=self.momentum,
                epsilon=self.epsilon,
                use_running_average=not training
            )(x)

class FeatureTransformer(nn.Module):
    """Feature transformer block for TabNet"""
    shared_layers: int = 2
    decision_layers: int = 2
    feature_dim: int = 50
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass through feature transformer"""
        
        # Shared layers (apply to all decision steps)
        for i in range(self.shared_layers):
            x = nn.Dense(self.feature_dim * 2)(x)
            x = GhostBatchNormalization()(x, training=training)
            x = nn.gelu(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Decision-specific layers
        for i in range(self.decision_layers):
            x = nn.Dense(self.feature_dim)(x)
            x = GhostBatchNormalization()(x, training=training)
            x = nn.gelu(x)
        
        return x

class AttentiveTransformer(nn.Module):
    """Attentive transformer for feature selection"""
    feature_dim: int = 50
    attention_dim: int = 32
    sparsity_coefficient: float = 1e-5
    
    @nn.compact
    def __call__(self, 
                 processed_features: jnp.ndarray,
                 prior_scales: jnp.ndarray,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute attention weights for feature selection
        
        Args:
            processed_features: Features from feature transformer
            prior_scales: Prior attention scales from previous steps
            training: Whether in training mode
            
        Returns:
            Tuple of (attention_weights, sparse_attention_weights)
        """
        
        # Compute attention scores
        attention_logits = nn.Dense(self.feature_dim)(processed_features)
        attention_logits = GhostBatchNormalization()(attention_logits, training=training)
        
        # Apply prior scaling
        attention_logits = attention_logits * prior_scales
        
        # Apply sparsemax for sparse attention
        sparse_attention = sparsemax(attention_logits, axis=-1)
        
        # Compute sparsity loss for regularization
        sparsity_loss = jnp.sum(sparse_attention * jnp.log(sparse_attention + 1e-10), axis=-1)
        sparsity_loss = jnp.mean(sparsity_loss) * self.sparsity_coefficient
        
        return sparse_attention, sparsity_loss

class DecisionStep(nn.Module):
    """Single decision step in TabNet"""
    config: TabNetConfig
    step_index: int
    
    @nn.compact  
    def __call__(self,
                 features: jnp.ndarray,
                 processed_features: jnp.ndarray,
                 prior_scales: jnp.ndarray,
                 training: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Execute single decision step
        
        Args:
            features: Original input features
            processed_features: Features from previous step
            prior_scales: Prior attention scales
            training: Whether in training mode
            
        Returns:
            Tuple of (decision_output, processed_features, attention_weights, sparsity_loss)
        """
        
        # Feature transformation
        feature_transformer = FeatureTransformer(
            shared_layers=self.config.n_shared_layers,
            decision_layers=self.config.n_decision_layers,
            feature_dim=self.config.feature_dim,
            dropout_rate=self.config.dropout_rate
        )
        
        transformed_features = feature_transformer(processed_features, training=training)
        
        # Attentive transformation for feature selection
        attentive_transformer = AttentiveTransformer(
            feature_dim=self.config.feature_dim,
            attention_dim=self.config.attention_dim,
            sparsity_coefficient=self.config.sparsity_coefficient
        )
        
        attention_weights, sparsity_loss = attentive_transformer(
            transformed_features, prior_scales, training=training
        )
        
        # Apply attention to original features
        selected_features = features * attention_weights
        
        # Split into decision and next step features
        decision_features = nn.Dense(self.config.feature_dim)(selected_features)
        decision_features = nn.relu(decision_features)
        
        # Output for this decision step
        decision_output = nn.Dense(
            int(self.config.feature_dim * self.config.relaxation_factor)
        )(decision_features)
        decision_output = GhostBatchNormalization()(decision_output, training=training)
        decision_output = nn.relu(decision_output)
        
        # Features for next step (if not last step)
        if self.step_index < self.config.n_decision_steps - 1:
            next_features = nn.Dense(self.config.feature_dim)(selected_features)
            next_features = GhostBatchNormalization()(next_features, training=training)
            processed_features = processed_features + next_features
        else:
            processed_features = transformed_features
        
        # Update prior scales for next step
        updated_prior_scales = prior_scales * (self.config.sparsity_coefficient - attention_weights)
        
        return decision_output, processed_features, attention_weights, sparsity_loss

class TabNet(nn.Module):
    """
    TabNet: Attentive Interpretable Tabular Learning
    
    Optimized for crypto trading feature selection and prediction.
    """
    config: TabNetConfig
    
    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray, 
                 training: bool = False) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through TabNet
        
        Args:
            x: Input features (batch_size, feature_dim)
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
            - 'output': Final predictions
            - 'attention_weights': Attention weights for each step
            - 'feature_importance': Overall feature importance scores
            - 'sparsity_loss': Sparsity regularization loss
        """
        batch_size, feature_dim = x.shape
        
        # Input feature normalization
        x = GhostBatchNormalization(
            virtual_batch_size=self.config.virtual_batch_size
        )(x, training=training)
        
        # Initialize processed features and prior scales
        processed_features = nn.Dense(self.config.feature_dim)(x)
        prior_scales = jnp.ones((batch_size, feature_dim))
        
        # Store outputs from each decision step
        decision_outputs = []
        attention_weights_list = []
        sparsity_losses = []
        
        # Execute decision steps
        for step_idx in range(self.config.n_decision_steps):
            decision_step = DecisionStep(
                config=self.config,
                step_index=step_idx
            )
            
            decision_output, processed_features, attention_weights, sparsity_loss = decision_step(
                x, processed_features, prior_scales, training=training
            )
            
            decision_outputs.append(decision_output)
            attention_weights_list.append(attention_weights)
            sparsity_losses.append(sparsity_loss)
            
            # Update prior scales
            prior_scales = prior_scales * (1.0 - attention_weights)
        
        # Aggregate decision outputs
        aggregated_output = jnp.sum(jnp.stack(decision_outputs, axis=1), axis=1)
        
        # Final output layer
        final_output = nn.Dense(64)(aggregated_output)
        final_output = nn.relu(final_output)
        final_output = nn.Dropout(self.config.dropout_rate)(final_output, deterministic=not training)
        
        # Multi-head output for comprehensive trading signals
        predictions = {
            'price_direction': nn.Dense(1)(final_output),     # Main trading signal
            'confidence': nn.Dense(1)(final_output),          # Prediction confidence
            'volatility': nn.Dense(1)(final_output),          # Expected volatility
            'feature_quality': nn.Dense(1)(final_output)      # Quality of input features
        }
        
        # Apply appropriate activations
        predictions['price_direction'] = nn.tanh(predictions['price_direction'])
        predictions['confidence'] = nn.sigmoid(predictions['confidence'])
        predictions['volatility'] = nn.softplus(predictions['volatility'])
        predictions['feature_quality'] = nn.sigmoid(predictions['feature_quality'])
        
        # Compute overall feature importance
        feature_importance = jnp.mean(jnp.stack(attention_weights_list, axis=1), axis=1)
        feature_importance = feature_importance / (jnp.sum(feature_importance, axis=-1, keepdims=True) + 1e-10)
        
        # Aggregate sparsity loss
        total_sparsity_loss = jnp.mean(jnp.array(sparsity_losses))
        
        return {
            'outputs': predictions,
            'attention_weights': jnp.stack(attention_weights_list, axis=1),  # (batch, steps, features)
            'feature_importance': feature_importance,                        # (batch, features)
            'sparsity_loss': total_sparsity_loss,
            'decision_outputs': jnp.stack(decision_outputs, axis=1)         # For analysis
        }

class TabNetFeatureSelector:
    """Feature selection utility using trained TabNet model"""
    
    def __init__(self, model_params: Dict[str, Any], config: TabNetConfig):
        self.model_params = model_params
        self.config = config
        self.feature_names = None
    
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names for interpretability"""
        self.feature_names = feature_names
    
    def get_feature_importance(self, features: jnp.ndarray) -> Dict[str, float]:
        """Get feature importance scores for given input"""
        tabnet = TabNet(config=self.config)
        result = tabnet.apply(self.model_params, features, training=False)
        
        importance_scores = jnp.mean(result['feature_importance'], axis=0)
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance_scores.tolist()))
        else:
            return {f'feature_{i}': score for i, score in enumerate(importance_scores.tolist())}
    
    def select_top_features(self, features: jnp.ndarray, top_k: int = 20) -> Tuple[jnp.ndarray, List[int]]:
        """Select top-k most important features"""
        tabnet = TabNet(config=self.config)
        result = tabnet.apply(self.model_params, features, training=False)
        
        importance_scores = jnp.mean(result['feature_importance'], axis=0)
        top_indices = jnp.argsort(importance_scores)[-top_k:]
        
        selected_features = features[:, top_indices]
        return selected_features, top_indices.tolist()
    
    def explain_prediction(self, features: jnp.ndarray) -> Dict[str, Any]:
        """Provide detailed explanation of prediction"""
        tabnet = TabNet(config=self.config)
        result = tabnet.apply(self.model_params, features, training=False)
        
        # Get step-by-step attention
        attention_by_step = result['attention_weights'][0]  # First sample
        
        explanation = {
            'prediction': {
                'price_direction': float(result['outputs']['price_direction'][0, 0]),
                'confidence': float(result['outputs']['confidence'][0, 0]),
                'volatility': float(result['outputs']['volatility'][0, 0])
            },
            'feature_importance': self.get_feature_importance(features),
            'attention_by_step': attention_by_step.tolist(),
            'sparsity_loss': float(result['sparsity_loss'])
        }
        
        return explanation

# Factory function
def create_tabnet_model(config_dict: Dict[str, Any]) -> TabNet:
    """Create TabNet model from configuration dictionary"""
    config = TabNetConfig(
        feature_dim=config_dict.get('feature_dim', 50),
        output_dim=config_dict.get('output_dim', 1),
        n_decision_steps=config_dict.get('n_decision_steps', 6),
        relaxation_factor=config_dict.get('relaxation_factor', 1.3),
        sparsity_coefficient=config_dict.get('sparsity_coefficient', 1e-5),
        attention_dim=config_dict.get('attention_dim', 32),
        n_shared_layers=config_dict.get('n_shared_layers', 2),
        n_decision_layers=config_dict.get('n_decision_layers', 2),
        batch_momentum=config_dict.get('batch_momentum', 0.7),
        virtual_batch_size=config_dict.get('virtual_batch_size', 256),
        dropout_rate=config_dict.get('dropout_rate', 0.1),
        enable_feature_reuse=config_dict.get('enable_feature_reuse', True)
    )
    
    logger.info(f"Creating TabNet with config: {config}")
    return TabNet(config=config)

if __name__ == "__main__":
    # Example usage and testing
    config = {
        'feature_dim': 50,
        'n_decision_steps': 4,  # Fewer steps for testing
        'attention_dim': 16,
        'sparsity_coefficient': 1e-4
    }
    
    print("Testing TabNet implementation...")
    
    # Create model
    tabnet = create_tabnet_model(config)
    
    # Generate test data
    key = jax.random.PRNGKey(42)
    test_features = jax.random.normal(key, (32, 50))  # batch_size=32, features=50
    
    # Initialize parameters
    params = tabnet.init(key, test_features, training=False)
    
    # Test inference
    result = tabnet.apply(params, test_features, training=False)
    
    print(f"Output shapes:")
    for key, value in result['outputs'].items():
        print(f"  {key}: {value.shape}")
    
    print(f"Attention weights shape: {result['attention_weights'].shape}")
    print(f"Feature importance shape: {result['feature_importance'].shape}")
    print(f"Sparsity loss: {result['sparsity_loss']:.6f}")
    
    # Test feature selection
    feature_selector = TabNetFeatureSelector(params, tabnet.config)
    feature_names = [f'crypto_feature_{i}' for i in range(50)]
    feature_selector.set_feature_names(feature_names)
    
    importance_scores = feature_selector.get_feature_importance(test_features[:1])
    print(f"\\nTop 5 most important features:")
    sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, score in sorted_features:
        print(f"  {name}: {score:.4f}")
    
    print("✅ TabNet test completed successfully")