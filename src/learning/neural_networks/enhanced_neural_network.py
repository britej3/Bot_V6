"""
EnhancedTradingNeuralNetwork: Multi-scale temporal processing with uncertainty estimation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TimeScale(Enum):
    """Different time scales for market analysis"""
    SECOND = "second"          # 1-second granularity
    TEN_SECOND = "ten_second"  # 10-second granularity
    MINUTE = "minute"          # 1-minute granularity
    FIVE_MINUTE = "five_minute" # 5-minute granularity

@dataclass
class NetworkConfig:
    """Configuration for enhanced trading neural network"""
    # Input parameters
    feature_dimensions: int = 50  # Number of input features
    sequence_lengths: Dict[TimeScale, int] = None  # Sequence lengths per time scale
    
    # Network architecture
    hidden_sizes: List[int] = None  # Hidden layer sizes
    attention_heads: int = 8  # Number of attention heads
    dropout_rate: float = 0.1  # Dropout rate
    
    # Multi-scale parameters
    scale_weights: Dict[TimeScale, float] = None  # Weights for each time scale
    
    # Bayesian parameters
    uncertainty_samples: int = 10  # Number of samples for uncertainty estimation
    
    # Graph parameters
    max_connections: int = 10  # Maximum connections in graph network
    
    def __post_init__(self):
        if self.sequence_lengths is None:
            self.sequence_lengths = {
                TimeScale.SECOND: 60,      # 60 seconds
                TimeScale.TEN_SECOND: 30,  # 300 seconds (5 minutes)
                TimeScale.MINUTE: 60,      # 60 minutes (1 hour)
                TimeScale.FIVE_MINUTE: 24  # 120 minutes (2 hours)
            }
        
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 128, 64]
            
        if self.scale_weights is None:
            self.scale_weights = {
                TimeScale.SECOND: 0.4,
                TimeScale.TEN_SECOND: 0.3,
                TimeScale.MINUTE: 0.2,
                TimeScale.FIVE_MINUTE: 0.1
            }

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for different market aspects"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention"""
        attended, attention_weights = self.attention(x, x, x)
        return attended, attention_weights

class TemporalEncoder(nn.Module):
    """Temporal encoder for specific time scale"""
    
    def __init__(self, input_size: int, hidden_size: int, sequence_length: int, dropout: float = 0.1):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(hidden_size, 4, dropout)
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attended, attention_weights = self.attention(lstm_out)
        
        # Global average pooling
        pooled = self.pool(attended.transpose(1, 2)).squeeze(-1)
        
        return pooled, attention_weights

class MultiScaleProcessor(nn.Module):
    """Multi-scale temporal processor"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        
        # Create temporal encoders for each time scale
        self.encoders = nn.ModuleDict()
        for time_scale, seq_length in config.sequence_lengths.items():
            encoder = TemporalEncoder(
                input_size=config.feature_dimensions,
                hidden_size=config.hidden_sizes[0],
                sequence_length=seq_length,
                dropout=config.dropout_rate
            )
            self.encoders[time_scale.value] = encoder
        
        # Scale fusion layer
        total_hidden = len(config.sequence_lengths) * config.hidden_sizes[0]
        self.fusion = nn.Linear(total_hidden, config.hidden_sizes[0])
        
    def forward(self, inputs: Dict[TimeScale, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through multi-scale processors"""
        scale_outputs = {}
        scale_attentions = {}
        
        # Process each time scale
        for time_scale, tensor in inputs.items():
            if time_scale.value in self.encoders:
                output, attention = self.encoders[time_scale.value](tensor)
                scale_outputs[time_scale.value] = output
                scale_attentions[time_scale.value] = attention
        
        # Concatenate scale outputs
        if scale_outputs:
            concatenated = torch.cat(list(scale_outputs.values()), dim=-1)
            fused = self.fusion(concatenated)
        else:
            # Fallback if no valid inputs
            batch_size = list(inputs.values())[0].size(0) if inputs else 1
            fused = torch.zeros(batch_size, self.config.hidden_sizes[0])
        
        return fused, scale_attentions

class BayesianEstimator(nn.Module):
    """Bayesian uncertainty estimator"""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int], dropout: float = 0.1):
        super().__init__()
        
        # Create hidden layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layers for mean and variance
        layers.append(nn.Linear(prev_size, output_size * 2))  # Mean and log-variance
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and uncertainty"""
        output = self.network(x)
        
        # Split into mean and log-variance
        mean = output[:, :output.size(1)//2]
        log_variance = output[:, output.size(1)//2:]
        
        # Ensure positive variance
        variance = torch.exp(log_variance)
        
        return mean, variance
    
    def sample_predictions(self, x: torch.Tensor, num_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample multiple predictions for uncertainty estimation"""
        means, variances = self.forward(x)
        
        # Sample from distribution
        samples = torch.randn(num_samples, *means.shape, device=means.device)
        samples = means.unsqueeze(0) + samples * torch.sqrt(variances).unsqueeze(0)
        
        # Calculate statistics
        sample_mean = samples.mean(dim=0)
        sample_std = samples.std(dim=0)
        
        return sample_mean, sample_std

class GraphNeuralNetwork(nn.Module):
    """Graph neural network for modeling relationships between trading pairs"""
    
    def __init__(self, node_features: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        
        # Graph convolutional layers
        self.gcns = nn.ModuleList()
        self.gcns.append(nn.Linear(node_features, hidden_size))
        
        for _ in range(num_layers - 1):
            self.gcns.append(nn.Linear(hidden_size, hidden_size))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass with graph convolution"""
        # Normalize adjacency matrix
        degree = torch.sum(adjacency, dim=1, keepdim=True)
        degree = torch.where(degree > 0, 1.0 / torch.sqrt(degree), torch.zeros_like(degree))
        norm_adj = degree * adjacency * degree.t()
        
        # Graph convolution
        h = x
        for i, gcn in enumerate(self.gcns):
            h = torch.matmul(norm_adj, h)
            h = gcn(h)
            if i < len(self.gcns) - 1:
                h = self.activation(h)
                h = self.dropout(h)
                
        return h

class EnhancedTradingNeuralNetwork(nn.Module):
    """Enhanced trading neural network with multi-scale processing and uncertainty estimation"""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        super().__init__()
        self.config = config or NetworkConfig()
        
        # Multi-scale processor
        self.multi_scale = MultiScaleProcessor(self.config)
        
        # Bayesian estimator
        self.bayesian_estimator = BayesianEstimator(
            input_size=self.config.hidden_sizes[0] * 2,  # This is 256 * 2 = 512
            output_size=3,  # Reverted to 3
            hidden_sizes=self.config.hidden_sizes[1:],
            dropout=self.config.dropout_rate
        )
        
        # Graph neural network for cross-pair relationships
        self.gnn = GraphNeuralNetwork(
            node_features=self.config.hidden_sizes[0],
            hidden_size=self.config.hidden_sizes[0],
            dropout=self.config.dropout_rate
        )
        
        # Final decision layer
        self.decision_layer = nn.Linear(3, 3)  # Changed input size to 3
        
        logger.info("EnhancedTradingNeuralNetwork initialized")
    
    def forward(
        self, 
        temporal_inputs: Dict[TimeScale, torch.Tensor],
        graph_inputs: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the enhanced network
        
        Args:
            temporal_inputs: Dictionary of tensors for each time scale
            graph_inputs: Node features for graph network
            adjacency_matrix: Adjacency matrix for graph connections
            
        Returns:
            mean: Predicted values
            uncertainty: Uncertainty estimates
            attention_weights: Attention weights from temporal processing
        """
        # Multi-scale temporal processing
        temporal_features, attention_weights = self.multi_scale(temporal_inputs)
        
        # Graph processing (if provided)
        if graph_inputs is not None and adjacency_matrix is not None:
            graph_features = self.gnn(graph_inputs, adjacency_matrix)
            # Combine temporal and graph features
            combined_features = torch.cat([temporal_features, graph_features], dim=-1)
        else:
            # Use only temporal features
            combined_features = torch.cat([temporal_features, temporal_features], dim=-1)

        # Bayesian estimation on combined features
        mean_features, variance_features = self.bayesian_estimator(combined_features)
        uncertainty = torch.sqrt(variance_features)

        # Make final decision based on mean features
        decision_output = self.decision_layer(mean_features)
        
        return decision_output, uncertainty, attention_weights
    
    def predict_with_uncertainty(
        self, 
        temporal_inputs: Dict[TimeScale, torch.Tensor],
        graph_inputs: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None,
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty quantification
        
        Returns:
            mean_prediction: Mean prediction
            uncertainty: Model uncertainty
            confidence: Confidence score based on uncertainty
        """
        mean, uncertainty, _ = self.forward(temporal_inputs, graph_inputs, adjacency_matrix)
        
        # Calculate confidence (inverse of uncertainty)
        confidence = torch.exp(-uncertainty)
        
        return mean, uncertainty, confidence
    
    def get_attention_weights(
        self, 
        temporal_inputs: Dict[TimeScale, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Get attention weights from temporal processing"""
        _, attention_weights = self.multi_scale(temporal_inputs)
        return attention_weights

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create network configuration
    config = NetworkConfig()
    
    # Initialize network
    network = EnhancedTradingNeuralNetwork(config)
    
    # Create dummy inputs for testing
    batch_size = 16
    
    # Temporal inputs for different time scales
    temporal_inputs = {
        TimeScale.SECOND: torch.randn(batch_size, config.sequence_lengths[TimeScale.SECOND], config.feature_dimensions),
        TimeScale.TEN_SECOND: torch.randn(batch_size, config.sequence_lengths[TimeScale.TEN_SECOND], config.feature_dimensions),
        TimeScale.MINUTE: torch.randn(batch_size, config.sequence_lengths[TimeScale.MINUTE], config.feature_dimensions),
        TimeScale.FIVE_MINUTE: torch.randn(batch_size, config.sequence_lengths[TimeScale.FIVE_MINUTE], config.feature_dimensions)
    }
    
    # Graph inputs (optional)
    num_pairs = 10
    graph_features = torch.randn(num_pairs, config.hidden_sizes[0])
    adjacency_matrix = torch.randint(0, 2, (num_pairs, num_pairs)).float()
    
    # Test forward pass
    mean, uncertainty, attention_weights = network(
        temporal_inputs=temporal_inputs,
        graph_inputs=graph_features,
        adjacency_matrix=adjacency_matrix
    )
    
    print(f"Mean prediction shape: {mean.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Attention weights keys: {list(attention_weights.keys())}")
    
    # Test prediction with uncertainty
    mean_pred, uncertainty_pred, confidence = network.predict_with_uncertainty(
        temporal_inputs=temporal_inputs,
        graph_inputs=graph_features,
        adjacency_matrix=adjacency_matrix
    )
    
    print(f"Mean prediction: {mean_pred.shape}")
    print(f"Uncertainty: {uncertainty_pred.shape}")
    print(f"Confidence: {confidence.shape}")
    
    logger.info("EnhancedTradingNeuralNetwork test completed successfully")