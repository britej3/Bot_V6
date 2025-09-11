"""
Unit tests for EnhancedTradingNeuralNetwork
"""
import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.learning.neural_networks.enhanced_neural_network import (
    EnhancedTradingNeuralNetwork,
    NetworkConfig,
    MultiHeadAttention,
    TemporalEncoder,
    MultiScaleProcessor,
    BayesianEstimator,
    GraphNeuralNetwork,
    TimeScale
)

class TestNetworkConfig:
    """Test NetworkConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = NetworkConfig()
        
        assert config.feature_dimensions == 50
        assert isinstance(config.sequence_lengths, dict)
        assert isinstance(config.hidden_sizes, list)
        assert config.attention_heads == 8
        assert config.dropout_rate == 0.1
        assert isinstance(config.scale_weights, dict)
        assert config.uncertainty_samples == 10
        assert config.max_connections == 10
        
        # Check time scales
        assert TimeScale.SECOND in config.sequence_lengths
        assert TimeScale.TEN_SECOND in config.sequence_lengths
        assert TimeScale.MINUTE in config.sequence_lengths
        assert TimeScale.FIVE_MINUTE in config.sequence_lengths
    
    def test_custom_values(self):
        """Test custom configuration values"""
        custom_config = NetworkConfig(
            feature_dimensions=30,
            hidden_sizes=[128, 64, 32],
            attention_heads=4,
            dropout_rate=0.2
        )
        
        assert custom_config.feature_dimensions == 30
        assert custom_config.hidden_sizes == [128, 64, 32]
        assert custom_config.attention_heads == 4
        assert custom_config.dropout_rate == 0.2

class TestMultiHeadAttention:
    """Test MultiHeadAttention module"""
    
    def test_initialization(self):
        """Test MultiHeadAttention initialization"""
        attention = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.1)
        
        assert isinstance(attention, MultiHeadAttention)
        assert isinstance(attention.attention, torch.nn.MultiheadAttention)
    
    def test_forward_pass(self):
        """Test forward pass through MultiHeadAttention"""
        attention = MultiHeadAttention(embed_dim=64, num_heads=4, dropout=0.1)
        
        # Create dummy input
        batch_size = 8
        seq_length = 10
        embed_dim = 64
        input_tensor = torch.randn(batch_size, seq_length, embed_dim)
        
        # Test forward pass
        attended, attention_weights = attention(input_tensor)
        
        assert attended.shape == (batch_size, seq_length, embed_dim)
        assert attention_weights.shape == (batch_size, seq_length, seq_length)
        assert not torch.isnan(attended).any()
        assert not torch.isinf(attended).any()
        assert not torch.isnan(attention_weights).any()
        assert not torch.isinf(attention_weights).any()

class TestTemporalEncoder:
    """Test TemporalEncoder module"""
    
    def test_initialization(self):
        """Test TemporalEncoder initialization"""
        encoder = TemporalEncoder(
            input_size=20,
            hidden_size=64,
            sequence_length=50,
            dropout=0.1
        )
        
        assert isinstance(encoder, TemporalEncoder)
        assert isinstance(encoder.lstm, torch.nn.LSTM)
        assert isinstance(encoder.attention, MultiHeadAttention)
        assert isinstance(encoder.pool, torch.nn.AdaptiveAvgPool1d)
    
    def test_forward_pass(self):
        """Test forward pass through TemporalEncoder"""
        encoder = TemporalEncoder(
            input_size=20,
            hidden_size=64,
            sequence_length=50,
            dropout=0.1
        )
        
        # Create dummy input
        batch_size = 4
        seq_length = 50
        input_size = 20
        input_tensor = torch.randn(batch_size, seq_length, input_size)
        
        # Test forward pass
        pooled_output, attention_weights = encoder(input_tensor)
        
        assert pooled_output.shape == (batch_size, 64)
        assert attention_weights.shape == (batch_size, seq_length, seq_length)
        assert not torch.isnan(pooled_output).any()
        assert not torch.isinf(pooled_output).any()

class TestMultiScaleProcessor:
    """Test MultiScaleProcessor module"""
    
    def test_initialization(self):
        """Test MultiScaleProcessor initialization"""
        config = NetworkConfig()
        processor = MultiScaleProcessor(config)
        
        assert isinstance(processor, MultiScaleProcessor)
        assert isinstance(processor.config, NetworkConfig)
        assert isinstance(processor.encoders, torch.nn.ModuleDict)
        
        # Check that encoders were created for each time scale
        for time_scale in TimeScale:
            assert time_scale.value in processor.encoders
    
    def test_forward_pass(self):
        """Test forward pass through MultiScaleProcessor"""
        config = NetworkConfig()
        processor = MultiScaleProcessor(config)
        
        # Create dummy inputs for each time scale
        batch_size = 2
        inputs = {
            TimeScale.SECOND: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.SECOND], 
                config.feature_dimensions
            ),
            TimeScale.TEN_SECOND: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.TEN_SECOND], 
                config.feature_dimensions
            ),
            TimeScale.MINUTE: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.MINUTE], 
                config.feature_dimensions
            ),
            TimeScale.FIVE_MINUTE: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.FIVE_MINUTE], 
                config.feature_dimensions
            )
        }
        
        # Test forward pass
        fused_output, attention_weights = processor(inputs)
        
        assert fused_output.shape == (batch_size, config.hidden_sizes[0])
        assert isinstance(attention_weights, dict)
        # Check that attention weights are returned for each time scale
        for time_scale in inputs.keys():
            assert time_scale.value in attention_weights

class TestBayesianEstimator:
    """Test BayesianEstimator module"""
    
    def test_initialization(self):
        """Test BayesianEstimator initialization"""
        estimator = BayesianEstimator(
            input_size=64,
            output_size=3,
            hidden_sizes=[32, 16],
            dropout=0.1
        )
        
        assert isinstance(estimator, BayesianEstimator)
        assert isinstance(estimator.network, torch.nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass through BayesianEstimator"""
        estimator = BayesianEstimator(
            input_size=64,
            output_size=3,
            hidden_sizes=[32, 16],
            dropout=0.1
        )
        
        # Create dummy input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 64)
        
        # Test forward pass
        mean, variance = estimator(input_tensor)
        
        assert mean.shape == (batch_size, 3)
        assert variance.shape == (batch_size, 3)
        assert not torch.isnan(mean).any()
        assert not torch.isinf(mean).any()
        assert torch.all(variance > 0)  # Variance should be positive
    
    def test_sample_predictions(self):
        """Test prediction sampling"""
        estimator = BayesianEstimator(
            input_size=64,
            output_size=3,
            hidden_sizes=[32, 16],
            dropout=0.1
        )
        
        # Create dummy input
        batch_size = 4
        input_tensor = torch.randn(batch_size, 64)
        
        # Test sampling
        sample_mean, sample_std = estimator.sample_predictions(input_tensor, num_samples=5)
        
        assert sample_mean.shape == (batch_size, 3)
        assert sample_std.shape == (batch_size, 3)
        assert not torch.isnan(sample_mean).any()
        assert not torch.isinf(sample_mean).any()
        assert not torch.isnan(sample_std).any()
        assert not torch.isinf(sample_std).any()
        assert torch.all(sample_std >= 0)  # Standard deviation should be non-negative

class TestGraphNeuralNetwork:
    """Test GraphNeuralNetwork module"""
    
    def test_initialization(self):
        """Test GraphNeuralNetwork initialization"""
        gnn = GraphNeuralNetwork(
            node_features=32,
            hidden_size=64,
            num_layers=2,
            dropout=0.1
        )
        
        assert isinstance(gnn, GraphNeuralNetwork)
        assert isinstance(gnn.gcns, torch.nn.ModuleList)
        assert len(gnn.gcns) == 2
    
    def test_forward_pass(self):
        """Test forward pass through GraphNeuralNetwork"""
        gnn = GraphNeuralNetwork(
            node_features=32,
            hidden_size=64,
            num_layers=2,
            dropout=0.1
        )
        
        # Create dummy inputs
        num_nodes = 10
        node_features = torch.randn(num_nodes, 32)
        adjacency_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
        
        # Test forward pass
        output = gnn(node_features, adjacency_matrix)
        
        assert output.shape == (num_nodes, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestEnhancedTradingNeuralNetwork:
    """Test EnhancedTradingNeuralNetwork main class"""
    
    @pytest.fixture
    def network(self):
        """Create EnhancedTradingNeuralNetwork instance for testing"""
        config = NetworkConfig()
        return EnhancedTradingNeuralNetwork(config)
    
    def test_initialization(self, network):
        """Test EnhancedTradingNeuralNetwork initialization"""
        assert isinstance(network, EnhancedTradingNeuralNetwork)
        assert isinstance(network.config, NetworkConfig)
        assert isinstance(network.multi_scale, MultiScaleProcessor)
        assert isinstance(network.bayesian_estimator, BayesianEstimator)
        assert isinstance(network.gnn, GraphNeuralNetwork)
        assert isinstance(network.decision_layer, torch.nn.Linear)
    
    def test_forward_pass(self, network):
        """Test forward pass through EnhancedTradingNeuralNetwork"""
        # Create dummy inputs
        batch_size = 2
        config = network.config
        
        temporal_inputs = {
            TimeScale.SECOND: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.SECOND], 
                config.feature_dimensions
            ),
            TimeScale.TEN_SECOND: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.TEN_SECOND], 
                config.feature_dimensions
            ),
            TimeScale.MINUTE: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.MINUTE], 
                config.feature_dimensions
            ),
            TimeScale.FIVE_MINUTE: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.FIVE_MINUTE], 
                config.feature_dimensions
            )
        }
        
        # Test forward pass without graph inputs
        mean, uncertainty, attention_weights = network(temporal_inputs)
        
        assert mean.shape == (batch_size, 3)
        assert uncertainty.shape == (batch_size, 3)
        assert isinstance(attention_weights, dict)
        
        # Test forward pass with graph inputs
        num_pairs = 5
        graph_inputs = torch.randn(num_pairs, config.hidden_sizes[0])
        adjacency_matrix = torch.randint(0, 2, (num_pairs, num_pairs)).float()
        
        mean, uncertainty, attention_weights = network(
            temporal_inputs, graph_inputs, adjacency_matrix
        )
        
        assert mean.shape == (batch_size, 3)
        assert uncertainty.shape == (batch_size, 3)
    
    def test_predict_with_uncertainty(self, network):
        """Test prediction with uncertainty"""
        # Create dummy inputs
        batch_size = 2
        config = network.config
        
        temporal_inputs = {
            TimeScale.SECOND: torch.randn(
                batch_size, 
                config.sequence_lengths[TimeScale.SECOND], 
                config.feature_dimensions
            )
        }
        
        # Test prediction
        mean_pred, uncertainty_pred, confidence = network.predict_with_uncertainty(
            temporal_inputs
        )
        
        assert mean_pred.shape == (batch_size, 3)
        assert uncertainty_pred.shape == (batch_size, 3)
        assert confidence.shape == (batch_size, 3)
        # Confidence should be between 0 and 1
        assert torch.all((confidence >= 0) & (confidence <= 1))

if __name__ == "__main__":
    pytest.main([__file__])