"""
Unit tests for MetaLearningEngine
"""
import pytest
import torch
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.learning.meta_learning.meta_learning_engine import (
    MetaLearningEngine, 
    MetaLearningConfig, 
    MetaLearner,
    MetaLearningEngine
)

class TestMetaLearningConfig:
    """Test MetaLearningConfig dataclass"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = MetaLearningConfig()
        
        assert config.meta_lr == 0.001
        assert config.fast_lr == 0.01
        assert config.num_inner_steps == 5
        assert config.meta_batch_size == 32
        assert config.adaptation_horizon == 60
        assert config.feature_window == 100
        assert config.prediction_horizon == 5
        assert config.hidden_size == 128
        assert config.num_layers == 2
        assert config.dropout_rate == 0.1

class TestMetaLearner:
    """Test MetaLearner neural network"""
    
    def test_initialization(self):
        """Test MetaLearner initialization"""
        config = MetaLearningConfig()
        model = MetaLearner(input_size=10, output_size=1, config=config)
        
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'output_layer')
        assert hasattr(model, 'dropout')
    
    def test_forward_pass(self):
        """Test forward pass through MetaLearner"""
        config = MetaLearningConfig()
        model = MetaLearner(input_size=10, output_size=1, config=config)
        
        # Create dummy input
        batch_size = 4
        seq_length = 50
        input_tensor = torch.randn(batch_size, seq_length, 10)
        
        # Test forward pass
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestMetaLearningEngine:
    """Test MetaLearningEngine functionality"""
    
    @pytest.fixture
    def engine(self):
        """Create MetaLearningEngine instance for testing"""
        config = MetaLearningConfig()
        return MetaLearningEngine(config)
    
    def test_initialization(self, engine):
        """Test MetaLearningEngine initialization"""
        assert isinstance(engine, MetaLearningEngine)
        assert isinstance(engine.config, MetaLearningConfig)
        assert engine.device is not None
        assert isinstance(engine.adaptation_history, list)
        assert isinstance(engine.performance_history, list)
    
    def test_create_meta_learner(self, engine):
        """Test creating meta-learner model"""
        model = engine.create_meta_learner(input_size=20, output_size=5)
        
        assert isinstance(model, MetaLearner)
        assert model.config == engine.config
        # Check that model is on the correct device
        assert next(model.parameters()).device.type in ['cpu', 'cuda']
    
    def test_compute_loss(self, engine):
        """Test loss computation"""
        predictions = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        
        loss = engine.compute_loss(predictions, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar tensor
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_clone_model(self, engine):
        """Test model cloning"""
        original_model = engine.create_meta_learner(input_size=10, output_size=1)
        cloned_model = engine.clone_model(original_model)
        
        assert isinstance(cloned_model, MetaLearner)
        assert cloned_model != original_model  # Different objects
        # Check that parameters are the same
        for (p1, p2) in zip(original_model.parameters(), cloned_model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_few_shot_adaptation(self, engine):
        """Test few-shot adaptation"""
        model = engine.create_meta_learner(input_size=10, output_size=1)
        
        # Create dummy data
        batch_size = 20
        seq_length = 50
        market_data = torch.randn(batch_size, seq_length, 10)
        target_data = torch.randn(batch_size, 1)
        
        # Test adaptation
        adapted_model = engine.few_shot_adaptation(
            model, market_data, target_data, num_support_samples=5
        )
        
        assert isinstance(adapted_model, MetaLearner)
        assert len(engine.adaptation_history) == 1
        
        # Check adaptation record
        record = engine.adaptation_history[0]
        assert "timestamp" in record
        assert "query_loss" in record
        assert "num_support_samples" in record
        assert "data_shape" in record
    
    def test_detect_market_regime_change(self, engine):
        """Test market regime change detection"""
        # Test with insufficient data
        assert not engine.detect_market_regime_change([0.1, 0.2])
        
        # Test with stable performance
        stable_performance = [0.05] * 15
        assert not engine.detect_market_regime_change(stable_performance)
        
        # Test with performance degradation
        degrading_performance = [0.1] * 5 + [0.05] * 5  # 50% improvement (lower is better)
        assert engine.detect_market_regime_change(degrading_performance)
    
    def test_get_history_methods(self, engine):
        """Test history retrieval methods"""
        # Add some dummy data
        engine.adaptation_history.append({"test": "data"})
        engine.performance_history.append({"test": "data"})
        
        adaptation_history = engine.get_adaptation_history()
        performance_history = engine.get_performance_history()
        
        assert isinstance(adaptation_history, list)
        assert isinstance(performance_history, list)
        assert len(adaptation_history) == 1
        assert len(performance_history) == 1
        # Check that returned lists are copies
        assert adaptation_history is not engine.adaptation_history
        assert performance_history is not engine.performance_history

if __name__ == "__main__":
    pytest.main([__file__])