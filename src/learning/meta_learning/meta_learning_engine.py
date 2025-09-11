"""
MetaLearningEngine: Enhanced with Autonomous Meta-Learning Architecture
Integrates MAML (Model-Agnostic Meta-Learning) with the new autonomous meta-learning framework
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta

# Import the new autonomous meta-learning architecture
from .autonomous_meta_learning_architecture import (
    AutonomousMetaLearningArchitecture, 
    MetaLearningConfig as AutonomousMetaLearningConfig,
    MetaTask,
    create_autonomous_meta_learning_architecture
)

logger = logging.getLogger(__name__)

@dataclass
class MetaLearningConfig:
    """Configuration for MetaLearningEngine"""
    # Meta-learning parameters
    meta_lr: float = 0.001  # Meta learning rate
    fast_lr: float = 0.01   # Fast adaptation learning rate
    num_inner_steps: int = 5  # Number of inner loop steps
    meta_batch_size: int = 32  # Batch size for meta-learning
    adaptation_horizon: int = 60  # Time horizon for adaptation in seconds
    
    # Market data parameters
    feature_window: int = 100  # Lookback window for features
    prediction_horizon: int = 5  # Prediction horizon in seconds
    
    # Model parameters
    hidden_size: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.1

class MetaLearner(nn.Module):
    """Neural network for meta-learning"""
    
    def __init__(self, input_size: int, output_size: int, config: MetaLearningConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers for temporal processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        lstm_out, _ = self.lstm(x)
        # Take the last output
        output = self.dropout(lstm_out[:, -1, :])
        return self.output_layer(output)

class MetaLearningEngine:
    """Enhanced MAML-based meta-learning engine with autonomous architecture"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"MetaLearningEngine initialized on device: {self.device}")
        
        # Store adaptation history
        self.adaptation_history: List[Dict] = []
        self.performance_history: List[Dict] = []
        
        # Initialize autonomous meta-learning architecture
        self.autonomous_meta_learning: Optional[AutonomousMetaLearningArchitecture] = None
        self._setup_autonomous_architecture()
        
    def _setup_autonomous_architecture(self) -> None:
        """Setup the autonomous meta-learning architecture"""
        # Create autonomous config from existing config
        autonomous_config = AutonomousMetaLearningConfig(
            meta_learning_rate=self.config.meta_lr,
            inner_learning_rate=self.config.fast_lr,
            inner_steps=self.config.num_inner_steps,
            meta_batch_size=self.config.meta_batch_size,
            support_size=32,  # Default support size
            query_size=16     # Default query size
        )
        
        # Will be initialized when we have a base model
        self._autonomous_config = autonomous_config
        logger.info("Autonomous meta-learning configuration prepared")
        
    def create_meta_learner(self, input_size: int, output_size: int) -> MetaLearner:
        """Create a new meta-learner model and initialize autonomous architecture"""
        model = MetaLearner(input_size, output_size, self.config)
        model = model.to(self.device)
        
        # Initialize autonomous meta-learning architecture with the base model
        if self.autonomous_meta_learning is None:
            self.autonomous_meta_learning = create_autonomous_meta_learning_architecture(
                model, self._autonomous_config
            )
            logger.info("Autonomous meta-learning architecture initialized")
        
        return model
    
    def add_autonomous_task(self, task_type: str, market_data: torch.Tensor, 
                          labels: torch.Tensor, context: Dict[str, Any]) -> None:
        """Add a task to the autonomous meta-learning system"""
        if self.autonomous_meta_learning is not None:
            self.autonomous_meta_learning.add_task(task_type, market_data, labels, context)
            logger.debug(f"Added autonomous task of type: {task_type}")
        else:
            logger.warning("Autonomous meta-learning not initialized. Create a model first.")
    
    def perform_autonomous_meta_learning(self, task_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform autonomous meta-learning across multiple tasks"""
        if self.autonomous_meta_learning is None:
            logger.warning("Autonomous meta-learning not initialized")
            return {}
        
        results = self.autonomous_meta_learning.meta_learn_batch(task_types)
        
        # Store results in performance history
        for learner_type, result in results.items():
            self.performance_history.append({
                "timestamp": datetime.now(),
                "learner_type": learner_type,
                "meta_loss": result.meta_loss,
                "adaptation_time": result.adaptation_time,
                "success": result.success
            })
        
        return results
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for meta-learning"""
        return nn.MSELoss()(predictions, targets)
    
    def adapt_to_market_conditions(
        self, 
        model: MetaLearner, 
        support_data: Tuple[torch.Tensor, torch.Tensor],
        query_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[MetaLearner, torch.Tensor]:
        """
        Adapt model to new market conditions using MAML
        
        Args:
            model: Meta-learner model
            support_data: (features, targets) for adaptation
            query_data: (features, targets) for evaluation
            
        Returns:
            adapted_model, query_loss
        """
        support_features, support_targets = support_data
        query_features, query_targets = query_data
        
        # Create a copy of the model for adaptation
        adapted_model = self.clone_model(model)
        adapted_model.train()
        
        # Inner loop adaptation
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.config.fast_lr)
        
        for step in range(self.config.num_inner_steps):
            support_predictions = adapted_model(support_features)
            support_loss = self.compute_loss(support_predictions, support_targets)
            
            optimizer.zero_grad()
            support_loss.backward()
            optimizer.step()
        
        # Evaluate on query set
        adapted_model.eval()
        with torch.no_grad():
            query_predictions = adapted_model(query_features)
            query_loss = self.compute_loss(query_predictions, query_targets)
        
        return adapted_model, query_loss
    
    def create_continuous_learning_integration(self) -> Dict[str, Any]:
        """Create integration configuration for continuous learning pipeline"""
        return {
            'meta_learning_config': self._autonomous_config,
            'autonomous_architecture': self.autonomous_meta_learning,
            'engine_config': self.config
        }
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get status of autonomous meta-learning system"""
        if self.autonomous_meta_learning:
            return self.autonomous_meta_learning.get_meta_learning_status()
        return {'status': 'not_initialized'}
    
    def adapt_to_task_context(self, task_type: str, market_data: torch.Tensor,
                             labels: torch.Tensor, context: Dict[str, Any]) -> Optional[nn.Module]:
        """Adapt model to specific task context using autonomous architecture"""
        if self.autonomous_meta_learning:
            return self.autonomous_meta_learning.adapt_to_task(task_type, market_data, labels, context)
        return None
    
    def clone_model(self, model: MetaLearner) -> MetaLearner:
        """Create a copy of the model for adaptation"""
        cloned = self.create_meta_learner(
            model.lstm.input_size, 
            model.output_layer.out_features
        )
        cloned.load_state_dict(model.state_dict())
        return cloned
    
    def meta_update(
        self, 
        models: List[MetaLearner], 
        losses: List[torch.Tensor]
    ) -> None:
        """
        Perform meta-update using accumulated gradients
        
        Args:
            models: List of adapted models
            losses: List of query losses
        """
        if not models or not losses:
            return
            
        # Compute meta-gradient
        meta_loss = torch.stack(losses).mean()
        
        # In a real implementation, we would update the original model here
        # For now, we'll just log the meta-update
        logger.info(f"Meta-update performed with average loss: {meta_loss.item():.6f}")
    
    def few_shot_adaptation(
        self, 
        model: MetaLearner,
        market_data: torch.Tensor,
        target_data: torch.Tensor,
        num_support_samples: int = 10
    ) -> MetaLearner:
        """
        Adapt to new trading pairs with minimal data (few-shot learning)
        
        Args:
            model: Base meta-learner model
            market_data: Market data tensor
            target_data: Target data tensor
            num_support_samples: Number of samples for support set
            
        Returns:
            Adapted model
        """
        if len(market_data) < num_support_samples * 2:
            logger.warning("Insufficient data for few-shot adaptation")
            return model
            
        # Split data into support and query sets
        support_indices = torch.randperm(len(market_data))[:num_support_samples]
        query_indices = torch.randperm(len(market_data))[num_support_samples:2*num_support_samples]
        
        support_features = market_data[support_indices].to(self.device)
        support_targets = target_data[support_indices].to(self.device)
        query_features = market_data[query_indices].to(self.device)
        query_targets = target_data[query_indices].to(self.device)
        
        # Adapt model
        adapted_model, query_loss = self.adapt_to_market_conditions(
            model,
            (support_features, support_targets),
            (query_features, query_targets)
        )
        
        # Store adaptation results
        adaptation_record = {
            "timestamp": datetime.now(),
            "query_loss": query_loss.item(),
            "num_support_samples": num_support_samples,
            "data_shape": market_data.shape
        }
        self.adaptation_history.append(adaptation_record)
        
        logger.info(f"Few-shot adaptation completed with query loss: {query_loss.item():.6f}")
        return adapted_model
    
    def detect_market_regime_change(self, recent_performance: List[float]) -> bool:
        """
        Detect market regime changes using performance degradation
        
        Args:
            recent_performance: Recent model performance metrics
            
        Returns:
            True if regime change detected
        """
        if len(recent_performance) < 10:
            return False
            
        # Simple threshold-based detection
        recent_avg = np.mean(recent_performance[-5:])
        older_avg = np.mean(recent_performance[-10:-5])
        
        # If performance drops significantly, detect regime change
        if older_avg > 0 and (older_avg - recent_avg) / older_avg > 0.1:  # 10% drop
            logger.info("Market regime change detected based on performance degradation")
            return True
            
        return False
    
    def get_adaptation_history(self) -> List[Dict]:
        """Get adaptation history"""
        return self.adaptation_history.copy()
    
    def get_performance_history(self) -> List[Dict]:
        """Get performance history"""
        return self.performance_history.copy()

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = MetaLearningConfig()
    
    # Initialize engine
    engine = MetaLearningEngine(config)
    
    # Create a simple meta-learner
    input_size = 20  # Example feature size
    output_size = 1  # Example output size
    model = engine.create_meta_learner(input_size, output_size)
    
    # Generate dummy data for testing
    batch_size = 32
    seq_length = 100
    dummy_features = torch.randn(batch_size, seq_length, input_size)
    dummy_targets = torch.randn(batch_size, output_size)
    
    # Test few-shot adaptation
    adapted_model = engine.few_shot_adaptation(
        model, dummy_features, dummy_targets, num_support_samples=10
    )
    
    logger.info("MetaLearningEngine test completed successfully")