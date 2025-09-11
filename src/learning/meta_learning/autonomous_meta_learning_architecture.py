"""
Autonomous Meta-Learning Architecture
====================================

This module implements the core meta-learning architecture that enables the autonomous
neural network to learn how to learn. It provides the foundation for all self-learning
capabilities in the trading system.

Key Features:
- Model-Agnostic Meta-Learning (MAML) for rapid adaptation
- Gradient-based meta-learning for strategy optimization  
- Experience-based meta-learning for market condition adaptation
- Hierarchical meta-learning for multi-level optimization
- Continuous meta-learning for ongoing improvement

Author: Autonomous Systems Team
Date: 2025-01-22
Task: 14.1.1 - Design autonomous meta-learning architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import time
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning architecture"""
    # Meta-learning parameters
    meta_learning_rate: float = 0.001
    inner_learning_rate: float = 0.01
    inner_steps: int = 5
    meta_batch_size: int = 16
    
    # Task parameters
    support_size: int = 32    # Support set size for few-shot learning
    query_size: int = 16      # Query set size for evaluation
    max_tasks: int = 1000     # Maximum number of meta-tasks to store
    
    # Adaptation parameters
    adaptation_steps: int = 3
    adaptation_threshold: float = 0.1  # Performance improvement threshold
    
    # Memory parameters
    memory_size: int = 10000
    meta_memory_size: int = 1000
    
    # Optimization parameters
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5


@dataclass
class MetaTask:
    """Represents a meta-learning task"""
    task_id: str
    task_type: str  # "market_regime", "strategy_adaptation", "risk_adjustment"
    support_data: torch.Tensor
    support_labels: torch.Tensor
    query_data: torch.Tensor
    query_labels: torch.Tensor
    task_context: Dict[str, Any]
    difficulty: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MetaLearningResult:
    """Results from meta-learning process"""
    task_id: str
    initial_loss: float
    adapted_loss: float
    adaptation_steps: int
    meta_loss: float
    adaptation_time: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetaLearner(ABC):
    """Abstract base class for meta-learners"""
    
    @abstractmethod
    def meta_learn(self, tasks: List[MetaTask]) -> MetaLearningResult:
        """Perform meta-learning on a batch of tasks"""
        pass
    
    @abstractmethod
    def adapt(self, task: MetaTask) -> nn.Module:
        """Adapt the model to a specific task"""
        pass
    
    @abstractmethod
    def get_meta_parameters(self) -> Dict[str, torch.Tensor]:
        """Get meta-parameters for the learner"""
        pass


class MAMLMetaLearner(MetaLearner):
    """
    Model-Agnostic Meta-Learning (MAML) implementation for trading strategies
    """
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        self.model = model
        self.config = config
        self.meta_optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.meta_learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Meta-learning statistics
        self.meta_learning_history = []
        self.adaptation_performance = defaultdict(list)
        
        logger.info("MAML Meta-Learner initialized")
    
    def meta_learn(self, tasks: List[MetaTask]) -> MetaLearningResult:
        """Perform MAML meta-learning update"""
        start_time = time.time()
        
        # Sample tasks for meta-batch
        if len(tasks) > self.config.meta_batch_size:
            indices = np.random.choice(
                len(tasks), self.config.meta_batch_size, replace=False
            )
            tasks = [tasks[i] for i in indices]
        
        meta_losses = []
        task_results = []
        
        # Zero meta-gradients
        self.meta_optimizer.zero_grad()
        
        for task in tasks:
            # Clone model for task adaptation
            adapted_model = copy.deepcopy(self.model)
            adapted_model.train()
            
            # Inner loop: adapt to task
            task_optimizer = optim.SGD(
                adapted_model.parameters(), 
                lr=self.config.inner_learning_rate
            )
            
            # Support set adaptation
            for step in range(self.config.inner_steps):
                task_optimizer.zero_grad()
                
                support_pred = adapted_model(task.support_data)
                support_loss = F.mse_loss(support_pred, task.support_labels)
                
                support_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    adapted_model.parameters(), self.config.gradient_clip
                )
                task_optimizer.step()
            
            # Query set evaluation
            query_pred = adapted_model(task.query_data)
            query_loss = F.mse_loss(query_pred, task.query_labels)
            
            # Accumulate meta-gradients
            query_loss.backward()
            meta_losses.append(query_loss.item())
            
            # Store task result
            task_results.append({
                'task_id': task.task_id,
                'query_loss': query_loss.item(),
                'adaptation_steps': self.config.inner_steps
            })
        
        # Meta-update
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        self.meta_optimizer.step()
        
        # Calculate results
        avg_meta_loss = np.mean(meta_losses)
        adaptation_time = time.time() - start_time
        
        # Store meta-learning history
        self.meta_learning_history.append({
            'timestamp': datetime.now(),
            'meta_loss': avg_meta_loss,
            'num_tasks': len(tasks),
            'adaptation_time': adaptation_time
        })
        
        return MetaLearningResult(
            task_id="meta_batch",
            initial_loss=0.0,  # Not tracked in MAML
            adapted_loss=float(avg_meta_loss),
            adaptation_steps=self.config.inner_steps,
            meta_loss=float(avg_meta_loss),
            adaptation_time=adaptation_time,
            success=bool(avg_meta_loss < self.config.adaptation_threshold),
            metadata={
                'task_results': task_results,
                'meta_batch_size': len(tasks)
            }
        )
    
    def adapt(self, task: MetaTask) -> nn.Module:
        """Adapt model to specific task using MAML"""
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()
        
        optimizer = optim.SGD(
            adapted_model.parameters(), 
            lr=self.config.inner_learning_rate
        )
        
        initial_loss = None
        final_loss = 0.0
        
        for step in range(self.config.adaptation_steps):
            optimizer.zero_grad()
            
            pred = adapted_model(task.support_data)
            loss = F.mse_loss(pred, task.support_labels)
            
            if step == 0:
                initial_loss = loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                adapted_model.parameters(), self.config.gradient_clip
            )
            optimizer.step()
            final_loss = loss.item()
        self.adaptation_performance[task.task_type].append({
            'initial_loss': initial_loss or 0.0,
            'final_loss': final_loss,
            'improvement': (initial_loss or 0.0) - final_loss
        })
        
        return adapted_model
    
    def get_meta_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current meta-parameters"""
        return {name: param.clone() for name, param in self.model.named_parameters()}


class GradientBasedMetaLearner(MetaLearner):
    """
    Gradient-based meta-learning for strategy optimization
    """
    
    def __init__(self, model: nn.Module, config: MetaLearningConfig):
        self.model = model
        self.config = config
        
        # Meta-parameters for gradient computation
        self.meta_parameters = nn.ParameterDict({
            'learning_rates': nn.Parameter(torch.ones(len(list(model.parameters()))) * config.inner_learning_rate),
            'adaptation_weights': nn.Parameter(torch.ones(config.adaptation_steps))
        })
        
        self.meta_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.meta_parameters.parameters()),
            lr=config.meta_learning_rate
        )
        
        logger.info("Gradient-based Meta-Learner initialized")
    
    def meta_learn(self, tasks: List[MetaTask]) -> MetaLearningResult:
        """Perform gradient-based meta-learning"""
        start_time = time.time()
        
        self.meta_optimizer.zero_grad()
        total_meta_loss = 0.0
        
        for task in tasks[:self.config.meta_batch_size]:
            # Compute task-specific gradients
            task_gradients = self._compute_task_gradients(task)
            
            # Apply learned learning rates
            adapted_params = {}
            for i, (name, param) in enumerate(self.model.named_parameters()):
                lr = torch.sigmoid(self.meta_parameters['learning_rates'][i]) * 0.1
                adapted_params[name] = param - lr * task_gradients[name]
            
            # Evaluate adapted model
            query_loss = self._evaluate_adapted_model(adapted_params, task)
            total_meta_loss += query_loss
        
        # Meta-update
        avg_meta_loss = total_meta_loss / min(len(tasks), self.config.meta_batch_size)
        if isinstance(avg_meta_loss, torch.Tensor):
            avg_meta_loss.backward()
            self.meta_optimizer.step()
            avg_loss_value = avg_meta_loss.item()
        else:
            avg_loss_value = float(avg_meta_loss)
        
        adaptation_time = time.time() - start_time
        
        return MetaLearningResult(
            task_id="gradient_meta_batch",
            initial_loss=0.0,
            adapted_loss=avg_loss_value,
            adaptation_steps=self.config.adaptation_steps,
            meta_loss=avg_loss_value,
            adaptation_time=adaptation_time,
            success=avg_loss_value < self.config.adaptation_threshold
        )
    
    def _compute_task_gradients(self, task: MetaTask) -> Dict[str, torch.Tensor]:
        """Compute gradients for a specific task"""
        self.model.zero_grad()
        
        pred = self.model(task.support_data)
        loss = F.mse_loss(pred, task.support_labels)
        loss.backward()
        
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
            else:
                gradients[name] = torch.zeros_like(param)
        
        return gradients
    
    def _evaluate_adapted_model(self, adapted_params: Dict[str, torch.Tensor], 
                              task: MetaTask) -> torch.Tensor:
        """Evaluate model with adapted parameters"""
        # Create adapted model (simplified - in practice would need functional API)
        adapted_model = copy.deepcopy(self.model)
        
        with torch.no_grad():
            for name, param in adapted_model.named_parameters():
                param.copy_(adapted_params[name])
        
        pred = adapted_model(task.query_data)
        return F.mse_loss(pred, task.query_labels)
    
    def adapt(self, task: MetaTask) -> nn.Module:
        """Adapt model using learned gradients"""
        task_gradients = self._compute_task_gradients(task)
        
        adapted_model = copy.deepcopy(self.model)
        
        with torch.no_grad():
            for i, (name, param) in enumerate(adapted_model.named_parameters()):
                lr = torch.sigmoid(self.meta_parameters['learning_rates'][i]) * 0.1
                param -= lr * task_gradients[name]
        
        return adapted_model
    
    def get_meta_parameters(self) -> Dict[str, torch.Tensor]:
        """Get meta-parameters including learned learning rates"""
        meta_params = {name: param.clone() for name, param in self.model.named_parameters()}
        meta_params.update({name: param.clone() for name, param in self.meta_parameters.items()})
        return meta_params


class TaskGenerator:
    """Generates meta-learning tasks from trading data"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.task_templates = {
            'market_regime': self._generate_regime_task,
            'strategy_adaptation': self._generate_strategy_task,
            'risk_adjustment': self._generate_risk_task
        }
        
        logger.info("Task Generator initialized")
    
    def generate_task(self, task_type: str, market_data: torch.Tensor, 
                     labels: torch.Tensor, context: Dict[str, Any]) -> MetaTask:
        """Generate a meta-learning task"""
        if task_type not in self.task_templates:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return self.task_templates[task_type](market_data, labels, context)
    
    def _generate_regime_task(self, market_data: torch.Tensor, 
                            labels: torch.Tensor, context: Dict[str, Any]) -> MetaTask:
        """Generate market regime adaptation task"""
        total_size = len(market_data)
        support_end = self.config.support_size
        
        return MetaTask(
            task_id=f"regime_{int(time.time())}",
            task_type="market_regime",
            support_data=market_data[:support_end],
            support_labels=labels[:support_end],
            query_data=market_data[support_end:support_end + self.config.query_size],
            query_labels=labels[support_end:support_end + self.config.query_size],
            task_context=context,
            difficulty=context.get('volatility', 1.0)
        )
    
    def _generate_strategy_task(self, market_data: torch.Tensor, 
                              labels: torch.Tensor, context: Dict[str, Any]) -> MetaTask:
        """Generate strategy adaptation task"""
        total_size = len(market_data)
        support_end = self.config.support_size
        
        return MetaTask(
            task_id=f"strategy_{int(time.time())}",
            task_type="strategy_adaptation",
            support_data=market_data[:support_end],
            support_labels=labels[:support_end],
            query_data=market_data[support_end:support_end + self.config.query_size],
            query_labels=labels[support_end:support_end + self.config.query_size],
            task_context=context,
            difficulty=context.get('complexity', 1.0)
        )
    
    def _generate_risk_task(self, market_data: torch.Tensor, 
                          labels: torch.Tensor, context: Dict[str, Any]) -> MetaTask:
        """Generate risk adjustment task"""
        total_size = len(market_data)
        support_end = self.config.support_size
        
        return MetaTask(
            task_id=f"risk_{int(time.time())}",
            task_type="risk_adjustment",
            support_data=market_data[:support_end],
            support_labels=labels[:support_end],
            query_data=market_data[support_end:support_end + self.config.query_size],
            query_labels=labels[support_end:support_end + self.config.query_size],
            task_context=context,
            difficulty=context.get('risk_level', 1.0)
        )


class AutonomousMetaLearningArchitecture:
    """
    Main architecture class that coordinates all meta-learning components
    for autonomous neural network trading system
    """
    
    def __init__(self, base_model: nn.Module, config: Optional[MetaLearningConfig] = None):
        self.config = config or MetaLearningConfig()
        self.base_model = base_model
        
        # Initialize meta-learners
        self.maml_learner = MAMLMetaLearner(base_model, self.config)
        self.gradient_learner = GradientBasedMetaLearner(base_model, self.config)
        
        # Task generation and management
        self.task_generator = TaskGenerator(self.config)
        self.task_memory = deque(maxlen=self.config.max_tasks)
        
        # Performance tracking
        self.meta_learning_performance = defaultdict(list)
        self.adaptation_success_rate = defaultdict(float)
        
        # Current active learner
        self.active_learner = self.maml_learner  # Default to MAML
        
        logger.info("Autonomous Meta-Learning Architecture initialized")
    
    def add_task(self, task_type: str, market_data: torch.Tensor, 
                labels: torch.Tensor, context: Dict[str, Any]) -> None:
        """Add a new meta-learning task"""
        task = self.task_generator.generate_task(task_type, market_data, labels, context)
        self.task_memory.append(task)
        
        logger.debug(f"Added task {task.task_id} of type {task_type}")
    
    def meta_learn_batch(self, task_types: Optional[List[str]] = None) -> Dict[str, MetaLearningResult]:
        """Perform meta-learning on a batch of tasks"""
        if not self.task_memory:
            logger.warning("No tasks available for meta-learning")
            return {}
        
        # Filter tasks by type if specified
        if task_types:
            available_tasks = [task for task in self.task_memory if task.task_type in task_types]
        else:
            available_tasks = list(self.task_memory)
        
        if not available_tasks:
            logger.warning(f"No tasks of types {task_types} available")
            return {}
        
        # Perform meta-learning
        results = {}
        
        # MAML meta-learning
        maml_result = self.maml_learner.meta_learn(available_tasks)
        results['maml'] = maml_result
        
        # Gradient-based meta-learning
        gradient_result = self.gradient_learner.meta_learn(available_tasks)
        results['gradient'] = gradient_result
        
        # Update performance tracking
        self._update_performance_tracking(results)
        
        return results
    
    def adapt_to_task(self, task_type: str, market_data: torch.Tensor, 
                     labels: torch.Tensor, context: Dict[str, Any]) -> nn.Module:
        """Adapt the model to a specific task"""
        # Generate task
        task = self.task_generator.generate_task(task_type, market_data, labels, context)
        
        # Use active learner for adaptation
        adapted_model = self.active_learner.adapt(task)
        
        # Evaluate adaptation success
        with torch.no_grad():
            adapted_model.eval()
            pred = adapted_model(task.query_data)
            adaptation_loss = F.mse_loss(pred, task.query_labels).item()
            
            # Update success rate
            previous_rate = self.adaptation_success_rate[task_type]
            success = adaptation_loss < self.config.adaptation_threshold
            new_rate = 0.9 * previous_rate + 0.1 * (1.0 if success else 0.0)
            self.adaptation_success_rate[task_type] = new_rate
        
        logger.info(f"Adapted to task {task.task_id}, loss: {adaptation_loss:.4f}")
        
        return adapted_model
    
    def select_best_learner(self) -> str:
        """Select the best performing meta-learner"""
        if len(self.meta_learning_performance['maml']) < 3:
            return 'maml'  # Default during warmup
        
        # Compare recent performance
        maml_recent = np.mean(self.meta_learning_performance['maml'][-3:])
        gradient_recent = np.mean(self.meta_learning_performance['gradient'][-3:])
        
        if gradient_recent < maml_recent:
            self.active_learner = self.gradient_learner
            return 'gradient'
        else:
            self.active_learner = self.maml_learner
            return 'maml'
    
    def _update_performance_tracking(self, results: Dict[str, MetaLearningResult]) -> None:
        """Update performance tracking metrics"""
        for learner_type, result in results.items():
            self.meta_learning_performance[learner_type].append(result.meta_loss)
            
            # Keep only recent performance history
            if len(self.meta_learning_performance[learner_type]) > 100:
                self.meta_learning_performance[learner_type] = \
                    self.meta_learning_performance[learner_type][-100:]
    
    def get_meta_learning_status(self) -> Dict[str, Any]:
        """Get current meta-learning status"""
        return {
            'active_learner': type(self.active_learner).__name__,
            'total_tasks': len(self.task_memory),
            'task_distribution': {
                task_type: len([t for t in self.task_memory if t.task_type == task_type])
                for task_type in ['market_regime', 'strategy_adaptation', 'risk_adjustment']
            },
            'adaptation_success_rates': dict(self.adaptation_success_rate),
            'recent_performance': {
                learner: np.mean(perf[-5:]) if perf else 0.0
                for learner, perf in self.meta_learning_performance.items()
            }
        }
    
    def save_meta_learning_state(self, filepath: str) -> None:
        """Save meta-learning state"""
        state = {
            'config': self.config,
            'maml_model_state': self.maml_learner.model.state_dict(),
            'gradient_model_state': self.gradient_learner.model.state_dict(),
            'gradient_meta_params': self.gradient_learner.meta_parameters.state_dict(),
            'performance_tracking': dict(self.meta_learning_performance),
            'adaptation_success_rate': dict(self.adaptation_success_rate),
            'active_learner': type(self.active_learner).__name__
        }
        
        torch.save(state, filepath)
        logger.info(f"Meta-learning state saved to {filepath}")
    
    def load_meta_learning_state(self, filepath: str) -> None:
        """Load meta-learning state"""
        state = torch.load(filepath)
        
        self.maml_learner.model.load_state_dict(state['maml_model_state'])
        self.gradient_learner.model.load_state_dict(state['gradient_model_state'])
        self.gradient_learner.meta_parameters.load_state_dict(state['gradient_meta_params'])
        
        self.meta_learning_performance = defaultdict(list, state['performance_tracking'])
        self.adaptation_success_rate = defaultdict(float, state['adaptation_success_rate'])
        
        # Set active learner
        if state['active_learner'] == 'GradientBasedMetaLearner':
            self.active_learner = self.gradient_learner
        else:
            self.active_learner = self.maml_learner
        
        logger.info(f"Meta-learning state loaded from {filepath}")


# Factory function for easy instantiation
def create_autonomous_meta_learning_architecture(
    base_model: nn.Module, 
    config: Optional[MetaLearningConfig] = None
) -> AutonomousMetaLearningArchitecture:
    """Create and return configured autonomous meta-learning architecture"""
    return AutonomousMetaLearningArchitecture(base_model, config)


if __name__ == "__main__":
    # Example usage and testing
    import torch.nn as nn
    
    # Create a simple base model for testing
    base_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 25),
        nn.ReLU(),
        nn.Linear(25, 1)
    )
    
    # Create meta-learning architecture
    config = MetaLearningConfig(
        meta_learning_rate=0.001,
        inner_learning_rate=0.01,
        support_size=16,
        query_size=8
    )
    
    meta_arch = create_autonomous_meta_learning_architecture(base_model, config)
    
    # Simulate adding tasks
    for i in range(10):
        market_data = torch.randn(32, 100)
        labels = torch.randn(32, 1)
        context = {'volatility': np.random.random(), 'regime': 'normal'}
        
        meta_arch.add_task('market_regime', market_data, labels, context)
    
    # Perform meta-learning
    results = meta_arch.meta_learn_batch()
    print(f"Meta-learning results: {results}")
    
    # Get status
    status = meta_arch.get_meta_learning_status()
    print(f"Meta-learning status: {status}")
    
    print("✅ Task 14.1.1 - Autonomous Meta-Learning Architecture - IMPLEMENTED")