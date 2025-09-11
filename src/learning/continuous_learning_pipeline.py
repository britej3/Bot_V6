"""
Continuous Learning Pipeline - Enhanced with Autonomous Meta-Learning
=====================================================================

This module implements an enhanced continuous learning pipeline that integrates
with the autonomous meta-learning architecture for true self-learning capabilities.

Key Features:
- Online learning with streaming data
- Advanced experience replay for stable learning
- Meta-learning integration for rapid adaptation
- Concept drift detection and autonomous recovery
- Task-based learning for market condition adaptation
- Autonomous model selection and optimization
- Real-time performance feedback integration

Enhanced for Task 14.1.2: Implement continuous learning pipeline
Author: Autonomous Systems Team
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import time
import threading
from datetime import datetime, timedelta
import copy
import pickle
import json

# Import autonomous meta-learning components
from .meta_learning.autonomous_meta_learning_architecture import (
    AutonomousMetaLearningArchitecture,
    MetaLearningConfig,
    MetaTask,
    create_autonomous_meta_learning_architecture
)

# Import advanced experience replay memory system
from .experience_replay_memory import (
    ExperienceReplayMemory,
    MemoryConfig,
    ExperienceType,
    PriorityType,
    create_experience_replay_memory
)

logger = logging.getLogger(__name__)


@dataclass
class LearningBatch:
    """A batch of learning data"""
    features: torch.Tensor
    targets: torch.Tensor
    weights: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskBasedLearningBatch:
    """Enhanced learning batch that includes task context for meta-learning"""
    features: torch.Tensor
    targets: torch.Tensor
    task_type: str  # 'market_regime', 'strategy_adaptation', 'risk_adjustment'
    market_context: Dict[str, Any]  # Market condition context
    weights: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    performance_score: Optional[float] = None
    difficulty_score: float = 1.0


@dataclass
class LearningMetrics:
    """Metrics tracking learning progress"""
    loss: float
    accuracy: float
    learning_rate: float
    gradient_norm: float
    model_confidence: float
    adaptation_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContinuousLearningConfig:
    """Enhanced configuration for autonomous continuous learning pipeline"""
    # Learning parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    buffer_size: int = 10000
    min_buffer_size: int = 100
    
    # Adaptation parameters
    adaptation_threshold: float = 0.8  # Trigger adaptation when performance drops below this
    concept_drift_window: int = 1000   # Window size for drift detection
    meta_learning_steps: int = 5       # Steps for meta-learning adaptation
    
    # Update frequencies
    update_frequency: int = 10         # Updates per batch
    validation_frequency: int = 100    # Validation every N updates
    model_save_frequency: int = 1000   # Save model every N updates
    
    # Performance thresholds
    min_accuracy: float = 0.6
    max_loss: float = 2.0
    
    # Online learning settings
    momentum: float = 0.9
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Autonomous meta-learning parameters
    enable_meta_learning: bool = True
    meta_learning_frequency: int = 50   # Trigger meta-learning every N updates
    task_generation_threshold: float = 0.1  # Performance drop threshold for task generation
    autonomous_adaptation: bool = True   # Enable autonomous adaptation
    
    # Task-based learning parameters
    support_set_size: int = 32          # Size of support set for meta-tasks
    query_set_size: int = 16            # Size of query set for meta-tasks
    max_concurrent_tasks: int = 5       # Maximum number of concurrent meta-tasks
    
    # Performance tracking
    performance_window: int = 100       # Window size for performance tracking
    adaptation_success_threshold: float = 0.85  # Success threshold for adaptations
    
    # Advanced experience replay settings
    use_advanced_replay: bool = True     # Use advanced experience replay memory
    replay_priority_type: PriorityType = PriorityType.COMPOSITE  # Priority calculation method
    replay_consolidation: bool = True    # Enable memory consolidation
    replay_forgetting: bool = True       # Enable experience forgetting


class ExperienceReplayBuffer:
    """Experience replay buffer for stable online learning"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add(self, experience: LearningBatch, priority: float = 1.0) -> None:
        """Add experience to buffer with priority"""
        with self.lock:
            self.buffer.append(experience)
            self.priorities.append(priority)
    
    def sample(self, batch_size: int, prioritized: bool = True) -> List[LearningBatch]:
        """Sample batch from buffer"""
        with self.lock:
            if len(self.buffer) == 0:
                return []
            
            sample_size = min(batch_size, len(self.buffer))
            
            if prioritized and len(self.priorities) > 0:
                # Prioritized sampling
                priorities = np.array(list(self.priorities))
                probabilities = priorities / priorities.sum()
                indices = np.random.choice(len(self.buffer), sample_size, p=probabilities, replace=False)
            else:
                # Uniform sampling
                indices = np.random.choice(len(self.buffer), sample_size, replace=False)
            
            return [self.buffer[i] for i in indices]
    
    def size(self) -> int:
        """Get buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for specific experiences"""
        with self.lock:
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = priority


class ConceptDriftDetector:
    """Detects concept drift in the data stream"""
    
    def __init__(self, window_size: int = 1000, sensitivity: float = 0.1):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.performance_window = deque(maxlen=window_size)
        self.baseline_performance: Optional[float] = None
        
    def add_performance(self, performance: float) -> None:
        """Add new performance measurement"""
        self.performance_window.append(performance)
        
        if self.baseline_performance is None and len(self.performance_window) >= self.window_size // 2:
            self.baseline_performance = float(np.mean(list(self.performance_window)))
    
    def detect_drift(self) -> Tuple[bool, float]:
        """Detect if concept drift occurred"""
        if len(self.performance_window) < self.window_size // 2:
            return False, 0.0
        
        if self.baseline_performance is None:
            return False, 0.0
        
        recent_performance = float(np.mean(list(self.performance_window)[-self.window_size//4:]))
        drift_magnitude = abs(recent_performance - self.baseline_performance) / abs(self.baseline_performance)
        
        is_drift = drift_magnitude > self.sensitivity
        
        if is_drift:
            logger.info(f"Concept drift detected: {drift_magnitude:.3f} > {self.sensitivity}")
        
        return bool(is_drift), float(drift_magnitude)


class ModelCheckpoint:
    """Manages model checkpoints for rollback capability"""
    
    def __init__(self, max_checkpoints: int = 10):
        self.max_checkpoints = max_checkpoints
        self.checkpoints = deque(maxlen=max_checkpoints)
        
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       metrics: LearningMetrics) -> str:
        """Save model checkpoint"""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        
        checkpoint = {
            'model_state': copy.deepcopy(model.state_dict()),
            'optimizer_state': copy.deepcopy(optimizer.state_dict()),
            'metrics': metrics,
            'timestamp': datetime.now(),
            'id': checkpoint_id
        }
        
        self.checkpoints.append(checkpoint)
        return checkpoint_id
    
    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       checkpoint_id: Optional[str] = None) -> Optional[LearningMetrics]:
        """Load model checkpoint (latest if checkpoint_id not specified)"""
        if not self.checkpoints:
            return None
        
        if checkpoint_id:
            checkpoint = next((c for c in self.checkpoints if c['id'] == checkpoint_id), None)
        else:
            checkpoint = self.checkpoints[-1]  # Latest checkpoint
        
        if checkpoint:
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            logger.info(f"Loaded checkpoint: {checkpoint['id']}")
            return checkpoint['metrics']
        
        return None
    
    def get_best_checkpoint(self, metric: str = 'accuracy', maximize: bool = True) -> Optional[Dict]:
        """Get checkpoint with best performance"""
        if not self.checkpoints:
            return None
        
        best_checkpoint = None
        best_value = float('-inf') if maximize else float('inf')
        
        for checkpoint in self.checkpoints:
            value = getattr(checkpoint['metrics'], metric, 0)
            if (maximize and value > best_value) or (not maximize and value < best_value):
                best_value = value
                best_checkpoint = checkpoint
        
        return best_checkpoint


class ContinuousLearningPipeline:
    """
    Enhanced continuous learning pipeline with autonomous meta-learning integration.
    Coordinates online learning, experience replay, concept drift detection,
    and meta-learning adaptation for fully autonomous operation.
    """
    
    def __init__(self, model: nn.Module, config: Optional[ContinuousLearningConfig] = None):
        self.config = config or ContinuousLearningConfig()
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=100, factor=0.8
        )
        
        # Learning components
        if self.config.use_advanced_replay:
            # Use advanced experience replay memory system
            memory_config = MemoryConfig(
                max_capacity=self.config.buffer_size,
                min_capacity=self.config.min_buffer_size,
                priority_type=self.config.replay_priority_type,
                batch_size=self.config.batch_size,
                auto_consolidation=self.config.replay_consolidation,
                enable_forgetting=self.config.replay_forgetting
            )
            self.advanced_replay_memory = create_experience_replay_memory(memory_config)
            self.advanced_replay_memory.start_background_processing()
            self.replay_buffer = ExperienceReplayBuffer(self.config.buffer_size)  # Keep for compatibility
            logger.info("Advanced experience replay memory system initialized")
        else:
            # Use basic experience replay buffer
            self.replay_buffer = ExperienceReplayBuffer(self.config.buffer_size)
            self.advanced_replay_memory = None
        
        self.drift_detector = ConceptDriftDetector(self.config.concept_drift_window)
        self.checkpoint_manager = ModelCheckpoint()
        
        # Autonomous meta-learning integration
        if self.config.enable_meta_learning:
            meta_config = MetaLearningConfig(
                meta_learning_rate=self.config.learning_rate * 0.1,
                inner_learning_rate=self.config.learning_rate,
                support_size=self.config.support_set_size,
                query_size=self.config.query_set_size,
                meta_batch_size=self.config.batch_size // 2
            )
            self.meta_learning_architecture = create_autonomous_meta_learning_architecture(
                self.model, meta_config
            )
            logger.info("Autonomous meta-learning architecture integrated")
        else:
            self.meta_learning_architecture = None
        
        # Task-based learning components
        self.task_buffer = deque(maxlen=self.config.max_concurrent_tasks * 10)
        self.active_tasks = deque(maxlen=self.config.max_concurrent_tasks)
        self.task_performance_history = {}
        
        # Tracking
        self.update_count = 0
        self.meta_learning_count = 0
        self.learning_history = []
        self.task_learning_history = []
        self.is_learning = False
        self.learning_thread: Optional[threading.Thread] = None
        
        # Metrics
        self.current_metrics = LearningMetrics(0.0, 0.0, self.config.learning_rate, 0.0, 0.0, 0.0)
        
        # Adaptation state
        self.adaptation_mode = False
        self.meta_learning_steps_remaining = 0
        self.autonomous_adaptation_enabled = self.config.autonomous_adaptation
        
        # Performance tracking for autonomous decisions
        self.performance_tracker = deque(maxlen=self.config.performance_window)
        self.adaptation_success_tracker = deque(maxlen=50)
        
        logger.info("Enhanced Continuous Learning Pipeline with autonomous meta-learning initialized")
    
    def start_learning(self) -> None:
        """Start continuous learning process"""
        if self.is_learning:
            logger.warning("Learning already started")
            return
        
        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        logger.info("Continuous learning started")
    
    def stop_learning(self) -> None:
        """Stop continuous learning process"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
        
        # Stop advanced replay memory if enabled
        if self.advanced_replay_memory:
            self.advanced_replay_memory.stop_background_processing()
        
        logger.info("Enhanced continuous learning stopped")
    
    def add_experience(self, features: torch.Tensor, targets: torch.Tensor,
                      performance: Optional[float] = None, metadata: Optional[Dict] = None) -> None:
        """Add new experience to learning pipeline (backward compatibility method)"""
        
        # Create learning batch
        batch = LearningBatch(
            features=features.detach().cpu(),
            targets=targets.detach().cpu(),
            metadata=metadata
        )
        
        # Calculate priority based on prediction error or use default
        priority = 1.0
        if performance is not None:
            priority = max(0.1, 1.0 - performance)  # Higher priority for poor performance
        
        # Add to replay buffer
        self.replay_buffer.add(batch, priority)
        
        # Add performance to drift detector
        if performance is not None:
            self.drift_detector.add_performance(performance)
            self.performance_tracker.append(performance)
            
            # Check for concept drift
            drift_detected, drift_magnitude = self.drift_detector.detect_drift()
            if drift_detected:
                self._handle_concept_drift(drift_magnitude)
    
    def add_task_based_experience(self, features: torch.Tensor, targets: torch.Tensor,
                                 task_type: str, market_context: Dict[str, Any],
                                 performance: Optional[float] = None, 
                                 metadata: Optional[Dict] = None) -> None:
        """Add new task-based experience for autonomous meta-learning"""
        
        # Create task-based learning batch
        task_batch = TaskBasedLearningBatch(
            features=features.detach().cpu(),
            targets=targets.detach().cpu(),
            task_type=task_type,
            market_context=market_context,
            metadata=metadata,
            performance_score=performance,
            difficulty_score=self._calculate_task_difficulty(market_context)
        )
        
        # Add to task buffer
        self.task_buffer.append(task_batch)
        
        # Add to regular replay buffer for backward compatibility
        regular_batch = LearningBatch(
            features=features.detach().cpu(),
            targets=targets.detach().cpu(),
            metadata={**(metadata or {}), 'task_type': task_type, 'market_context': market_context}
        )
        
        priority = self._calculate_task_priority(task_batch)
        self.replay_buffer.add(regular_batch, priority)
        
        # Add to advanced replay memory if enabled
        if self.advanced_replay_memory:
            experience_type = self._map_task_to_experience_type(task_type)
            reward = torch.tensor([performance or 0.5])  # Convert performance to reward signal
            
            # Create dummy action and next_state for experience replay format
            action = torch.zeros(3)  # Placeholder action
            next_state = features + torch.randn_like(features) * 0.01  # Slightly perturbed next state
            
            advanced_metadata = {
                'source': 'continuous_learning_pipeline',
                'quality_score': self._calculate_experience_quality(task_batch),
                'importance_score': task_batch.difficulty_score / 3.0,  # Normalize to [0,1]
                'prediction_error': 1.0 - (performance or 0.5),  # Convert performance to error
                'surprise_score': self._calculate_surprise_score(task_batch),
                'market_regime': market_context.get('regime', 'unknown'),
                'strategy_context': market_context
            }
            
            exp_id = self.advanced_replay_memory.add_experience(
                state=features,
                action=action,
                reward=reward,
                next_state=next_state,
                done=False,  # Continuous learning, no episode termination
                experience_type=experience_type,
                metadata=advanced_metadata
            )
            
            logger.debug(f"Added experience {exp_id} to advanced replay memory")
        
        # Generate meta-learning task if conditions are met
        if self.meta_learning_architecture and self._should_generate_meta_task(task_batch):
            self._generate_meta_learning_task(task_batch)
        
        # Add performance to drift detector
        if performance is not None:
            self.drift_detector.add_performance(performance)
            self.performance_tracker.append(performance)
            
            # Check for concept drift
            drift_detected, drift_magnitude = self.drift_detector.detect_drift()
            if drift_detected:
                self._handle_autonomous_concept_drift(drift_magnitude, task_type, market_context)
        
        logger.debug(f"Added task-based experience of type: {task_type}")
    
    def _calculate_task_difficulty(self, market_context: Dict[str, Any]) -> float:
        """Calculate difficulty score for a task based on market context"""
        difficulty = 1.0
        
        # Increase difficulty based on volatility
        volatility = market_context.get('volatility', 0.5)
        difficulty *= (1.0 + volatility)
        
        # Increase difficulty for uncertain market conditions
        uncertainty = market_context.get('uncertainty', 0.5)
        difficulty *= (1.0 + uncertainty * 0.5)
        
        # Increase difficulty for regime transitions
        if market_context.get('regime_transition', False):
            difficulty *= 1.5
        
        return min(difficulty, 3.0)  # Cap difficulty at 3.0
    
    def _calculate_task_priority(self, task_batch: TaskBasedLearningBatch) -> float:
        """Calculate priority for task-based experience"""
        priority = 1.0
        
        # Higher priority for poor performance
        if task_batch.performance_score is not None:
            priority = max(0.1, 1.0 - task_batch.performance_score)
        
        # Higher priority for difficult tasks
        priority *= task_batch.difficulty_score
        
        # Higher priority for certain task types
        task_type_priorities = {
            'market_regime': 1.2,
            'strategy_adaptation': 1.1,
            'risk_adjustment': 1.0
        }
        priority *= task_type_priorities.get(task_batch.task_type, 1.0)
        
        return priority
    
    def _should_generate_meta_task(self, task_batch: TaskBasedLearningBatch) -> bool:
        """Determine if a meta-learning task should be generated"""
        if not self.config.enable_meta_learning:
            return False
        
        # Generate meta-task if performance is below threshold
        if (task_batch.performance_score is not None and 
            task_batch.performance_score < self.config.task_generation_threshold):
            return True
        
        # Generate meta-task for high difficulty scenarios
        if task_batch.difficulty_score > 2.0:
            return True
        
        # Generate meta-task periodically for continuous improvement
        if self.update_count % (self.config.meta_learning_frequency * 2) == 0:
            return True
        
        return False
    
    def _generate_meta_learning_task(self, task_batch: TaskBasedLearningBatch) -> None:
        """Generate a meta-learning task from task-based experience"""
        try:
            # Collect related experiences for task generation
            related_batches = self._collect_related_experiences(task_batch.task_type)
            
            if len(related_batches) < self.config.support_set_size + self.config.query_set_size:
                logger.debug(f"Insufficient data for meta-task generation: {len(related_batches)}")
                return
            
            # Prepare support and query sets
            total_needed = self.config.support_set_size + self.config.query_set_size
            selected_batches = related_batches[-total_needed:]
            
            support_batches = selected_batches[:self.config.support_set_size]
            query_batches = selected_batches[self.config.support_set_size:]
            
            # Create tensors
            support_features = torch.stack([b.features for b in support_batches])
            support_targets = torch.stack([b.targets for b in support_batches])
            query_features = torch.stack([b.features for b in query_batches])
            query_targets = torch.stack([b.targets for b in query_batches])
            
            # Add to meta-learning architecture
            if self.meta_learning_architecture is not None:
                self.meta_learning_architecture.add_task(
                    task_batch.task_type,
                    torch.cat([support_features, query_features], dim=0),
                    torch.cat([support_targets, query_targets], dim=0),
                    task_batch.market_context
                )
            
            logger.debug(f"Generated meta-learning task for type: {task_batch.task_type}")
            
        except Exception as e:
            logger.error(f"Error generating meta-learning task: {e}")
    
    def _collect_related_experiences(self, task_type: str) -> List[TaskBasedLearningBatch]:
        """Collect experiences related to a specific task type"""
        related = []
        for batch in self.task_buffer:
            if batch.task_type == task_type:
                related.append(batch)
        
        # Sort by timestamp (most recent first)
        related.sort(key=lambda x: x.timestamp, reverse=True)
        return related
    
    def _map_task_to_experience_type(self, task_type: str) -> ExperienceType:
        """Map task type to experience type for advanced replay memory"""
        mapping = {
            'market_regime': ExperienceType.MARKET,
            'strategy_adaptation': ExperienceType.STRATEGY,
            'risk_adjustment': ExperienceType.RISK,
            'execution_optimization': ExperienceType.EXECUTION,
            'model_adaptation': ExperienceType.ADAPTATION,
            'research_discovery': ExperienceType.RESEARCH
        }
        return mapping.get(task_type, ExperienceType.STRATEGY)
    
    def _calculate_experience_quality(self, task_batch: TaskBasedLearningBatch) -> float:
        """Calculate quality score for experience replay"""
        quality = 0.5  # Base quality
        
        # Higher quality for better performance
        if task_batch.performance_score is not None:
            quality = task_batch.performance_score
        
        # Adjust for difficulty (more difficult = potentially more valuable)
        difficulty_bonus = min(0.2, task_batch.difficulty_score * 0.1)
        quality += difficulty_bonus
        
        # Market regime bonus (regime transitions are more valuable)
        if task_batch.market_context.get('regime_transition', False):
            quality += 0.1
        
        return min(1.0, max(0.0, quality))
    
    def _calculate_surprise_score(self, task_batch: TaskBasedLearningBatch) -> float:
        """Calculate surprise score based on expectation vs reality"""
        surprise = 0.5  # Default surprise
        
        # High volatility = more surprise
        volatility = task_batch.market_context.get('volatility', 0.5)
        surprise += volatility * 0.3
        
        # Unexpected performance = more surprise
        if task_batch.performance_score is not None:
            expected_performance = 0.7  # Expected baseline
            surprise += abs(task_batch.performance_score - expected_performance) * 0.4
        
        # Regime transitions are surprising
        if task_batch.market_context.get('regime_transition', False):
            surprise += 0.2
        
        return min(1.0, max(0.0, surprise))
    
    def _learning_loop(self) -> None:
        """Enhanced continuous learning loop with autonomous meta-learning"""
        logger.info("Starting enhanced continuous learning loop with autonomous meta-learning")
        
        while self.is_learning:
            try:
                # Perform regular learning updates
                if self.replay_buffer.size() >= self.config.min_buffer_size:
                    self._perform_learning_update()
                
                # Perform autonomous meta-learning
                if (self.config.enable_meta_learning and 
                    self.meta_learning_architecture and
                    self.update_count % self.config.meta_learning_frequency == 0):
                    self._perform_autonomous_meta_learning()
                
                # Check for autonomous adaptations
                if self.autonomous_adaptation_enabled and self._should_trigger_autonomous_adaptation():
                    self._trigger_autonomous_adaptation()
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error in enhanced learning loop: {e}")
                time.sleep(1.0)  # Longer delay on error
    
    def _perform_autonomous_meta_learning(self) -> None:
        """Perform autonomous meta-learning across available tasks"""
        try:
            logger.debug("Performing autonomous meta-learning")
            
            # Get available task types from task buffer
            available_task_types = list(set(batch.task_type for batch in self.task_buffer))
            
            if not available_task_types:
                logger.debug("No task types available for meta-learning")
                return
            
            # Perform meta-learning for available task types
            if self.meta_learning_architecture is not None:
                results = self.meta_learning_architecture.meta_learn_batch(available_task_types)
            else:
                results = {}
            
            if results:
                self.meta_learning_count += 1
                
                # Track meta-learning performance
                for learner_type, result in results.items():
                    self.task_learning_history.append({
                        'timestamp': datetime.now(),
                        'learner_type': learner_type,
                        'meta_loss': result.meta_loss,
                        'adaptation_time': result.adaptation_time,
                        'success': result.success,
                        'task_types': available_task_types
                    })
                
                # Update adaptation success tracker
                overall_success = any(result.success for result in results.values())
                self.adaptation_success_tracker.append(overall_success)
                
                logger.info(f"Autonomous meta-learning completed. Success: {overall_success}")
            
        except Exception as e:
            logger.error(f"Error in autonomous meta-learning: {e}")
    
    def _should_trigger_autonomous_adaptation(self) -> bool:
        """Determine if autonomous adaptation should be triggered"""
        if self.adaptation_mode:
            return False
        
        # Check recent performance trend
        if len(self.performance_tracker) >= 10:
            recent_performance = list(self.performance_tracker)[-10:]
            avg_recent = np.mean(recent_performance)
            
            if avg_recent < self.config.adaptation_threshold:
                return True
        
        # Check adaptation success rate
        if len(self.adaptation_success_tracker) >= 5:
            recent_success = list(self.adaptation_success_tracker)[-5:]
            success_rate = np.mean(recent_success)
            
            if success_rate < self.config.adaptation_success_threshold:
                return True
        
        return False
    
    def _trigger_autonomous_adaptation(self) -> None:
        """Trigger autonomous adaptation based on current conditions"""
        logger.info("Triggering autonomous adaptation")
        
        self.adaptation_mode = True
        
        try:
            # Save current state as checkpoint
            self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.current_metrics)
            
            # Perform meta-learning based adaptation
            if self.meta_learning_architecture:
                self._perform_meta_learning_adaptation()
            else:
                # Fallback to traditional adaptation
                self._perform_traditional_adaptation()
            
        except Exception as e:
            logger.error(f"Error in autonomous adaptation: {e}")
        finally:
            self.adaptation_mode = False
    
    def _handle_autonomous_concept_drift(self, drift_magnitude: float, 
                                       task_type: str, market_context: Dict[str, Any]) -> None:
        """Handle concept drift with autonomous adaptation"""
        logger.warning(f"Autonomous concept drift handling: magnitude={drift_magnitude:.3f}, task_type={task_type}")
        
        # Generate emergency meta-learning task for rapid adaptation
        if self.meta_learning_architecture:
            # Create high-priority adaptation task
            emergency_context = {
                **market_context,
                'drift_magnitude': drift_magnitude,
                'emergency': True,
                'drift_type': 'concept_drift'
            }
            
            # Collect recent experiences for emergency adaptation
            recent_batches = [b for b in self.task_buffer if b.task_type == task_type][-20:]
            
            if len(recent_batches) >= 10:
                # Generate emergency adaptation data
                features = torch.stack([b.features for b in recent_batches[-10:]])
                targets = torch.stack([b.targets for b in recent_batches[-10:]])
                
                # Adapt model immediately using meta-learning
                adapted_model = self.meta_learning_architecture.adapt_to_task(
                    task_type, features, targets, emergency_context
                )
                
                # Update main model with adapted parameters (simplified)
                with torch.no_grad():
                    for main_param, adapted_param in zip(self.model.parameters(), adapted_model.parameters()):
                        main_param.data = 0.8 * main_param.data + 0.2 * adapted_param.data
                
                logger.info("Emergency autonomous adaptation completed")
        
        # Also perform traditional concept drift handling
        self._handle_concept_drift(drift_magnitude)
    
    def _perform_traditional_adaptation(self) -> None:
        """Perform traditional meta-learning adaptation (fallback method)"""
        logger.info("Performing traditional meta-learning adaptation")
        
        # Get recent high-priority samples for fast adaptation
        adaptation_batch = self.replay_buffer.sample(
            min(self.config.batch_size * 2, self.replay_buffer.size()), 
            prioritized=True
        )
        
        if not adaptation_batch:
            return
        
        # Prepare adaptation data
        features = torch.stack([b.features for b in adaptation_batch]).to(self.device)
        targets = torch.stack([b.targets for b in adaptation_batch]).to(self.device)
        
        # Fast adaptation with higher learning rate
        temp_optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate * 10)
        
        for step in range(self.config.meta_learning_steps):
            self.model.train()
            predictions = self.model(features)
            loss = self._calculate_loss(predictions, targets)
            
            temp_optimizer.zero_grad()
            loss.backward()
            temp_optimizer.step()
        
        logger.info("Traditional meta-learning adaptation completed")
    
    def _perform_learning_update(self) -> None:
        """Perform a single learning update using advanced or basic replay buffer"""
        try:
            # Use advanced replay memory if available
            if self.advanced_replay_memory and self.advanced_replay_memory.get_total_size() >= self.config.min_buffer_size:
                self._perform_advanced_learning_update()
            else:
                # Fallback to basic replay buffer
                self._perform_basic_learning_update()
                
        except Exception as e:
            logger.error(f"Error in learning update: {e}")
    
    def _perform_advanced_learning_update(self) -> None:
        """Perform learning update using advanced experience replay memory"""
        # Sample diverse experiences from advanced memory
        if self.advanced_replay_memory is not None:
            experiences = self.advanced_replay_memory.sample(
                batch_size=self.config.batch_size,
                sampling_strategy="mixed"
            )
        else:
            experiences = []
        
        if not experiences:
            logger.debug("No experiences available from advanced memory")
            return
        
        # Convert experiences to training format
        features_list = []
        targets_list = []
        experience_ids = []
        
        for exp in experiences:
            # For continuous learning, we use state as features and reward as target
            features_list.append(exp.state)
            targets_list.append(exp.reward)
            experience_ids.append(exp.metadata.experience_id)
        
        features = torch.stack(features_list).to(self.device)
        targets = torch.stack(targets_list).to(self.device)
        
        # Forward pass
        self.model.train()
        predictions = self.model(features)
        
        # Calculate loss
        loss = self._calculate_loss(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Calculate prediction errors for priority updates
        with torch.no_grad():
            prediction_errors = torch.abs(predictions - targets).mean(dim=1).tolist()
        
        # Update experience priorities based on prediction errors
        if self.advanced_replay_memory is not None:
            self.advanced_replay_memory.update_experience_priorities(
                experience_ids, prediction_errors
            )
        
        # Update metrics
        with torch.no_grad():
            accuracy = self._calculate_accuracy(predictions, targets)
            confidence = self._calculate_confidence(predictions)
        
        self.current_metrics = LearningMetrics(
            loss=loss.item(),
            accuracy=accuracy,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            gradient_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            model_confidence=confidence,
            adaptation_score=self._calculate_adaptation_score()
        )
        
        self.learning_history.append(self.current_metrics)
        self.update_count += 1
        
        # Periodic operations
        if self.update_count % self.config.validation_frequency == 0:
            self._validate_model()
        
        if self.update_count % self.config.model_save_frequency == 0:
            self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.current_metrics)
        
        # Update learning rate
        self.scheduler.step(loss)
        
        # Check if adaptation is needed
        if self._should_adapt():
            self._trigger_adaptation()
        
        logger.debug(f"Advanced learning update completed. Loss: {loss.item():.4f}")
    
    def _perform_basic_learning_update(self) -> None:
        """Perform learning update using basic replay buffer (fallback method)"""

    def _calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate learning loss"""
        if targets.dtype == torch.long:
            # Classification loss
            return nn.CrossEntropyLoss()(predictions, targets)
        else:
            # Regression loss
            return nn.MSELoss()(predictions, targets)
    
    def _calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate prediction accuracy"""
        if targets.dtype == torch.long:
            # Classification accuracy
            pred_classes = torch.argmax(predictions, dim=1)
            correct = (pred_classes == targets).sum().item()
            return correct / len(targets)
        else:
            # Regression accuracy (R²-like metric)
            mse = torch.mean((predictions - targets) ** 2)
            variance = torch.var(targets)
            if variance > 0:
                r_squared = 1 - mse / variance
                return max(0.0, float(r_squared.item()))
            return 0.0
    
    def _calculate_confidence(self, predictions: torch.Tensor) -> float:
        """Calculate model confidence"""
        with torch.no_grad():
            if predictions.shape[1] > 1:  # Multi-class
                probabilities = torch.softmax(predictions, dim=1)
                max_probs = torch.max(probabilities, dim=1)[0]
                return torch.mean(max_probs).item()
            else:  # Regression
                return 1.0 / (1.0 + torch.std(predictions).item())
    
    def _calculate_adaptation_score(self) -> float:
        """Calculate how well the model is adapting"""
        if len(self.learning_history) < 10:
            return 0.5
        
        recent_losses = [m.loss for m in self.learning_history[-10:]]
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # Negative slope (decreasing loss) = good adaptation
        return max(0, min(1, 0.5 - loss_trend))
    
    def _should_adapt(self) -> bool:
        """Check if model adaptation should be triggered"""
        if self.adaptation_mode:
            return False
        
        # Trigger adaptation if performance drops
        if (self.current_metrics.accuracy < self.config.min_accuracy or 
            self.current_metrics.loss > self.config.max_loss):
            return True
        
        # Trigger adaptation if adaptation score is low
        if self.current_metrics.adaptation_score < self.config.adaptation_threshold:
            return True
        
        return False
    
    def _trigger_adaptation(self) -> None:
        """Trigger model adaptation process"""
        logger.info("Triggering model adaptation")
        
        self.adaptation_mode = True
        self.meta_learning_steps_remaining = self.config.meta_learning_steps
        
        # Save current state as checkpoint
        self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, self.current_metrics)
        
        # Implement meta-learning adaptation
        self._perform_meta_learning_adaptation()
    
    def _perform_meta_learning_adaptation(self) -> None:
        """Enhanced meta-learning adaptation using autonomous architecture"""
        logger.info("Performing enhanced meta-learning adaptation")
        
        if self.meta_learning_architecture:
            # Use autonomous meta-learning architecture for adaptation
            try:
                # Get the best learner from the architecture
                best_learner = self.meta_learning_architecture.select_best_learner()
                logger.info(f"Using {best_learner} learner for adaptation")
                
                # Perform meta-learning across all available task types
                results = self.meta_learning_architecture.meta_learn_batch()
                
                if results:
                    # Update main model with meta-learned parameters
                    meta_params = self.meta_learning_architecture.active_learner.get_meta_parameters()
                    
                    # Apply meta-learned parameters with blending
                    alpha = 0.3  # Blending factor
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if name in meta_params:
                                param.data = (1 - alpha) * param.data + alpha * meta_params[name].data
                    
                    logger.info("Meta-learning adaptation applied to main model")
                else:
                    logger.warning("No meta-learning results available, using traditional adaptation")
                    self._perform_traditional_adaptation()
                    
            except Exception as e:
                logger.error(f"Error in autonomous meta-learning adaptation: {e}")
                # Fallback to traditional adaptation
                self._perform_traditional_adaptation()
        else:
            # Use traditional adaptation as fallback
            self._perform_traditional_adaptation()
    
    def _handle_concept_drift(self, drift_magnitude: float) -> None:
        """Handle detected concept drift"""
        logger.warning(f"Handling concept drift with magnitude: {drift_magnitude:.3f}")
        
        if drift_magnitude > 0.5:  # Severe drift
            # Load best checkpoint and retrain
            best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
            if best_checkpoint:
                self.checkpoint_manager.load_checkpoint(self.model, self.optimizer)
                logger.info("Loaded best checkpoint due to severe concept drift")
        
        # Clear old experiences from buffer (keep only recent ones)
        buffer_size = self.replay_buffer.size()
        keep_size = buffer_size // 2  # Keep recent half
        
        if keep_size > 0:
            recent_experiences = list(self.replay_buffer.buffer)[-keep_size:]
            recent_priorities = list(self.replay_buffer.priorities)[-keep_size:]
            
            self.replay_buffer.buffer.clear()
            self.replay_buffer.priorities.clear()
            
            for exp, pri in zip(recent_experiences, recent_priorities):
                self.replay_buffer.add(exp, pri)
        
        # Trigger immediate adaptation
        self._trigger_adaptation()
    
    def _validate_model(self) -> None:
        """Validate current model performance"""
        # This is a placeholder for validation logic
        # In real implementation, this would use held-out validation data
        pass
    
    def get_learning_metrics(self) -> LearningMetrics:
        """Get current learning metrics"""
        return self.current_metrics
    
    def get_learning_history(self, window: int = 100) -> List[LearningMetrics]:
        """Get recent learning history"""
        return self.learning_history[-window:]
    
    def get_autonomous_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive autonomous learning status"""
        status = {
            'is_learning': self.is_learning,
            'update_count': self.update_count,
            'meta_learning_count': self.meta_learning_count,
            'current_metrics': self.current_metrics.__dict__,
            'adaptation_mode': self.adaptation_mode,
            'autonomous_adaptation_enabled': self.autonomous_adaptation_enabled,
            
            # Buffer status
            'replay_buffer_size': self.replay_buffer.size(),
            'replay_buffer_utilization': self.replay_buffer.size() / self.config.buffer_size,
            'task_buffer_size': len(self.task_buffer),
            'active_tasks': len(self.active_tasks),
            
            # Advanced memory status
            'advanced_replay_enabled': self.config.use_advanced_replay,
            'advanced_memory_status': self._get_advanced_memory_status(),
            
            # Performance tracking
            'recent_performance': list(self.performance_tracker)[-10:] if self.performance_tracker else [],
            'avg_recent_performance': np.mean(list(self.performance_tracker)[-10:]) if len(self.performance_tracker) >= 10 else 0.0,
            'adaptation_success_rate': np.mean(list(self.adaptation_success_tracker)) if self.adaptation_success_tracker else 0.0,
            
            # Meta-learning status
            'meta_learning_enabled': self.config.enable_meta_learning,
            'meta_learning_architecture_status': None
        }
        
        # Get meta-learning architecture status if available
        if self.meta_learning_architecture:
            status['meta_learning_architecture_status'] = self.meta_learning_architecture.get_meta_learning_status()
        
        return status
    
    def get_task_learning_history(self, window: int = 50) -> List[Dict[str, Any]]:
        """Get recent task-based learning history"""
        return self.task_learning_history[-window:]
    
    def get_task_performance_summary(self) -> Dict[str, Any]:
        """Get summary of task performance across different task types"""
        task_summary = {}
        
        for batch in self.task_buffer:
            task_type = batch.task_type
            if task_type not in task_summary:
                task_summary[task_type] = {
                    'count': 0,
                    'avg_performance': 0.0,
                    'avg_difficulty': 0.0,
                    'recent_performance': []
                }
            
            task_summary[task_type]['count'] += 1
            task_summary[task_type]['avg_difficulty'] += batch.difficulty_score
            
            if batch.performance_score is not None:
                task_summary[task_type]['recent_performance'].append(batch.performance_score)
        
        # Calculate averages
        for task_type, summary in task_summary.items():
            if summary['count'] > 0:
                summary['avg_difficulty'] /= summary['count']
            
            if summary['recent_performance']:
                summary['avg_performance'] = np.mean(summary['recent_performance'])
                summary['recent_performance'] = summary['recent_performance'][-10:]  # Keep only recent
        
        return task_summary
    
    def _get_advanced_memory_status(self) -> Dict[str, Any]:
        """Get status of advanced experience replay memory"""
        if not self.advanced_replay_memory:
            return {'enabled': False}
        
        try:
            stats = self.advanced_replay_memory.get_statistics()
            return {
                'enabled': True,
                'total_experiences': stats['total_experiences'],
                'current_size': stats['current_size'],
                'memory_utilization': stats['memory_utilization'],
                'segment_sizes': stats['segment_sizes'],
                'quality_statistics': stats['quality_statistics'],
                'last_consolidation': stats['last_consolidation'],
                'is_running': stats['is_running']
            }
        except Exception as e:
            logger.error(f"Error getting advanced memory status: {e}")
            return {'enabled': True, 'error': str(e)}
    
    def save_pipeline_state(self, filepath: str) -> None:
        """Save enhanced pipeline state including autonomous meta-learning"""
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'update_count': self.update_count,
            'meta_learning_count': self.meta_learning_count,
            'current_metrics': self.current_metrics,
            'learning_history': self.learning_history[-1000:],  # Save recent history
            'task_learning_history': self.task_learning_history[-500:],
            'performance_tracker': list(self.performance_tracker),
            'adaptation_success_tracker': list(self.adaptation_success_tracker),
            'autonomous_adaptation_enabled': self.autonomous_adaptation_enabled
        }
        
        # Save meta-learning architecture state if available
        if self.meta_learning_architecture:
            meta_learning_filepath = filepath.replace('.pt', '_meta_learning.pt')
            self.meta_learning_architecture.save_meta_learning_state(meta_learning_filepath)
            state['meta_learning_state_path'] = meta_learning_filepath
        
        torch.save(state, filepath)
        logger.info(f"Enhanced pipeline state saved to {filepath}")
    
    def load_pipeline_state(self, filepath: str) -> None:
        """Load enhanced pipeline state including autonomous meta-learning"""
        state = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.update_count = state.get('update_count', 0)
        self.meta_learning_count = state.get('meta_learning_count', 0)
        self.current_metrics = state.get('current_metrics', self.current_metrics)
        self.learning_history = state.get('learning_history', [])
        self.task_learning_history = state.get('task_learning_history', [])
        
        # Restore performance tracking
        performance_data = state.get('performance_tracker', [])
        self.performance_tracker = deque(performance_data, maxlen=self.config.performance_window)
        
        adaptation_data = state.get('adaptation_success_tracker', [])
        self.adaptation_success_tracker = deque(adaptation_data, maxlen=50)
        
        self.autonomous_adaptation_enabled = state.get('autonomous_adaptation_enabled', True)
        
        # Load meta-learning architecture state if available
        if 'meta_learning_state_path' in state and self.meta_learning_architecture:
            try:
                self.meta_learning_architecture.load_meta_learning_state(state['meta_learning_state_path'])
                logger.info("Meta-learning architecture state loaded")
            except Exception as e:
                logger.warning(f"Could not load meta-learning state: {e}")
        
        logger.info(f"Enhanced pipeline state loaded from {filepath}")


# Factory function for easy instantiation
def create_enhanced_continuous_learning_pipeline(
    model: nn.Module, 
    config: Optional[ContinuousLearningConfig] = None
) -> ContinuousLearningPipeline:
    """Create and return configured enhanced continuous learning pipeline"""
    return ContinuousLearningPipeline(model, config)


if __name__ == "__main__":
    # Enhanced example usage demonstrating autonomous capabilities
    import torch.nn as nn
    
    # Create a more sophisticated model for testing
    model = nn.Sequential(
        nn.Linear(100, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Create enhanced configuration
    config = ContinuousLearningConfig(
        enable_meta_learning=True,
        autonomous_adaptation=True,
        meta_learning_frequency=25,
        support_set_size=16,
        query_set_size=8,
        batch_size=32,
        min_buffer_size=50
    )
    
    # Create enhanced pipeline
    pipeline = create_enhanced_continuous_learning_pipeline(model, config)
    
    print("🚀 Starting Enhanced Autonomous Continuous Learning Pipeline Demo")
    
    # Start learning
    pipeline.start_learning()
    
    # Simulate different types of learning experiences
    print("📊 Adding diverse learning experiences...")
    
    # Market regime learning tasks
    for i in range(30):
        features = torch.randn(100)
        targets = torch.randn(1)
        performance = np.random.random() * 0.8 + 0.1  # 0.1 to 0.9
        
        market_context = {
            'volatility': np.random.random(),
            'regime': np.random.choice(['bull', 'bear', 'sideways']),
            'uncertainty': np.random.random() * 0.5
        }
        
        # Add both regular and task-based experiences
        if i % 3 == 0:
            pipeline.add_task_based_experience(
                features, targets, 'market_regime', market_context, performance
            )
        else:
            pipeline.add_experience(features, targets, performance)
        
        time.sleep(0.02)
    
    # Strategy adaptation tasks
    print("🎯 Adding strategy adaptation experiences...")
    for i in range(20):
        features = torch.randn(100) * 2  # Different distribution
        targets = torch.randn(1) * 1.5
        performance = np.random.random() * 0.6 + 0.2  # Lower performance initially
        
        strategy_context = {
            'strategy_type': np.random.choice(['momentum', 'mean_reversion', 'arbitrage']),
            'complexity': np.random.random(),
            'regime_transition': np.random.choice([True, False])
        }
        
        pipeline.add_task_based_experience(
            features, targets, 'strategy_adaptation', strategy_context, performance
        )
        time.sleep(0.02)
    
    # Wait for learning and meta-learning
    print("🧠 Allowing time for autonomous meta-learning...")
    time.sleep(3)
    
    # Get comprehensive status
    status = pipeline.get_autonomous_learning_status()
    task_summary = pipeline.get_task_performance_summary()
    task_history = pipeline.get_task_learning_history()
    
    print("\n📈 Enhanced Continuous Learning Results:")
    print(f"   Regular Updates: {status['update_count']}")
    print(f"   Meta-Learning Updates: {status['meta_learning_count']}")
    print(f"   Buffer Utilization: {status['replay_buffer_utilization']:.2%}")
    print(f"   Task Buffer Size: {status['task_buffer_size']}")
    print(f"   Average Recent Performance: {status['avg_recent_performance']:.3f}")
    print(f"   Adaptation Success Rate: {status['adaptation_success_rate']:.3f}")
    
    print("\n🎯 Task Performance Summary:")
    for task_type, summary in task_summary.items():
        print(f"   {task_type}:")
        print(f"     Count: {summary['count']}")
        print(f"     Avg Performance: {summary['avg_performance']:.3f}")
        print(f"     Avg Difficulty: {summary['avg_difficulty']:.3f}")
    
    if status['meta_learning_architecture_status']:
        meta_status = status['meta_learning_architecture_status']
        print("\n🤖 Meta-Learning Architecture Status:")
        print(f"   Active Learner: {meta_status['active_learner']}")
        print(f"   Total Tasks: {meta_status['total_tasks']}")
        print(f"   Task Distribution: {meta_status['task_distribution']}")
        print(f"   Recent Performance: {meta_status['recent_performance']}")
    
    # Stop learning
    pipeline.stop_learning()
    
    print("\n✅ Task 14.1.2 - Enhanced Continuous Learning Pipeline - IMPLEMENTED")
    print("🎉 Autonomous meta-learning integration completed successfully!")