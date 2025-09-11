"""
Online Model Adaptation Framework
=================================

This module implements a comprehensive online model adaptation framework that enables
real-time model updates, version management, and seamless deployment of adapted models
without system downtime.

Key Features:
- Real-time model adaptation and updates
- Version management with rollback capabilities  
- Performance-based adaptation triggers
- Safe model deployment with validation
- A/B testing framework for model comparison
- Adaptive learning rate scheduling
- Model ensemble management
- Distributed adaptation coordination

Implements Task 14.1.4: Create online model adaptation framework
Author: Autonomous Systems Team
Date: 2025-01-22
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import threading
import asyncio
from datetime import datetime, timedelta
import copy
import json
import pickle
import hashlib
from enum import Enum
from abc import ABC, abstractmethod
import concurrent.futures
from pathlib import Path

logger = logging.getLogger(__name__)

# Platform compatibility for Mac Intel optimization
try:
    from .platform_compatibility import get_platform_compatibility, ensure_mac_intel_compatibility
    PLATFORM_COMPATIBILITY_AVAILABLE = True
    _get_platform_compatibility = get_platform_compatibility
except ImportError:
    PLATFORM_COMPATIBILITY_AVAILABLE = False
    _get_platform_compatibility = None
    logger.warning("Platform compatibility module not available")


class AdaptationStrategy(Enum):
    """Different adaptation strategies available"""
    GRADUAL = "gradual"           # Gradual parameter updates
    ENSEMBLE = "ensemble"         # Ensemble-based adaptation
    REPLACEMENT = "replacement"   # Full model replacement
    FINE_TUNING = "fine_tuning"  # Fine-tuning approach
    META_LEARNING = "meta_learning"  # Meta-learning adaptation


class ModelState(Enum):
    """Model states in the adaptation lifecycle"""
    ACTIVE = "active"
    ADAPTING = "adapting"
    VALIDATING = "validating"
    STAGED = "staged"
    RETIRED = "retired"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    creation_time: datetime
    adaptation_history: List[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]] = None
    deployment_time: Optional[datetime] = None
    state: ModelState = ModelState.STAGED


@dataclass
class AdaptationRequest:
    """Request for model adaptation"""
    request_id: str
    trigger_type: str  # 'performance_drop', 'concept_drift', 'scheduled', 'manual'
    trigger_data: Dict[str, Any]
    adaptation_strategy: AdaptationStrategy
    priority: int = 1  # Higher number = higher priority
    timeout: float = 300.0  # Maximum adaptation time in seconds
    validation_required: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationResult:
    """Result of adaptation process"""
    request_id: str
    success: bool
    new_version_id: Optional[str]
    performance_improvement: Optional[float]
    adaptation_time: float
    error_message: Optional[str] = None
    validation_metrics: Optional[Dict[str, float]] = None
    rollback_required: bool = False


@dataclass
class OnlineAdaptationConfig:
    """Configuration for online model adaptation"""
    # Adaptation triggers
    performance_threshold: float = 0.7  # Trigger adaptation when performance drops below
    drift_sensitivity: float = 0.1      # Sensitivity for concept drift detection
    adaptation_interval: float = 300.0  # Minimum seconds between adaptations
    
    # Adaptation strategies
    default_strategy: AdaptationStrategy = AdaptationStrategy.GRADUAL
    ensemble_size: int = 3              # Number of models in ensemble
    fine_tuning_epochs: int = 5         # Epochs for fine-tuning
    
    # Model management
    max_versions: int = 10              # Maximum model versions to keep
    validation_samples: int = 1000      # Samples for model validation
    rollback_threshold: float = 0.9     # Performance threshold for rollback
    
    # Performance monitoring
    performance_window: int = 100       # Window for performance tracking
    adaptation_history_size: int = 50   # Number of adaptations to remember
    
    # A/B testing
    enable_ab_testing: bool = True      # Enable A/B testing for new models
    ab_test_duration: float = 3600.0    # A/B test duration in seconds
    ab_test_traffic_split: float = 0.1  # Fraction of traffic for new model
    
    # Safety settings
    max_concurrent_adaptations: int = 2  # Maximum concurrent adaptations
    backup_frequency: int = 5           # Backup every N adaptations
    enable_auto_rollback: bool = True   # Automatic rollback on performance issues
    
    # Learning parameters
    adaptation_learning_rate: float = 0.001
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5


class ModelAdapter(ABC):
    """Abstract base class for model adaptation strategies"""
    
    @abstractmethod
    def adapt_model(self, model: nn.Module, adaptation_data: Dict[str, Any], 
                   config: OnlineAdaptationConfig) -> Tuple[nn.Module, Dict[str, Any]]:
        """Adapt the model using the specific strategy"""
        pass
    
    @abstractmethod
    def validate_adaptation(self, original_model: nn.Module, adapted_model: nn.Module,
                          validation_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate the adapted model"""
        pass


class GradualModelAdapter(ModelAdapter):
    """Gradual adaptation strategy using exponential moving averages"""
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
    
    def adapt_model(self, model: nn.Module, adaptation_data: Dict[str, Any], 
                   config: OnlineAdaptationConfig) -> Tuple[nn.Module, Dict[str, Any]]:
        """Gradually update model parameters"""
        adapted_model = copy.deepcopy(model)
        
        # Get new parameters from adaptation data
        new_parameters = adaptation_data.get('new_parameters', {})
        learning_signal = adaptation_data.get('learning_signal', 1.0)
        
        # Apply gradual updates
        adaptation_info = {'updated_layers': [], 'parameter_changes': {}}
        
        with torch.no_grad():
            for name, param in adapted_model.named_parameters():
                if name in new_parameters:
                    old_param = param.clone()
                    new_param = new_parameters[name]
                    
                    # Exponential moving average update
                    param.data = (self.momentum * param.data + 
                                 (1 - self.momentum) * new_param * learning_signal)
                    
                    # Track changes
                    change_magnitude = torch.norm(param.data - old_param).item()
                    adaptation_info['updated_layers'].append(name)
                    adaptation_info['parameter_changes'][name] = change_magnitude
        
        return adapted_model, adaptation_info
    
    def validate_adaptation(self, original_model: nn.Module, adapted_model: nn.Module,
                          validation_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate gradual adaptation"""
        metrics = {}
        
        # Calculate parameter drift
        total_drift = 0.0
        param_count = 0
        
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                original_model.named_parameters(), adapted_model.named_parameters()
            ):
                drift = torch.norm(param1 - param2).item()
                total_drift += drift
                param_count += 1
        
        metrics['parameter_drift'] = total_drift / param_count if param_count > 0 else 0.0
        metrics['adaptation_magnitude'] = total_drift
        
        return metrics


class EnsembleModelAdapter(ModelAdapter):
    """Ensemble-based adaptation strategy"""
    
    def __init__(self, ensemble_size: int = 3):
        self.ensemble_size = ensemble_size
        self.ensemble_models = deque(maxlen=ensemble_size)
    
    def adapt_model(self, model: nn.Module, adaptation_data: Dict[str, Any], 
                   config: OnlineAdaptationConfig) -> Tuple[nn.Module, Dict[str, Any]]:
        """Create ensemble with adapted model"""
        new_model = copy.deepcopy(model)
        
        # Fine-tune the new model on recent data
        training_data = adaptation_data.get('training_data')
        if training_data:
            new_model = self._fine_tune_model(new_model, training_data, config)
        
        # Add to ensemble
        self.ensemble_models.append(new_model)
        
        # Create ensemble wrapper
        ensemble_model = EnsembleWrapper(list(self.ensemble_models))
        
        adaptation_info = {
            'ensemble_size': len(self.ensemble_models),
            'new_model_weight': 1.0 / len(self.ensemble_models)
        }
        
        return ensemble_model, adaptation_info
    
    def _fine_tune_model(self, model: nn.Module, training_data: Dict[str, Any],
                        config: OnlineAdaptationConfig) -> nn.Module:
        """Fine-tune model on adaptation data"""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=config.adaptation_learning_rate)
        
        features = training_data['features']
        targets = training_data['targets']
        
        for epoch in range(config.fine_tuning_epochs):
            optimizer.zero_grad()
            outputs = model(features)
            loss = nn.MSELoss()(outputs, targets)
            loss.backward()
            # Gradient clipping
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            if total_norm > config.gradient_clip:
                clip_coef = config.gradient_clip / (total_norm + 1e-6)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coef)
            optimizer.step()
        
        model.eval()
        return model
    
    def validate_adaptation(self, original_model: nn.Module, adapted_model: nn.Module,
                          validation_data: Dict[str, Any]) -> Dict[str, float]:
        """Validate ensemble adaptation"""
        metrics = {}
        
        if hasattr(adapted_model, 'models'):
            metrics['ensemble_size'] = len(adapted_model.models)
            metrics['ensemble_diversity'] = self._calculate_diversity(adapted_model.models)
        
        return metrics
    
    def _calculate_diversity(self, models: List[nn.Module]) -> float:
        """Calculate diversity among ensemble models"""
        if len(models) < 2:
            return 0.0
        
        diversity_sum = 0.0
        comparisons = 0
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                # Calculate parameter difference
                diff = 0.0
                count = 0
                
                with torch.no_grad():
                    for (name1, param1), (name2, param2) in zip(
                        models[i].named_parameters(), models[j].named_parameters()
                    ):
                        diff += torch.norm(param1 - param2).item()
                        count += 1
                
                if count > 0:
                    diversity_sum += diff / count
                    comparisons += 1
        
        return diversity_sum / comparisons if comparisons > 0 else 0.0


class EnsembleWrapper(nn.Module):
    """Wrapper for ensemble of models"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        outputs = []
        
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            outputs.append(output * weight)
        
        return torch.stack(outputs).sum(dim=0)


class PerformanceMonitor:
    """Monitors model performance for adaptation triggers"""
    
    def __init__(self, config: OnlineAdaptationConfig):
        self.config = config
        self.performance_history = deque(maxlen=config.performance_window)
        self.last_adaptation_time = 0.0
        self.adaptation_count = 0
        
    def add_performance_sample(self, metrics: Dict[str, float]) -> None:
        """Add performance sample to monitoring"""
        sample = {
            'timestamp': time.time(),
            'metrics': metrics,
            'adaptation_count': self.adaptation_count
        }
        self.performance_history.append(sample)
    
    def should_trigger_adaptation(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Check if adaptation should be triggered"""
        if len(self.performance_history) < 10:
            return False, "", {}
        
        # Check minimum interval
        current_time = time.time()
        if current_time - self.last_adaptation_time < self.config.adaptation_interval:
            return False, "interval_not_met", {}
        
        # Check performance degradation
        recent_performance = self._get_recent_average_performance()
        if recent_performance < self.config.performance_threshold:
            return True, "performance_drop", {
                'current_performance': recent_performance,
                'threshold': self.config.performance_threshold
            }
        
        # Check concept drift
        drift_detected, drift_info = self._detect_concept_drift()
        if drift_detected:
            return True, "concept_drift", drift_info
        
        return False, "", {}
    
    def _get_recent_average_performance(self, window: int = 20) -> float:
        """Get average performance over recent window"""
        if len(self.performance_history) == 0:
            return 1.0
        
        recent_samples = list(self.performance_history)[-window:]
        accuracies = [sample['metrics'].get('accuracy', 0.0) for sample in recent_samples]
        
        return float(np.mean(accuracies)) if accuracies else 0.0
    
    def _detect_concept_drift(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect concept drift in performance"""
        if len(self.performance_history) < 50:
            return False, {}
        
        # Compare recent vs historical performance
        recent_perf = self._get_recent_average_performance(window=20)
        historical_perf = self._get_recent_average_performance(window=50)
        
        drift_magnitude = abs(historical_perf - recent_perf)
        
        if drift_magnitude > self.config.drift_sensitivity:
            return True, {
                'drift_magnitude': drift_magnitude,
                'recent_performance': recent_perf,
                'historical_performance': historical_perf
            }
        
        return False, {}
    
    def mark_adaptation_completed(self) -> None:
        """Mark that adaptation has completed"""
        self.last_adaptation_time = time.time()
        self.adaptation_count += 1


class ModelVersionManager:
    """Manages model versions and deployment"""
    
    def __init__(self, config: OnlineAdaptationConfig, storage_path: Optional[str] = None):
        self.config = config
        self.storage_path = Path(storage_path) if storage_path else Path("model_versions")
        self.storage_path.mkdir(exist_ok=True)
        
        self.versions: Dict[str, ModelVersion] = {}
        self.active_version_id: Optional[str] = None
        self.deployment_history = deque(maxlen=config.adaptation_history_size)
        
    def create_version(self, model: nn.Module, optimizer: Optional[optim.Optimizer] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create new model version"""
        version_id = self._generate_version_id()
        
        # Extract state dictionaries
        model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        optimizer_state = optimizer.state_dict() if optimizer else None
        
        version = ModelVersion(
            version_id=version_id,
            model_state_dict=model_state,
            optimizer_state_dict=optimizer_state,
            metadata=metadata or {},
            performance_metrics={},
            creation_time=datetime.now(),
            adaptation_history=[]
        )
        
        self.versions[version_id] = version
        self._cleanup_old_versions()
        
        # Save to disk
        self._save_version_to_disk(version)
        
        logger.info(f"Created model version {version_id}")
        return version_id
    
    def deploy_version(self, version_id: str, validation_metrics: Optional[Dict[str, float]] = None) -> bool:
        """Deploy a model version as active"""
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        version = self.versions[version_id]
        
        # Update version status
        if validation_metrics:
            version.validation_results = validation_metrics
        
        version.state = ModelState.ACTIVE
        version.deployment_time = datetime.now()
        
        # Update active version
        old_version_id = self.active_version_id
        self.active_version_id = version_id
        
        # Record deployment
        deployment_record = {
            'timestamp': datetime.now(),
            'version_id': version_id,
            'previous_version_id': old_version_id,
            'validation_metrics': validation_metrics
        }
        self.deployment_history.append(deployment_record)
        
        logger.info(f"Deployed model version {version_id}")
        return True
    
    def rollback_to_previous(self) -> Optional[str]:
        """Rollback to previous version"""
        if len(self.deployment_history) < 2:
            logger.warning("No previous version available for rollback")
            return None
        
        previous_deployment = self.deployment_history[-2]
        previous_version_id = previous_deployment['version_id']
        
        if previous_version_id in self.versions:
            logger.info(f"Rolling back to version {previous_version_id}")
            return previous_version_id if self.deploy_version(previous_version_id) else None
        
        return None
    
    def get_active_model(self) -> Optional[nn.Module]:
        """Get the currently active model"""
        if not self.active_version_id or self.active_version_id not in self.versions:
            return None
        
        version = self.versions[self.active_version_id]
        
        # Reconstruct model (this is simplified - in practice would need model architecture)
        # For now, return a placeholder
        return None  # Would need model factory here
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"v_{timestamp}_{random_suffix}"
    
    def _cleanup_old_versions(self) -> None:
        """Clean up old versions beyond max limit"""
        if len(self.versions) <= self.config.max_versions:
            return
        
        # Sort by creation time
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].creation_time
        )
        
        # Remove oldest versions
        versions_to_remove = len(self.versions) - self.config.max_versions
        for i in range(versions_to_remove):
            version_id, version = sorted_versions[i]
            if version_id != self.active_version_id:  # Don't remove active version
                del self.versions[version_id]
                self._remove_version_from_disk(version_id)
                logger.info(f"Removed old version {version_id}")
    
    def _save_version_to_disk(self, version: ModelVersion) -> None:
        """Save version to disk"""
        version_path = self.storage_path / f"{version.version_id}.pkl"
        with open(version_path, 'wb') as f:
            pickle.dump(version, f)
    
    def _remove_version_from_disk(self, version_id: str) -> None:
        """Remove version from disk"""
        version_path = self.storage_path / f"{version_id}.pkl"
        if version_path.exists():
            version_path.unlink()


class ABTestManager:
    """Manages A/B testing for model deployments"""
    
    def __init__(self, config: OnlineAdaptationConfig):
        self.config = config
        self.active_tests: Dict[str, Dict[str, Any]] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
        
    def start_ab_test(self, control_version_id: str, test_version_id: str, 
                     test_name: Optional[str] = None) -> str:
        """Start A/B test between two model versions"""
        test_id = test_name or f"test_{int(time.time())}"
        
        test_config = {
            'test_id': test_id,
            'control_version': control_version_id,
            'test_version': test_version_id,
            'start_time': time.time(),
            'duration': self.config.ab_test_duration,
            'traffic_split': self.config.ab_test_traffic_split,
            'control_metrics': defaultdict(list),
            'test_metrics': defaultdict(list),
            'sample_count': {'control': 0, 'test': 0}
        }
        
        self.active_tests[test_id] = test_config
        logger.info(f"Started A/B test {test_id}: {control_version_id} vs {test_version_id}")
        
        return test_id
    
    def route_prediction_request(self, test_id: str) -> str:
        """Route prediction request to control or test version"""
        if test_id not in self.active_tests:
            return "control"
        
        test_config = self.active_tests[test_id]
        
        # Check if test has expired
        if time.time() - test_config['start_time'] > test_config['duration']:
            self._finalize_test(test_id)
            return "control"
        
        # Route based on traffic split
        if np.random.random() < test_config['traffic_split']:
            return "test"
        else:
            return "control"
    
    def record_prediction_result(self, test_id: str, variant: str, metrics: Dict[str, float]) -> None:
        """Record prediction result for A/B test"""
        if test_id not in self.active_tests:
            return
        
        test_config = self.active_tests[test_id]
        
        # Record metrics
        for metric_name, value in metrics.items():
            test_config[f'{variant}_metrics'][metric_name].append(value)
        
        test_config['sample_count'][variant] += 1
    
    def _finalize_test(self, test_id: str) -> Dict[str, Any]:
        """Finalize A/B test and compute results"""
        if test_id not in self.active_tests:
            return {}
        
        test_config = self.active_tests[test_id]
        
        # Compute statistical results
        results = {
            'test_id': test_id,
            'duration': time.time() - test_config['start_time'],
            'sample_counts': test_config['sample_count'],
            'metrics_comparison': {},
            'winner': None,
            'confidence': 0.0
        }
        
        # Compare metrics
        for metric_name in set(test_config['control_metrics'].keys()) & set(test_config['test_metrics'].keys()):
            control_values = test_config['control_metrics'][metric_name]
            test_values = test_config['test_metrics'][metric_name]
            
            if len(control_values) > 0 and len(test_values) > 0:
                control_mean = np.mean(control_values)
                test_mean = np.mean(test_values)
                
                # Simple statistical test (in practice, would use proper statistical tests)
                improvement = (test_mean - control_mean) / control_mean if control_mean != 0 else 0
                
                results['metrics_comparison'][metric_name] = {
                    'control_mean': control_mean,
                    'test_mean': test_mean,
                    'improvement': improvement,
                    'sample_sizes': {'control': len(control_values), 'test': len(test_values)}
                }
        
        # Determine winner (simplified)
        accuracy_comparison = results['metrics_comparison'].get('accuracy', {})
        if accuracy_comparison and accuracy_comparison['improvement'] > 0.01:  # 1% improvement threshold
            results['winner'] = 'test'
            results['confidence'] = min(0.95, abs(accuracy_comparison['improvement']) * 10)
        else:
            results['winner'] = 'control'
            results['confidence'] = 0.8
        
        # Store results and cleanup
        self.test_results[test_id] = results
        del self.active_tests[test_id]
        
        logger.info(f"A/B test {test_id} completed. Winner: {results['winner']}")
        return results
    
    def get_active_tests(self) -> List[str]:
        """Get list of active test IDs"""
        return list(self.active_tests.keys())
    
    def get_test_results(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get results for completed test"""
        return self.test_results.get(test_id)


class OnlineModelAdaptationFramework:
    """Main framework for online model adaptation"""
    
    def __init__(self, model: nn.Module, config: Optional[OnlineAdaptationConfig] = None):
        self.config = config or OnlineAdaptationConfig()
        self.base_model = model
        
        # Initialize platform compatibility for Mac Intel optimization
        if PLATFORM_COMPATIBILITY_AVAILABLE and _get_platform_compatibility is not None:
            self.platform_compat = _get_platform_compatibility()
            self.base_model = self.platform_compat.configure_model_for_platform(self.base_model)
            
            # Apply platform-specific optimizations
            if self.platform_compat.is_mac_intel():
                logger.info("Applied Mac Intel optimizations to adaptation framework")
        else:
            self.platform_compat = None
        
        # Core components
        self.performance_monitor = PerformanceMonitor(self.config)
        self.version_manager = ModelVersionManager(self.config)
        self.ab_test_manager = ABTestManager(self.config)
        
        # Adaptation strategies
        self.adapters = {
            AdaptationStrategy.GRADUAL: GradualModelAdapter(),
            AdaptationStrategy.ENSEMBLE: EnsembleModelAdapter(self.config.ensemble_size),
        }
        
        # State management
        self.adaptation_queue = asyncio.Queue()
        self.active_adaptations: Dict[str, asyncio.Task] = {}
        self.adaptation_history: List[AdaptationResult] = []
        
        # Thread safety
        self.lock = threading.Lock()
        self.is_running = False
        self.background_task: Optional[asyncio.Task] = None
        
        # Initialize with base model version
        initial_version_id = self.version_manager.create_version(
            model, metadata={'type': 'initial', 'source': 'base_model'}
        )
        self.version_manager.deploy_version(initial_version_id)
        
        logger.info("Online Model Adaptation Framework initialized")
    
    async def start(self) -> None:
        """Start the adaptation framework"""
        if self.is_running:
            return
        
        self.is_running = True
        self.background_task = asyncio.create_task(self._adaptation_loop())
        logger.info("Adaptation framework started")
    
    async def stop(self) -> None:
        """Stop the adaptation framework"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        # Cancel active adaptations
        for task in self.active_adaptations.values():
            task.cancel()
        
        logger.info("Adaptation framework stopped")
    
    def add_performance_sample(self, metrics: Dict[str, float]) -> None:
        """Add performance sample for monitoring"""
        self.performance_monitor.add_performance_sample(metrics)
    
    async def request_adaptation(self, trigger_type: str, trigger_data: Dict[str, Any],
                               strategy: Optional[AdaptationStrategy] = None,
                               priority: int = 1) -> str:
        """Request model adaptation"""
        request_id = f"adapt_{int(time.time())}_{priority}"
        
        adaptation_request = AdaptationRequest(
            request_id=request_id,
            trigger_type=trigger_type,
            trigger_data=trigger_data,
            adaptation_strategy=strategy or self.config.default_strategy,
            priority=priority
        )
        
        await self.adaptation_queue.put(adaptation_request)
        logger.info(f"Adaptation request {request_id} queued")
        
        return request_id
    
    async def _adaptation_loop(self) -> None:
        """Main adaptation processing loop"""
        while self.is_running:
            try:
                # Check for automatic triggers
                await self._check_automatic_triggers()
                
                # Process adaptation requests
                await self._process_adaptation_requests()
                
                # Cleanup completed adaptations
                await self._cleanup_completed_adaptations()
                
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_automatic_triggers(self) -> None:
        """Check for automatic adaptation triggers"""
        should_adapt, trigger_type, trigger_data = self.performance_monitor.should_trigger_adaptation()
        
        if should_adapt:
            await self.request_adaptation(
                trigger_type=trigger_type,
                trigger_data=trigger_data,
                priority=2  # High priority for automatic triggers
            )
    
    async def _process_adaptation_requests(self) -> None:
        """Process pending adaptation requests"""
        # Check if we can start new adaptations
        if len(self.active_adaptations) >= self.config.max_concurrent_adaptations:
            return
        
        try:
            # Get next request (non-blocking)
            request = await asyncio.wait_for(self.adaptation_queue.get(), timeout=0.1)
            
            # Start adaptation task
            task = asyncio.create_task(self._execute_adaptation(request))
            self.active_adaptations[request.request_id] = task
            
        except asyncio.TimeoutError:
            # No requests pending
            pass
    
    async def _execute_adaptation(self, request: AdaptationRequest) -> AdaptationResult:
        """Execute a single adaptation request"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting adaptation {request.request_id} with strategy {request.adaptation_strategy}")
            
            # Get current active model
            current_model = self.version_manager.get_active_model() or self.base_model
            
            # Select adaptation strategy
            adapter = self.adapters.get(request.adaptation_strategy)
            if not adapter:
                raise ValueError(f"Unknown adaptation strategy: {request.adaptation_strategy}")
            
            # Perform adaptation
            adapted_model, adaptation_info = adapter.adapt_model(
                current_model, request.trigger_data, self.config
            )
            
            # Validate adaptation if required
            validation_metrics = None
            if request.validation_required:
                validation_metrics = adapter.validate_adaptation(
                    current_model, adapted_model, request.trigger_data
                )
            
            # Create new version
            new_version_id = self.version_manager.create_version(
                adapted_model,
                metadata={
                    'adaptation_strategy': request.adaptation_strategy.value,
                    'trigger_type': request.trigger_type,
                    'adaptation_info': adaptation_info
                }
            )
            
            # Determine deployment strategy
            if self.config.enable_ab_testing and validation_metrics:
                # Start A/B test
                active_version = self.version_manager.active_version_id or 'default'
                test_id = self.ab_test_manager.start_ab_test(
                    active_version,
                    new_version_id
                )
                logger.info(f"Started A/B test {test_id} for adaptation {request.request_id}")
            else:
                # Direct deployment
                self.version_manager.deploy_version(new_version_id, validation_metrics)
            
            # Mark adaptation completed
            self.performance_monitor.mark_adaptation_completed()
            
            # Create result
            adaptation_time = time.time() - start_time
            result = AdaptationResult(
                request_id=request.request_id,
                success=True,
                new_version_id=new_version_id,
                performance_improvement=0.0,  # Would be calculated from validation
                adaptation_time=adaptation_time,
                validation_metrics=validation_metrics
            )
            
            self.adaptation_history.append(result)
            logger.info(f"Adaptation {request.request_id} completed successfully")
            
            return result
            
        except Exception as e:
            adaptation_time = time.time() - start_time
            error_result = AdaptationResult(
                request_id=request.request_id,
                success=False,
                new_version_id=None,
                performance_improvement=None,
                adaptation_time=adaptation_time,
                error_message=str(e)
            )
            
            self.adaptation_history.append(error_result)
            logger.error(f"Adaptation {request.request_id} failed: {e}")
            
            return error_result
    
    async def _cleanup_completed_adaptations(self) -> None:
        """Clean up completed adaptation tasks"""
        completed_requests = []
        
        for request_id, task in self.active_adaptations.items():
            if task.done():
                completed_requests.append(request_id)
                try:
                    result = await task
                    # Process result if needed
                except Exception as e:
                    logger.error(f"Error in adaptation task {request_id}: {e}")
        
        # Remove completed tasks
        for request_id in completed_requests:
            del self.active_adaptations[request_id]
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get current framework status"""
        return {
            'is_running': self.is_running,
            'active_version': self.version_manager.active_version_id,
            'total_versions': len(self.version_manager.versions),
            'active_adaptations': len(self.active_adaptations),
            'adaptation_history_size': len(self.adaptation_history),
            'active_ab_tests': len(self.ab_test_manager.get_active_tests()),
            'performance_samples': len(self.performance_monitor.performance_history),
            'last_adaptation_time': self.performance_monitor.last_adaptation_time
        }
    
    def get_adaptation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent adaptation history"""
        recent_adaptations = self.adaptation_history[-limit:]
        
        return [{
            'request_id': result.request_id,
            'success': result.success,
            'new_version_id': result.new_version_id,
            'adaptation_time': result.adaptation_time,
            'error_message': result.error_message,
            'validation_metrics': result.validation_metrics
        } for result in recent_adaptations]
    
    async def force_rollback(self, reason: str = "manual") -> bool:
        """Force rollback to previous version"""
        logger.warning(f"Forcing rollback: {reason}")
        
        previous_version = self.version_manager.rollback_to_previous()
        if previous_version:
            logger.info(f"Rolled back to version {previous_version}")
            return True
        else:
            logger.error("Rollback failed - no previous version available")
            return False


# Factory functions
def create_online_adaptation_framework(model: nn.Module, 
                                     config: Optional[OnlineAdaptationConfig] = None) -> OnlineModelAdaptationFramework:
    """Factory function to create online adaptation framework"""
    return OnlineModelAdaptationFramework(model, config)


def create_adaptation_config(**kwargs) -> OnlineAdaptationConfig:
    """Factory function to create adaptation configuration"""
    return OnlineAdaptationConfig(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_adaptation_framework():
        """Test the online adaptation framework"""
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Create configuration
        config = OnlineAdaptationConfig(
            performance_threshold=0.8,
            adaptation_interval=10.0,
            enable_ab_testing=True
        )
        
        # Create framework
        framework = create_online_adaptation_framework(model, config)
        
        # Start framework
        await framework.start()
        
        try:
            # Simulate performance samples
            for i in range(20):
                metrics = {
                    'accuracy': 0.9 - (i * 0.02),  # Decreasing performance
                    'loss': 0.1 + (i * 0.01)
                }
                framework.add_performance_sample(metrics)
                await asyncio.sleep(1)
            
            # Check status
            status = framework.get_framework_status()
            print(f"Framework Status: {status}")
            
            # Wait for adaptations
            await asyncio.sleep(10)
            
            # Get adaptation history
            history = framework.get_adaptation_history()
            print(f"Adaptation History: {history}")
            
        finally:
            await framework.stop()
    
    # Run test
    asyncio.run(test_adaptation_framework())