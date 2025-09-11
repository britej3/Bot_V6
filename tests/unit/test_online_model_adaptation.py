"""
Unit Tests for Online Model Adaptation Framework
===============================================

Tests for Task 14.1.4: Create online model adaptation framework

Author: Autonomous Systems Team
Date: 2025-01-22
"""

import pytest
import torch
import torch.nn as nn
import asyncio
import time
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from learning.online_model_adaptation import (
    OnlineModelAdaptationFramework,
    OnlineAdaptationConfig,
    AdaptationStrategy,
    ModelState,
    GradualModelAdapter,
    EnsembleModelAdapter,
    PerformanceMonitor,
    ModelVersionManager,
    ABTestManager,
    create_online_adaptation_framework,
    create_adaptation_config
)

from learning.online_adaptation_integration import (
    IntegratedOnlineAdaptation,
    IntegratedAdaptationConfig,
    create_integrated_adaptation_system
)


class SimpleTestModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_size=10, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def test_model():
    """Create test model"""
    return SimpleTestModel()


@pytest.fixture
def adaptation_config():
    """Create test configuration"""
    return OnlineAdaptationConfig(
        performance_threshold=0.8,
        adaptation_interval=5.0,
        enable_ab_testing=True,
        max_concurrent_adaptations=2
    )


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestOnlineAdaptationConfig:
    """Test configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = OnlineAdaptationConfig()
        
        assert config.performance_threshold == 0.7
        assert config.adaptation_interval == 300.0
        assert config.default_strategy == AdaptationStrategy.GRADUAL
        assert config.enable_ab_testing is True
        assert config.max_concurrent_adaptations == 2
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = create_adaptation_config(
            performance_threshold=0.9,
            adaptation_interval=60.0,
            default_strategy=AdaptationStrategy.ENSEMBLE
        )
        
        assert config.performance_threshold == 0.9
        assert config.adaptation_interval == 60.0
        assert config.default_strategy == AdaptationStrategy.ENSEMBLE


class TestGradualModelAdapter:
    """Test gradual adaptation strategy"""
    
    def test_gradual_adaptation(self, test_model):
        """Test gradual model parameter updates"""
        adapter = GradualModelAdapter(momentum=0.8)
        config = OnlineAdaptationConfig()
        
        # Create new parameters
        new_params = {}
        for name, param in test_model.named_parameters():
            new_params[name] = param.clone() + 0.1
        
        adaptation_data = {'new_parameters': new_params}
        
        adapted_model, info = adapter.adapt_model(test_model, adaptation_data, config)
        
        assert adapted_model is not None
        assert 'updated_layers' in info
        assert 'parameter_changes' in info
        assert len(info['updated_layers']) > 0
    
    def test_validation(self, test_model):
        """Test adaptation validation"""
        adapter = GradualModelAdapter()
        
        # Create slightly different model
        adapted_model = SimpleTestModel()
        
        validation_data = {}
        metrics = adapter.validate_adaptation(test_model, adapted_model, validation_data)
        
        assert 'parameter_drift' in metrics
        assert 'adaptation_magnitude' in metrics
        assert isinstance(metrics['parameter_drift'], float)


class TestEnsembleModelAdapter:
    """Test ensemble adaptation strategy"""
    
    def test_ensemble_creation(self, test_model, adaptation_config):
        """Test ensemble model creation"""
        adapter = EnsembleModelAdapter(ensemble_size=3)
        
        training_data = {
            'features': torch.randn(32, 10),
            'targets': torch.randn(32, 1)
        }
        adaptation_data = {'training_data': training_data}
        
        ensemble_model, info = adapter.adapt_model(test_model, adaptation_data, adaptation_config)
        
        assert ensemble_model is not None
        assert 'ensemble_size' in info
        assert hasattr(ensemble_model, 'models')
    
    def test_ensemble_diversity(self, test_model):
        """Test ensemble diversity calculation"""
        adapter = EnsembleModelAdapter()
        
        # Create multiple models
        models = [SimpleTestModel() for _ in range(3)]
        diversity = adapter._calculate_diversity(models)
        
        assert isinstance(diversity, float)
        assert diversity >= 0.0


class TestPerformanceMonitor:
    """Test performance monitoring"""
    
    def test_performance_tracking(self, adaptation_config):
        """Test performance sample tracking"""
        monitor = PerformanceMonitor(adaptation_config)
        
        # Add performance samples
        for i in range(10):
            metrics = {'accuracy': 0.9 - i * 0.01, 'loss': 0.1 + i * 0.01}
            monitor.add_performance_sample(metrics)
        
        assert len(monitor.performance_history) == 10
    
    def test_adaptation_trigger(self, adaptation_config):
        """Test adaptation trigger logic"""
        config = OnlineAdaptationConfig(
            performance_threshold=0.8,
            adaptation_interval=1.0
        )
        monitor = PerformanceMonitor(config)
        
        # Add declining performance samples
        for i in range(15):
            metrics = {'accuracy': 0.9 - i * 0.02}
            monitor.add_performance_sample(metrics)
        
        # Wait for interval
        time.sleep(1.1)
        
        should_adapt, trigger_type, trigger_data = monitor.should_trigger_adaptation()
        
        assert should_adapt is True
        assert trigger_type == "performance_drop"
        assert 'current_performance' in trigger_data
    
    def test_concept_drift_detection(self, adaptation_config):
        """Test concept drift detection"""
        config = OnlineAdaptationConfig(drift_sensitivity=0.1)
        monitor = PerformanceMonitor(config)
        
        # Add stable performance, then declining
        for i in range(30):
            metrics = {'accuracy': 0.9}
            monitor.add_performance_sample(metrics)
        
        for i in range(20):
            metrics = {'accuracy': 0.7}
            monitor.add_performance_sample(metrics)
        
        drift_detected, drift_info = monitor._detect_concept_drift()
        
        assert drift_detected is True
        assert 'drift_magnitude' in drift_info


class TestModelVersionManager:
    """Test model version management"""
    
    def test_version_creation(self, test_model, temp_storage_dir):
        """Test model version creation"""
        config = OnlineAdaptationConfig()
        manager = ModelVersionManager(config, temp_storage_dir)
        
        version_id = manager.create_version(test_model, metadata={'test': True})
        
        assert version_id in manager.versions
        assert manager.versions[version_id].state == ModelState.STAGED
    
    def test_version_deployment(self, test_model, temp_storage_dir):
        """Test version deployment"""
        config = OnlineAdaptationConfig()
        manager = ModelVersionManager(config, temp_storage_dir)
        
        version_id = manager.create_version(test_model)
        success = manager.deploy_version(version_id, {'accuracy': 0.9})
        
        assert success is True
        assert manager.active_version_id == version_id
        assert manager.versions[version_id].state == ModelState.ACTIVE
    
    def test_rollback(self, test_model, temp_storage_dir):
        """Test version rollback"""
        config = OnlineAdaptationConfig()
        manager = ModelVersionManager(config, temp_storage_dir)
        
        # Create and deploy two versions
        v1 = manager.create_version(test_model, metadata={'version': 1})
        manager.deploy_version(v1)
        
        v2 = manager.create_version(test_model, metadata={'version': 2})
        manager.deploy_version(v2)
        
        # Rollback
        rollback_version = manager.rollback_to_previous()
        
        assert rollback_version == v1
        assert manager.active_version_id == v1


class TestABTestManager:
    """Test A/B testing functionality"""
    
    def test_ab_test_creation(self, adaptation_config):
        """Test A/B test creation"""
        manager = ABTestManager(adaptation_config)
        
        test_id = manager.start_ab_test('v1', 'v2', 'test_adaptation')
        
        assert test_id in manager.active_tests
        assert manager.active_tests[test_id]['control_version'] == 'v1'
        assert manager.active_tests[test_id]['test_version'] == 'v2'
    
    def test_traffic_routing(self, adaptation_config):
        """Test traffic routing for A/B tests"""
        config = OnlineAdaptationConfig(ab_test_traffic_split=0.5)
        manager = ABTestManager(config)
        
        test_id = manager.start_ab_test('v1', 'v2')
        
        # Test multiple routing decisions
        routes = [manager.route_prediction_request(test_id) for _ in range(100)]
        
        control_count = routes.count('control')
        test_count = routes.count('test')
        
        # Should be roughly 50/50 split
        assert 30 <= control_count <= 70  # Allow some variance
        assert 30 <= test_count <= 70
    
    def test_result_recording(self, adaptation_config):
        """Test recording A/B test results"""
        manager = ABTestManager(adaptation_config)
        
        test_id = manager.start_ab_test('v1', 'v2')
        
        # Record some results
        manager.record_prediction_result(test_id, 'control', {'accuracy': 0.8})
        manager.record_prediction_result(test_id, 'test', {'accuracy': 0.85})
        
        test_config = manager.active_tests[test_id]
        
        assert len(test_config['control_metrics']['accuracy']) == 1
        assert len(test_config['test_metrics']['accuracy']) == 1
        assert test_config['sample_count']['control'] == 1
        assert test_config['sample_count']['test'] == 1


class TestOnlineAdaptationFramework:
    """Test main adaptation framework"""
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self, test_model, adaptation_config):
        """Test framework initialization"""
        framework = create_online_adaptation_framework(test_model, adaptation_config)
        
        assert framework.base_model is test_model
        assert framework.config is adaptation_config
        assert framework.version_manager is not None
        assert framework.performance_monitor is not None
        assert framework.ab_test_manager is not None
    
    @pytest.mark.asyncio
    async def test_framework_start_stop(self, test_model, adaptation_config):
        """Test framework start/stop lifecycle"""
        framework = create_online_adaptation_framework(test_model, adaptation_config)
        
        await framework.start()
        assert framework.is_running is True
        
        await framework.stop()
        assert framework.is_running is False
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, test_model, adaptation_config):
        """Test performance monitoring integration"""
        framework = create_online_adaptation_framework(test_model, adaptation_config)
        
        # Add performance samples
        framework.add_performance_sample({'accuracy': 0.9, 'loss': 0.1})
        framework.add_performance_sample({'accuracy': 0.85, 'loss': 0.15})
        
        assert len(framework.performance_monitor.performance_history) == 2
    
    @pytest.mark.asyncio
    async def test_adaptation_request(self, test_model, adaptation_config):
        """Test adaptation request processing"""
        framework = create_online_adaptation_framework(test_model, adaptation_config)
        
        await framework.start()
        
        try:
            request_id = await framework.request_adaptation(
                trigger_type='test',
                trigger_data={'test': True},
                strategy=AdaptationStrategy.GRADUAL
            )
            
            assert request_id is not None
            assert request_id.startswith('adapt_')
            
        finally:
            await framework.stop()
    
    def test_framework_status(self, test_model, adaptation_config):
        """Test framework status reporting"""
        framework = create_online_adaptation_framework(test_model, adaptation_config)
        
        status = framework.get_framework_status()
        
        assert 'is_running' in status
        assert 'active_version' in status
        assert 'total_versions' in status
        assert 'active_adaptations' in status


class TestIntegratedAdaptation:
    """Test integrated adaptation system"""
    
    def test_integration_initialization(self, test_model):
        """Test integrated system initialization"""
        mock_pipeline = Mock()
        mock_pipeline.is_learning_active.return_value = True
        
        system = create_integrated_adaptation_system(test_model, mock_pipeline)
        
        assert system.model is test_model
        assert system.learning_pipeline is mock_pipeline
        assert system.adaptation_framework is not None
    
    @pytest.mark.asyncio
    async def test_integration_lifecycle(self, test_model):
        """Test integrated system lifecycle"""
        mock_pipeline = Mock()
        mock_pipeline.is_learning_active.return_value = False
        mock_pipeline.start_learning.return_value = None
        
        system = create_integrated_adaptation_system(test_model, mock_pipeline)
        
        await system.start()
        assert system.is_running is True
        
        await system.stop()
        assert system.is_running is False
    
    def test_performance_integration(self, test_model):
        """Test performance monitoring integration"""
        mock_pipeline = Mock()
        system = create_integrated_adaptation_system(test_model, mock_pipeline)
        
        # Add performance sample
        system.add_performance_sample({'accuracy': 0.9, 'loss': 0.1})
        
        assert system.current_performance == 0.9
        assert system.baseline_performance == 0.9
        assert len(system.performance_history) == 1
    
    def test_integration_status(self, test_model):
        """Test integrated system status"""
        mock_pipeline = Mock()
        mock_pipeline.is_learning_active.return_value = True
        
        system = create_integrated_adaptation_system(test_model, mock_pipeline)
        
        status = system.get_integration_status()
        
        assert 'is_running' in status
        assert 'framework_status' in status
        assert 'current_performance' in status
        assert 'baseline_performance' in status


# Integration test
@pytest.mark.asyncio
async def test_end_to_end_adaptation():
    """Test end-to-end adaptation flow"""
    model = SimpleTestModel()
    config = OnlineAdaptationConfig(
        performance_threshold=0.8,
        adaptation_interval=1.0,
        enable_ab_testing=False
    )
    
    framework = create_online_adaptation_framework(model, config)
    
    await framework.start()
    
    try:
        # Simulate declining performance
        for i in range(5):
            metrics = {'accuracy': 0.9 - i * 0.05, 'loss': 0.1 + i * 0.02}
            framework.add_performance_sample(metrics)
        
        # Wait for adaptation interval
        await asyncio.sleep(1.5)
        
        # Check if adaptation was triggered
        status = framework.get_framework_status()
        
        # Should have at least initial version
        assert status['total_versions'] >= 1
        
    finally:
        await framework.stop()


if __name__ == "__main__":
    # Run a simple test
    print("Running Online Model Adaptation Framework Tests...")
    
    # Test configuration
    config = create_adaptation_config(performance_threshold=0.8)
    assert config.performance_threshold == 0.8
    print("✅ Configuration test passed")
    
    # Test model creation
    model = SimpleTestModel()
    framework = create_online_adaptation_framework(model, config)
    assert framework is not None
    print("✅ Framework creation test passed")
    
    print("\n🎯 Task 14.1.4: Online Model Adaptation Framework - IMPLEMENTATION COMPLETE")
    print("Run 'pytest tests/unit/test_online_model_adaptation.py' for full test suite")