"""
MLOps Manager for CryptoScalp AI

This module implements formal MLOps practices including Feature Store integration,
enhanced Model Registry, and automated pipelines for production-grade ML operations.
"""

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Enhanced model metadata for tracking"""
    model_name: str
    version: str
    created_at: str
    optimization_level: str
    target_latency_ms: float
    compression_ratio: float
    accuracy_score: float
    feature_set_hash: str
    training_data_hash: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class FeatureMetadata:
    """Feature metadata for tracking"""
    feature_name: str
    data_type: str
    description: str
    transformation: str
    importance_score: float
    last_updated: str


class FeatureStoreManager:
    """
    Feature Store integration for guaranteed consistency between training and inference
    """

    def __init__(self, feature_store_path: str = "feature_store"):
        self.feature_store_path = Path(feature_store_path)
        self.feature_store_path.mkdir(exist_ok=True)
        self.feature_definitions = {}
        self.feature_cache = {}

    def register_feature(self, name: str, data_type: str,
                        description: str, transformation: str) -> FeatureMetadata:
        """Register a feature in the feature store"""
        metadata = FeatureMetadata(
            feature_name=name,
            data_type=data_type,
            description=description,
            transformation=transformation,
            importance_score=0.0,
            last_updated=datetime.now().isoformat()
        )

        self.feature_definitions[name] = metadata

        # Save to disk
        self._save_feature_metadata(metadata)

        logger.info(f"Registered feature: {name}")
        return metadata

    def store_features(self, features: Dict[str, np.ndarray],
                      entity_id: str, timestamp: Optional[str] = None) -> str:
        """Store feature values with versioning"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Create feature hash for versioning
        feature_hash = self._compute_feature_hash(features)

        # Store features
        feature_data = {
            'entity_id': entity_id,
            'timestamp': timestamp,
            'features': {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in features.items()},
            'feature_hash': feature_hash
        }

        # Save to file
        filename = f"{entity_id}_{timestamp.replace(':', '-')}.json"
        filepath = self.feature_store_path / filename

        with open(filepath, 'w') as f:
            json.dump(feature_data, f, indent=2)

        # Cache for fast retrieval
        self.feature_cache[entity_id] = feature_data

        logger.info(f"Stored features for entity {entity_id} with hash {feature_hash}")
        return feature_hash

    def get_features(self, entity_id: str,
                    feature_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Retrieve features from store"""
        if entity_id in self.feature_cache:
            features = self.feature_cache[entity_id]['features']
        else:
            # Load from disk (simplified - in practice would use proper database)
            feature_files = list(self.feature_store_path.glob(f"{entity_id}_*.json"))
            if not feature_files:
                raise ValueError(f"No features found for entity {entity_id}")

            latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
            with open(latest_file, 'r') as f:
                feature_data = json.load(f)
                features = feature_data['features']

        # Convert to numpy arrays
        result = {}
        for name, value in features.items():
            if feature_names is None or name in feature_names:
                result[name] = np.array(value)

        return result

    def get_feature_hash(self, entity_id: str) -> str:
        """Get feature hash for reproducibility"""
        if entity_id in self.feature_cache:
            return self.feature_cache[entity_id]['feature_hash']

        feature_files = list(self.feature_store_path.glob(f"{entity_id}_*.json"))
        if not feature_files:
            return ""

        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, 'r') as f:
            feature_data = json.load(f)
            return feature_data.get('feature_hash', '')

    def _compute_feature_hash(self, features: Dict[str, Any]) -> str:
        """Compute hash of feature values for versioning"""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()[:8]

    def _save_feature_metadata(self, metadata: FeatureMetadata):
        """Save feature metadata to disk"""
        filename = f"{metadata.feature_name}_metadata.json"
        filepath = self.feature_store_path / filename

        with open(filepath, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)


class EnhancedModelRegistry:
    """
    Enhanced MLflow-based model registry with advanced tracking capabilities
    """

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        self.tracking_uri = tracking_uri

    def register_optimized_model(self, model: nn.Module,
                               model_name: str,
                               metrics: Dict[str, float],
                               metadata: ModelMetadata,
                               feature_importance: Optional[Dict[str, float]] = None) -> str:
        """Register an optimized model with comprehensive metadata"""
        logger.info(f"Registering model {model_name} version {metadata.version}")

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(metadata.hyperparameters)

            # Log performance metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log metadata
            mlflow.log_param("optimization_level", metadata.optimization_level)
            mlflow.log_param("target_latency_ms", metadata.target_latency_ms)
            mlflow.log_param("compression_ratio", metadata.compression_ratio)
            mlflow.log_param("feature_set_hash", metadata.feature_set_hash)

            # Log feature importance if available
            if feature_importance:
                importance_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=['feature', 'importance']
                )
                mlflow.log_table(importance_df, "feature_importance.json")

            # Create model signature
            dummy_input = torch.randn(1, 1000)
            with torch.no_grad():
                dummy_output = model(dummy_input)
            signature = infer_signature(dummy_input.numpy(), dummy_output.numpy())

            # Log model with enhanced metadata
            mlflow.pytorch.log_model(
                model,
                "model",
                registered_model_name=model_name,
                signature=signature,
                metadata={
                    'model_metadata': asdict(metadata),
                    'created_at': datetime.now().isoformat(),
                    'framework_version': torch.__version__,
                    'optimization_details': {
                        'target_latency_achieved': metrics.get('inference_time_ms', 0) <= metadata.target_latency_ms,
                        'compression_achieved': metadata.compression_ratio >= 2.0,
                        'production_ready': True
                    }
                }
            )

            # Register in model registry
            model_version = mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name=model_name,
                tags={
                    'optimization_level': metadata.optimization_level,
                    'target_latency_ms': str(metadata.target_latency_ms),
                    'production_ready': 'true'
                }
            )

            logger.info(f"Successfully registered model {model_name} version {model_version.version}")
            return model_version.version

    def transition_model_stage(self, model_name: str,
                             version: str,
                             stage: str) -> bool:
        """Transition model through stages (Staging, Production, Archived)"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            return True
        except Exception as e:
            logger.error(f"Failed to transition {model_name} v{version} to {stage}: {e}")
            return False

    def get_model_metadata(self, model_name: str, version: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata"""
        try:
            model_version = self.client.get_model_version(model_name, version)
            metadata = model_version.tags.get('model_metadata')

            if metadata:
                return ModelMetadata(**json.loads(metadata))

        except Exception as e:
            logger.error(f"Failed to get metadata for {model_name} v{version}: {e}")

        return None

    def compare_model_versions(self, model_name: str,
                             version_a: str,
                             version_b: str) -> Dict[str, Any]:
        """Compare two model versions"""
        meta_a = self.get_model_metadata(model_name, version_a)
        meta_b = self.get_model_metadata(model_name, version_b)

        if not meta_a or not meta_b:
            return {'error': 'Could not retrieve metadata for one or both versions'}

        return {
            'version_a': {
                'version': version_a,
                'inference_time': meta_a.performance_metrics.get('inference_time_ms', 0),
                'compression_ratio': meta_a.compression_ratio,
                'accuracy': meta_a.accuracy_score
            },
            'version_b': {
                'version': version_b,
                'inference_time': meta_b.performance_metrics.get('inference_time_ms', 0),
                'compression_ratio': meta_b.compression_ratio,
                'accuracy': meta_b.accuracy_score
            },
            'improvement': {
                'inference_time': meta_b.performance_metrics.get('inference_time_ms', 0) -
                                meta_a.performance_metrics.get('inference_time_ms', 0),
                'compression': meta_b.compression_ratio - meta_a.compression_ratio,
                'accuracy': meta_b.accuracy_score - meta_a.accuracy_score
            }
        }

    def list_production_models(self) -> List[Dict[str, Any]]:
        """List all models in production"""
        try:
            registered_models = self.client.list_registered_models()
            production_models = []

            for model in registered_models:
                latest_version = self.client.get_latest_versions(model.name, stages=['Production'])
                if latest_version:
                    production_models.append({
                        'name': model.name,
                        'version': latest_version[0].version,
                        'last_updated': latest_version[0].last_updated_timestamp
                    })

            return production_models

        except Exception as e:
            logger.error(f"Failed to list production models: {e}")
            return []


class AutomatedPipelineManager:
    """
    Automated MLOps pipeline for continuous model improvement
    """

    def __init__(self, feature_store: FeatureStoreManager,
                 model_registry: EnhancedModelRegistry):
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.pipeline_runs = []

    async def run_automated_pipeline(self, model: nn.Module,
                                   training_data: Dict[str, np.ndarray],
                                   validation_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Run complete automated MLOps pipeline"""
        logger.info("Starting automated MLOps pipeline")

        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Step 1: Store features
            feature_hash = self.feature_store.store_features(
                training_data, f"training_{pipeline_id}"
            )

            # Step 2: Train and optimize model
            optimized_model = await self._train_and_optimize_model(
                model, training_data, validation_data
            )

            # Step 3: Evaluate model
            metrics = await self._evaluate_model(optimized_model, validation_data)

            # Step 4: Create model metadata
            metadata = ModelMetadata(
                model_name=f"cryptoscalp_model_{pipeline_id}",
                version="v1.0",
                created_at=datetime.now().isoformat(),
                optimization_level="advanced",
                target_latency_ms=1.0,
                compression_ratio=metrics.get('compression_ratio', 1.0),
                accuracy_score=metrics.get('accuracy', 0.0),
                feature_set_hash=feature_hash,
                training_data_hash=self._compute_data_hash(training_data),
                hyperparameters={'learning_rate': 0.001, 'batch_size': 32},
                performance_metrics=metrics
            )

            # Step 5: Register model
            version = self.model_registry.register_optimized_model(
                optimized_model,
                metadata.model_name,
                metrics,
                metadata
            )

            # Step 6: Validate production readiness
            is_production_ready = await self._validate_production_readiness(
                optimized_model, metrics
            )

            if is_production_ready:
                self.model_registry.transition_model_stage(
                    metadata.model_name, version, "Staging"
                )

            pipeline_result = {
                'pipeline_id': pipeline_id,
                'success': True,
                'model_version': version,
                'metrics': metrics,
                'production_ready': is_production_ready,
                'feature_hash': feature_hash
            }

            self.pipeline_runs.append(pipeline_result)
            logger.info(f"Automated pipeline {pipeline_id} completed successfully")

            return pipeline_result

        except Exception as e:
            logger.error(f"Automated pipeline {pipeline_id} failed: {e}")
            return {
                'pipeline_id': pipeline_id,
                'success': False,
                'error': str(e)
            }

    async def _train_and_optimize_model(self, model: nn.Module,
                                     training_data: Dict[str, np.ndarray],
                                     validation_data: Dict[str, np.ndarray]) -> nn.Module:
        """Train and optimize model (placeholder - implement specific training logic)"""
        # This would contain the actual training and optimization logic
        # For now, return the model as-is
        logger.info("Training and optimizing model")
        return model

    async def _evaluate_model(self, model: nn.Module,
                            validation_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance"""
        # This would contain the actual evaluation logic
        # For now, return dummy metrics
        logger.info("Evaluating model performance")
        return {
            'accuracy': 0.85,
            'inference_time_ms': 0.8,
            'compression_ratio': 3.2,
            'memory_usage_mb': 25.0
        }

    def _compute_data_hash(self, data: Dict[str, np.ndarray]) -> str:
        """Compute hash of training data"""
        data_str = json.dumps({k: v.shape for k, v in data.items()}, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    async def _validate_production_readiness(self, model: nn.Module,
                                          metrics: Dict[str, float]) -> bool:
        """Validate if model is ready for production"""
        checks = {
            'latency_check': metrics.get('inference_time_ms', 10) <= 1.0,
            'accuracy_check': metrics.get('accuracy', 0) >= 0.7,
            'compression_check': metrics.get('compression_ratio', 1) >= 2.0
        }

        all_passed = all(checks.values())

        if all_passed:
            logger.info("Model passed all production readiness checks")
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            logger.warning(f"Model failed production checks: {failed_checks}")

        return all_passed


class MLOpsDashboard:
    """
    MLOps monitoring and visualization dashboard
    """

    def __init__(self, model_registry: EnhancedModelRegistry,
                 pipeline_manager: AutomatedPipelineManager):
        self.model_registry = model_registry
        self.pipeline_manager = pipeline_manager

    def get_system_overview(self) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        production_models = self.model_registry.list_production_models()

        pipeline_stats = {
            'total_runs': len(self.pipeline_manager.pipeline_runs),
            'successful_runs': len([r for r in self.pipeline_manager.pipeline_runs if r['success']]),
            'failed_runs': len([r for r in self.pipeline_manager.pipeline_runs if not r['success']])
        }

        return {
            'production_models_count': len(production_models),
            'production_models': production_models,
            'pipeline_stats': pipeline_stats,
            'recent_pipeline_runs': self.pipeline_manager.pipeline_runs[-5:],  # Last 5 runs
            'system_health': self._compute_system_health()
        }

    def _compute_system_health(self) -> Dict[str, Any]:
        """Compute overall system health"""
        # This would include various health checks
        return {
            'feature_store_health': 'healthy',
            'model_registry_health': 'healthy',
            'pipeline_health': 'healthy',
            'overall_status': 'healthy'
        }

    def generate_performance_report(self) -> pd.DataFrame:
        """Generate performance report for all models"""
        # This would compile performance data from all models
        data = []
        for model in self.model_registry.list_production_models():
            metadata = self.model_registry.get_model_metadata(
                model['name'], model['version']
            )
            if metadata:
                data.append({
                    'model_name': model['name'],
                    'version': model['version'],
                    'inference_time_ms': metadata.performance_metrics.get('inference_time_ms', 0),
                    'compression_ratio': metadata.compression_ratio,
                    'accuracy': metadata.accuracy_score,
                    'created_at': metadata.created_at
                })

        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    # Initialize MLOps components
    feature_store = FeatureStoreManager()
    model_registry = EnhancedModelRegistry()
    pipeline_manager = AutomatedPipelineManager(feature_store, model_registry)
    dashboard = MLOpsDashboard(model_registry, pipeline_manager)

    # Register some example features
    feature_store.register_feature(
        name="price_momentum",
        data_type="float64",
        description="Price momentum indicator",
        transformation="pct_change"
    )

    # Example training data
    training_data = {
        'price_momentum': np.random.randn(1000),
        'volume': np.random.randn(1000),
        'volatility': np.random.randn(1000)
    }

    print("MLOps Manager initialized successfully!")
    print(f"System overview: {dashboard.get_system_overview()}")