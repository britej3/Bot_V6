"""
XGBoost Ensemble for Advanced Crypto Futures Scalping
Provides ensemble methods, hyperparameter optimization, and confidence-based predictions
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pickle
import os
import json

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import mlflow
    import mlflow.xgboost
except ImportError as e:
    logging.error(f"Missing required dependencies: {e}")
    raise

# Optional dependencies
try:
    import ray
    from ray import tune
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import pytorch_lightning as pl
    PYTORCH_LIGHTNING_AVAILABLE = True
except ImportError:
    PYTORCH_LIGHTNING_AVAILABLE = False

from src.config.trading_config import AdvancedTradingConfig

logger = logging.getLogger(__name__)


class XGBoostEnsemble:
    """XGBoost Ensemble with advanced features for crypto futures trading"""

    def __init__(self, config: AdvancedTradingConfig):
        self.config = config
        self.primary_model = None
        self.secondary_models = []
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False

        # Model performance tracking
        self.model_metrics = {}
        self.prediction_history = []

        # Ensemble settings
        self.n_models = 3  # Number of models in ensemble
        self.ensemble_method = 'weighted'  # 'voting', 'weighted', 'stacking'

        logger.info("🧠 XGBoost Ensemble initialized")

    def create_training_data(self, features_list: List[np.ndarray], prices_list: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data from features and price data"""
        try:
            # Convert to numpy arrays
            X = np.array(features_list)
            prices = np.array(prices_list)

            # Create target variable: price direction after feature_horizon seconds
            y = np.zeros(len(prices) - self.config.feature_horizon)
            for i in range(len(y)):
                current_price = prices[i]
                future_price = prices[i + self.config.feature_horizon]
                y[i] = 1 if future_price > current_price else 0

            # Align features with targets
            X = X[:len(y)]

            logger.info(f"📊 Created training data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y

        except Exception as e:
            logger.error(f"❌ Failed to create training data: {e}")
            raise

    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Train XGBoost ensemble with multiple models"""
        try:
            logger.info("🚀 Starting XGBoost ensemble training...")

            # Store feature names
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train primary model
            self.primary_model = self._train_single_model(X_train_scaled, y_train, X_val_scaled, y_val)

            # Train secondary models with different parameters
            self.secondary_models = []
            for i in range(self.n_models - 1):
                model = self._train_single_model(
                    X_train_scaled, y_train, X_val_scaled, y_val,
                    variation=i+1
                )
                self.secondary_models.append(model)

            # Evaluate ensemble
            metrics = self._evaluate_ensemble(X_val_scaled, y_val)

            self.is_trained = True
            logger.info("✅ XGBoost ensemble training completed")
            return metrics

        except Exception as e:
            logger.error(f"❌ Failed to train ensemble: {e}")
            raise

    def _train_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           variation: int = 0) -> xgb.XGBClassifier:
        """Train a single XGBoost model"""
        try:
            # Get base parameters
            params = self.config.get_xgboost_params().copy()

            # Add variation for ensemble diversity
            if variation > 0:
                params['learning_rate'] *= (0.9 ** variation)
                params['max_depth'] = max(3, params['max_depth'] - variation)
                params['subsample'] = min(0.95, params['subsample'] + 0.05 * variation)

            # Create DMatrix for efficiency
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

            # Training parameters
            train_params = params.copy()
            train_params.update({
                'early_stopping_rounds': 50,
                'verbose': False
            })

            # Train model
            model = xgb.train(
                train_params,
                dtrain,
                num_boost_round=params['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                verbose_eval=False
            )

            logger.info(f"✅ Single model trained (variation {variation})")
            return model

    def save_model(self, dir_path: str) -> None:
        """Persist primary/secondary models and scaler to disk.

        Files written:
        - primary.xgb (XGBoost Booster)
        - secondary_#.xgb (if any)
        - scaler.pkl (StandardScaler)
        - meta.json (feature_names, ensemble config)
        """
        try:
            os.makedirs(dir_path, exist_ok=True)

            if self.primary_model is not None:
                primary_path = os.path.join(dir_path, "primary.xgb")
                self.primary_model.save_model(primary_path)

            for i, booster in enumerate(self.secondary_models):
                try:
                    booster.save_model(os.path.join(dir_path, f"secondary_{i}.xgb"))
                except Exception:
                    continue

            with open(os.path.join(dir_path, "scaler.pkl"), "wb") as f:
                pickle.dump(self.scaler, f)

            meta = {
                "feature_names": self.feature_names,
                "n_models": self.n_models,
                "ensemble_method": self.ensemble_method,
            }
            with open(os.path.join(dir_path, "meta.json"), "w") as f:
                json.dump(meta, f)

            logger.info(f"💾 Ensemble saved to {dir_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise

    def load_model(self, dir_path: str) -> None:
        """Load primary/secondary models and scaler from disk."""
        try:
            primary_path = os.path.join(dir_path, "primary.xgb")
            booster = xgb.Booster()
            booster.load_model(primary_path)
            self.primary_model = booster

            # Load secondary boosters if present
            secondaries = []
            i = 0
            while True:
                path = os.path.join(dir_path, f"secondary_{i}.xgb")
                if not os.path.exists(path):
                    break
                b = xgb.Booster()
                b.load_model(path)
                secondaries.append(b)
                i += 1
            self.secondary_models = secondaries

            with open(os.path.join(dir_path, "scaler.pkl"), "rb") as f:
                self.scaler = pickle.load(f)

            meta_path = os.path.join(dir_path, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    self.feature_names = meta.get("feature_names")
                    self.n_models = meta.get("n_models", self.n_models)
                    self.ensemble_method = meta.get("ensemble_method", self.ensemble_method)

            self.is_trained = True
            logger.info(f"📥 Ensemble loaded from {dir_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

        except Exception as e:
            logger.error(f"❌ Failed to train single model: {e}")
            raise

    def predict_with_confidence(self, features: np.ndarray) -> Dict[str, Any]:
        """Make prediction with confidence estimation"""
        try:
            if not self.is_trained or self.primary_model is None:
                return {'signal': 0, 'confidence': 0.0}

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Get predictions from all models
            predictions = []
            confidences = []

            # Primary model prediction
            dtest = xgb.DMatrix(features_scaled, feature_names=self.feature_names)
            primary_pred = self.primary_model.predict(dtest)[0]
            primary_conf = abs(primary_pred - 0.5) * 2  # Convert to 0-1 scale
            predictions.append(primary_pred)
            confidences.append(primary_conf)

            # Secondary model predictions
            for model in self.secondary_models:
                pred = model.predict(dtest)[0]
                conf = abs(pred - 0.5) * 2
                predictions.append(pred)
                confidences.append(conf)

            # Ensemble prediction based on method
            if self.ensemble_method == 'voting':
                ensemble_pred = np.mean(predictions)
                ensemble_conf = np.mean(confidences)
            elif self.ensemble_method == 'weighted':
                weights = np.array([0.5] + [0.5/(self.n_models-1)] * (self.n_models-1))
                ensemble_pred = np.average(predictions, weights=weights)
                ensemble_conf = np.average(confidences, weights=weights)
            else:  # stacking
                ensemble_pred = np.mean(predictions)
                ensemble_conf = np.mean(confidences)

            # Convert to binary signal
            signal = 1 if ensemble_pred > 0.5 else -1

            # Adjust confidence based on agreement
            agreement = np.mean([1 if (p > 0.5) == (ensemble_pred > 0.5) else 0 for p in predictions])
            final_confidence = ensemble_conf * agreement

            result = {
                'signal': signal,
                'confidence': float(final_confidence),
                'raw_prediction': float(ensemble_pred),
                'agreement': float(agreement),
                'timestamp': datetime.utcnow().isoformat()
            }

            # Store prediction for tracking
            self.prediction_history.append(result)

            return result

        except Exception as e:
            logger.error(f"❌ Failed to make prediction: {e}")
            return {'signal': 0, 'confidence': 0.0}

    def _evaluate_ensemble(self, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        try:
            predictions = []
            confidences = []

            for i in range(len(X_val)):
                pred = self.predict_with_confidence(X_val[i])
                predictions.append(1 if pred['signal'] > 0 else 0)
                confidences.append(pred['confidence'])

            # Calculate metrics
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(y_val, predictions, zero_division=0)
            recall = recall_score(y_val, predictions, zero_division=0)
            f1 = f1_score(y_val, predictions, zero_division=0)

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mean_confidence': np.mean(confidences)
            }

            self.model_metrics = metrics
            logger.info(f"📈 Ensemble metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"❌ Failed to evaluate ensemble: {e}")
            return {}

    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using available methods"""
        try:
            if self.config.ray_tune_enabled and RAY_AVAILABLE:
                return self._optimize_with_ray(X_train, y_train)
            else:
                return self._optimize_with_grid_search(X_train, y_train)

        except Exception as e:
            logger.error(f"❌ Failed to optimize hyperparameters: {e}")
            return self.config.get_xgboost_params()

    def _optimize_with_ray(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using Ray Tune"""
        try:
            logger.info("🎯 Starting Ray Tune hyperparameter optimization...")

            def objective(config):
                # Train model with given config
                params = self.config.get_xgboost_params().copy()
                params.update(config)

                model = xgb.XGBClassifier(**params)
                scores = []

                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                for train_idx, val_idx in tscv.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

                    model.fit(X_train_fold, y_train_fold)
                    pred = model.predict(X_val_fold)
                    scores.append(f1_score(y_val_fold, pred, zero_division=0))

                return {"f1_score": np.mean(scores)}

            # Define search space
            search_space = {
                "learning_rate": tune.uniform(0.001, 0.1),
                "max_depth": tune.randint(3, 10),
                "min_child_weight": tune.uniform(1, 10),
                "subsample": tune.uniform(0.6, 1.0),
                "colsample_bytree": tune.uniform(0.6, 1.0),
            }

            # Run optimization
            analysis = tune.run(
                objective,
                config=search_space,
                num_samples=self.config.ray_tune_samples,
                metric="f1_score",
                mode="max",
                verbose=1
            )

            best_config = analysis.best_config
            logger.info(f"✅ Ray Tune optimization completed. Best config: {best_config}")
            return best_config

        except Exception as e:
            logger.error(f"❌ Ray Tune optimization failed: {e}")
            return self.config.get_xgboost_params()

    def _optimize_with_grid_search(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Simple grid search for hyperparameter optimization"""
        try:
            logger.info("🔍 Starting grid search hyperparameter optimization...")

            # Define parameter grid
            param_grid = {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [4, 6, 8],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }

            best_score = 0
            best_params = self.config.get_xgboost_params()

            # Simple grid search (in production, use sklearn's GridSearchCV)
            for lr in param_grid['learning_rate']:
                for md in param_grid['max_depth']:
                    for mcw in param_grid['min_child_weight']:
                        for ss in param_grid['subsample']:
                            for cbt in param_grid['colsample_bytree']:
                                params = {
                                    'learning_rate': lr,
                                    'max_depth': md,
                                    'min_child_weight': mcw,
                                    'subsample': ss,
                                    'colsample_bytree': cbt,
                                    'objective': 'binary:logistic',
                                    'eval_metric': 'logloss',
                                    'random_state': 42,
                                    'n_jobs': -1
                                }

                                # Quick evaluation
                                model = xgb.XGBClassifier(**params)
                                model.fit(X_train, y_train)
                                pred = model.predict(X_train)  # Simple evaluation

                                score = f1_score(y_train, pred, zero_division=0)
                                if score > best_score:
                                    best_score = score
                                    best_params = params

            logger.info(f"✅ Grid search completed. Best score: {best_score}")
            return best_params

        except Exception as e:
            logger.error(f"❌ Grid search optimization failed: {e}")
            return self.config.get_xgboost_params()

    def save_model(self, filepath: str):
        """Save trained ensemble to disk"""
        try:
            model_data = {
                'primary_model': self.primary_model,
                'secondary_models': self.secondary_models,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'config': self.config,
                'model_metrics': self.model_metrics,
                'is_trained': self.is_trained
            }

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"💾 Model saved to {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Load trained ensemble from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.primary_model = model_data['primary_model']
            self.secondary_models = model_data['secondary_models']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_metrics = model_data.get('model_metrics', {})
            self.is_trained = model_data.get('is_trained', True)

            logger.info(f"📂 Model loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        return {
            'is_trained': self.is_trained,
            'n_models': len(self.secondary_models) + 1 if self.primary_model else 0,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'ensemble_method': self.ensemble_method,
            'n_predictions': len(self.prediction_history)
        }
