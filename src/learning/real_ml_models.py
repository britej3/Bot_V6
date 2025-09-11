"""
Real ML Models for Trading Strategy Integration
==============================================

This module provides actual implementations of the 4 ML models:
- Logistic Regression
- Random Forest
- LSTM Neural Network
- XGBoost

These models are trained on tick-level trading data to provide
real predictions for trading decisions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import xgboost as xgb
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class RealLogisticRegression:
    """Real Logistic Regression model for trading signals"""

    def __init__(self):
        self.model = SklearnLogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            penalty='l2'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the logistic regression model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            self.is_trained = True
            self.feature_names = feature_names

            logger.info(f"Logistic Regression trained - Accuracy: {accuracy:.4f}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}")

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }

        except Exception as e:
            logger.error(f"Logistic Regression training failed: {e}")
            return None

    def predict(self, X: np.ndarray) -> Dict[str, float]:
        """Make prediction"""
        if not self.is_trained:
            return {'probability': 0.5, 'confidence': 0.0}

        try:
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)

            # Get probability for positive class (buy signal)
            buy_probability = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]
            confidence = abs(buy_probability - 0.5) * 2  # Scale to 0-1

            return {
                'probability': float(buy_probability),
                'confidence': float(confidence)
            }

        except Exception as e:
            logger.error(f"Logistic Regression prediction failed: {e}")
            return {'probability': 0.5, 'confidence': 0.0}

    def save_model(self, filepath: str):
        """Save model to file"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'created_at': datetime.now().isoformat()
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Logistic Regression model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']

            logger.info(f"Logistic Regression model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class RealRandomForest:
    """Real Random Forest model for trading signals"""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = None

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the random forest model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            self.is_trained = True
            self.feature_names = feature_names

            logger.info(f"Random Forest trained - Accuracy: {accuracy:.4f}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}")

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'feature_importance': dict(zip(feature_names or [], self.model.feature_importances_))
            }

        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return None

    def predict(self, X: np.ndarray) -> Dict[str, float]:
        """Make prediction"""
        if not self.is_trained:
            return {'probability': 0.5, 'confidence': 0.0}

        try:
            probabilities = self.model.predict_proba(X)

            # Get probability for positive class (buy signal)
            buy_probability = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]
            confidence = abs(buy_probability - 0.5) * 2  # Scale to 0-1

            return {
                'probability': float(buy_probability),
                'confidence': float(confidence)
            }

        except Exception as e:
            logger.error(f"Random Forest prediction failed: {e}")
            return {'probability': 0.5, 'confidence': 0.0}

    def save_model(self, filepath: str):
        """Save model to file"""
        try:
            model_data = {
                'model': self.model,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'created_at': datetime.now().isoformat()
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Random Forest model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']

            logger.info(f"Random Forest model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


class RealLSTM(nn.Module):
    """Real LSTM Neural Network for sequential trading data"""

    def __init__(self, input_size: int = 25, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class RealLSTMModel:
    """Wrapper for LSTM model training and inference"""

    def __init__(self, input_size: int = 25):
        self.model = RealLSTM(input_size=input_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_trained = False
        self.input_size = input_size

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model"""
        try:
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Create sequences (use sliding window of 10)
            sequence_length = 10
            if X_tensor.shape[0] > sequence_length:
                X_sequences = []
                y_sequences = []
                for i in range(len(X_tensor) - sequence_length + 1):
                    X_sequences.append(X_tensor[i:i+sequence_length])
                    y_sequences.append(y_tensor[i+sequence_length-1])  # Predict next step

                X_tensor = torch.stack(X_sequences)
                y_tensor = torch.stack(y_sequences)

            # Split data
            train_size = int(0.8 * len(X_tensor))
            X_train = X_tensor[:train_size]
            X_test = X_tensor[train_size:]
            y_train = y_tensor[:train_size]
            y_test = y_tensor[train_size:]

            # Loss and optimizer
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)

            # Training loop
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if epoch % 10 == 0:
                    logger.info(f"LSTM Epoch {epoch}/{epochs}, Loss: {total_loss/len(X_train):.4f}")

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                predictions = (test_outputs.squeeze() > 0.5).float()
                accuracy = (predictions == y_test).float().mean().item()

            self.is_trained = True
            logger.info(f"LSTM trained - Accuracy: {accuracy:.4f}")

            return {'accuracy': accuracy}

        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return None

    def predict(self, X: np.ndarray) -> Dict[str, float]:
        """Make prediction"""
        if not self.is_trained:
            return {'probability': 0.5, 'confidence': 0.0}

        try:
            self.model.eval()
            with torch.no_grad():
                # Ensure we have enough data for a sequence
                if X.shape[0] < 10:
                    # Pad with zeros if needed
                    padding = np.zeros((10 - X.shape[0], X.shape[1]))
                    X_padded = np.vstack([padding, X])
                else:
                    X_padded = X[-10:]  # Use last 10 samples

                X_tensor = torch.FloatTensor(X_padded).unsqueeze(0).to(self.device)
                output = self.model(X_tensor)
                probability = output.item()

                confidence = abs(probability - 0.5) * 2  # Scale to 0-1

                return {
                    'probability': float(probability),
                    'confidence': float(confidence)
                }

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {'probability': 0.5, 'confidence': 0.0}

    def save_model(self, filepath: str):
        """Save model to file"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'input_size': self.input_size,
                'is_trained': self.is_trained,
                'created_at': datetime.now().isoformat()
            }, filepath)
            logger.info(f"LSTM model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save LSTM model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Load model from file"""
        try:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.input_size = checkpoint['input_size']
            self.is_trained = checkpoint['is_trained']

            logger.info(f"LSTM model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            return False


class RealXGBoostModel:
    """Real XGBoost model for trading signals"""

    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_names = None

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the XGBoost model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            self.is_trained = True
            self.feature_names = feature_names

            logger.info(f"XGBoost trained - Accuracy: {accuracy:.4f}, "
                       f"Precision: {precision:.4f}, Recall: {recall:.4f}")

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'feature_importance': dict(zip(feature_names or [], self.model.feature_importances_))
            }

        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return None

    def predict(self, X: np.ndarray) -> Dict[str, float]:
        """Make prediction"""
        if not self.is_trained:
            return {'probability': 0.5, 'confidence': 0.0}

        try:
            probabilities = self.model.predict_proba(X)

            # Get probability for positive class (buy signal)
            buy_probability = probabilities[0][1] if len(probabilities[0]) > 1 else probabilities[0][0]
            confidence = abs(buy_probability - 0.5) * 2  # Scale to 0-1

            return {
                'probability': float(buy_probability),
                'confidence': float(confidence)
            }

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return {'probability': 0.5, 'confidence': 0.0}

    def save_model(self, filepath: str):
        """Save model to file"""
        try:
            self.model.save_model(filepath)
            # Save additional metadata
            metadata = {
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'created_at': datetime.now().isoformat()
            }
            metadata_path = filepath.replace('.json', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            logger.info(f"XGBoost model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Load model from file"""
        try:
            self.model.load_model(filepath)
            # Load metadata
            metadata_path = filepath.replace('.json', '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.is_trained = metadata['is_trained']
            self.feature_names = metadata['feature_names']

            logger.info(f"XGBoost model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            return False


# Factory functions
def create_logistic_regression_model() -> RealLogisticRegression:
    """Create logistic regression model"""
    return RealLogisticRegression()


def create_random_forest_model() -> RealRandomForest:
    """Create random forest model"""
    return RealRandomForest()


def create_lstm_model(input_size: int = 25) -> RealLSTMModel:
    """Create LSTM model"""
    return RealLSTMModel(input_size=input_size)


def create_xgboost_model() -> RealXGBoostModel:
    """Create XGBoost model"""
    return RealXGBoostModel()


def generate_sample_training_data(n_samples: int = 1000, n_features: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample training data for demonstration"""
    np.random.seed(42)

    # Generate features (simulating tick data)
    X = np.random.randn(n_samples, n_features)

    # Add some realistic patterns
    for i in range(n_features):
        if i < 5:  # Price-related features
            X[:, i] = np.cumsum(X[:, i]) * 0.01  # Random walk
        elif i < 10:  # Volume features
            X[:, i] = np.abs(X[:, i]) * 1000  # Positive values
        elif i < 15:  # Technical indicators
            X[:, i] = np.tanh(X[:, i])  # Bounded between -1 and 1

    # Generate labels based on some features (simple rule)
    y = ((X[:, 0] > 0) & (X[:, 5] > 0.5)).astype(int)

    return X, y