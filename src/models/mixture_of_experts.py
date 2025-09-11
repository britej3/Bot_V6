"""
Mixture of Experts (MoE) Architecture for CryptoScalp AI

This module implements a Mixture of Experts architecture that dynamically routes
trading signals to specialized neural networks based on market regime detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
import learn2learn as l2l
from learn2learn.data.transforms import (
    NWays, KShots, LoadData, RemapLabels, ConsecutiveLabels
)
from learn2learn.data import MetaDataset
import higher

logger = logging.getLogger(__name__)


def get_optimal_device() -> torch.device:
    """
    Detect the optimal device for PyTorch operations.
    Priority order: CUDA > MPS (Apple Metal) > CPU

    Returns:
        torch.device: The optimal available device
    """
    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device

    # Check for Apple Metal Performance Shaders (MPS) availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal Performance Shaders (MPS) for GPU acceleration")
        return device

    # Fallback to CPU
    device = torch.device("cpu")
    logger.info("Using CPU (CUDA and MPS not available)")
    return device


@dataclass
class RegimeClassification:
    """Market regime classification result"""
    regime: str
    confidence: float
    probabilities: Dict[str, float]

    def __post_init__(self):
        valid_regimes = {'high_volatility', 'low_volatility', 'trending', 'ranging'}
        if self.regime not in valid_regimes:
            raise ValueError(f"Invalid regime: {self.regime}")


@dataclass
class MoESignal:
    """Enhanced trading signal from MoE"""
    direction: float  # -1 to 1 (sell to buy)
    confidence: float  # 0 to 1
    size: float       # Position size recommendation
    regime: str       # Market regime used
    regime_confidence: float
    expert_contributions: Dict[str, float]  # Contribution from each expert


class MarketRegimeDetector(nn.Module):
    """
    Fast, lightweight model to detect current market regime.
    Uses simplified architecture for speed.
    """

    def __init__(self, input_dim: int = 1000, hidden_dim: int = 128):
        super().__init__()

        self.regime_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 regimes
        )

        # Regime mapping
        self.regime_names = ['high_volatility', 'low_volatility', 'trending', 'ranging']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Regime probabilities of shape (batch_size, 4)
        """
        logits = self.regime_classifier(x)
        return F.softmax(logits, dim=-1)

    def classify_regime(self, market_data: torch.Tensor) -> RegimeClassification:
        """Classify current market regime"""
        with torch.no_grad():
            probs = self.forward(market_data.unsqueeze(0)).squeeze(0)
            regime_idx = torch.argmax(probs).item()
            regime = self.regime_names[regime_idx]
            confidence = probs[regime_idx].item()

            probabilities = {
                name: prob.item()
                for name, prob in zip(self.regime_names, probs)
            }

            return RegimeClassification(
                regime=regime,
                confidence=confidence,
                probabilities=probabilities
            )


class RegimeSpecificExpert(nn.Module):
    """
    Specialized neural network for a specific market regime.
    Smaller and faster than the monolithic model.
    """

    def __init__(self, regime_name: str, input_dim: int = 1000):
        super().__init__()
        self.regime = regime_name

        # Lightweight architecture for speed
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Regime-specific LSTM configuration
        lstm_hidden = 32 if regime_name in ['high_volatility', 'ranging'] else 64
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.1
        )

        # Output layers
        self.signal_head = nn.Linear(lstm_hidden, 3)  # [direction, confidence, size]

        # Regime-specific adaptations
        self.apply_regime_specific_init()

    def apply_regime_specific_init(self):
        """Apply regime-specific weight initialization"""
        if self.regime == 'high_volatility':
            # More conservative, lower confidence
            nn.init.xavier_uniform_(self.signal_head.weight, gain=0.5)
        elif self.regime == 'trending':
            # More aggressive, higher confidence
            nn.init.xavier_uniform_(self.signal_head.weight, gain=1.2)
        elif self.regime == 'ranging':
            # Balanced approach
            nn.init.xavier_uniform_(self.signal_head.weight, gain=0.8)
        elif self.regime == 'low_volatility':
            # Conservative but confident
            nn.init.xavier_uniform_(self.signal_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Signal tensor of shape (batch_size, 3)
        """
        # Reshape for Conv1d: (batch, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Extract features
        features = self.feature_extractor(x)  # (batch, 32, seq_len)

        # Reshape for LSTM: (batch, seq_len, features)
        features = features.permute(0, 2, 1)

        # LSTM processing
        lstm_out, _ = self.lstm(features)

        # Use last timestep
        final_features = lstm_out[:, -1, :]

        # Generate signal
        signal = self.signal_head(final_features)

        # Apply regime-specific constraints
        signal = self.apply_regime_constraints(signal)

        return signal

    def apply_regime_constraints(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply regime-specific constraints to output"""
        direction, confidence, size = signal.split(1, dim=-1)

        if self.regime == 'high_volatility':
            # Reduce confidence and position size
            confidence = confidence * 0.7
            size = size * 0.5
        elif self.regime == 'low_volatility':
            # Increase confidence, moderate size
            confidence = torch.clamp(confidence * 1.3, 0, 1)
            size = size * 0.8
        elif self.regime == 'trending':
            # Higher confidence, larger size
            confidence = torch.clamp(confidence * 1.2, 0, 1)
            size = size * 1.2
        elif self.regime == 'ranging':
            # Balanced approach
            confidence = confidence * 0.9
            size = size * 0.9

        # Ensure confidence bounds
        confidence = torch.clamp(confidence, 0, 1)

        # Ensure size is positive
        size = torch.clamp(size, 0.01, 1.0)

        return torch.cat([direction, confidence, size], dim=-1)


class MixtureOfExperts(nn.Module):
    """
    Main Mixture of Experts architecture that routes data to specialized experts
    based on market regime detection.
    """

    def __init__(self, input_dim: int = 1000):
        super().__init__()

        # Market regime detector
        self.regime_detector = MarketRegimeDetector(input_dim)

        # Specialized experts for each regime
        self.experts = nn.ModuleDict({
            'high_volatility': RegimeSpecificExpert('high_volatility', input_dim),
            'low_volatility': RegimeSpecificExpert('low_volatility', input_dim),
            'trending': RegimeSpecificExpert('trending', input_dim),
            'ranging': RegimeSpecificExpert('ranging', input_dim)
        })

        # Gating network for expert combination
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )

        # Expert contribution tracking
        self.expert_contributions = {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass through MoE architecture.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Tuple of (final_signal, regime_probabilities)
        """
        # Detect market regime
        regime_probs = self.regime_detector(x)
        regime_classification = self.regime_detector.classify_regime(x[0] if x.dim() > 1 else x)

        # Get signals from all experts
        expert_signals = {}
        expert_contributions = {}

        for regime_name, expert in self.experts.items():
            expert_signals[regime_name] = expert(x)
            expert_contributions[regime_name] = regime_probs[0, self.regime_detector.regime_names.index(regime_name)].item()

        # Dynamic routing based on regime
        regime_idx = self.regime_detector.regime_names.index(regime_classification.regime)
        primary_expert = self.experts[regime_classification.regime]
        primary_signal = primary_expert(x)

        # Gate-based combination for ensemble effect
        gate_weights = self.gate_network(x)
        combined_signal = sum(
            gate_weights[0, i] * expert_signals[regime_name]
            for i, regime_name in enumerate(self.regime_detector.regime_names)
        )

        # Adaptive combination based on regime confidence
        if regime_classification.confidence > 0.8:
            # High confidence: use primary expert
            final_signal = primary_signal
        else:
            # Low confidence: use weighted combination
            final_signal = 0.7 * primary_signal + 0.3 * combined_signal

        self.expert_contributions = expert_contributions

        return final_signal, regime_classification.probabilities

    def generate_trading_signal(self, market_data: torch.Tensor) -> MoESignal:
        """Generate enhanced trading signal with regime awareness"""
        with torch.no_grad():
            signal_output, regime_probs = self.forward(market_data.unsqueeze(0))
            signal_output = signal_output.squeeze(0)

            regime_classification = self.regime_detector.classify_regime(market_data)

            direction, confidence, size = signal_output.tolist()

            return MoESignal(
                direction=direction,
                confidence=confidence,
                size=size,
                regime=regime_classification.regime,
                regime_confidence=regime_classification.confidence,
                expert_contributions=self.expert_contributions
            )


class MetaLearningTrainer:
    """
    Handles meta-training of RegimeSpecificExpert models using MAML.
    """
    def __init__(self, model_template: nn.Module, input_dim: int, device: torch.device, inner_lr: float = 0.01, meta_lr: float = 0.001):
        self.model_template = model_template
        self.input_dim = input_dim
        self.device = device
        self.loss_fn = nn.MSELoss()  # Example loss function for regression tasks
        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(self.model_template.parameters(), lr=meta_lr)

    def train_meta_batch(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], adaptation_steps: int = 1):
        """
        Performs a meta-training step on a batch of tasks.
        Each task consists of (support_data, support_labels, query_data, query_labels).
        """
        self.meta_optimizer.zero_grad()
        meta_batch_loss = 0.0

        for support_data, support_labels, query_data, query_labels in tasks:
            # Move data to device
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)

            # Create a functional model for inner loop updates
            with higher.innerloop_ctx(self.model_template, self.meta_optimizer, copy_initial_weights=False) as (fmodel, diff_optimizer):
                # Inner loop: Adaptation
                for step in range(adaptation_steps):
                    predictions_support = fmodel(support_data)
                    loss_support = self.loss_fn(predictions_support, support_labels)
                    diff_optimizer.step(loss_support)

                # Outer loop: Meta-update
                predictions_query = fmodel(query_data)
                loss_query = self.loss_fn(predictions_query, query_labels)
                meta_batch_loss += loss_query

        meta_batch_loss.backward()
        self.meta_optimizer.step()
        return meta_batch_loss.item()

class MoETradingEngine:
    """
    High-level interface for the MoE trading system.
    Integrates with existing trading infrastructure.
    """

    def __init__(self, input_dim: int = 1000, device: Optional[str] = None, meta_learning_enabled: bool = False):
        # Auto-detect optimal device if not specified
        if device is None:
            self.device = get_optimal_device()
        else:
            self.device = torch.device(device)

        self.model = MixtureOfExperts(input_dim).to(self.device)
        self.model.eval()

        self.meta_learning_enabled = meta_learning_enabled
        if self.meta_learning_enabled:
            # For meta-learning, we'll train the experts.
            # We need to ensure the experts are trainable.
            # A simplified approach for now: pass one expert as a template.
            # In a full MAML setup, the outer model would be the one being optimized.
            # Here, we'll assume meta-learning applies to the structure of RegimeSpecificExpert.
            # We'll instantiate a dummy expert to pass its structure to the trainer.
            dummy_expert = RegimeSpecificExpert("trending", input_dim).to(self.device)
            self.meta_trainer = MetaLearningTrainer(dummy_expert, input_dim, self.device)
            self.model.train() # Set MoE to train mode if meta-learning is enabled

        # Performance tracking
        self.inference_times = []
        self.regime_distribution = {}

    async def generate_signal(self, market_data: Dict[str, np.ndarray]) -> MoESignal:
        """Generate trading signal from market data"""
        import time

        # Prepare input tensor
        input_tensor = self.prepare_input_tensor(market_data)

        # Measure inference time
        start_time = time.time()

        signal = self.model.generate_trading_signal(input_tensor)

        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)

        # Track regime distribution
        self.regime_distribution[signal.regime] = self.regime_distribution.get(signal.regime, 0) + 1

        logger.info(f"MoE Signal: {signal.direction:.3f} confidence: {signal.confidence:.3f} "
                   f"size: {signal.size:.3f} regime: {signal.regime} "
                   f"inference: {inference_time:.2f}ms")

        return signal

    def prepare_input_tensor(self, market_data: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert market data dict to model input tensor"""
        # Flatten and concatenate all features
        features = []
        for key, value in market_data.items():
            if isinstance(value, np.ndarray):
                features.append(value.flatten())
            elif isinstance(value, (int, float)):
                features.append([value])

        # Concatenate all features
        combined_features = np.concatenate(features)

        # Ensure correct input dimension
        if len(combined_features) < self.model.regime_detector.regime_classifier[0].in_features:
            # Pad with zeros if needed
            padding = self.model.regime_detector.regime_classifier[0].in_features - len(combined_features)
            combined_features = np.pad(combined_features, (0, padding), 'constant')

        return torch.FloatTensor(combined_features).to(self.device)

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring"""
        if not self.inference_times:
            return {'avg_inference_time_ms': 0, 'regime_distribution': {}}

        return {
            'avg_inference_time_ms': np.mean(self.inference_times),
            'p95_inference_time_ms': np.percentile(self.inference_times, 95),
            'regime_distribution': self.regime_distribution.copy()
        }

    def load_pretrained_weights(self, weights_path: str):
        """Load pretrained weights for the MoE model"""
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded MoE weights from {weights_path}")
        except Exception as e:
            logger.error(f"Failed to load MoE weights: {e}")

    def save_weights(self, save_path: str):
        """Save current model weights"""
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Saved MoE weights to {save_path}")

    def meta_train_experts(self, meta_dataset: MetaDataset, epochs: int = 100, tasks_per_epoch: int = 10):
        """
        Orchestrates the meta-training process for the experts.
        meta_dataset: A learn2learn MetaDataset that provides tasks.
        """
        if not self.meta_learning_enabled:
            logger.warning("Meta-learning is not enabled. Cannot meta-train experts.")
            return

        logger.info(f"Starting meta-training for {epochs} epochs...")

        for epoch in range(epochs):
            self.meta_trainer.meta_optimizer.zero_grad()
            meta_batch_loss = 0.0

            for _ in range(tasks_per_epoch):
                # In a real scenario, you would sample a task from a MetaDataset.
                # A task would represent a specific market condition, trading pair, or time period.
                # For demonstration, we generate dummy data for support and query sets.
                
                # Dummy data for a task (support_data, support_labels, query_data, query_labels)
                # Assuming input_dim for features and 3 outputs (direction, confidence, size)
                support_data = torch.randn(10, self.input_dim)
                support_labels = torch.randn(10, 3)
                query_data = torch.randn(10, self.input_dim)
                query_labels = torch.randn(10, 3)

                # Perform one meta-training step for this task
                meta_batch_loss += self.meta_trainer.train_meta_batch(
                    [(support_data, support_labels, query_data, query_labels)],
                    adaptation_steps=1 # Number of inner loop steps
                )

            meta_batch_loss /= tasks_per_epoch
            meta_batch_loss.backward()
            self.meta_trainer.meta_optimizer.step()

            logger.info(f"Epoch {epoch+1}/{epochs}, Meta-Batch Loss: {meta_batch_loss.item():.4f}")

        logger.info("Meta-training complete.")
        # After meta-training, the self.model (MixtureOfExperts) would have its experts
        # initialized with meta-learned parameters, ready for fast adaptation.
        # You might want to save these meta-learned initializations.
        self.save_weights("meta_learned_moe_weights.pth")

# Example usage and testing
if __name__ == "__main__":
    # Example market data
    example_data = {
        'price': np.random.randn(100),
        'volume': np.random.randn(100),
        'volatility': np.array([0.05]),
        'rsi': np.array([65.0]),
        'macd': np.array([0.002])
    }

    # Initialize MoE engine
    moe_engine = MoETradingEngine(input_dim=1000)

    # Generate signal (asynchronous in practice)
    import asyncio

    async def test_moe():
        signal = await moe_engine.generate_signal(example_data)
        print(f"Generated signal: {signal}")
        print(f"Performance metrics: {moe_engine.get_performance_metrics()}")

    # Run test
    asyncio.run(test_moe())