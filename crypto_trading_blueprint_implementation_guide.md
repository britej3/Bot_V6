# Crypto Trading Blueprint Implementation Guide

## Overview

This implementation guide provides step-by-step instructions for integrating the crypto trading blueprint specifications into the existing CryptoScalp AI codebase. The guide focuses on leveraging existing components while implementing missing blueprint requirements.

## Prerequisites

### System Requirements
- **Intel MacBook Pro** (i7 6-core, 16GB RAM, AMD 4GB GPU)
- **Python 3.9+** with conda/virtual environment
- **Docker** for containerized services
- **Git** for version control

### Existing Codebase Components
Ensure these components are present and functional:
- Nautilus Trader integration (`src/trading/nautilus_integration.py`)
- XGBoost ensemble (`src/learning/xgboost_ensemble.py`)
- Adaptive risk management (`src/learning/adaptive_risk_management.py`)
- Binance data pipeline (`src/data_pipeline/binance_data_manager.py`)

## Phase 1: Core Infrastructure Setup

### Step 1.1: Enhanced Data Pipeline Implementation

#### 1.1.1 Install Required Dependencies
```bash
# Add to requirements.txt
pip install confluent-kafka redis dragonfly
pip install timescaledb clickhouse-driver
```

#### 1.1.2 Create Kafka Configuration
```python
# src/config/kafka_config.py
from dataclasses import dataclass

@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    topic_prefix: str = "cryptoscalp"
    consumer_group: str = "trading_engine"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
```

#### 1.1.3 Implement Real-Time Data Pipeline
```python
# src/data_pipeline/real_time_pipeline.py
import asyncio
from confluent_kafka import Producer, Consumer
import redis
import json
from typing import Dict, Any

class RealTimeDataPipeline:
    def __init__(self, kafka_config, redis_config):
        self.producer = Producer({'bootstrap.servers': kafka_config.bootstrap_servers})
        self.redis = redis.Redis(**redis_config)
        self.consumer = Consumer({
            'bootstrap.servers': kafka_config.bootstrap_servers,
            'group.id': kafka_config.consumer_group,
            'auto.offset.reset': kafka_config.auto_offset_reset
        })

    async def publish_market_data(self, exchange: str, symbol: str, data: Dict[str, Any]):
        """Publish market data to Kafka"""
        topic = f"{self.kafka_config.topic_prefix}.market.{exchange}.{symbol}"
        message = json.dumps(data).encode('utf-8')
        self.producer.produce(topic, message)
        self.producer.flush()

    async def consume_signals(self, callback):
        """Consume trading signals from Kafka"""
        topic = f"{self.kafka_config.topic_prefix}.signals"
        self.consumer.subscribe([topic])

        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            signal = json.loads(msg.value().decode('utf-8'))
            await callback(signal)
```

### Step 1.2: ML Model Ensemble Expansion

#### 1.2.1 Implement Temporal Convolutional Network (TCN)
```python
# src/learning/temporal_convolutional_network.py
import torch
import torch.nn as nn
from typing import List

class TemporalConvolutionalNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.tcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.tcn_layers.append(
                nn.Conv1d(
                    in_channels=input_size if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=3,
                    padding=1
                )
            )
            self.tcn_layers.append(nn.ReLU())
            self.tcn_layers.append(nn.Dropout(0.2))

    def forward(self, x):
        """Forward pass through TCN"""
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)

        for layer in self.tcn_layers:
            x = layer(x)

        # Global average pooling
        x = torch.mean(x, dim=2)  # (batch_size, hidden_size)
        return x
```

#### 1.2.2 Implement TabNet Model
```python
# src/learning/tabnet_model.py
import torch
import torch.nn as nn
import numpy as np

class TabNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_steps: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps

        # Feature transformer
        self.feature_transformer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Attentive transformer
        self.attentive_transformer = nn.Sequential(
            nn.Linear(64, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Softmax(dim=1)
        )

        # Output layer
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, x):
        """Forward pass with feature selection"""
        batch_size = x.size(0)

        # Initial feature transformation
        features = self.feature_transformer(x)

        # Attentive feature selection
        attention_weights = self.attentive_transformer(features)
        selected_features = features * attention_weights

        # Final prediction
        output = self.output_layer(selected_features)
        return output, attention_weights
```

#### 1.2.3 Implement PPO Trading Agent
```python
# src/learning/ppo_trading_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class PPOTradingAgent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Mean and std for each action
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def get_action(self, state):
        """Get action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_params = self.actor(state)

        mean = action_params[:, :self.action_dim]
        std = torch.exp(action_params[:, self.action_dim:])

        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy()[0], log_prob

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO"""
        action_params = self.actor(states)
        mean = action_params[:, :self.action_dim]
        std = torch.exp(action_params[:, self.action_dim:])

        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        values = self.critic(states)

        return log_probs, values, entropy
```

## Phase 2: Advanced Feature Implementation

### Step 2.1: Order Flow Analysis

#### 2.1.1 Implement Order Flow Analyzer
```python
# src/analysis/order_flow_analyzer.py
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class OrderFlowMetrics:
    order_imbalance: float
    aggressive_buy_ratio: float
    passive_sell_ratio: float
    whale_activity_score: float
    iceberg_probability: float

class OrderFlowAnalyzer:
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.order_history = []

    def analyze_order_book(self, order_book: Dict) -> OrderFlowMetrics:
        """Analyze order book for institutional activity"""
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        # Calculate order imbalance
        bid_volume = sum([level[1] for level in bids[:10]])  # Top 10 levels
        ask_volume = sum([level[1] for level in asks[:10]])
        order_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        # Detect whale activity (large orders)
        whale_threshold = np.percentile([level[1] for level in bids + asks], 95)
        whale_orders = [level for level in bids + asks if level[1] > whale_threshold]
        whale_activity_score = len(whale_orders) / len(bids + asks)

        # Analyze aggressive vs passive flow
        aggressive_buy_ratio = self._calculate_aggressive_buy_ratio(bids, asks)
        passive_sell_ratio = self._calculate_passive_sell_ratio(bids, asks)

        # Iceberg order detection
        iceberg_probability = self._detect_iceberg_orders(bids, asks)

        return OrderFlowMetrics(
            order_imbalance=order_imbalance,
            aggressive_buy_ratio=aggressive_buy_ratio,
            passive_sell_ratio=passive_sell_ratio,
            whale_activity_score=whale_activity_score,
            iceberg_probability=iceberg_probability
        )

    def _calculate_aggressive_buy_ratio(self, bids, asks):
        """Calculate ratio of aggressive buy orders"""
        # Implementation for aggressive buy detection
        return 0.5  # Placeholder

    def _calculate_passive_sell_ratio(self, bids, asks):
        """Calculate ratio of passive sell orders"""
        # Implementation for passive sell detection
        return 0.3  # Placeholder

    def _detect_iceberg_orders(self, bids, asks):
        """Detect potential iceberg orders"""
        # Implementation for iceberg detection
        return 0.1  # Placeholder
```

### Step 2.2: Volume Profile Analysis

#### 2.2.1 Implement Volume Profile Analyzer
```python
# src/analysis/volume_profile_analyzer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class VolumeProfile:
    poc_price: float  # Point of Control
    value_area_high: float
    value_area_low: float
    volume_nodes: List[Tuple[float, float]]  # (price, volume)

class VolumeProfileAnalyzer:
    def __init__(self, price_bins: int = 50):
        self.price_bins = price_bins

    def calculate_volume_profile(self, price_data: pd.Series, volume_data: pd.Series) -> VolumeProfile:
        """Calculate volume profile for given time period"""
        # Create price bins
