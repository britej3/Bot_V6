# Minimum Viable Vertical Slice: BTC/USDT Trading Implementation

## Overview

This document outlines the implementation of a Minimum Viable Vertical Slice (MVVS) for the CryptoScalp AI system. The goal is to build one complete end-to-end path through the system before expanding to the full breadth of 15+ agents and 4 expert models.

**Selected Slice: BTC/USDT Trading**
- **Exchange**: Binance Futures (single exchange for simplicity)
- **Trading Pair**: BTC/USDT (most liquid pair)
- **Market Regime**: Trending Market (one regime to start)
- **Execution**: Paper trading validation

## Architecture for MVVS

### Simplified Component Stack
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Binance       │───▶│  Market Data    │───▶│   Trending      │
│   WebSocket     │    │  Gateway        │    │   Expert       │
│   Feed          │    │  (BTC/USDT)     │    │   Model        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Protocol      │    │   Protocol      │    │   Protocol      │
│   Buffer        │    │   Buffer        │    │   Buffer        │
│   Messages      │    │   Messages      │    │   Messages      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   NATS          │    │   NATS          │    │   Alpha        │
│   Message       │    │   Message       │    │   Agent        │
│   Bus           │    │   Bus           │    │   (Decision)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │   Execution     │
                                              │   Core          │
                                              │   (Paper Trade) │
                                              └─────────────────┘
```

## Implementation Phases

### Phase 1: Data Infrastructure (Weeks 1-2)

#### 1.1 Binance WebSocket Connection
```python
# src/data_ingestion/binance_feed.py
import asyncio
import websockets
import json
from typing import Callable, Dict, Any

class BinanceWebSocketFeed:
    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol
        self.ws_url = f"wss://fstream.binance.com/ws/{symbol}@depth10@100ms"
        self.callbacks: List[Callable] = []

    async def connect(self):
        """Connect to Binance WebSocket and stream data"""
        async with websockets.connect(self.ws_url) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    await self._process_message(data)
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    await asyncio.sleep(5)  # Reconnect delay

    async def _process_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        for callback in self.callbacks:
            await callback(data)
```

#### 1.2 Market Data Gateway (Simplified)
```python
# src/data_gateway/market_data_gateway.py
import asyncio
import nats
from binance_feed import BinanceWebSocketFeed

class MarketDataGateway:
    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol
        self.feed = BinanceWebSocketFeed(symbol)
        self.nc = None

    async def start(self):
        """Start the market data gateway"""
        # Connect to NATS
        self.nc = await nats.connect("nats://localhost:4222")

        # Register callback for WebSocket data
        self.feed.callbacks.append(self._on_market_data)

        # Start WebSocket connection
        await self.feed.connect()

    async def _on_market_data(self, data: Dict[str, Any]):
        """Handle incoming market data and publish to NATS"""
        # Convert to protobuf message
        proto_message = self._convert_to_protobuf(data)

        # Publish to NATS
        await self.nc.publish(f"market.{self.symbol}", proto_message.SerializeToString())
```

### Phase 2: Model Implementation (Weeks 3-4)

#### 2.1 Trending Market Expert Model
```python
# src/models/experts/trending_expert.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

class TrendingMarketExpert(nn.Module):
    def __init__(self, input_dim: int = 1000):
        super().__init__()
        self.input_dim = input_dim

        # Simplified architecture for MVVS
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128)
        )

        self.lstm = nn.LSTM(32, 16, batch_first=True)
        self.output = nn.Linear(16, 3)  # [direction, confidence, size]

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward pass through the model"""
        # x shape: [batch, sequence, features]
        batch_size, seq_len, features = x.shape

        # Extract features
        x_reshaped = x.permute(0, 2, 1)  # [batch, features, sequence]
        features = self.feature_extractor(x_reshaped)

        # LSTM processing
        features = features.permute(0, 2, 1)  # [batch, sequence, features]
        lstm_out, _ = self.lstm(features)

        # Generate prediction
        prediction = self.output(lstm_out[:, -1, :])

        # Apply softmax for direction and confidence
        direction = torch.softmax(prediction[:, 0:3], dim=-1)
        confidence = torch.sigmoid(prediction[:, 2:3])
        position_size = torch.tanh(prediction[:, 3:4])

        return {
            "direction": direction,  # [HOLD, BUY, SELL] probabilities
            "confidence": confidence,
            "position_size": position_size,
            "raw_prediction": prediction
        }
```

#### 2.2 Model Training Script
```python
# scripts/train_trending_expert.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.experts.trending_expert import TrendingMarketExpert

def train_trending_expert():
    """Train the trending market expert model"""

    # Initialize model
    model = TrendingMarketExpert()
    criterion = nn.MSELoss()  # Simplified loss for MVVS
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load training data (simplified for MVVS)
    # In practice, this would load from your data pipeline
    train_loader = get_training_data()

    # Training loop
    model.train()
    for epoch in range(10):  # Short training for MVVS
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_x)
            loss = criterion(outputs["direction"], batch_y)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "models/trained/trending_expert_v1.pth")
    return model
```

### Phase 3: Signal Flow Integration (Weeks 5-6)

#### 3.1 Signal Generation Service
```python
# src/signals/signal_generator.py
import asyncio
import nats
import torch
from models.experts.trending_expert import TrendingMarketExpert

class SignalGenerator:
    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol
        self.model = TrendingMarketExpert()
        self.model.load_state_dict(torch.load("models/trained/trending_expert_v1.pth"))
        self.model.eval()

        self.nc = None
        self.price_history = []

    async def start(self):
        """Start the signal generator"""
        self.nc = await nats.connect("nats://localhost:4222")

        # Subscribe to market data
        await self.nc.subscribe(f"market.{self.symbol}", cb=self._on_market_data)

    async def _on_market_data(self, msg):
        """Handle incoming market data"""
        # Deserialize protobuf message
        market_data = MarketDataProto().FromString(msg.data)

        # Update price history
        self.price_history.append(market_data.price)
        if len(self.price_history) > 1000:  # Keep last 1000 points
            self.price_history = self.price_history[-1000:]

        # Generate signals when we have enough data
        if len(self.price_history) >= 100:
            await self._generate_signal()

    async def _generate_signal(self):
        """Generate trading signal using the model"""
        # Prepare input tensor
        input_data = torch.tensor(self.price_history[-100:]).unsqueeze(0).unsqueeze(-1)
        input_data = input_data.float()

        # Get model prediction
        with torch.no_grad():
            prediction = self.model(input_data)

        # Create trading signal
        signal = TradingSignal()
        signal.symbol = self.symbol
        signal.direction = prediction["direction"].argmax().item()
        signal.confidence = prediction["confidence"].item()
        signal.position_size = prediction["position_size"].item()

        # Publish signal to NATS
        await self.nc.publish(f"signals.{self.symbol}", signal.SerializeToString())
```

#### 3.2 Alpha Agent (Decision Making)
```python
# src/agents/alpha_agent.py
import asyncio
import nats

class AlphaAgent:
    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol
        self.nc = None
        self.current_position = 0.0
        self.pending_signals = []

    async def start(self):
        """Start the alpha agent"""
        self.nc = await nats.connect("nats://localhost:4222")

        # Subscribe to trading signals
        await self.nc.subscribe(f"signals.{self.symbol}", cb=self._on_trading_signal)

    async def _on_trading_signal(self, msg):
        """Handle incoming trading signal"""
        signal = TradingSignal().FromString(msg.data)

        # Simple decision logic for MVVS
        if signal.confidence > 0.7:  # High confidence threshold
            if signal.direction == 1 and self.current_position <= 0:  # BUY signal
                await self._send_order(signal, "BUY")
            elif signal.direction == 2 and self.current_position >= 0:  # SELL signal
                await self._send_order(signal, "SELL")

    async def _send_order(self, signal, side):
        """Send order to execution core"""
        order = Order()
        order.symbol = signal.symbol
        order.side = side
        order.size = abs(signal.position_size)
        order.price = 0.0  # Market order
        order.order_type = "MARKET"

        await self.nc.publish(f"orders.{self.symbol}", order.SerializeToString())

        # Update position (simplified)
        if side == "BUY":
            self.current_position += order.size
        else:
            self.current_position -= order.size
```

### Phase 4: Paper Trading Validation (Weeks 7-8)

#### 4.1 Execution Core (Paper Trading)
```python
# src/execution/paper_trader.py
import asyncio
import nats
from datetime import datetime

class PaperTrader:
    def __init__(self, symbol: str = "btcusdt", initial_balance: float = 10000.0):
        self.symbol = symbol
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trades = []

        self.nc = None
        self.current_price = 0.0

    async def start(self):
        """Start the paper trader"""
        self.nc = await nats.connect("nats://localhost:4222")

        # Subscribe to orders
        await self.nc.subscribe(f"orders.{self.symbol}", cb=self._on_order)

        # Subscribe to market data for current price
        await self.nc.subscribe(f"market.{self.symbol}", cb=self._on_price_update)

    async def _on_order(self, msg):
        """Execute paper trade"""
        order = Order().FromString(msg.data)

        if order.order_type == "MARKET":
            execution_price = self.current_price
        else:
            execution_price = order.price

        # Calculate trade value
        trade_value = order.size * execution_price

        if order.side == "BUY" and self.balance >= trade_value:
            # Execute buy order
            self.position += order.size
            self.balance -= trade_value
            self.entry_price = execution_price

            trade = {
                "timestamp": datetime.now().isoformat(),
                "side": "BUY",
                "size": order.size,
                "price": execution_price,
                "value": trade_value
            }
            self.trades.append(trade)
            print(f"Executed BUY: {trade}")

        elif order.side == "SELL" and self.position >= order.size:
            # Execute sell order
            pnl = (execution_price - self.entry_price) * order.size
            self.position -= order.size
            self.balance += trade_value + pnl

            trade = {
                "timestamp": datetime.now().isoformat(),
                "side": "SELL",
                "size": order.size,
                "price": execution_price,
                "value": trade_value,
                "pnl": pnl
            }
            self.trades.append(trade)
            print(f"Executed SELL: {trade}, PnL: ${pnl:.2f}")

    async def _on_price_update(self, msg):
        """Update current market price"""
        market_data = MarketDataProto().FromString(msg.data)
        self.current_price = market_data.price
```

## Validation Metrics

### Success Criteria for MVVS
- [ ] **Data Flow**: Successfully ingest BTC/USDT data from Binance
- [ ] **Model Training**: Train trending expert model with reasonable accuracy (>60%)
- [ ] **Signal Generation**: Generate trading signals with confidence scores
- [ ] **Order Execution**: Successfully execute paper trades
- [ ] **Performance**: End-to-end latency < 50ms for signal to order
- [ ] **Stability**: System runs continuously for 24+ hours without errors

### Performance Benchmarks
- **Signal Generation**: < 10ms per signal
- **Message Processing**: < 1ms per NATS message
- **Order Execution**: < 5ms paper trade execution
- **Memory Usage**: < 2GB total system memory
- **CPU Usage**: < 30% average CPU utilization

## Expansion Path

After validating the MVVS, the expansion path follows:

### Immediate Next Steps (After MVVS Validation)
1. **Add Second Expert**: Implement "Ranging Market" expert
2. **Multi-Symbol Support**: Add ETH/USDT and SOL/USDT
3. **Additional Exchanges**: Integrate OKX and Bybit
4. **Advanced Features**: Add stop-loss, take-profit, position sizing

### Long-term Expansion
1. **Full MoE Implementation**: Add all 4 expert models
2. **Production Deployment**: Move from paper trading to live execution
3. **Advanced Risk Management**: Implement 7-layer risk controls
4. **Monitoring & Observability**: Add comprehensive monitoring stack

## Risk Mitigation

### Technical Risks
- **Model Performance**: Start with simple models, optimize iteratively
- **Data Quality**: Implement comprehensive data validation early
- **Integration Complexity**: Use well-defined Protobuf contracts
- **Performance Issues**: Monitor and optimize bottlenecks early

### Operational Risks
- **Data Continuity**: Implement reconnection logic and error handling
- **Resource Constraints**: Start with minimal resource requirements
- **Debugging Complexity**: Add comprehensive logging and monitoring

This MVVS approach ensures we validate the core architecture and integrations before investing significant time in the full implementation, reducing overall project risk and providing early feedback on system design decisions.