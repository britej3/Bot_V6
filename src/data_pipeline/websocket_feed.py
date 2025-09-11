"""
WebSocket Data Feed for Real-time Crypto Data

This module handles WebSocket connections to cryptocurrency exchanges
for real-time market data streaming with automatic reconnection and
failover mechanisms.
"""

import asyncio
import logging
import json
import websockets
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import ccxt
import ccxt.pro as ccxtpro
from typing import Optional

# Optional Kafka integration
from src.config.kafka_config import kafka_config
try:
    from src.data_pipeline.kafka_io import make_kafka_callback
    _KAFKA_INTEGRATION_AVAILABLE = True
except Exception:
    _KAFKA_INTEGRATION_AVAILABLE = False

from src.data_pipeline.data_validator import DataValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WebSocketConfig:
    """Configuration for WebSocket connections"""
    exchange_name: str
    ws_url: str
    symbols: List[str]
    channels: List[str] = None
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    ping_interval: float = 30.0
    pong_timeout: float = 10.0

    def __post_init__(self):
        if self.channels is None:
            self.channels = ['ticker', 'orderbook']

class WebSocketDataFeed:
    """
    WebSocket data feed manager for real-time crypto data

    Features:
    - Multi-exchange WebSocket connections
    - Automatic reconnection and failover
    - Heartbeat monitoring
    - Data normalization and validation
    - Performance metrics
    """

    def __init__(self, configs: List[WebSocketConfig]):
        self.configs = configs
        self.connections = {}  # exchange_name -> connection
        self.subscriptions = {}  # exchange_name -> subscription_info
        self.is_running = False
        self.data_callbacks = []
        self.validator = DataValidator()

        # Performance tracking
        self.message_counts = {config.exchange_name: 0 for config in configs}
        self.error_counts = {config.exchange_name: 0 for config in configs}
        self.reconnect_counts = {config.exchange_name: 0 for config in configs}

        # Auto-wire Kafka callback when enabled via config
        if kafka_config.enabled:
            if _KAFKA_INTEGRATION_AVAILABLE:
                try:
                    self.add_data_callback(make_kafka_callback())
                    logger.info("Kafka publishing enabled; callback registered")
                except Exception as e:
                    logger.warning("Kafka enabled but callback setup failed: %s", e)
            else:
                logger.warning("Kafka enabled but kafka_io module is unavailable")

    def add_data_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback function for processing incoming data"""
        self.data_callbacks.append(callback)

    async def start(self):
        """Start WebSocket connections"""
        self.is_running = True
        logger.info(f"Starting WebSocket data feed for {len(self.configs)} exchanges")

        # Start all WebSocket connections
        tasks = []
        for config in self.configs:
            tasks.append(asyncio.create_task(self._manage_connection(config)))

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"WebSocket data feed error: {e}")
            await self.stop()

    async def stop(self):
        """Stop all WebSocket connections"""
        self.is_running = False
        logger.info("Stopping WebSocket data feed")

        # Close all connections
        for exchange_name, connection in self.connections.items():
            if connection and not connection.closed:
                await connection.close()
                logger.info(f"Closed WebSocket connection for {exchange_name}")

        self.connections.clear()
        self.subscriptions.clear()

    async def _manage_connection(self, config: WebSocketConfig):
        """Manage WebSocket connection with reconnection logic"""
        reconnect_attempts = 0

        while self.is_running and reconnect_attempts < config.max_reconnect_attempts:
            try:
                await self._connect_and_listen(config)
                reconnect_attempts = 0  # Reset on successful connection

            except Exception as e:
                reconnect_attempts += 1
                self.error_counts[config.exchange_name] += 1
                self.reconnect_counts[config.exchange_name] += 1

                logger.warning(
                    f"WebSocket connection failed for {config.exchange_name} "
                    f"(attempt {reconnect_attempts}/{config.max_reconnect_attempts}): {e}"
                )

                if reconnect_attempts < config.max_reconnect_attempts:
                    await asyncio.sleep(config.reconnect_delay)
                else:
                    logger.error(f"Max reconnection attempts reached for {config.exchange_name}")

    async def _connect_and_listen(self, config: WebSocketConfig):
        """Connect to WebSocket and start listening for data"""
        logger.info(f"Connecting to {config.exchange_name} WebSocket: {config.ws_url}")

        # Create WebSocket connection
        connection = await websockets.connect(
            config.ws_url,
            ping_interval=config.ping_interval,
            ping_timeout=config.pong_timeout or (config.ping_interval * 2)
        )

        self.connections[config.exchange_name] = connection

        # Subscribe to channels
        await self._subscribe_to_channels(connection, config)

        # Listen for messages
        async for message in connection:
            try:
                await self._process_message(config.exchange_name, message)
                self.message_counts[config.exchange_name] += 1

            except Exception as e:
                logger.error(f"Error processing message from {config.exchange_name}: {e}")
                self.error_counts[config.exchange_name] += 1

    async def _subscribe_to_channels(self, connection, config: WebSocketConfig):
        """Subscribe to specified channels for the exchange"""
        if config.exchange_name.lower() == 'binance':
            await self._subscribe_binance(connection, config)
        elif config.exchange_name.lower() == 'okx':
            await self._subscribe_okx(connection, config)
        elif config.exchange_name.lower() == 'bybit':
            await self._subscribe_bybit(connection, config)
        else:
            logger.warning(f"No subscription logic for {config.exchange_name}")

    async def _subscribe_binance(self, connection, config: WebSocketConfig):
        """Subscribe to Binance WebSocket channels"""
        subscriptions = []

        for symbol in config.symbols:
            for channel in config.channels:
                if channel == 'ticker':
                    subscriptions.append(f"{symbol.lower()}@ticker")
                elif channel == 'orderbook':
                    subscriptions.append(f"{symbol.lower()}@depth20@100ms")

        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": subscriptions,
            "id": 1
        }

        await connection.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to Binance channels: {subscriptions}")

    async def _subscribe_okx(self, connection, config: WebSocketConfig):
        """Subscribe to OKX WebSocket channels"""
        args = []

        for symbol in config.symbols:
            for channel in config.channels:
                if channel == 'ticker':
                    args.append({"channel": "tickers", "instId": symbol})
                elif channel == 'orderbook':
                    args.append({"channel": "books5", "instId": symbol})

        subscribe_message = {
            "op": "subscribe",
            "args": args
        }

        await connection.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to OKX channels: {args}")

    async def _subscribe_bybit(self, connection, config: WebSocketConfig):
        """Subscribe to Bybit WebSocket channels"""
        topics = []

        for symbol in config.symbols:
            for channel in config.channels:
                if channel == 'ticker':
                    topics.append(f"tickers.{symbol}")
                elif channel == 'orderbook':
                    topics.append(f"orderbook.50.{symbol}")

        subscribe_message = {
            "op": "subscribe",
            "args": topics
        }

        await connection.send(json.dumps(subscribe_message))
        logger.info(f"Subscribed to Bybit topics: {topics}")

    async def _process_message(self, exchange_name: str, message: str):
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)

            # Normalize data format across exchanges
            normalized_data = self._normalize_data(exchange_name, data)

            if normalized_data:
                # Optional validation step
                try:
                    if normalized_data.get('type') == 'ticker':
                        symbol = normalized_data.get('symbol', '')
                        vres = self.validator.validate_market_data(exchange_name, symbol, normalized_data)
                        if not vres.is_valid:
                            logger.debug(f"Validation failed for {exchange_name}/{symbol}: {vres.issues}")
                            return
                    elif normalized_data.get('type') == 'orderbook':
                        symbol = normalized_data.get('symbol', '')
                        vres = self.validator.validate_orderbook_data(exchange_name, symbol, normalized_data)
                        if not vres.is_valid:
                            logger.debug(f"Orderbook validation failed for {exchange_name}/{symbol}: {vres.issues}")
                            return
                except Exception as e:
                    logger.debug(f"Validation exception ignored (pass-through): {e}")

                # Call all registered callbacks
                for callback in self.data_callbacks:
                    try:
                        callback(exchange_name, normalized_data)
                    except Exception as e:
                        logger.error(f"Error in data callback: {e}")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse message from {exchange_name}: {e}")
        except Exception as e:
            logger.error(f"Error processing message from {exchange_name}: {e}")

    def _normalize_data(self, exchange_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize data format across different exchanges"""
        try:
            if exchange_name.lower() == 'binance':
                return self._normalize_binance_data(data)
            elif exchange_name.lower() == 'okx':
                return self._normalize_okx_data(data)
            elif exchange_name.lower() == 'bybit':
                return self._normalize_bybit_data(data)
            else:
                logger.warning(f"No normalization logic for {exchange_name}")
                return None

        except Exception as e:
            logger.error(f"Error normalizing data from {exchange_name}: {e}")
            return None

    def _normalize_binance_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Binance WebSocket data"""
        if 'e' in data:  # Event type
            event_type = data['e']
            if event_type == '24hrTicker':
                return {
                    'type': 'ticker',
                    'symbol': data['s'],
                    'timestamp': datetime.fromtimestamp(data['E'] / 1000),
                    'price': float(data['c']),
                    'bid_price': float(data['b']),
                    'ask_price': float(data['a']),
                    'volume': float(data['v'])
                }
            elif event_type == 'depthUpdate':
                return {
                    'type': 'orderbook',
                    'symbol': data['s'],
                    'timestamp': datetime.fromtimestamp(data['E'] / 1000),
                    'bids': [[float(price), float(qty)] for price, qty in data['b']],
                    'asks': [[float(price), float(qty)] for price, qty in data['a']]
                }
        return None

    def _normalize_okx_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize OKX WebSocket data"""
        if 'event' in data and data['event'] == 'subscribe':
            return None  # Skip subscription confirmations

        if 'arg' in data:
            channel = data['arg']['channel']
            if channel == 'tickers' and 'data' in data:
                ticker = data['data'][0]
                return {
                    'type': 'ticker',
                    'symbol': ticker['instId'],
                    'timestamp': datetime.fromtimestamp(int(ticker['ts']) / 1000),
                    'price': float(ticker['last']),
                    'bid_price': float(ticker['bidPx']),
                    'ask_price': float(ticker['askPx']),
                    'volume': float(ticker['vol24h'])
                }
            elif channel == 'books5' and 'data' in data:
                book = data['data'][0]
                return {
                    'type': 'orderbook',
                    'symbol': book['instId'],
                    'timestamp': datetime.now(),
                    'bids': [[float(level[0]), float(level[1])] for level in book['bids']],
                    'asks': [[float(level[0]), float(level[1])] for level in book['asks']]
                }
        return None

    def _normalize_bybit_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize Bybit WebSocket data"""
        if 'topic' in data:
            topic = data['topic']
            if topic.startswith('tickers.') and 'data' in data:
                ticker = data['data']
                return {
                    'type': 'ticker',
                    'symbol': ticker['symbol'],
                    'timestamp': datetime.fromtimestamp(int(ticker['timestampE6']) / 1000000),
                    'price': float(ticker['lastPrice']),
                    'bid_price': float(ticker['bid1Price']),
                    'ask_price': float(ticker['ask1Price']),
                    'volume': float(ticker['volume24h'])
                }
            elif topic.startswith('orderbook.') and 'data' in data:
                book = data['data']
                return {
                    'type': 'orderbook',
                    'symbol': book['s'],
                    'timestamp': datetime.now(),
                    'bids': [[float(level[0]), float(level[1])] for level in book['b']],
                    'asks': [[float(level[0]), float(level[1])] for level in book['a']]
                }
        return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all WebSocket connections"""
        metrics = {}

        for config in self.configs:
            exchange_name = config.exchange_name
            metrics[exchange_name] = {
                'messages_received': self.message_counts[exchange_name],
                'errors': self.error_counts[exchange_name],
                'reconnections': self.reconnect_counts[exchange_name],
                'is_connected': exchange_name in self.connections and
                              not self.connections[exchange_name].closed
            }

        return metrics

    def get_active_connections(self) -> List[str]:
        """Get list of exchanges with active WebSocket connections"""
        return [
            exchange_name for exchange_name, connection in self.connections.items()
            if connection and not connection.closed
        ]
