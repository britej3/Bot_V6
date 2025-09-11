"""
Binance Data Manager for Live Crypto Futures Data
Handles WebSocket connections, historical data download, and real-time data streaming
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from collections import deque
import json
import time

try:
    import ccxt
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError as e:
    logging.error(f"Missing required dependencies: {e}")
    raise

from .config import AdvancedTradingConfig

logger = logging.getLogger(__name__)


class BinanceDataManager:
    """Manages Binance data connections and streaming"""

    def __init__(self, config: AdvancedTradingConfig):
        self.config = config
        self.binance_config = config.get_binance_config()

        # Initialize CCXT exchange
        self.exchange = None
        self.websocket = None

        # Data buffers
        self.tick_buffer = deque(maxlen=config.tick_buffer_size)
        self.order_book_buffer = deque(maxlen=100)
        self.trade_buffer = deque(maxlen=1000)

        # Connection state
        self.is_connected = False
        self.is_streaming = False
        self.last_update = datetime.utcnow()

        # Data callbacks
        self.data_callbacks = []

        # WebSocket URLs
        self.ws_base_url = "wss://fstream.binance.com/ws" if config.binance_testnet else "wss://fstream.binance.com/ws"
        self.rest_base_url = "https://fapi.binance.com" if config.binance_testnet else "https://fapi.binance.com"

        logger.info("🧠 Binance Data Manager initialized")

    async def initialize(self) -> bool:
        """Initialize Binance connections"""
        try:
            logger.info("🚀 Initializing Binance Data Manager...")

            # Initialize CCXT exchange
            await self._initialize_exchange()

            # Test connection
            await self._test_connection()

            logger.info("✅ Binance Data Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize Binance Data Manager: {e}")
            return False

    async def _initialize_exchange(self):
        """Initialize CCXT Binance exchange"""
        try:
            exchange_config = {
                'apiKey': self.binance_config['api_key'],
                'secret': self.binance_config['secret_key'],
                'testnet': self.binance_config['testnet'],
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                    'recvWindow': self.binance_config['recv_window']
                }
            }

            self.exchange = ccxt.binance(exchange_config)

            # Load markets
            await self.exchange.load_markets()

            logger.info("✅ CCXT Binance exchange initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize CCXT exchange: {e}")
            raise

    async def _test_connection(self):
        """Test Binance API connection"""
        try:
            # Test API connectivity
            ticker = await self.exchange.fetch_ticker(self.config.symbol)
            if ticker:
                logger.info(f"✅ Binance API connection successful. Current price: ${ticker['last']}")
            else:
                raise Exception("Failed to fetch ticker data")

        except Exception as e:
            logger.error(f"❌ Binance API connection test failed: {e}")
            raise

    async def start_live_data_streams(self):
        """Start live data streaming"""
        try:
            logger.info("📡 Starting live data streams...")

            # Start WebSocket connection
            await self._start_websocket_connection()

            self.is_streaming = True
            logger.info("✅ Live data streams started")

        except Exception as e:
            logger.error(f"❌ Failed to start live data streams: {e}")
            raise

    async def stop_streams(self):
        """Stop all data streams"""
        try:
            logger.info("🛑 Stopping data streams...")

            self.is_streaming = False

            if self.websocket:
                await self.websocket.close()

            logger.info("✅ Data streams stopped")

        except Exception as e:
            logger.error(f"❌ Failed to stop data streams: {e}")

    async def _start_websocket_connection(self):
        """Start WebSocket connection for real-time data"""
        try:
            # WebSocket streams to subscribe to
            streams = [
                f"{self.config.symbol.lower()}@ticker",  # 24hr ticker
                f"{self.config.symbol.lower()}@depth20",  # Order book depth
                f"{self.config.symbol.lower()}@trade",    # Recent trades
            ]

            stream_url = f"{self.ws_base_url}/stream?streams={'/'.join(streams)}"

            logger.info(f"🔌 Connecting to WebSocket: {stream_url}")

            # Connect to WebSocket
            self.websocket = await websockets.connect(stream_url)

            # Start message handler
            asyncio.create_task(self._handle_websocket_messages())

            logger.info("✅ WebSocket connection established")

        except Exception as e:
            logger.error(f"❌ Failed to start WebSocket connection: {e}")
            raise

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)

                    # Process different stream types
                    if 'stream' in data:
                        stream_type = data['stream'].split('@')[1] if '@' in data['stream'] else 'unknown'
                        await self._process_stream_data(stream_type, data['data'])

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")

        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            if self.is_streaming:
                # Attempt to reconnect
                asyncio.create_task(self._reconnect_websocket())
        except Exception as e:
            logger.error(f"WebSocket message handler error: {e}")

    async def _process_stream_data(self, stream_type: str, data: Dict[str, Any]):
        """Process data from different streams"""
        try:
            self.last_update = datetime.utcnow()

            if stream_type == 'ticker':
                await self._process_ticker_data(data)
            elif stream_type == 'depth20':
                await self._process_orderbook_data(data)
            elif stream_type == 'trade':
                await self._process_trade_data(data)

            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(stream_type, data)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing stream data: {e}")

    async def _process_ticker_data(self, data: Dict[str, Any]):
        """Process 24hr ticker data"""
        try:
            tick_data = {
                'timestamp': datetime.utcnow(),
                'price': float(data['c']),  # Close price
                'quantity': float(data['v']),  # Volume
                'is_buyer_maker': True,  # Placeholder
                'event_type': 'ticker',
                'raw_data': data
            }

            self.tick_buffer.append(tick_data)

        except Exception as e:
            logger.error(f"Error processing ticker data: {e}")

    async def _process_orderbook_data(self, data: Dict[str, Any]):
        """Process order book depth data"""
        try:
            orderbook_data = {
                'timestamp': datetime.utcnow(),
                'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                'asks': [[float(price), float(qty)] for price, qty in data['asks']],
                'event_type': 'orderbook',
                'raw_data': data
            }

            self.order_book_buffer.append(orderbook_data)

        except Exception as e:
            logger.error(f"Error processing orderbook data: {e}")

    async def _process_trade_data(self, data: Dict[str, Any]):
        """Process recent trade data"""
        try:
            trade_data = {
                'timestamp': datetime.utcnow(),
                'price': float(data['p']),
                'quantity': float(data['q']),
                'is_buyer_maker': data['m'],  # True if sell order, False if buy order
                'event_type': 'trade',
                'raw_data': data
            }

            self.trade_buffer.append(trade_data)
            self.tick_buffer.append(trade_data)

        except Exception as e:
            logger.error(f"Error processing trade data: {e}")

    async def _reconnect_websocket(self):
        """Reconnect WebSocket after connection loss"""
        try:
            logger.info("🔄 Attempting WebSocket reconnection...")

            if self.websocket:
                await self.websocket.close()

            # Wait before reconnecting
            await asyncio.sleep(5)

            if self.is_streaming:
                await self._start_websocket_connection()

        except Exception as e:
            logger.error(f"❌ WebSocket reconnection failed: {e}")

    def get_live_data(self) -> Optional[Dict[str, Any]]:
        """Get latest live data point"""
        try:
            if not self.tick_buffer:
                return None

            latest_tick = self.tick_buffer[-1]

            # Format for feature engine
            data_point = {
                'timestamp': latest_tick['timestamp'],
                'price': latest_tick['price'],
                'quantity': latest_tick['quantity'],
                'is_buyer_maker': latest_tick['is_buyer_maker']
            }

            return latest_tick['event_type'], data_point

        except Exception as e:
            logger.error(f"Error getting live data: {e}")
            return None

    async def download_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download historical data from Binance"""
        try:
            logger.info(f"📥 Downloading historical data from {start_date} to {end_date}")

            # Parse dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Download data in chunks to avoid rate limits
            all_data = []
            current_dt = start_dt

            while current_dt < end_dt:
                chunk_end = min(current_dt + timedelta(days=1), end_dt)

                try:
                    # Fetch OHLCV data
                    ohlcv = await self.exchange.fetch_ohlcv(
                        self.config.symbol,
                        timeframe='1m',
                        since=int(current_dt.timestamp() * 1000),
                        limit=1440  # Max per request
                    )

                    if ohlcv:
                        df_chunk = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                        all_data.append(df_chunk)

                    # Rate limiting
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.warning(f"Error downloading chunk {current_dt}: {e}")

                current_dt = chunk_end

            if all_data:
                df = pd.concat(all_data, ignore_index=True)

                # Add required columns for tick data format
                df['price'] = df['close']
                df['quantity'] = df['volume']
                df['is_buyer_maker'] = True  # Placeholder

                logger.info(f"✅ Downloaded {len(df)} historical data points")
                return df
            else:
                logger.warning("No historical data downloaded")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"❌ Failed to download historical data: {e}")
            return pd.DataFrame()

    def get_recent_ticks(self, n_ticks: int = 100) -> List[Dict[str, Any]]:
        """Get recent tick data"""
        return list(self.tick_buffer)[-n_ticks:]

    def get_recent_trades(self, n_trades: int = 100) -> List[Dict[str, Any]]:
        """Get recent trade data"""
        return list(self.trade_buffer)[-n_trades:]

    def get_order_book_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get latest order book snapshot"""
        return self.order_book_buffer[-1] if self.order_book_buffer else None

    def add_data_callback(self, callback: Callable):
        """Add callback for new data"""
        self.data_callbacks.append(callback)

    def remove_data_callback(self, callback: Callable):
        """Remove data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)

    async def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        try:
            if not self.exchange:
                raise Exception("Exchange not initialized")

            balance = await self.exchange.fetch_balance()
            return balance

        except Exception as e:
            logger.error(f"❌ Failed to get account balance: {e}")
            return {}

    async def get_position_info(self, symbol: str) -> Dict[str, Any]:
        """Get position information"""
        try:
            if not self.exchange:
                raise Exception("Exchange not initialized")

            positions = await self.exchange.fetch_positions([symbol])
            return positions[0] if positions else {}

        except Exception as e:
            logger.error(f"❌ Failed to get position info: {e}")
            return {}

    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information"""
        return {
            'is_connected': self.is_connected,
            'is_streaming': self.is_streaming,
            'last_update': self.last_update,
            'tick_buffer_size': len(self.tick_buffer),
            'order_book_buffer_size': len(self.order_book_buffer),
            'trade_buffer_size': len(self.trade_buffer),
            'websocket_url': self.ws_base_url,
            'symbol': self.config.symbol
        }

    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.stop_streams()

            if self.exchange:
                await self.exchange.close()

            logger.info("🧹 Binance Data Manager cleaned up")

        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")