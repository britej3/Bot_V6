"""
Multi-Source Data Loader for CryptoScalp AI

This module provides a unified interface for acquiring market data from multiple
cryptocurrency exchanges including Binance, OKX, and Bybit with automatic
failover and load balancing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import ccxt
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Data structure for market data"""
    exchange: str
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    last_price: float
    volume_24h: float

@dataclass
class OrderBookData:
    """Data structure for order book data"""
    exchange: str
    symbol: str
    timestamp: datetime
    bids: List[List[float]]  # [[price, volume], ...]
    asks: List[List[float]]  # [[price, volume], ...]

class ExchangeConfig:
    """Configuration for exchange connections"""

    def __init__(self, name: str, api_key: str = None, secret: str = None, testnet: bool = False):
        self.name = name
        self.api_key = api_key
        self.secret = secret
        self.testnet = testnet
        self.client = self._create_client()

    def _create_client(self):
        """Create exchange client with configuration"""
        config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        }

        if self.api_key and self.secret:
            config.update({
                'apiKey': self.api_key,
                'secret': self.secret
            })

        if self.testnet:
            config['testnet'] = True

        # Create the appropriate exchange client
        if self.name.lower() == 'binance':
            return ccxt.binance(config)
        elif self.name.lower() == 'okx':
            return ccxt.okx(config)
        elif self.name.lower() == 'bybit':
            return ccxt.bybit(config)
        else:
            raise ValueError(f"Unsupported exchange: {self.name}")

class MultiSourceDataLoader:
    """
    Multi-source data loader with failover and load balancing

    Features:
    - Concurrent data acquisition from multiple exchanges
    - Automatic failover on exchange failures
    - Load balancing across exchanges
    - Real-time data validation
    - Performance monitoring
    """

    def __init__(self, exchanges: List[ExchangeConfig], symbols: List[str]):
        self.exchanges = exchanges
        self.symbols = symbols
        self.active_exchanges = []
        self.data_buffer = asyncio.Queue(maxsize=10000)
        self.is_running = False

        # Performance tracking
        self.request_counts = {ex.name: 0 for ex in exchanges}
        self.error_counts = {ex.name: 0 for ex in exchanges}
        self.response_times = {ex.name: [] for ex in exchanges}

    async def start(self):
        """Start the data acquisition process"""
        self.is_running = True
        logger.info(f"Starting MultiSourceDataLoader with {len(self.exchanges)} exchanges")

        # Test all exchanges and identify active ones
        await self._test_exchanges()

        # Start data acquisition tasks
        tasks = []
        for exchange in self.active_exchanges:
            tasks.append(asyncio.create_task(self._acquire_data(exchange)))

        # Start data processing task
        tasks.append(asyncio.create_task(self._process_data()))

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Data acquisition error: {e}")
            await self.stop()

    async def stop(self):
        """Stop the data acquisition process"""
        self.is_running = False
        logger.info("Stopping MultiSourceDataLoader")

    async def _test_exchanges(self):
        """Test all exchanges and identify active ones"""
        logger.info("Testing exchange connectivity...")

        for exchange_config in self.exchanges:
            try:
                # Test basic connectivity
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    exchange_config.client.load_markets
                )
                self.active_exchanges.append(exchange_config)
                logger.info(f"✅ {exchange_config.name} connection successful")
            except Exception as e:
                logger.warning(f"❌ {exchange_config.name} connection failed: {e}")
                self.error_counts[exchange_config.name] += 1

        if not self.active_exchanges:
            raise RuntimeError("No exchanges available for data acquisition")

        logger.info(f"Active exchanges: {[ex.name for ex in self.active_exchanges]}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError))
    )
    async def _acquire_data(self, exchange: ExchangeConfig):
        """Acquire data from a specific exchange"""
        logger.info(f"Starting data acquisition from {exchange.name}")

        while self.is_running:
            try:
                start_time = datetime.now()

                # Acquire data from all symbols
                tasks = []
                for symbol in self.symbols:
                    tasks.append(self._get_symbol_data(exchange, symbol))

                # Wait for all symbol data to be acquired
                symbol_data = await asyncio.gather(*tasks, return_exceptions=True)

                # Process successful results
                successful_data = [data for data in symbol_data if not isinstance(data, Exception)]
                failed_data = [data for data in symbol_data if isinstance(data, Exception)]

                # Log failures
                for i, error in enumerate(failed_data):
                    logger.warning(f"Failed to get data for {self.symbols[i]} from {exchange.name}: {error}")
                    self.error_counts[exchange.name] += 1

                # Add successful data to buffer
                for data in successful_data:
                    if data:
                        await self.data_buffer.put(data)

                # Update metrics
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds() * 1000
                self.response_times[exchange.name].append(response_time)
                self.request_counts[exchange.name] += len(successful_data)

                # Keep only last 100 response times for memory efficiency
                if len(self.response_times[exchange.name]) > 100:
                    self.response_times[exchange.name] = self.response_times[exchange.name][-100:]

                # Wait before next acquisition cycle
                await asyncio.sleep(0.1)  # 10Hz data acquisition

            except Exception as e:
                logger.error(f"Data acquisition error for {exchange.name}: {e}")
                self.error_counts[exchange.name] += 1
                await asyncio.sleep(5)  # Wait before retry

    async def _get_symbol_data(self, exchange: ExchangeConfig, symbol: str) -> Optional[MarketData]:
        """Get market data for a specific symbol from an exchange"""
        try:
            # Get ticker data
            ticker = await asyncio.get_event_loop().run_in_executor(
                None,
                exchange.client.fetch_ticker,
                symbol
            )

            return MarketData(
                exchange=exchange.name,
                symbol=symbol,
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                bid_price=float(ticker['bid']) if ticker['bid'] else 0.0,
                ask_price=float(ticker['ask']) if ticker['ask'] else 0.0,
                bid_volume=float(ticker['bidVolume']) if ticker['bidVolume'] else 0.0,
                ask_volume=float(ticker['askVolume']) if ticker['askVolume'] else 0.0,
                last_price=float(ticker['last']) if ticker['last'] else 0.0,
                volume_24h=float(ticker['baseVolume']) if ticker['baseVolume'] else 0.0
            )

        except Exception as e:
            logger.debug(f"Error getting data for {symbol} from {exchange.name}: {e}")
            raise

    async def _process_data(self):
        """Process data from the buffer and apply validation"""
        logger.info("Starting data processing...")

        while self.is_running:
            try:
                # Get data from buffer
                data = await self.data_buffer.get()

                # Validate data
                if self._validate_data(data):
                    # Process validated data (store, forward, etc.)
                    await self._handle_validated_data(data)
                else:
                    logger.warning(f"Invalid data received: {data}")

                self.data_buffer.task_done()

            except Exception as e:
                logger.error(f"Data processing error: {e}")
                await asyncio.sleep(1)

    def _validate_data(self, data: MarketData) -> bool:
        """Validate market data for consistency and anomalies"""
        try:
            # Basic validation checks
            if not all([
                data.bid_price > 0,
                data.ask_price > 0,
                data.bid_price <= data.ask_price,
                data.last_price > 0,
                data.volume_24h >= 0
            ]):
                return False

            # Check for reasonable spread
            spread = (data.ask_price - data.bid_price) / data.bid_price
            if spread > 0.1:  # 10% spread is too high
                logger.warning(f"Unreasonable spread detected: {spread:.4f} for {data.symbol}")
                return False

            return True

        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False

    async def _handle_validated_data(self, data: MarketData):
        """Handle validated market data (store to database, forward to processing, etc.)"""
        # This would typically store to database or forward to other components
        # For now, just log the data
        logger.debug(f"Processed data: {data.exchange} {data.symbol} {data.last_price}")

    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, OrderBookData]:
        """Get order book data from all active exchanges"""
        order_books = {}

        for exchange in self.active_exchanges:
            try:
                orderbook = await asyncio.get_event_loop().run_in_executor(
                    None,
                    exchange.client.fetch_order_book,
                    symbol,
                    depth
                )

                order_books[exchange.name] = OrderBookData(
                    exchange=exchange.name,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bids=orderbook['bids'][:depth],
                    asks=orderbook['asks'][:depth]
                )

            except Exception as e:
                logger.warning(f"Failed to get order book for {symbol} from {exchange.name}: {e}")

        return order_books

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all exchanges"""
        metrics = {}

        for exchange_name in self.request_counts.keys():
            total_requests = self.request_counts[exchange_name]
            total_errors = self.error_counts[exchange_name]
            response_times = self.response_times[exchange_name]

            success_rate = ((total_requests - total_errors) / total_requests) * 100 if total_requests > 0 else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            metrics[exchange_name] = {
                'total_requests': total_requests,
                'total_errors': total_errors,
                'success_rate': success_rate,
                'avg_response_time_ms': avg_response_time,
                'is_active': exchange_name in [ex.name for ex in self.active_exchanges]
            }

        return metrics

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols across all exchanges"""
        all_symbols = set()

        for exchange in self.active_exchanges:
            try:
                symbols = list(exchange.client.markets.keys())
                all_symbols.update(symbols)
            except Exception as e:
                logger.warning(f"Failed to get symbols from {exchange.name}: {e}")

        return sorted(list(all_symbols))