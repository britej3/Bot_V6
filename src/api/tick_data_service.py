"""
Tick Data Service with CCXT Integration

This module provides a configurable tick data service that integrates with
cryptocurrency exchanges via CCXT to fetch real-time tick data while
maintaining safety and performance.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import ccxt
import ccxt.pro as ccxtpro
from dataclasses import dataclass
from collections import defaultdict, deque

from src.api.models import (
    TickDataPoint,
    TickDataConfig,
    TickDataError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExchangeConnection:
    """Exchange connection wrapper"""
    exchange: Any
    is_connected: bool = False
    last_used: float = 0
    error_count: int = 0
    rate_limiter: Optional[Any] = None

class TickDataCache:
    """In-memory cache for tick data"""

    def __init__(self, max_size: int = 10000, ttl: int = 30):
        self.cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_size))
        self.timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_size))
        self.ttl = ttl

    def add_tick(self, symbol: str, tick: TickDataPoint):
        """Add tick to cache"""
        self.cache[symbol].append(tick)
        self.timestamps[symbol].append(time.time())

    def get_ticks(self, symbol: str, limit: int) -> List[TickDataPoint]:
        """Get recent ticks from cache"""
        if symbol not in self.cache:
            return []

        # Remove expired entries
        current_time = time.time()
        valid_ticks = []

        for tick, timestamp in zip(self.cache[symbol], self.timestamps[symbol]):
            if current_time - timestamp < self.ttl:
                valid_ticks.append(tick)

        # Update cache with only valid entries
        self.cache[symbol].clear()
        self.timestamps[symbol].clear()

        for tick in valid_ticks[-limit:]:
            self.cache[symbol].append(tick)
            self.timestamps[symbol].append(current_time)

        return list(valid_ticks[-limit:])

    def clear_symbol(self, symbol: str):
        """Clear cache for specific symbol"""
        if symbol in self.cache:
            del self.cache[symbol]
            del self.timestamps[symbol]

class TickDataService:
    """
    Configurable tick data service with CCXT integration

    Features:
    - Multi-exchange support with failover
    - Rate limiting and safety controls
    - In-memory caching with TTL
    - Data validation and normalization
    - Configurable parameters
    """

    def __init__(self, config: TickDataConfig):
        self.config = config
        self.cache = TickDataCache(ttl=config.cache_ttl)
        self.exchanges: Dict[str, ExchangeConnection] = {}
        self.is_running = False

        # Rate limiting
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        self.request_timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))

        # Initialize exchanges
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        for exchange_name in self.config.supported_exchanges:
            try:
                # Create both sync and async versions
                exchange_class = getattr(ccxt, exchange_name.lower(), None)
                if exchange_class:
                    # Configure exchange for production use
                    exchange_config = {
                        'enableRateLimit': True,
                        'rateLimit': 1000,  # 1 request per second
                        'timeout': 30000,
                    }

                    # Only use sandbox mode in development environment
                    if config.environment != "production":
                        exchange_config['sandbox'] = True

                    exchange = exchange_class(exchange_config)

                    self.exchanges[exchange_name.lower()] = ExchangeConnection(
                        exchange=exchange,
                        rate_limiter=self._create_rate_limiter(exchange_name)
                    )
                    logger.info(f"Initialized {exchange_name} exchange")
                else:
                    logger.warning(f"Exchange {exchange_name} not found in CCXT")

            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {e}")

    def _create_rate_limiter(self, exchange_name: str) -> Any:
        """Create rate limiter for exchange"""
        # Simple token bucket rate limiter
        return {
            'tokens': 60,  # 60 requests per minute
            'last_refill': time.time(),
            'refill_rate': 1  # 1 token per second
        }

    async def start(self):
        """Start the tick data service"""
        self.is_running = True
        logger.info("Tick data service started")

        # Start background tasks for data collection
        asyncio.create_task(self._background_data_collection())

    async def stop(self):
        """Stop the tick data service"""
        self.is_running = False
        logger.info("Tick data service stopped")

        # Close exchange connections
        for conn in self.exchanges.values():
            if conn.exchange and hasattr(conn.exchange, 'close'):
                await conn.exchange.close()

    async def _background_data_collection(self):
        """Background task for continuous data collection"""
        while self.is_running:
            try:
                await self._collect_all_exchanges_data()
                await asyncio.sleep(1)  # Collect every second
            except Exception as e:
                logger.error(f"Error in background data collection: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def _collect_all_exchanges_data(self):
        """Collect tick data from all exchanges"""
        tasks = []
        for exchange_name, connection in self.exchanges.items():
            if connection.is_connected:
                tasks.append(self._collect_exchange_data(exchange_name, connection))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _collect_exchange_data(self, exchange_name: str, connection: ExchangeConnection):
        """Collect tick data from specific exchange"""
        try:
            # Check rate limit
            if not self._check_rate_limit(exchange_name):
                return

            # Get tickers for configured symbols
            symbols = getattr(self.config, 'supported_symbols', ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT'])
            tickers = await connection.exchange.fetch_tickers(symbols)

            for symbol, ticker in tickers.items():
                if ticker and 'last' in ticker:
                    tick = TickDataPoint(
                        timestamp=time.time(),
                        symbol=symbol,
                        price=float(ticker['last']),
                        volume=float(ticker.get('baseVolume', 0)),
                        side=None,  # Not available in ticker
                        exchange_timestamp=ticker.get('timestamp', time.time()),
                        source_exchange=exchange_name
                    )

                    self.cache.add_tick(symbol, tick)
                    connection.last_used = time.time()

        except Exception as e:
            logger.error(f"Error collecting data from {exchange_name}: {e}")
            connection.error_count += 1
            connection.is_connected = False

    def _check_rate_limit(self, exchange_name: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()

        # Clean old requests (older than 60 seconds)
        cutoff_time = current_time - 60
        self.request_timestamps[exchange_name] = deque(
            [t for t in self.request_timestamps[exchange_name] if t > cutoff_time],
            maxlen=60
        )

        # Check if we're within limits
        if len(self.request_timestamps[exchange_name]) >= self.config.rate_limit_per_minute:
            return False

        # Add current request
        self.request_timestamps[exchange_name].append(current_time)
        return True

    async def get_tick_data(
        self,
        symbol: str,
        limit: int,
        exchange_preference: Optional[str] = None
    ) -> Tuple[List[TickDataPoint], str]:
        """
        Get tick data for a symbol

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Number of ticks to retrieve
            exchange_preference: Preferred exchange (optional)

        Returns:
            Tuple of (tick_data, message)
        """
        try:
            # Validate inputs
            if limit > self.config.max_limit:
                limit = self.config.max_limit
            elif limit < 1:
                limit = self.config.default_limit

            # Check cache first
            cached_data = self.cache.get_ticks(symbol, limit)
            if len(cached_data) >= limit:
                return cached_data[:limit], "Data from cache"

            # If not enough cached data, try to fetch from exchange
            if exchange_preference and exchange_preference.lower() in self.exchanges:
                exchange_name = exchange_preference.lower()
            else:
                # Find best available exchange
                exchange_name = self._get_best_exchange()

            if not exchange_name:
                return cached_data, "No exchange available, using cached data"

            # Fetch from exchange
            fresh_data = await self._fetch_from_exchange(exchange_name, symbol, limit)

            if fresh_data:
                # Add to cache
                for tick in fresh_data:
                    self.cache.add_tick(symbol, tick)

                # Combine with cached data
                all_data = fresh_data + cached_data
                all_data.sort(key=lambda x: x.timestamp, reverse=True)
                return all_data[:limit], f"Data from {exchange_name} exchange"

            return cached_data, "Using cached data (exchange fetch failed)"

        except Exception as e:
            logger.error(f"Error getting tick data for {symbol}: {e}")
            return [], f"Error: {str(e)}"

    def _get_best_exchange(self) -> Optional[str]:
        """Get the best available exchange based on health metrics"""
        available_exchanges = []

        for name, conn in self.exchanges.items():
            if conn.is_connected and conn.error_count < 5:
                # Score based on recent usage and error count
                score = (time.time() - conn.last_used) + (conn.error_count * 10)
                available_exchanges.append((name, score))

        if not available_exchanges:
            return None

        # Return exchange with lowest score (most recently used, fewest errors)
        return min(available_exchanges, key=lambda x: x[1])[0]

    async def _fetch_from_exchange(
        self,
        exchange_name: str,
        symbol: str,
        limit: int
    ) -> List[TickDataPoint]:
        """Fetch tick data from specific exchange"""
        if exchange_name not in self.exchanges:
            return []

        connection = self.exchanges[exchange_name]
        if not connection.is_connected:
            return []

        try:
            # Use CCXT to fetch ticker data
            ticker = await connection.exchange.fetch_ticker(symbol)

            if ticker and 'last' in ticker:
                tick = TickDataPoint(
                    timestamp=time.time(),
                    symbol=symbol,
                    price=float(ticker['last']),
                    volume=float(ticker.get('baseVolume', 0)),
                    side=None,
                    exchange_timestamp=ticker.get('timestamp', time.time()),
                    source_exchange=exchange_name
                )

                return [tick]

        except Exception as e:
            logger.error(f"Error fetching from {exchange_name}: {e}")
            connection.error_count += 1

        return []

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        stats = {
            'is_running': self.is_running,
            'cache_size': sum(len(deque) for deque in self.cache.cache.values()),
            'exchanges': {}
        }

        for name, conn in self.exchanges.items():
            stats['exchanges'][name] = {
                'is_connected': conn.is_connected,
                'error_count': conn.error_count,
                'last_used': conn.last_used,
                'request_count': len(self.request_timestamps[name])
            }

        return stats

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache for symbol or all symbols"""
        if symbol:
            self.cache.clear_symbol(symbol)
        else:
            self.cache = TickDataCache(ttl=self.config.cache_ttl)