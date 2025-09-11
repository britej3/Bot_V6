"""
Exchange Connector Module
========================

Real exchange integration for live trading with multiple exchange support.
Provides unified interface for order execution, market data, and account management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import aiohttp
import json
import hashlib
import hmac
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str = "binance"
    base_url: str = "https://api.binance.com"
    api_key: str = ""
    secret_key: str = ""
    testnet: bool = True
    max_retries: int = 3
    timeout: float = 1.0  # 1 second timeout for ultra-low latency


class ExchangeConnector:
    """Real exchange connector for live trading"""

    def __init__(self, exchange_config: ExchangeConfig):
        self.config = exchange_config
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_connected = False
        self.last_request_time = 0
        self.request_count = 0

    async def connect(self) -> bool:
        """Establish connection to exchange"""
        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Test connection
            if await self._test_connection():
                self.is_connected = True
                logger.info(f"✅ Connected to {self.config.name}: {self.config.base_url}")
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"❌ Failed to connect to {self.config.name}: {e}")
            return False

    async def disconnect(self):
        """Disconnect from exchange"""
        if self.session:
            await self.session.close()
        self.is_connected = False
        logger.info(f"🔌 Disconnected from {self.config.name}")

    async def _test_connection(self) -> bool:
        """Test exchange connection"""
        if not self.config.api_key:
            # For demo/simulation mode
            await asyncio.sleep(0.001)
            return True

        try:
            # Real connection test
            timestamp = int(time.time() * 1000)
            params = {'timestamp': timestamp}
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            headers = {'X-MBX-APIKEY': self.config.api_key}

            async with self.session.get(
                f"{self.config.base_url}/api/v3/time",
                headers=headers,
                params={**params, 'signature': signature}
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Submit order to exchange"""
        if not self.is_connected:
            return {'status': 'error', 'message': 'Not connected to exchange'}

        try:
            # Rate limiting
            await self._rate_limit()

            # Prepare order parameters
            order_params = self._prepare_order_params(order)
            timestamp = int(time.time() * 1000)

            # Create signature for authentication
            query_string = '&'.join([f"{k}={v}" for k, v in order_params.items()])
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            headers = {'X-MBX-APIKEY': self.config.api_key}
            params = {**order_params, 'signature': signature}

            # Submit order to exchange
            async with self.session.post(
                f"{self.config.base_url}/api/v3/order",
                headers=headers,
                params=params
            ) as response:
                response_data = await response.json()

                if response.status == 200:
                    logger.info(f"📤 Order submitted: {order.get('symbol')} {order.get('side')} {order.get('quantity')}")
                    return {
                        'status': 'success',
                        'order_id': response_data.get('orderId'),
                        'executed_quantity': float(response_data.get('executedQty', 0)),
                        'executed_price': float(response_data.get('price', 0)),
                        'status_exchange': response_data.get('status', 'UNKNOWN').lower(),
                        'fees': float(response_data.get('commission', 0)),
                        'timestamp': datetime.utcnow()
                    }
                else:
                    error_msg = response_data.get('msg', 'Unknown error')
                    logger.error(f"Order rejected: {error_msg}")
                    return {
                        'status': 'rejected',
                        'message': error_msg,
                        'timestamp': datetime.utcnow()
                    }

        except Exception as e:
            logger.error(f"Order submission error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow()
            }

    def _prepare_order_params(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare order parameters for exchange API"""
        params = {
            'symbol': order['symbol'].upper(),
            'side': order['side'].upper(),
            'type': order['type'].upper(),
            'quantity': str(order['quantity']),
            'timestamp': int(time.time() * 1000)
        }

        if 'price' in order:
            params['price'] = str(order['price'])
        if 'stop_price' in order:
            params['stopPrice'] = str(order['stop_price'])

        # Add time in force for limit orders
        if order['type'] in ['LIMIT', 'STOP_LIMIT']:
            params['timeInForce'] = 'GTC'

        return params

    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get order status from exchange"""
        if not self.is_connected:
            return None

        try:
            await self._rate_limit()

            timestamp = int(time.time() * 1000)
            params = {
                'symbol': symbol.upper(),
                'orderId': order_id,
                'timestamp': timestamp
            }

            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            headers = {'X-MBX-APIKEY': self.config.api_key}

            async with self.session.get(
                f"{self.config.base_url}/api/v3/order",
                headers=headers,
                params={**params, 'signature': signature}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get order status: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on exchange"""
        if not self.is_connected:
            return False

        try:
            await self._rate_limit()

            timestamp = int(time.time() * 1000)
            params = {
                'symbol': symbol.upper(),
                'orderId': order_id,
                'timestamp': timestamp
            }

            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            headers = {'X-MBX-APIKEY': self.config.api_key}

            async with self.session.delete(
                f"{self.config.base_url}/api/v3/order",
                headers=headers,
                params={**params, 'signature': signature}
            ) as response:
                if response.status == 200:
                    logger.info(f"❌ Order cancelled: {order_id}")
                    return True
                else:
                    logger.error(f"Failed to cancel order: {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

    async def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance"""
        if not self.is_connected:
            return None

        try:
            await self._rate_limit()

            timestamp = int(time.time() * 1000)
            params = {'timestamp': timestamp}
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            headers = {'X-MBX-APIKEY': self.config.api_key}

            async with self.session.get(
                f"{self.config.base_url}/api/v3/account",
                headers=headers,
                params={**params, 'signature': signature}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get account balance: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return None

    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        if current_time - self.last_request_time < 0.1:  # Max 10 requests per second
            await asyncio.sleep(0.1)
        self.last_request_time = time.time()
        self.request_count += 1


# Factory function
def create_exchange_connector(config: ExchangeConfig) -> ExchangeConnector:
    """Create exchange connector instance"""
    return ExchangeConnector(config)