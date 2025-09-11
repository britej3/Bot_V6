# Tick Data API Documentation

## Overview

The Tick Data API provides real-time and historical tick data from cryptocurrency exchanges via CCXT integration. This API is designed for high-frequency trading applications and provides configurable, safe, and efficient access to market data.

## Base URL

```
/api/v1/tick-data
```

## Authentication

Currently, the API operates in read-only mode without authentication. Rate limiting is implemented to prevent abuse.

## Endpoints

### Health Check

**GET** `/health`

Check the health status of the tick data service.

**Response:**
```json
{
  "status": "healthy",
  "service": "tick_data_api",
  "timestamp": 1693526400.123,
  "stats": {
    "is_running": true,
    "cache_size": 1500,
    "exchanges": {
      "binance": {
        "is_connected": true,
        "error_count": 0,
        "last_used": 1693526390.456,
        "request_count": 45
      }
    }
  }
}
```

### Supported Symbols

**GET** `/symbols`

Get list of supported trading symbols.

**Parameters:**
- `exchange` (optional): Filter by specific exchange

**Response:**
```json
[
  "BTC/USDT",
  "ETH/USDT",
  "BNB/USDT",
  "ADA/USDT",
  "SOL/USDT"
]
```

### Supported Exchanges

**GET** `/exchanges`

Get list of supported exchanges.

**Response:**
```json
[
  "binance",
  "okx",
  "bybit",
  "coinbase",
  "kraken"
]
```

### Get Tick Data

**GET** `/{symbol:path}`

Get real-time tick data for a specific symbol.

**Parameters:**
- `symbol` (path): Trading pair symbol (e.g., BTC/USDT)
- `limit` (query): Number of ticks to retrieve (1-1000, default: 100)
- `exchange` (query): Preferred exchange (optional)

**Example Request:**
```
GET /api/v1/tick-data/BTC/USDT?limit=50&exchange=binance
```

**Response:**
```json
{
  "symbol": "BTC/USDT",
  "limit": 50,
  "data": [
    {
      "timestamp": 1693526400.123,
      "symbol": "BTC/USDT",
      "price": 50000.25,
      "volume": 1.5,
      "side": "buy",
      "exchange_timestamp": 1693526400.100,
      "source_exchange": "binance"
    }
  ],
  "message": "Data from binance exchange",
  "total_count": 1,
  "request_timestamp": 1693526400.123
}
```

### Service Statistics

**GET** `/stats`

Get tick data service statistics.

**Response:**
```json
{
  "total_requests": 100,
  "successful_requests": 95,
  "failed_requests": 5,
  "average_response_time": 0.123,
  "last_updated": 1693526400.123
}
```

### Service Configuration

**GET** `/config`

Get current service configuration.

**Response:**
```json
{
  "max_limit": 1000,
  "default_limit": 100,
  "cache_ttl": 30,
  "rate_limit_per_minute": 60,
  "supported_exchanges": [
    "binance",
    "okx",
    "bybit",
    "coinbase",
    "kraken"
  ]
}
```

### Clear Cache

**DELETE** `/cache/{symbol:path}`

Clear cache for specific symbol (admin operation).

**Parameters:**
- `symbol` (path): Trading pair symbol to clear cache for

**Response:**
```json
{
  "message": "Cache cleared for symbol: BTC/USDT",
  "timestamp": 1693526400.123
}
```

## Data Models

### TickDataPoint

```typescript
interface TickDataPoint {
  timestamp: number;           // Unix timestamp in seconds
  symbol: string;              // Trading pair symbol
  price: number;               // Last trade price (> 0)
  volume: number;              // Trade volume (>= 0)
  side?: string;               // Trade side (buy/sell)
  exchange_timestamp?: number; // Exchange timestamp
  source_exchange: string;     // Exchange name
}
```

### TickDataResponse

```typescript
interface TickDataResponse {
  symbol: string;              // Trading pair symbol
  limit: number;               // Number of ticks requested
  data: TickDataPoint[];       // Array of tick data points
  message: string;             // Response message
  total_count: number;         // Total number of ticks available
  request_timestamp: number;   // Request timestamp
}
```

### TickDataError

```typescript
interface TickDataError {
  error_code: string;          // Error code
  message: string;             // Error message
  details?: Record<string, any>; // Additional error details
  timestamp: number;           // Error timestamp
}
```

## Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_SYMBOL` | Invalid symbol format | 400 |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | 429 |
| `INTERNAL_ERROR` | Internal server error | 500 |

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Per Minute**: 60 requests per minute per IP
- **Per Hour**: 1000 requests per hour per IP
- **Burst**: 10 concurrent requests

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time until reset

## Caching

The API uses in-memory caching with configurable TTL:

- **Default TTL**: 30 seconds
- **Max Cache Size**: 10,000 entries per symbol
- **Cache Cleanup**: Every 60 seconds

## Supported Exchanges

- **Binance**: High-volume exchange with comprehensive API
- **OKX**: Fast execution with competitive fees
- **Bybit**: Popular for derivatives trading
- **Coinbase**: Regulated exchange with high liquidity
- **Kraken**: Security-focused exchange

## Safety Features

1. **Read-Only Mode**: API only provides data access, no trading capabilities
2. **Input Validation**: All inputs are validated and sanitized
3. **Rate Limiting**: Prevents API abuse and ensures fair usage
4. **Error Handling**: Comprehensive error handling with detailed messages
5. **Sandbox Mode**: Uses exchange sandbox environments when available

## Configuration

The service is configurable via environment variables with the prefix `TICK_DATA_`:

```bash
# Rate limiting
TICK_DATA_RATE_LIMIT_PER_MINUTE=60
TICK_DATA_RATE_LIMIT_PER_HOUR=1000

# Data limits
TICK_DATA_MAX_LIMIT=1000
TICK_DATA_DEFAULT_LIMIT=100

# Caching
TICK_DATA_CACHE_TTL=30
TICK_DATA_CACHE_MAX_SIZE=10000

# Supported exchanges (comma-separated)
TICK_DATA_SUPPORTED_EXCHANGES=binance,okx,bybit,coinbase,kraken
```

## Performance Considerations

- **Response Time**: Typically < 100ms for cached data
- **Throughput**: Up to 1000 requests per minute per IP
- **Memory Usage**: ~1MB per 1000 cached ticks
- **Network**: Minimal bandwidth usage for tick data

## Best Practices

1. **Caching**: Implement client-side caching to reduce API calls
2. **Rate Limits**: Respect rate limits to avoid throttling
3. **Error Handling**: Implement proper error handling and retry logic
4. **Symbol Validation**: Validate symbols before making requests
5. **Connection Pooling**: Reuse connections for better performance

## Examples

### Python

```python
import requests

# Get tick data
response = requests.get(
    "http://localhost:8000/api/v1/tick-data/BTC/USDT",
    params={"limit": 50, "exchange": "binance"}
)

if response.status_code == 200:
    data = response.json()
    print(f"Symbol: {data['symbol']}")
    print(f"Data points: {len(data['data'])}")
    for tick in data['data']:
        print(f"Price: {tick['price']}, Volume: {tick['volume']}")
```

### JavaScript

```javascript
// Get tick data
fetch('/api/v1/tick-data/BTC/USDT?limit=50')
  .then(response => response.json())
  .then(data => {
    console.log(`Symbol: ${data.symbol}`);
    console.log(`Data points: ${data.data.length}`);
    data.data.forEach(tick => {
      console.log(`Price: ${tick.price}, Volume: ${tick.volume}`);
    });
  })
  .catch(error => console.error('Error:', error));
```

## Monitoring

The API provides comprehensive monitoring:

- **Health Check**: `/health` endpoint
- **Statistics**: `/stats` endpoint
- **Metrics**: Prometheus metrics (optional)
- **Logging**: Structured logging with configurable levels

## Troubleshooting

### Common Issues

1. **429 Rate Limit Exceeded**
   - Reduce request frequency
   - Implement exponential backoff
   - Check rate limit headers

2. **400 Invalid Symbol**
   - Verify symbol format (BASE/QUOTE)
   - Check supported symbols list
   - Ensure symbol exists on exchange

3. **500 Internal Error**
   - Check service health
   - Review logs for details
   - Contact support if persistent

### Debugging

Enable debug logging by setting:
```bash
TICK_DATA_LOG_LEVEL=DEBUG
```

Check service health:
```bash
curl http://localhost:8000/api/v1/tick-data/health
```

## Future Enhancements

- **WebSocket Streams**: Real-time tick data streaming
- **Historical Data**: Support for historical tick data
- **Advanced Filtering**: Filter by time range, volume, price
- **Multiple Symbols**: Batch requests for multiple symbols
- **Order Book Data**: Integration with order book snapshots