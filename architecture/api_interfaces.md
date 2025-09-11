# API Interfaces Specification

## 1. RESTful API Design

### 1.1 Versioning
All APIs are versioned with `/v1/` prefix to allow for future changes while maintaining backward compatibility.

### 1.2 Authentication
All API endpoints (except authentication endpoints) require authentication via JWT tokens.

### 1.3 Rate Limiting
Rate limiting is implemented per user and per IP address to prevent abuse.

### 1.4 Error Handling
All API responses follow a consistent error format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": {
      "field": "field_name",
      "reason": "reason_for_error"
    }
  }
}
```

## 2. Authentication API

### 2.1 Login
```
POST /api/v1/auth/login
```

**Request:**
```json
{
  "username": "string",
  "password": "string",
  "mfa_token": "string"  // Optional, required if MFA enabled
}
```

**Response:**
```json
{
  "access_token": "jwt_token",
  "refresh_token": "refresh_token",
  "expires_in": 3600,
  "user": {
    "id": "integer",
    "username": "string",
    "email": "string"
  }
}
```

### 2.2 Logout
```
POST /api/v1/auth/logout
```

**Request:**
```json
{
  "refresh_token": "refresh_token"
}
```

**Response:**
```json
{
  "message": "Successfully logged out"
}
```

### 2.3 Refresh Token
```
POST /api/v1/auth/refresh
```

**Request:**
```json
{
  "refresh_token": "refresh_token"
}
```

**Response:**
```json
{
  "access_token": "new_jwt_token",
  "expires_in": 3600
}
```

## 3. Trading API

### 3.1 Get Positions
```
GET /api/v1/trading/positions
```

**Query Parameters:**
- `status` (optional): Filter by position status (open, closed, all)
- `symbol` (optional): Filter by symbol
- `limit` (optional): Number of results (default: 50, max: 100)

**Response:**
```json
{
  "positions": [
    {
      "id": "string",
      "symbol": "string",
      "exchange": "string",
      "side": "buy|sell",
      "quantity": "number",
      "entry_price": "number",
      "current_price": "number",
      "pnl": "number",
      "pnl_percentage": "number",
      "opened_at": "timestamp",
      "updated_at": "timestamp"
    }
  ]
}
```

### 3.2 Place Order
```
POST /api/v1/trading/orders
```

**Request:**
```json
{
  "symbol": "string",
  "exchange": "string",
  "side": "buy|sell",
  "type": "market|limit|stop_loss|take_profit",
  "quantity": "number",
  "price": "number",  // Required for limit orders
  "stop_price": "number",  // Required for stop orders
  "time_in_force": "gtc|ioc|fok",  // Good Till Cancelled, Immediate or Cancel, Fill or Kill
  "strategy_id": "string"  // Optional, link to strategy
}
```

**Response:**
```json
{
  "order_id": "string",
  "status": "placed|rejected",
  "placed_at": "timestamp"
}
```

### 3.3 Get Order
```
GET /api/v1/trading/orders/{order_id}
```

**Response:**
```json
{
  "order_id": "string",
  "symbol": "string",
  "exchange": "string",
  "side": "buy|sell",
  "type": "market|limit|stop_loss|take_profit",
  "quantity": "number",
  "price": "number",
  "stop_price": "number",
  "status": "placed|filled|partially_filled|cancelled|rejected",
  "filled_quantity": "number",
  "avg_fill_price": "number",
  "fees": "number",
  "placed_at": "timestamp",
  "updated_at": "timestamp"
}
```

### 3.4 Cancel Order
```
DELETE /api/v1/trading/orders/{order_id}
```

**Response:**
```json
{
  "order_id": "string",
  "status": "cancelled|cancel_failed",
  "cancelled_at": "timestamp"
}
```

## 4. Strategies API

### 4.1 List Strategies
```
GET /api/v1/strategies
```

**Query Parameters:**
- `active` (optional): Filter by active status (true, false)
- `limit` (optional): Number of results (default: 50, max: 100)

**Response:**
```json
{
  "strategies": [
    {
      "id": "string",
      "name": "string",
      "description": "string",
      "is_active": "boolean",
      "created_at": "timestamp",
      "updated_at": "timestamp"
    }
  ]
}
```

### 4.2 Create Strategy
```
POST /api/v1/strategies
```

**Request:**
```json
{
  "name": "string",
  "description": "string",
  "config": {
    // Strategy-specific configuration
  }
}
```

**Response:**
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "config": {},
  "is_active": "boolean",
  "created_at": "timestamp"
}
```

### 4.3 Update Strategy
```
PUT /api/v1/strategies/{id}
```

**Request:**
```json
{
  "name": "string",
  "description": "string",
  "config": {},
  "is_active": "boolean"
}
```

**Response:**
```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "config": {},
  "is_active": "boolean",
  "updated_at": "timestamp"
}
```

## 5. Market Data API

### 5.1 List Symbols
```
GET /api/v1/market-data/symbols
```

**Query Parameters:**
- `exchange` (optional): Filter by exchange
- `limit` (optional): Number of results (default: 100, max: 1000)

**Response:**
```json
{
  "symbols": [
    {
      "symbol": "string",
      "exchange": "string",
      "base_currency": "string",
      "quote_currency": "string",
      "status": "active|inactive"
    }
  ]
}
```

### 5.2 Get OHLCV Data
```
GET /api/v1/market-data/symbols/{symbol}/ohlcv
```

**Query Parameters:**
- `exchange` (required): Exchange name
- `interval` (required): Time interval (1m, 5m, 15m, 1h, 4h, 1d, 1w)
- `start_time` (optional): Start time (Unix timestamp)
- `end_time` (optional): End time (Unix timestamp)
- `limit` (optional): Number of candles (default: 100, max: 1000)

**Response:**
```json
{
  "symbol": "string",
  "exchange": "string",
  "interval": "string",
  "candles": [
    {
      "timestamp": "timestamp",
      "open": "number",
      "high": "number",
      "low": "number",
      "close": "number",
      "volume": "number"
    }
  ]
}
```

## 6. Risk API

### 6.1 Get Risk Limits
```
GET /api/v1/risk/limits
```

**Response:**
```json
{
  "max_position_value": "number",
  "max_daily_loss": "number",
  "max_drawdown": "number",
  "updated_at": "timestamp"
}
```

### 6.2 Update Risk Limits
```
PUT /api/v1/risk/limits
```

**Request:**
```json
{
  "max_position_value": "number",
  "max_daily_loss": "number",
  "max_drawdown": "number"
}
```

**Response:**
```json
{
  "max_position_value": "number",
  "max_daily_loss": "number",
  "max_drawdown": "number",
  "updated_at": "timestamp"
}
```

## 7. Monitoring API

### 7.1 Get System Metrics
```
GET /api/v1/monitoring/metrics
```

**Query Parameters:**
- `component` (optional): Filter by component
- `metric` (optional): Filter by metric name
- `start_time` (optional): Start time (Unix timestamp)
- `end_time` (optional): End time (Unix timestamp)

**Response:**
```json
{
  "metrics": [
    {
      "component": "string",
      "metric": "string",
      "value": "number",
      "timestamp": "timestamp"
    }
  ]
}
```

### 7.2 Get Active Alerts
```
GET /api/v1/monitoring/alerts
```

**Query Parameters:**
- `status` (optional): Filter by status (active, resolved)
- `severity` (optional): Filter by severity (info, warning, critical)

**Response:**
```json
{
  "alerts": [
    {
      "id": "string",
      "component": "string",
      "severity": "info|warning|critical",
      "message": "string",
      "status": "active|resolved",
      "triggered_at": "timestamp",
      "resolved_at": "timestamp"  // Optional
    }
  ]
}
```

## 8. WebSocket API

### 8.1 Connection
WebSocket connections are established at:
```
wss://api.example.com/ws/v1
```

### 8.2 Authentication
After connecting, clients must authenticate with:
```json
{
  "type": "auth",
  "token": "jwt_token"
}
```

### 8.3 Subscriptions
Clients can subscribe to various data streams:

#### Market Data
```json
{
  "type": "subscribe",
  "stream": "market_data",
  "symbols": ["BTC/USD", "ETH/USD"],
  "exchange": "binance"
}
```

#### Order Updates
```json
{
  "type": "subscribe",
  "stream": "order_updates"
}
```

#### Risk Alerts
```json
{
  "type": "subscribe",
  "stream": "risk_alerts"
}
```

### 8.4 Data Messages
Market data updates:
```json
{
  "type": "market_data",
  "symbol": "BTC/USD",
  "exchange": "binance",
  "price": 50000.0,
  "volume": 10.5,
  "timestamp": "timestamp"
}
```

Order updates:
```json
{
  "type": "order_update",
  "order_id": "string",
  "status": "filled",
  "filled_quantity": 1.0,
  "avg_fill_price": 50000.0,
  "timestamp": "timestamp"
}
```

Risk alerts:
```json
{
  "type": "risk_alert",
  "level": "warning",
  "message": "Portfolio drawdown exceeded threshold",
  "timestamp": "timestamp"
}
```