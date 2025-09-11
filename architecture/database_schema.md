# Database Schema Design

## 1. Time-Series Database (TimescaleDB)

### 1.1 Market Data Tables

#### Cryptocurrency Price Data
```sql
-- Cryptocurrency price data (candles)
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    open DECIMAL NOT NULL,
    high DECIMAL NOT NULL,
    low DECIMAL NOT NULL,
    close DECIMAL NOT NULL,
    volume DECIMAL NOT NULL,
    trades INTEGER
);

SELECT create_hypertable('market_data', 'time');
CREATE INDEX ON market_data (symbol, time DESC);
CREATE INDEX ON market_data (exchange, time DESC);
```

#### Order Book Data
```sql
-- Order book snapshots
CREATE TABLE order_book (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    bids JSONB,  -- [{price: decimal, amount: decimal}, ...]
    asks JSONB   -- [{price: decimal, amount: decimal}, ...]
);

SELECT create_hypertable('order_book', 'time');
CREATE INDEX ON order_book (symbol, time DESC);
```

#### Trade Data
```sql
-- Individual trades from exchanges
CREATE TABLE trades (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    trade_id VARCHAR(100),
    side VARCHAR(4) NOT NULL,  -- buy or sell
    price DECIMAL NOT NULL,
    quantity DECIMAL NOT NULL,
    FOREIGN KEY (symbol, exchange) REFERENCES symbols(symbol, exchange)
);

SELECT create_hypertable('trades', 'time');
CREATE INDEX ON trades (symbol, time DESC);
CREATE INDEX ON trades (exchange, time DESC);
```

### 1.2 Trading History Tables

#### Executed Orders
```sql
-- Executed orders
CREATE TABLE executed_orders (
    time TIMESTAMPTZ NOT NULL,
    order_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(4) NOT NULL,
    type VARCHAR(20) NOT NULL,
    price DECIMAL NOT NULL,
    quantity DECIMAL NOT NULL,
    filled_quantity DECIMAL NOT NULL,
    fees DECIMAL NOT NULL,
    status VARCHAR(20) NOT NULL,
    strategy_id INTEGER REFERENCES strategies(id)
);

SELECT create_hypertable('executed_orders', 'time');
CREATE INDEX ON executed_orders (order_id);
CREATE INDEX ON executed_orders (symbol, time DESC);
```

#### Positions
```sql
-- Open and closed positions
CREATE TABLE positions (
    id SERIAL PRIMARY KEY,
    position_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity DECIMAL NOT NULL,
    entry_price DECIMAL NOT NULL,
    exit_price DECIMAL,
    fees DECIMAL NOT NULL,
    pnl DECIMAL,
    pnl_percentage DECIMAL,
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    strategy_id INTEGER REFERENCES strategies(id)
);

CREATE INDEX ON positions (position_id);
CREATE INDEX ON positions (symbol, opened_at DESC);
CREATE INDEX ON positions (strategy_id);
```

#### Performance Metrics
```sql
-- Strategy performance metrics
CREATE TABLE performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    strategy_id INTEGER REFERENCES strategies(id),
    metric_name VARCHAR(50) NOT NULL,
    value DECIMAL NOT NULL
);

SELECT create_hypertable('performance_metrics', 'time');
CREATE INDEX ON performance_metrics (strategy_id, time DESC);
CREATE INDEX ON performance_metrics (metric_name, time DESC);
```

## 2. Relational Database (PostgreSQL)

### 2.1 User Management Tables

#### Users
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

CREATE INDEX ON users (username);
CREATE INDEX ON users (email);
```

#### User Sessions
```sql
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_accessed TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON user_sessions (session_token);
CREATE INDEX ON user_sessions (user_id);
CREATE INDEX ON user_sessions (expires_at);
```

#### API Keys
```sql
CREATE TABLE api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions JSONB,  -- { "read": true, "trade": true, "manage": false }
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX ON api_keys (key_hash);
CREATE INDEX ON api_keys (user_id);
```

### 2.2 Strategy Management Tables

#### Symbols
```sql
CREATE TABLE symbols (
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(50) NOT NULL,
    base_currency VARCHAR(10) NOT NULL,
    quote_currency VARCHAR(10) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',  -- active, inactive, deprecated
    min_order_size DECIMAL,
    max_order_size DECIMAL,
    price_precision INTEGER,
    quantity_precision INTEGER,
    PRIMARY KEY (symbol, exchange)
);

CREATE INDEX ON symbols (exchange);
CREATE INDEX ON symbols (status);
```

#### Strategies
```sql
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    config JSONB,  -- Strategy-specific configuration
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON strategies (user_id);
CREATE INDEX ON strategies (is_active);
```

#### Strategy Versions
```sql
CREATE TABLE strategy_versions (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    version_number INTEGER NOT NULL,
    config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by INTEGER REFERENCES users(id)
);

CREATE INDEX ON strategy_versions (strategy_id);
CREATE UNIQUE INDEX ON strategy_versions (strategy_id, version_number);
```

### 2.3 Risk Management Tables

#### Risk Limits
```sql
CREATE TABLE risk_limits (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    max_position_value DECIMAL,
    max_daily_loss DECIMAL,
    max_drawdown DECIMAL,
    max_orders_per_hour INTEGER,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON risk_limits (user_id);
```

#### Risk Events
```sql
CREATE TABLE risk_events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL,  -- limit_breach, compliance_violation, etc.
    severity VARCHAR(20) NOT NULL,    -- info, warning, critical
    description TEXT,
    details JSONB,
    triggered_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolved_by INTEGER REFERENCES users(id)
);

CREATE INDEX ON risk_events (user_id, triggered_at DESC);
CREATE INDEX ON risk_events (event_type);
CREATE INDEX ON risk_events (severity);
```

### 2.4 System Configuration Tables

#### Exchange Connections
```sql
CREATE TABLE exchange_connections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    exchange_name VARCHAR(50) NOT NULL,
    api_key_encrypted BYTEA NOT NULL,
    api_secret_encrypted BYTEA NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON exchange_connections (user_id);
CREATE INDEX ON exchange_connections (exchange_name);
```

#### System Settings
```sql
CREATE TABLE system_settings (
    id SERIAL PRIMARY KEY,
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON system_settings (setting_key);
```

## 3. Cache Database (Redis)

### 3.1 Session Storage
```
# User sessions
session:{session_token} -> {
  user_id: integer,
  expires_at: timestamp,
  last_accessed: timestamp
}

# JWT refresh tokens
refresh_token:{token} -> {
  user_id: integer,
  expires_at: timestamp
}
```

### 3.2 Real-Time Market Data
```
# Latest price for symbol
price:{exchange}:{symbol} -> {
  price: decimal,
  timestamp: timestamp,
  volume: decimal
}

# Order book for symbol
orderbook:{exchange}:{symbol} -> {
  bids: [[price, amount], ...],
  asks: [[price, amount], ...],
  timestamp: timestamp
}
```

### 3.3 Configuration Caching
```
# Active strategies
strategies:active -> [strategy_id, ...]

# Strategy configuration
strategy:{strategy_id} -> {
  config: json,
  is_active: boolean
}

# Risk limits
risk_limits:{user_id} -> {
  max_position_value: decimal,
  max_daily_loss: decimal,
  max_drawdown: decimal
}
```

### 3.4 Metrics Aggregation
```
# Real-time metrics
metrics:{component}:{metric_name} -> {
  value: decimal,
  timestamp: timestamp
}

# Rate limiting
rate_limit:{user_id}:{endpoint} -> {
  count: integer,
  reset_time: timestamp
}
```

## 4. Data Retention Policies

### 4.1 Time-Series Data
- Market data (1m candles): 2 years
- Market data (5m+ candles): 5 years
- Order book snapshots: 30 days
- Individual trades: 1 year

### 4.2 Trading History
- Executed orders: 7 years (compliance requirement)
- Positions: 7 years (compliance requirement)
- Performance metrics: 5 years

### 4.3 System Data
- User sessions: 30 days
- API keys: Until revoked
- Risk events: 7 years (compliance requirement)
- System logs: 1 year

## 5. Indexing Strategy

### 5.1 Time-Series Tables
- Primary index on time column (hypertable)
- Secondary indexes on symbol and exchange for filtering
- Composite indexes for common query patterns

### 5.2 Relational Tables
- Primary keys on ID columns
- Foreign key constraints for data integrity
- Indexes on frequently queried columns
- Composite indexes for complex queries

## 6. Partitioning Strategy

### 6.1 Time-Series Data
- Automatic partitioning by time using TimescaleDB hypertables
- Monthly partitions for most tables
- Daily partitions for high-frequency data (order book)

### 6.2 Relational Data
- No partitioning for small tables
- Consider partitioning for large tables based on access patterns