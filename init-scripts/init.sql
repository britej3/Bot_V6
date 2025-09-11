-- CryptoScalp AI Database Initialization
-- PostgreSQL initialization script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create database and user for the application
DO $$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'cryptoscalp') THEN

      CREATE ROLE cryptoscalp LOGIN PASSWORD 'devpassword';
   END IF;
END
$$;

-- Create database if it doesn't exist
SELECT 'CREATE DATABASE cryptoscalp_dev OWNER cryptoscalp'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'cryptoscalp_dev')\gexec

-- Connect to the application database
\c cryptoscalp_dev;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE cryptoscalp_dev TO cryptoscalp;
GRANT ALL PRIVILEGES ON SCHEMA public TO cryptoscalp;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading AUTHORIZATION cryptoscalp;
CREATE SCHEMA IF NOT EXISTS market_data AUTHORIZATION cryptoscalp;
CREATE SCHEMA IF NOT EXISTS models AUTHORIZATION cryptoscalp;
CREATE SCHEMA IF NOT EXISTS analytics AUTHORIZATION cryptoscalp;

-- Set default search path
ALTER ROLE cryptoscalp SET search_path TO trading, market_data, models, analytics, public;

-- Create basic tables
CREATE TABLE IF NOT EXISTS market_data.exchanges (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    api_url VARCHAR(255) NOT NULL,
    api_key_encrypted TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS market_data.symbols (
    id SERIAL PRIMARY KEY,
    exchange_id INTEGER REFERENCES market_data.exchanges(id),
    symbol VARCHAR(20) NOT NULL,
    base_asset VARCHAR(10) NOT NULL,
    quote_asset VARCHAR(10) NOT NULL,
    price_precision INTEGER DEFAULT 8,
    quantity_precision INTEGER DEFAULT 8,
    min_quantity DECIMAL(20,8) DEFAULT 0.00000001,
    max_quantity DECIMAL(20,8) DEFAULT 90000000.00000000,
    min_price DECIMAL(20,8) DEFAULT 0.00000001,
    max_price DECIMAL(20,8) DEFAULT 90000000.00000000,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(exchange_id, symbol)
);

-- Insert default exchanges
INSERT INTO market_data.exchanges (name, api_url) VALUES
    ('binance', 'https://api.binance.com'),
    ('okx', 'https://www.okx.com'),
    ('bybit', 'https://api.bybit.com')
ON CONFLICT (name) DO NOTHING;

-- Create hypertable for market data (TimescaleDB)
CREATE TABLE IF NOT EXISTS market_data.market_data_l1 (
    time TIMESTAMPTZ NOT NULL,
    symbol_id INTEGER NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    exchange_ts TIMESTAMPTZ,
    local_ts TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable if TimescaleDB is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable('market_data.market_data_l1', 'time', if_not_exists => TRUE);
    END IF;
END
$$;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_market_data_l1_symbol_time
ON market_data.market_data_l1 (symbol_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_market_data_l1_time
ON market_data.market_data_l1 (time DESC);

-- Create trading tables
CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol_id INTEGER NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    stop_loss DECIMAL(20,8),
    take_profit DECIMAL(20,8),
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    pnl DECIMAL(20,8) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    closed_at TIMESTAMPTZ
);

-- Create model performance table
CREATE TABLE IF NOT EXISTS models.model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    symbol_id INTEGER NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    start_date TIMESTAMPTZ NOT NULL,
    end_date TIMESTAMPTZ NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20,8) DEFAULT 0,
    max_drawdown DECIMAL(20,8) DEFAULT 0,
    sharpe_ratio DECIMAL(10,4),
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for trading tables
CREATE INDEX IF NOT EXISTS idx_positions_symbol_status
ON trading.positions (symbol_id, status);

CREATE INDEX IF NOT EXISTS idx_positions_created_at
ON trading.positions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_performance_model_date
ON models.model_performance (model_name, model_version, start_date);

-- Grant permissions on all tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO cryptoscalp;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO cryptoscalp;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA models TO cryptoscalp;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO cryptoscalp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA market_data TO cryptoscalp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO cryptoscalp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA models TO cryptoscalp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO cryptoscalp;

-- Create a view for active positions
CREATE OR REPLACE VIEW trading.active_positions AS
SELECT
    p.id,
    s.symbol,
    p.side,
    p.quantity,
    p.entry_price,
    p.current_price,
    p.stop_loss,
    p.take_profit,
    p.pnl,
    p.created_at
FROM trading.positions p
JOIN market_data.symbols s ON p.symbol_id = s.id
WHERE p.status = 'OPEN';

-- Create a view for model performance summary
CREATE OR REPLACE VIEW models.model_summary AS
SELECT
    model_name,
    model_version,
    COUNT(*) as total_evaluations,
    AVG(total_trades) as avg_trades,
    AVG(accuracy) as avg_accuracy,
    AVG(total_pnl) as avg_pnl,
    AVG(sharpe_ratio) as avg_sharpe,
    MAX(created_at) as last_updated
FROM models.model_performance
GROUP BY model_name, model_version
ORDER BY avg_pnl DESC;

-- Insert some sample symbols for testing
INSERT INTO market_data.symbols (exchange_id, symbol, base_asset, quote_asset)
SELECT
    e.id, 'BTC/USDT', 'BTC', 'USDT'
FROM market_data.exchanges e
WHERE e.name = 'binance'
ON CONFLICT (exchange_id, symbol) DO NOTHING;

INSERT INTO market_data.symbols (exchange_id, symbol, base_asset, quote_asset)
SELECT
    e.id, 'ETH/USDT', 'ETH', 'USDT'
FROM market_data.exchanges e
WHERE e.name = 'binance'
ON CONFLICT (exchange_id, symbol) DO NOTHING;