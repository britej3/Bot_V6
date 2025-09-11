# Database Design - CryptoScalp AI

## Overview

The CryptoScalp AI database is designed to support high-frequency algorithmic trading with autonomous learning capabilities. The schema is optimized for sub-millisecond query performance, real-time analytics, and comprehensive audit trails required for trading systems.

## Database Architecture

### Database Type
**Type**: PostgreSQL with TimescaleDB extension
**Reason**: ACID compliance, advanced indexing, JSON support, and time-series optimization
**Alternatives Considered**: ClickHouse (for pure analytics), MongoDB (for document flexibility)

### Schema Versioning
**Strategy**: Migration-based with Alembic
**Tool**: Alembic (SQLAlchemy migration tool)
**Location**: migrations/ directory with versioned scripts

## Entity-Relationship Model

### Core Trading Entities

#### Market Data
**Description**: Real-time and historical market data from exchanges
**Attributes**:
- `timestamp` (Primary Key): TIMESTAMPTZ, market data timestamp
- `symbol`: VARCHAR(20), trading pair (e.g., BTCUSDT)
- `exchange`: VARCHAR(20), exchange identifier
- `price`: DECIMAL(20,10), last traded price
- `volume`: DECIMAL(20,10), trading volume
- `bid_price`: DECIMAL(20,10), best bid price
- `ask_price`: DECIMAL(20,10), best ask price
- `bid_volume`: DECIMAL(20,10), bid volume at best price
- `ask_volume`: DECIMAL(20,10), ask volume at best price

**Relationships**:
- Many-to-one with exchanges table
- One-to-many with order_book_l1 table

#### Order Book Data (L1)
**Description**: Level 1 order book snapshots for real-time trading
**Attributes**:
- `timestamp` (Primary Key): TIMESTAMPTZ
- `symbol`: VARCHAR(20)
- `exchange`: VARCHAR(20)
- `bid_price`: DECIMAL(20,10)
- `bid_volume`: DECIMAL(20,10)
- `ask_price`: DECIMAL(20,10)
- `ask_volume`: DECIMAL(20,10)

#### Trading Positions
**Description**: Current and historical trading positions
**Attributes**:
- `position_id` (Primary Key): BIGSERIAL
- `timestamp`: TIMESTAMPTZ
- `symbol`: VARCHAR(20)
- `exchange`: VARCHAR(20)
- `side`: VARCHAR(10), LONG/SHORT
- `quantity`: DECIMAL(20,10)
- `entry_price`: DECIMAL(20,10)
- `current_price`: DECIMAL(20,10)
- `unrealized_pnl`: DECIMAL(20,10)
- `stop_loss`: DECIMAL(20,10)
- `take_profit`: DECIMAL(20,10)
- `status`: VARCHAR(20), OPEN/CLOSED

#### Trading History
**Description**: Complete trading transaction history
**Attributes**:
- `trade_id` (Primary Key): BIGSERIAL
- `timestamp`: TIMESTAMPTZ
- `symbol`: VARCHAR(20)
- `exchange`: VARCHAR(20)
- `side`: VARCHAR(10)
- `quantity`: DECIMAL(20,10)
- `price`: DECIMAL(20,10)
- `commission`: DECIMAL(20,10)
- `pnl`: DECIMAL(20,10)

### AI/ML Entities

#### Model Versions
**Description**: Version control and tracking for ML models
**Attributes**:
- `model_id` (Primary Key): BIGSERIAL
- `model_name`: VARCHAR(100)
- `version`: VARCHAR(50)
- `created_at`: TIMESTAMPTZ
- `deployed_at`: TIMESTAMPTZ
- `status`: VARCHAR(20), TRAINING/DEPLOYED/RETIRED
- `performance_metrics`: JSONB
- `hyperparameters`: JSONB
- `model_path`: VARCHAR(500)

#### Model Performance
**Description**: Real-time model prediction tracking and validation
**Attributes**:
- `timestamp` (Primary Key): TIMESTAMPTZ
- `model_version`: VARCHAR(50)
- `symbol`: VARCHAR(20)
- `prediction`: DECIMAL(10,5)
- `actual`: DECIMAL(10,5)
- `confidence`: DECIMAL(10,5)
- `latency_ms`: INTEGER

### Autonomous Learning Entities

#### Experience Replay Memory
**Description**: Hierarchical memory system for reinforcement learning
**Attributes**:
- `memory_id` (Primary Key): BIGSERIAL
- `timestamp`: TIMESTAMPTZ
- `tier`: INTEGER, 1-6 (hierarchical levels)
- `state`: JSONB, market state representation
- `action`: JSONB, trading action taken
- `reward`: DECIMAL(10,5), reward received
- `next_state`: JSONB, resulting state
- `priority`: DECIMAL(10,5), experience priority

#### Adaptation History
**Description**: Track model adaptation and learning events
**Attributes**:
- `adaptation_id` (Primary Key): BIGSERIAL
- `timestamp`: TIMESTAMPTZ
- `model_version`: VARCHAR(50)
- `trigger_event`: VARCHAR(100), DRIFT/REGIME_CHANGE/PERFORMANCE
- `adaptation_type`: VARCHAR(50), RETRAINING/FINETUNING/PARAMETER_UPDATE
- `performance_before`: JSONB
- `performance_after`: JSONB
- `success`: BOOLEAN

## Data Models

### Market Data Tables
```sql
-- Market Data Table
CREATE TABLE market_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    price DECIMAL(20,10) NOT NULL,
    volume DECIMAL(20,10) NOT NULL,
    bid_price DECIMAL(20,10),
    ask_price DECIMAL(20,10),
    bid_volume DECIMAL(20,10),
    ask_volume DECIMAL(20,10),
    PRIMARY KEY (timestamp, symbol, exchange)
) PARTITION BY RANGE (timestamp);

-- Order Book L1 Table
CREATE TABLE order_book_l1 (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    bid_price DECIMAL(20,10) NOT NULL,
    bid_volume DECIMAL(20,10) NOT NULL,
    ask_price DECIMAL(20,10) NOT NULL,
    ask_volume DECIMAL(20,10) NOT NULL,
    PRIMARY KEY (timestamp, symbol, exchange)
) PARTITION BY RANGE (timestamp);
```

### Trading Tables
```sql
-- Positions Table
CREATE TABLE positions (
    position_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,10) NOT NULL,
    entry_price DECIMAL(20,10) NOT NULL,
    current_price DECIMAL(20,10) NOT NULL,
    unrealized_pnl DECIMAL(20,10) NOT NULL,
    stop_loss DECIMAL(20,10),
    take_profit DECIMAL(20,10),
    status VARCHAR(20) NOT NULL DEFAULT 'open'
);

-- Trades Table
CREATE TABLE trades (
    trade_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    exchange VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20,10) NOT NULL,
    price DECIMAL(20,10) NOT NULL,
    commission DECIMAL(20,10),
    pnl DECIMAL(20,10)
);
```

### AI/ML Tables
```sql
-- Model Registry
CREATE TABLE model_versions (
    model_id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deployed_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'training',
    performance_metrics JSONB,
    hyperparameters JSONB,
    model_path VARCHAR(500)
);

-- Model Performance Tracking
CREATE TABLE model_performance (
    timestamp TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    prediction DECIMAL(10,5) NOT NULL,
    actual DECIMAL(10,5) NOT NULL,
    confidence DECIMAL(10,5) NOT NULL,
    latency_ms INTEGER NOT NULL,
    PRIMARY KEY (timestamp, model_version, symbol)
) PARTITION BY RANGE (timestamp);
```

### Autonomous Learning Tables
```sql
-- Experience Replay Memory
CREATE TABLE experience_replay (
    memory_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    tier INTEGER NOT NULL CHECK (tier >= 1 AND tier <= 6),
    state JSONB NOT NULL,
    action JSONB NOT NULL,
    reward DECIMAL(10,5) NOT NULL,
    next_state JSONB NOT NULL,
    priority DECIMAL(10,5) NOT NULL DEFAULT 1.0
);

-- Adaptation History
CREATE TABLE adaptation_history (
    adaptation_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    trigger_event VARCHAR(100) NOT NULL,
    adaptation_type VARCHAR(50) NOT NULL,
    performance_before JSONB,
    performance_after JSONB,
    success BOOLEAN
);

-- Market Regimes
CREATE TABLE market_regimes (
    regime_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    regime_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(10,5) NOT NULL,
    features JSONB,
    duration_minutes INTEGER
);
```

### Risk Management Tables
```sql
-- Risk Metrics
CREATE TABLE risk_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    portfolio_value DECIMAL(20,10) NOT NULL,
    total_exposure DECIMAL(20,10) NOT NULL,
    margin_used DECIMAL(20,10) NOT NULL,
    available_margin DECIMAL(20,10) NOT NULL,
    leverage_ratio DECIMAL(10,5) NOT NULL,
    max_drawdown DECIMAL(10,5) NOT NULL,
    var_95 DECIMAL(20,10) NOT NULL,
    expected_shortfall DECIMAL(20,10) NOT NULL,
    PRIMARY KEY (timestamp)
);

-- Stress Test Results
CREATE TABLE stress_test_results (
    test_id BIGSERIAL PRIMARY KEY,
    test_date TIMESTAMPTZ NOT NULL,
    scenario_name VARCHAR(100) NOT NULL,
    portfolio_value DECIMAL(20,10) NOT NULL,
    loss_amount DECIMAL(20,10) NOT NULL,
    loss_percentage DECIMAL(10,5) NOT NULL,
    recovery_time_minutes INTEGER,
    risk_metrics JSONB
);
```

### System Monitoring Tables
```sql
-- System Logs
CREATE TABLE system_logs (
    log_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level VARCHAR(20) NOT NULL,
    component VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB
) PARTITION BY RANGE (timestamp);

-- Performance Analytics
CREATE TABLE performance_analytics (
    timestamp TIMESTAMPTZ NOT NULL,
    period VARCHAR(20) NOT NULL,
    total_return DECIMAL(10,5) NOT NULL,
    sharpe_ratio DECIMAL(10,5) NOT NULL,
    max_drawdown DECIMAL(10,5) NOT NULL,
    win_rate DECIMAL(10,5) NOT NULL,
    profit_factor DECIMAL(10,5) NOT NULL,
    total_trades INTEGER NOT NULL,
    avg_win DECIMAL(20,10) NOT NULL,
    avg_loss DECIMAL(20,10) NOT NULL,
    PRIMARY KEY (timestamp, period)
);
```

## Indexing Strategy

### Primary Indexes
- `market_data.(timestamp, symbol, exchange)`: Time-series market data access
- `positions.position_id`: Position tracking and updates
- `trades.trade_id`: Trade history and audit trail
- `model_versions.model_id`: Model version management

### Secondary Indexes
- `market_data.symbol_timestamp_idx`: Symbol-specific time queries
  - Columns: symbol, timestamp
  - Type: BRIN (Block Range INdex)
  - Usage: Historical data retrieval for specific symbols

- `positions.symbol_status_idx`: Active position queries
  - Columns: symbol, status
  - Type: BTREE
  - Usage: Real-time position monitoring

- `trades.symbol_timestamp_idx`: Trade history analysis
  - Columns: symbol, timestamp
  - Type: BRIN
  - Usage: Performance analysis and backtesting

### Composite Indexes
- `experience_replay.tier_priority_timestamp_idx`: Memory sampling optimization
  - Columns: tier, priority, timestamp
  - Usage: Prioritized experience replay sampling

- `model_performance.model_symbol_timestamp_idx`: Model evaluation queries
  - Columns: model_version, symbol, timestamp
  - Usage: Model performance analysis

### Partial Indexes
- `positions.active_positions_idx`: Active positions only
  - Condition: WHERE status = 'open'
  - Usage: Real-time portfolio monitoring

- `system_logs.error_logs_idx`: Error log filtering
  - Condition: WHERE level = 'ERROR'
  - Usage: Error monitoring and alerting

## Data Types and Constraints

### Custom Types
```sql
-- Trading Enums
CREATE TYPE trade_side AS ENUM ('BUY', 'SELL');
CREATE TYPE position_status AS ENUM ('OPEN', 'CLOSED', 'PARTIAL');
CREATE TYPE model_status AS ENUM ('TRAINING', 'VALIDATING', 'DEPLOYED', 'RETIRED');
CREATE TYPE regime_type AS ENUM ('TRENDING', 'RANGING', 'VOLATILE', 'CRISIS');

-- Risk Enums
CREATE TYPE risk_level AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
CREATE TYPE adaptation_trigger AS ENUM ('DRIFT', 'REGIME_CHANGE', 'PERFORMANCE', 'MANUAL');
```

### Check Constraints
```sql
-- Market Data Validation
ALTER TABLE market_data ADD CONSTRAINT chk_positive_prices
CHECK (price > 0 AND bid_price > 0 AND ask_price > 0);

-- Position Validation
ALTER TABLE positions ADD CONSTRAINT chk_position_quantity
CHECK (quantity > 0);

-- Model Performance Validation
ALTER TABLE model_performance ADD CONSTRAINT chk_confidence_range
CHECK (confidence >= 0 AND confidence <= 1);
```

## Normalization

### Normal Forms Compliance
**1NF**: All tables in first normal form with atomic values
**2NF**: All non-key attributes depend on the full primary key
**3NF**: No transitive dependencies on non-key attributes
**BCNF**: All determinants are candidate keys

### Denormalization Decisions
**Table**: performance_analytics
**Reason**: Aggregated data for fast dashboard queries
**Impact**: Storage vs. query performance optimization
**Maintenance**: Updated via scheduled jobs and triggers

## Relationships and Foreign Keys

### Foreign Key Constraints
```sql
-- Model Performance References
ALTER TABLE model_performance ADD CONSTRAINT fk_model_performance_versions
FOREIGN KEY (model_version) REFERENCES model_versions(version)
ON DELETE CASCADE;

-- Position References
ALTER TABLE positions ADD CONSTRAINT fk_positions_exchanges
FOREIGN KEY (exchange) REFERENCES exchanges(exchange_id)
ON DELETE RESTRICT;

-- Risk Metrics References
ALTER TABLE stress_test_results ADD CONSTRAINT fk_stress_portfolio
FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
ON DELETE CASCADE;
```

### Relationship Types
- **One-to-One**: Model versions to deployment configs
- **One-to-Many**: Portfolio to positions, Model to performance metrics
- **Many-to-Many**: Symbols to strategies (via junction tables)

## Data Integrity

### Triggers
```sql
-- Update Position P&L
CREATE OR REPLACE FUNCTION update_position_pnl()
RETURNS TRIGGER AS $$
BEGIN
    NEW.unrealized_pnl = (NEW.current_price - NEW.entry_price) * NEW.quantity *
                        CASE WHEN NEW.side = 'SHORT' THEN -1 ELSE 1 END;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_position_pnl
    BEFORE UPDATE ON positions
    FOR EACH ROW
    WHEN (OLD.current_price IS DISTINCT FROM NEW.current_price)
    EXECUTE FUNCTION update_position_pnl();
```

### Views
```sql
-- Active Portfolio View
CREATE VIEW active_portfolio AS
SELECT
    symbol,
    SUM(CASE WHEN side = 'LONG' THEN quantity ELSE -quantity END) as net_position,
    SUM(unrealized_pnl) as total_pnl,
    AVG(entry_price) as avg_entry_price
FROM positions
WHERE status = 'OPEN'
GROUP BY symbol;

-- Model Performance Summary
CREATE VIEW model_performance_summary AS
SELECT
    model_version,
    symbol,
    AVG(ABS(prediction - actual)) as avg_error,
    AVG(latency_ms) as avg_latency,
    COUNT(*) as prediction_count
FROM model_performance
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY model_version, symbol;
```

## Performance Considerations

### Partitioning Strategy
**Table**: market_data, order_book_l1, model_performance
**Partition Key**: timestamp (daily partitions)
**Partition Type**: RANGE partitioning
**Benefits**: Improved query performance, easier data management

### Query Optimization
**Common Queries**:
- Latest market data: Index on (symbol, timestamp DESC)
- Position updates: Index on position_id with HOT updates
- Performance analytics: Pre-aggregated views with materialized results

### Caching Strategy
**Cache Type**: Redis Cluster with persistence
**Cache Keys**: Market data snapshots, feature computations, model predictions
**Invalidation**: Time-based expiration with event-driven updates

## Backup and Recovery

### Backup Strategy
**Type**: Continuous backup with periodic snapshots
**Frequency**: Continuous WAL archiving, daily full backups
**Retention**: 30 days for operational data, 7 years for regulatory data
**Storage**: Multi-region S3 with cross-region replication

### Recovery Procedures
**Point-in-Time Recovery**: Using WAL archives and backup snapshots
**Data Corruption Recovery**: Parallel recovery from multiple regions
**Disaster Recovery**: Multi-region failover with automated DNS updates

## Migration Scripts

### Version 1.0.0 - Initial Schema
**Date**: 2025-01-21
**Changes**:
- Core trading tables (market_data, positions, trades)
- AI/ML tables (model_versions, model_performance)
- Basic indexing and partitioning

**Forward Migration**:
```sql
-- Core schema creation
CREATE TABLE market_data (...);
CREATE TABLE positions (...);
-- ... additional table creations
```

**Rollback Migration**:
```sql
-- Drop all tables in reverse order
DROP TABLE IF EXISTS performance_analytics;
DROP TABLE IF EXISTS system_logs;
-- ... continue with other drops
```

## References
- [Entity-Relationship Diagrams](../../../docs/database/erd/)
- [Data Flow Diagrams](../../../docs/database/flows/)
- [Migration Scripts](../../../migrations/)
- [Backup Procedures](../../../docs/operations/backup/)
- [Performance Benchmarks](../../../docs/database/benchmarks/)