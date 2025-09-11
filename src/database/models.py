from sqlalchemy import Column, Float, Integer, String, DateTime, BigInteger, Boolean, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy import PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSON

Base = declarative_base()

class MarketData(Base):
    __tablename__ = "market_data"

    timestamp = Column(DateTime(timezone=True), primary_key=True, default=func.now())
    symbol = Column(String(20), primary_key=True)
    exchange = Column(String(20), primary_key=True)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    bid_price = Column(Float)
    ask_price = Column(Float)
    bid_volume = Column(Float)
    ask_volume = Column(Float)

    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', exchange='{self.exchange}', price={self.price})>"

class OrderBookL1(Base):
    __tablename__ = "order_book_l1"

    timestamp = Column(DateTime(timezone=True), primary_key=True, default=func.now())
    symbol = Column(String(20), primary_key=True)
    exchange = Column(String(20), primary_key=True)
    bid_price = Column(Float, nullable=False)
    bid_volume = Column(Float, nullable=False)
    ask_price = Column(Float, nullable=False)
    ask_volume = Column(Float, nullable=False)

    def __repr__(self):
        return f"<OrderBookL1(symbol='{self.symbol}', exchange='{self.exchange}', bid={self.bid_price}, ask={self.ask_price})>"

class Trade(Base):
    __tablename__ = "trades"

    trade_id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float)
    pnl = Column(Float)

    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', side='{self.side}', quantity={self.quantity}, price={self.price})>"

class ModelPerformance(Base):
    __tablename__ = "model_performance"

    timestamp = Column(DateTime(timezone=True), primary_key=True, default=func.now())
    model_version = Column(String(50), primary_key=True)
    symbol = Column(String(20), primary_key=True)
    prediction = Column(Float, nullable=False)
    actual = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    latency_ms = Column(Integer, nullable=False)

    def __repr__(self):
        return f"<ModelPerformance(model='{self.model_version}', symbol='{self.symbol}', prediction={self.prediction})>"

class Position(Base):
    __tablename__ = "positions"

    position_id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    status = Column(String(20), nullable=False, default='open')

    def __repr__(self):
        return f"<Position(symbol='{self.symbol}', side='{self.side}', quantity={self.quantity})>"

class RiskMetric(Base):
    __tablename__ = "risk_metrics"

    timestamp = Column(DateTime(timezone=True), primary_key=True, default=func.now())
    portfolio_value = Column(Float, nullable=False)
    total_exposure = Column(Float, nullable=False)
    margin_used = Column(Float, nullable=False)
    available_margin = Column(Float, nullable=False)
    leverage_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    var_95 = Column(Float, nullable=False)
    expected_shortfall = Column(Float, nullable=False)

    def __repr__(self):
        return f"<RiskMetric(timestamp='{self.timestamp}', portfolio_value={self.portfolio_value})>"

class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    deployed_at = Column(DateTime(timezone=True))
    status = Column(String(20), nullable=False, default='training')
    performance_metrics = Column(JSON)
    hyperparameters = Column(JSON)
    model_path = Column(String(500))

    __table_args__ = (UniqueConstraint('model_name', 'version', name='_model_version_uc'),)

    def __repr__(self):
        return f"<ModelVersion(name='{self.model_name}', version='{self.version}')>"

class AlternativeData(Base):
    __tablename__ = "alternative_data"

    timestamp = Column(DateTime(timezone=True), primary_key=True, default=func.now())
    symbol = Column(String(20), primary_key=True)
    data_type = Column(String(50), primary_key=True) # 'sentiment', 'onchain', 'whale', 'news'
    source = Column(String(100), nullable=False)
    value = Column(Float)
    text_content = Column(String)
    confidence = Column(Float)

    def __repr__(self):
        return f"<AlternativeData(symbol='{self.symbol}', type='{self.data_type}', value={self.value})>"

class SystemLog(Base):
    __tablename__ = "system_logs"

    log_id = Column(BigInteger, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    level = Column(String(20), nullable=False)
    component = Column(String(100), nullable=False)
    message = Column(String, nullable=False)
    metadata_ = Column(JSON, name='metadata') # Renamed to avoid conflict with Python keyword

    def __repr__(self):
        return f"<SystemLog(level='{self.level}', component='{self.component}', message='{self.message[:50]}...')>"

class PerformanceAnalytic(Base):
    __tablename__ = "performance_analytics"

    timestamp = Column(DateTime(timezone=True), primary_key=True, default=func.now())
    period = Column(String(20), primary_key=True) # 'daily', 'weekly', 'monthly'
    total_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=False)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    total_trades = Column(Integer, nullable=False)
    avg_win = Column(Float, nullable=False)
    avg_loss = Column(Float, nullable=False)

    def __repr__(self):
        return f"<PerformanceAnalytic(period='{self.period}', total_return={self.total_return})>"

# --- Embeddings (portable baseline; migrate to pgvector if available) ---
class EmbeddingRecord(Base):
    __tablename__ = "embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    namespace = Column(String(64), nullable=False)
    item_id = Column(String(128), nullable=False)
    vector = Column(JSON, nullable=False)  # list[float]; replace with pgvector later
    metadata_ = Column(JSON, name='metadata')

    __table_args__ = (
        UniqueConstraint('namespace', 'item_id', name='_emb_ns_item_uc'),
        Index('ix_embeddings_ns', 'namespace'),
    )

    def __repr__(self):
        return f"<EmbeddingRecord(ns='{self.namespace}', item='{self.item_id}')>"
