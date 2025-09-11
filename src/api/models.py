from pydantic import BaseModel, Field
from typing import List, Optional

class OrderCreate(BaseModel):
    symbol: str = Field(..., description="Trading pair symbol (e.g., BTC/USDT)")
    side: str = Field(..., description="Order side (buy or sell)")
    quantity: float = Field(..., gt=0, description="Quantity of the asset to trade")
    price: Optional[float] = Field(None, gt=0, description="Price for limit orders")
    order_type: str = Field("MARKET", description="Type of order (e.g., MARKET, LIMIT)")

class MarketDataPoint(BaseModel):
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

class MarketDataResponse(BaseModel):
    symbol: str
    limit: int
    data: List[MarketDataPoint]
    message: str

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import hashlib
import secrets

# Tick Data Models
class TickDataPoint(BaseModel):
    """Individual tick data point"""
    timestamp: float = Field(..., description="Unix timestamp in seconds")
    symbol: str = Field(..., description="Trading pair symbol")
    price: float = Field(..., gt=0, description="Last trade price")
    volume: float = Field(..., ge=0, description="Trade volume")
    side: Optional[str] = Field(None, description="Trade side (buy/sell)")
    exchange_timestamp: Optional[float] = Field(None, description="Exchange timestamp")
    source_exchange: str = Field(..., description="Exchange name")

class TickDataResponse(BaseModel):
    """Response model for tick data requests"""
    symbol: str = Field(..., description="Trading pair symbol")
    limit: int = Field(..., ge=1, le=1000, description="Number of tick data points")
    data: List[TickDataPoint] = Field(default_factory=list, description="List of tick data points")
    message: str = Field(..., description="Response message")
    total_count: int = Field(default=0, description="Total number of ticks available")
    request_timestamp: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Request timestamp")

class TickDataConfig(BaseModel):
    """Configuration for tick data parameters"""
    max_limit: int = Field(default=1000, ge=1, le=5000, description="Maximum number of ticks per request")
    default_limit: int = Field(default=100, ge=1, le=1000, description="Default number of ticks per request")
    cache_ttl: int = Field(default=30, ge=1, le=300, description="Cache TTL in seconds")
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000, description="Rate limit per minute")
    supported_exchanges: List[str] = Field(
        default=["binance", "okx", "bybit", "coinbase", "kraken"],
        description="Supported exchanges"
    )
    supported_symbols: List[str] = Field(
        default=["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT", "MATIC/USDT", "DOT/USDT"],
        description="Supported trading symbols"
    )

class TickDataError(BaseModel):
    """Error response model for tick data"""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Error timestamp")

class TickDataStats(BaseModel):
    """Statistics for tick data endpoints"""
    total_requests: int = Field(default=0, description="Total number of requests")
    successful_requests: int = Field(default=0, description="Number of successful requests")
    failed_requests: int = Field(default=0, description="Number of failed requests")
    average_response_time: float = Field(default=0.0, description="Average response time in seconds")
    last_updated: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Last updated timestamp")


# Security and Authentication Models
class APIKey(BaseModel):
    """API Key model for authentication"""
    key_id: str = Field(..., description="Unique API key identifier")
    hashed_key: str = Field(..., description="SHA-256 hash of the API key")
    name: str = Field(..., description="Human-readable name for the key")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    permissions: List[str] = Field(default_factory=lambda: ["read"], description="List of permissions")
    rate_limit: int = Field(default=1000, ge=1, le=10000, description="Requests per hour")
    is_active: bool = Field(default=True, description="Whether the key is active")
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Creation timestamp")
    expires_at: Optional[float] = Field(None, description="Expiration timestamp")
    last_used_at: Optional[float] = Field(None, description="Last usage timestamp")

    @classmethod
    def generate_key(cls, name: str, permissions: List[str] = None) -> tuple:
        """Generate a new API key pair"""
        if permissions is None:
            permissions = ["read"]

        raw_key = f"cryptoscalp_{secrets.token_urlsafe(32)}"
        hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()

        return raw_key, cls(
            key_id=f"key_{secrets.token_hex(8)}",
            hashed_key=hashed_key,
            name=name,
            permissions=permissions
        )

    def verify_key(self, provided_key: str) -> bool:
        """Verify a provided API key"""
        return self.hashed_key == hashlib.sha256(provided_key.encode()).hexdigest()

    def is_expired(self) -> bool:
        """Check if the API key is expired"""
        if self.expires_at is None:
            return False
        return datetime.now().timestamp() > self.expires_at


class SecurityEvent(BaseModel):
    """Security event model for audit logging"""
    event_id: str = Field(default_factory=lambda: f"sec_{secrets.token_hex(8)}", description="Unique event ID")
    event_type: str = Field(..., description="Type of security event (auth_failure, rate_limit, suspicious_activity)")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    ip_address: Optional[str] = Field(None, description="IP address of the request")
    user_agent: Optional[str] = Field(None, description="User agent string")
    endpoint: Optional[str] = Field(None, description="API endpoint accessed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")
    severity: str = Field(default="medium", description="Event severity (low, medium, high, critical)")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Event timestamp")


class RateLimit(BaseModel):
    """Rate limiting model"""
    identifier: str = Field(..., description="Rate limit identifier (IP, user, API key)")
    limit_type: str = Field(..., description="Type of limit (requests_per_minute, requests_per_hour)")
    current_count: int = Field(default=0, description="Current request count")
    limit: int = Field(..., description="Maximum allowed requests")
    window_start: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Window start timestamp")
    window_size: int = Field(..., description="Window size in seconds")
    blocked_until: Optional[float] = Field(None, description="Timestamp until which requests are blocked")

    def is_blocked(self) -> bool:
        """Check if the identifier is currently blocked"""
        if self.blocked_until is None:
            return False
        return datetime.now().timestamp() < self.blocked_until

    def increment(self) -> bool:
        """Increment the counter and check if limit is exceeded"""
        if self.is_blocked():
            return False

        current_time = datetime.now().timestamp()
        if current_time - self.window_start >= self.window_size:
            # Reset window
            self.current_count = 1
            self.window_start = current_time
            return True

        self.current_count += 1
        if self.current_count > self.limit:
            self.blocked_until = current_time + 300  # Block for 5 minutes
            return False

        return True


# Audit and Compliance Models
class AuditLog(BaseModel):
    """Audit log entry for compliance"""
    log_id: str = Field(default_factory=lambda: f"audit_{secrets.token_hex(8)}", description="Unique log ID")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Log timestamp")
    user_id: Optional[str] = Field(None, description="User ID")
    action: str = Field(..., description="Action performed")
    resource: str = Field(..., description="Resource affected")
    method: str = Field(..., description="HTTP method")
    status_code: int = Field(..., description="HTTP status code")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class TradingAuditLog(AuditLog):
    """Specialized audit log for trading activities"""
    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Trade side (buy/sell)")
    quantity: float = Field(..., description="Trade quantity")
    price: Optional[float] = Field(None, description="Trade price")
    order_type: str = Field(..., description="Order type")
    strategy_name: Optional[str] = Field(None, description="Trading strategy used")
    risk_score: Optional[float] = Field(None, description="Risk score at time of trade")
    compliance_flags: List[str] = Field(default_factory=list, description="Compliance flags raised")


class SystemHealthCheck(BaseModel):
    """System health check model"""
    check_id: str = Field(default_factory=lambda: f"health_{secrets.token_hex(8)}", description="Unique check ID")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Check timestamp")
    component: str = Field(..., description="Component being checked")
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional health details")


class ComplianceReport(BaseModel):
    """Compliance reporting model"""
    report_id: str = Field(default_factory=lambda: f"comp_{secrets.token_hex(8)}", description="Unique report ID")
    report_type: str = Field(..., description="Type of compliance report")
    period_start: float = Field(..., description="Report period start timestamp")
    period_end: float = Field(..., description="Report period end timestamp")
    generated_at: float = Field(default_factory=lambda: datetime.now().timestamp(), description="Report generation timestamp")
    total_trades: int = Field(default=0, description="Total number of trades")
    compliance_violations: int = Field(default=0, description="Number of compliance violations")
    risk_exposures: List[Dict[str, Any]] = Field(default_factory=list, description="Risk exposure details")
    recommendations: List[str] = Field(default_factory=list, description="Compliance recommendations")
    status: str = Field(default="draft", description="Report status (draft, final, approved)")
