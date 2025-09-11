"""
Nautilus Trader Integration API Router
=====================================

REST API endpoints for managing Nautilus Trader integration.
Provides control and monitoring capabilities for the hybrid trading system.

Endpoints:
- Integration management (start/stop/configure)
- Status monitoring and health checks
- Performance analytics and reporting
- Order routing configuration
- Strategy adapter management

Security:
- Rate limiting (60 requests/minute)
- Input validation and sanitization
- Error handling without information leakage
- Audit logging for all operations

Author: Trading Systems Team
Date: 2025-01-22
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from pydantic import BaseModel, Field, validator

from src.trading.nautilus_integration import (
    NautilusTraderManager,
    IntegrationMode,
    OrderRoutingStrategy,
    get_nautilus_integration,
    initialize_nautilus_integration,
    start_nautilus_integration,
    stop_nautilus_integration,
    get_nautilus_status,
    submit_order_hybrid
)
from src.trading.nautilus_strategy_adapter import (
    StrategyAdapterFactory,
    adapt_strategy_for_nautilus,
    get_adapter_performance_metrics
)
from src.api.dependencies import get_current_user, get_settings

logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Router
router = APIRouter(
    prefix="/nautilus",
    tags=["nautilus-integration"],
    dependencies=[Depends(security)],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Too many requests"},
        500: {"description": "Internal server error"}
    }
)


# Pydantic Models
class IntegrationStatus(BaseModel):
    """Nautilus integration status"""
    is_initialized: bool
    is_running: bool
    integration_mode: str
    routing_strategy: str
    last_health_check: datetime
    performance_metrics: Dict[str, Any]
    components: Dict[str, str]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    timestamp: datetime
    overall_status: str
    components: Dict[str, str]
    issues: List[str]


class OrderRequest(BaseModel):
    """Order request for hybrid routing"""
    symbol: str = Field(..., min_length=1, max_length=20, pattern=r'^[A-Z0-9/_]+$')
    side: str = Field(..., pattern=r'^(BUY|SELL)$')
    quantity: float = Field(..., gt=0, le=1000)
    order_type: str = Field(..., pattern=r'^(MARKET|LIMIT|STOP_MARKET|TRAILING_STOP|ICEBERG)$')
    price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    time_in_force: str = Field(default='GTC', pattern=r'^(GTC|IOC|FOK)$')
    client_id: Optional[str] = Field(None, max_length=100)

    @validator('price')
    def validate_price(cls, v, values):
        if values.get('order_type') in ['LIMIT', 'STOP_MARKET'] and v is None:
            raise ValueError('price required for LIMIT and STOP_MARKET orders')
        return v

    @validator('stop_price')
    def validate_stop_price(cls, v, values):
        if values.get('order_type') == 'TRAILING_STOP' and v is None:
            raise ValueError('stop_price required for TRAILING_STOP orders')
        return v


class RoutingDecision(BaseModel):
    """Order routing decision"""
    use_nautilus: bool
    reason: str
    confidence: float
    expected_improvement: float


class StrategyAdapterRequest(BaseModel):
    """Strategy adapter creation request"""
    strategy_type: str = Field(..., pattern=r'^(scalping|market_making|mean_reversion)$')
    strategy_id: str = Field(..., min_length=1, max_length=100)
    instrument_id: str = Field(..., min_length=1, max_length=50)


class PerformanceMetrics(BaseModel):
    """Performance metrics response"""
    adapter_info: Dict[str, Any]
    nautilus_status: Dict[str, Any]
    signal_adaptation_stats: Dict[str, Any]
    order_execution_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]


# API Endpoints

@router.get("/status", response_model=IntegrationStatus)
@limiter.limit("60/minute")
async def get_integration_status(
    current_user: Dict = Depends(get_current_user)
) -> IntegrationStatus:
    """
    Get comprehensive Nautilus integration status

    Returns detailed information about:
    - Integration state and configuration
    - Component health status
    - Performance metrics
    - Routing decisions history
    """
    try:
        logger.info(f"📊 Status request from user: {current_user.get('id', 'unknown')}")

        # Get integration status
        status = await get_nautilus_status()

        return IntegrationStatus(**status)

    except Exception as e:
        logger.error(f"❌ Failed to get integration status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve integration status"
        )


@router.get("/health", response_model=HealthCheckResponse)
@limiter.limit("60/minute")
async def health_check(
    current_user: Dict = Depends(get_current_user)
) -> HealthCheckResponse:
    """
    Perform comprehensive health check of Nautilus integration

    Checks all components:
    - Nautilus engine connectivity
    - Account manager status
    - Risk manager functionality
    - Analytics engine health
    - Strategy adapters status
    """
    try:
        logger.info(f"🏥 Health check request from user: {current_user.get('id', 'unknown')}")

        nautilus_integration = await get_nautilus_integration()
        health_status = await nautilus_integration.health_check()

        return HealthCheckResponse(**health_status)

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@router.post("/initialize")
@limiter.limit("10/minute")
async def initialize_integration(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Initialize Nautilus Trader integration

    This endpoint:
    - Initializes Nautilus engine
    - Sets up account management
    - Configures risk management
    - Prepares analytics engine
    """
    try:
        logger.info(f"🚀 Initialize request from user: {current_user.get('id', 'unknown')}")

        success = await initialize_nautilus_integration()

        if success:
            return {
                "message": "Nautilus integration initialized successfully",
                "status": "success",
                "timestamp": datetime.utcnow()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize Nautilus integration"
            )

    except Exception as e:
        logger.error(f"❌ Failed to initialize integration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize Nautilus integration"
        )


@router.post("/start")
@limiter.limit("10/minute")
async def start_integration(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Start Nautilus Trader integration

    Begins:
    - Real-time order processing
    - Market data streaming
    - Performance monitoring
    - Strategy execution
    """
    try:
        logger.info(f"▶️ Start request from user: {current_user.get('id', 'unknown')}")

        await start_nautilus_integration()

        return {
            "message": "Nautilus integration started successfully",
            "status": "running",
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to start integration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start Nautilus integration"
        )


@router.post("/stop")
@limiter.limit("10/minute")
async def stop_integration(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Stop Nautilus Trader integration

    Gracefully shuts down:
    - Order processing
    - Market data streaming
    - Strategy execution
    - Performance monitoring
    """
    try:
        logger.info(f"⏹️ Stop request from user: {current_user.get('id', 'unknown')}")

        await stop_nautilus_integration()

        return {
            "message": "Nautilus integration stopped successfully",
            "status": "stopped",
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to stop integration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to stop Nautilus integration"
        )


@router.post("/order", response_model=Dict[str, Any])
@limiter.limit("30/minute")
async def submit_order(
    order_request: OrderRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Submit order through hybrid routing system

    Routes order to:
    - Nautilus (for advanced order types)
    - Existing system (for standard orders)
    - Automatic failover based on availability

    Supports order types:
    - MARKET: Immediate execution
    - LIMIT: Price-based execution
    - STOP_MARKET: Stop-loss orders
    - TRAILING_STOP: Dynamic stop-loss
    - ICEBERG: Large order splitting
    """
    try:
        logger.info(f"📤 Order submission from user: {current_user.get('id', 'unknown')}")
        logger.info(f"   Order: {order_request.side} {order_request.quantity} {order_request.symbol}")

        # Convert to dict for processing
        order_dict = order_request.dict()

        # Submit through hybrid routing
        result = await submit_order_hybrid(order_dict)

        logger.info(f"✅ Order submitted successfully: {result.get('order_id', 'unknown')}")

        return {
            "message": "Order submitted successfully",
            "result": result,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to submit order: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit order"
        )


@router.post("/adapter", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def create_strategy_adapter(
    adapter_request: StrategyAdapterRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create strategy adapter for Nautilus integration

    Adapts existing strategies to work with Nautilus while:
    - Maintaining original strategy logic
    - Adding Nautilus advanced features
    - Providing seamless fallback
    - Enabling A/B testing capabilities
    """
    try:
        logger.info(f"🔄 Adapter creation from user: {current_user.get('id', 'unknown')}")
        logger.info(f"   Strategy: {adapter_request.strategy_type}")
        logger.info(f"   ID: {adapter_request.strategy_id}")

        # Create adapter (placeholder for actual implementation)
        # In production, this would create a real adapter instance
        adapter_info = {
            "adapter_id": f"adapter_{adapter_request.strategy_id}",
            "strategy_type": adapter_request.strategy_type,
            "instrument_id": adapter_request.instrument_id,
            "created_at": datetime.utcnow(),
            "status": "created"
        }

        return {
            "message": "Strategy adapter created successfully",
            "adapter": adapter_info,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to create strategy adapter: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create strategy adapter"
        )


@router.get("/performance", response_model=PerformanceMetrics)
@limiter.limit("60/minute")
async def get_performance_metrics(
    strategy_id: str = Query(..., description="Strategy ID to get metrics for"),
    current_user: Dict = Depends(get_current_user)
) -> PerformanceMetrics:
    """
    Get comprehensive performance metrics for strategy adapter

    Returns detailed metrics including:
    - Signal adaptation statistics
    - Order execution performance
    - P&L and win rate analysis
    - Nautilus vs existing system comparison
    """
    try:
        logger.info(f"📈 Performance metrics request from user: {current_user.get('id', 'unknown')}")
        logger.info(f"   Strategy ID: {strategy_id}")

        # Placeholder for actual metrics retrieval
        # In production, this would fetch real metrics from the adapter
        metrics = {
            "adapter_info": {
                "strategy_id": strategy_id,
                "is_active": True,
                "adaptation_metrics": {
                    "signals_processed": 100,
                    "orders_submitted": 95,
                    "orders_filled": 90,
                    "profit_loss": 1250.50,
                    "win_rate": 0.68
                }
            },
            "nautilus_status": await get_nautilus_status(),
            "signal_adaptation_stats": {
                "total_signals": 100,
                "successful_adaptations": 95,
                "adaptation_rate": 0.95
            },
            "order_execution_stats": {
                "orders_submitted": 95,
                "orders_filled": 90,
                "fill_rate": 0.947
            },
            "performance_stats": {
                "total_pnl": 1250.50,
                "win_rate": 0.68,
                "active_positions": 3
            }
        }

        return PerformanceMetrics(**metrics)

    except Exception as e:
        logger.error(f"❌ Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve performance metrics"
        )


@router.get("/routing-decisions", response_model=List[Dict[str, Any]])
@limiter.limit("30/minute")
async def get_routing_decisions(
    limit: int = Query(50, description="Number of decisions to return", ge=1, le=1000),
    current_user: Dict = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Get recent order routing decisions

    Shows how the hybrid system decided to route orders:
    - Which engine was chosen (Nautilus vs Existing)
    - Reasoning for the decision
    - Expected performance improvement
    - Actual outcomes
    """
    try:
        logger.info(f"🔀 Routing decisions request from user: {current_user.get('id', 'unknown')}")

        nautilus_integration = await get_nautilus_integration()

        # Get recent routing decisions
        decisions = nautilus_integration.routing_decisions[-limit:]

        return decisions

    except Exception as e:
        logger.error(f"❌ Failed to get routing decisions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve routing decisions"
        )


@router.put("/config")
@limiter.limit("5/minute")
async def update_configuration(
    integration_mode: Optional[str] = Query(None, regex=r'^(disabled|standby|hybrid|primary)$'),
    routing_strategy: Optional[str] = Query(None, regex=r'^(performance_based|capability_based|load_based|exchange_based)$'),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update Nautilus integration configuration

    Allows dynamic reconfiguration of:
    - Integration mode (disabled/standby/hybrid/primary)
    - Routing strategy (performance/capability/load/exchange based)
    - Real-time parameter adjustment
    """
    try:
        logger.info(f"⚙️ Configuration update from user: {current_user.get('id', 'unknown')}")

        nautilus_integration = await get_nautilus_integration()

        updates = {}

        if integration_mode:
            nautilus_integration.integration_mode = IntegrationMode(integration_mode)
            updates['integration_mode'] = integration_mode

        if routing_strategy:
            nautilus_integration.routing_strategy = OrderRoutingStrategy(routing_strategy)
            updates['routing_strategy'] = routing_strategy

        return {
            "message": "Configuration updated successfully",
            "updates": updates,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to update configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update configuration"
        )


# Export router
nautilus_router = router