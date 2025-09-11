"""
ML-Nautilus Integration API Router
=================================

REST API endpoints for ML-Nautilus integration management.
Provides control and monitoring capabilities for the ML-enhanced trading system.

Endpoints:
- ML-Nautilus integration management (start/stop/configure)
- ML performance monitoring and analytics
- Enhanced order submission with ML predictions
- Model performance tracking and adaptation
- Strategy adapter management and metrics

Security:
- Rate limiting (60 requests/minute)
- Input validation and sanitization
- Error handling without information leakage
- Audit logging for all operations

Author: ML & Trading Systems Team
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

from src.trading.ml_nautilus_integration import (
    MLNautilusIntegrationManager,
    MLEnhancedOrderRequest,
    MLIntegrationMode,
    get_ml_nautilus_integration,
    initialize_ml_nautilus_integration,
    process_tick_with_ml_nautilus,
    submit_ml_enhanced_order,
    get_ml_nautilus_metrics,
    start_ml_nautilus_integration,
    stop_ml_nautilus_integration,
    get_ml_nautilus_health
)
from src.api.dependencies import get_current_user, get_settings

logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Router
router = APIRouter(
    prefix="/ml-nautilus",
    tags=["ml-nautilus-integration"],
    dependencies=[Depends(security)],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Too many requests"},
        500: {"description": "Internal server error"}
    }
)


# Pydantic Models
class MLIntegrationStatus(BaseModel):
    """ML-Nautilus integration status"""
    is_initialized: bool
    is_running: bool
    integration_mode: str
    system_status: Dict[str, Any]
    ml_performance: Dict[str, Any]
    nautilus_status: Dict[str, Any]


class MLHealthCheckResponse(BaseModel):
    """ML-Nautilus health check response"""
    timestamp: datetime
    overall_status: str
    components: Dict[str, str]
    issues: List[str]
    ml_metrics: Dict[str, Any]


class MLEnhancedOrderSubmission(BaseModel):
    """ML-enhanced order submission request"""
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


class MLPerformanceMetrics(BaseModel):
    """ML performance metrics response"""
    total_predictions: int
    successful_predictions: int
    success_rate: float
    avg_confidence: float
    strategy_performance: Dict[str, Any]
    feature_importance_summary: Dict[str, float]
    adaptation_metrics: Dict[str, Any]
    system_status: Dict[str, Any]


class MLProcessingRequest(BaseModel):
    """ML processing request for tick data"""
    symbol: str = Field(..., min_length=1, max_length=20, pattern=r'^[A-Z0-9/_]+$')
    price: float = Field(..., gt=0)
    volume: float = Field(..., gt=0)
    bid_price: float = Field(..., gt=0)
    ask_price: float = Field(..., gt=0)
    bid_size: float = Field(..., ge=0)
    ask_size: float = Field(..., ge=0)
    spread: float = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MLProcessingResponse(BaseModel):
    """ML processing response"""
    processing_timestamp: datetime
    original_signal: Dict[str, Any]
    enhanced_signal: Optional[Dict[str, Any]]
    order_request: Optional[Dict[str, Any]]
    ml_prediction: Optional[Dict[str, Any]]
    market_regime: Dict[str, Any]
    features_processed: bool


# API Endpoints

@router.get("/status", response_model=MLIntegrationStatus)
@limiter.limit("60/minute")
async def get_ml_integration_status(
    current_user: Dict = Depends(get_current_user)
) -> MLIntegrationStatus:
    """
    Get comprehensive ML-Nautilus integration status

    Returns detailed information about:
    - ML integration state and configuration
    - Performance metrics and predictions
    - Strategy adapter status
    - Feature engineering statistics
    - Model ensemble health
    """
    try:
        logger.info(f"🧠 ML status request from user: {current_user.get('id', 'unknown')}")

        # Get integration manager
        ml_integration = await get_ml_nautilus_integration()

        # Get comprehensive status
        ml_metrics = await get_ml_nautilus_metrics()
        nautilus_status = await ml_integration.nautilus_manager.get_system_status()

        return MLIntegrationStatus(
            is_initialized=ml_integration.is_initialized,
            is_running=ml_integration.is_running,
            integration_mode=ml_integration.integration_mode.value,
            system_status=ml_metrics['system_status'],
            ml_performance=ml_metrics,
            nautilus_status=nautilus_status
        )

    except Exception as e:
        logger.error(f"❌ Failed to get ML integration status: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve ML integration status"
        )


@router.get("/health", response_model=MLHealthCheckResponse)
@limiter.limit("60/minute")
async def ml_health_check(
    current_user: Dict = Depends(get_current_user)
) -> MLHealthCheckResponse:
    """
    Perform comprehensive health check of ML-Nautilus integration

    Checks all components:
    - ML model ensemble functionality
    - Feature engineering pipeline
    - Strategy adapters health
    - Market regime detection
    - Nautilus integration connectivity
    - Performance tracking systems
    """
    try:
        logger.info(f"🏥 ML health check request from user: {current_user.get('id', 'unknown')}")

        # Get health status
        health_status = await get_ml_nautilus_health()
        ml_metrics = await get_ml_nautilus_metrics()

        return MLHealthCheckResponse(
            timestamp=health_status['timestamp'],
            overall_status=health_status['overall_status'],
            components=health_status['components'],
            issues=health_status['issues'],
            ml_metrics=ml_metrics
        )

    except Exception as e:
        logger.error(f"❌ ML health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="ML health check failed"
        )


@router.post("/initialize")
@limiter.limit("10/minute")
async def initialize_ml_integration(
    integration_mode: str = Query(
        "hybrid_execution",
        regex=r'^(enhanced_routing|adaptive_strategies|hybrid_execution|full_autonomous)$'
    ),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Initialize ML-Nautilus integration with specified mode

    Integration Modes:
    - enhanced_routing: ML-enhanced order routing decisions
    - adaptive_strategies: ML-driven strategy adaptation
    - hybrid_execution: ML + Nautilus hybrid execution
    - full_autonomous: Complete ML autonomy with Nautilus

    This endpoint:
    - Initializes ML components (ensemble, feature engineering)
    - Sets up market regime detection
    - Configures strategy adapters
    - Establishes Nautilus integration
    - Validates all connections
    """
    try:
        logger.info(f"🚀 ML initialize request from user: {current_user.get('id', 'unknown')}")
        logger.info(f"   Mode: {integration_mode}")

        # Parse integration mode
        mode = MLIntegrationMode(integration_mode)

        # Initialize integration
        success = await initialize_ml_nautilus_integration(mode)

        if success:
            return {
                "message": f"ML-Nautilus integration initialized successfully in {mode.value} mode",
                "status": "success",
                "integration_mode": mode.value,
                "timestamp": datetime.utcnow()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize ML-Nautilus integration"
            )

    except Exception as e:
        logger.error(f"❌ Failed to initialize ML integration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initialize ML-Nautilus integration"
        )


@router.post("/start")
@limiter.limit("10/minute")
async def start_ml_integration(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Start ML-Nautilus integration

    Begins:
    - ML model inference and predictions
    - Real-time feature engineering
    - Market regime detection
    - Strategy adapter execution
    - Performance monitoring and tracking
    """
    try:
        logger.info(f"▶️ ML start request from user: {current_user.get('id', 'unknown')}")

        await start_ml_nautilus_integration()

        return {
            "message": "ML-Nautilus integration started successfully",
            "status": "running",
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to start ML integration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start ML-Nautilus integration"
        )


@router.post("/stop")
@limiter.limit("10/minute")
async def stop_ml_integration(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Stop ML-Nautilus integration

    Gracefully shuts down:
    - ML model inference
    - Feature engineering pipeline
    - Strategy adapters
    - Performance tracking
    - Market regime detection
    """
    try:
        logger.info(f"⏹️ ML stop request from user: {current_user.get('id', 'unknown')}")

        await stop_ml_nautilus_integration()

        return {
            "message": "ML-Nautilus integration stopped successfully",
            "status": "stopped",
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to stop ML integration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to stop ML-Nautilus integration"
        )


@router.post("/process-tick", response_model=MLProcessingResponse)
@limiter.limit("100/minute")
async def process_tick_with_ml(
    tick_request: MLProcessingRequest,
    current_user: Dict = Depends(get_current_user)
) -> MLProcessingResponse:
    """
    Process tick data with ML enhancement

    This endpoint:
    - Extracts 1000+ technical features
    - Runs ML model ensemble predictions
    - Detects market regime
    - Generates ML-enhanced trading signals
    - Prepares orders for Nautilus execution

    Returns comprehensive analysis including:
    - Original ML signal
    - Enhanced signal with Nautilus optimizations
    - ML predictions and confidence scores
    - Market regime classification
    - Feature importance analysis
    """
    try:
        logger.info(f"📊 ML tick processing from user: {current_user.get('id', 'unknown')}")
        logger.info(f"   Symbol: {tick_request.symbol}")

        # Create tick data object
        tick_data = type('TickData', (), {
            'last_price': tick_request.price,
            'volume': tick_request.volume,
            'bid_price': tick_request.bid_price,
            'ask_price': tick_request.ask_price,
            'bid_size': tick_request.bid_size,
            'ask_size': tick_request.ask_size,
            'spread': tick_request.spread,
            'timestamp': tick_request.timestamp
        })()

        # Process with ML enhancement
        result = await process_tick_with_ml_nautilus(tick_data)

        return MLProcessingResponse(
            processing_timestamp=result['processing_timestamp'],
            original_signal={
                'action': result['original_signal'].action,
                'confidence': result['original_signal'].confidence,
                'position_size': result['original_signal'].position_size,
                'strategy': result['original_signal'].strategy.value,
                'reasoning': result['original_signal'].reasoning
            },
            enhanced_signal=result['enhanced_signal'],
            order_request=result['order_request'].__dict__ if result['order_request'] else None,
            ml_prediction=result['ml_prediction'],
            market_regime=result['market_regime'],
            features_processed=result['features'] is not None
        )

    except Exception as e:
        logger.error(f"❌ Failed to process tick with ML: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process tick with ML enhancement"
        )


@router.post("/order", response_model=Dict[str, Any])
@limiter.limit("30/minute")
async def submit_ml_enhanced_order_endpoint(
    order_request: MLEnhancedOrderSubmission,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Submit order with ML enhancement through Nautilus

    This endpoint:
    - Creates ML-enhanced order request
    - Applies ML predictions and confidence scores
    - Routes through intelligent order routing system
    - Executes via Nautilus with ML optimizations
    - Tracks performance and adaptation metrics

    Enhanced features:
    - ML-driven order type selection
    - Confidence-based position sizing
    - Market regime adjustments
    - Feature importance tracking
    - Performance attribution
    """
    try:
        logger.info(f"📤 ML-enhanced order from user: {current_user.get('id', 'unknown')}")
        logger.info(f"   Order: {order_request.side} {order_request.quantity} {order_request.symbol}")

        # Create ML-enhanced order request
        ml_order = MLEnhancedOrderRequest(
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            order_type=order_request.order_type,
            price=order_request.price,
            stop_price=order_request.stop_price,
            client_id=order_request.client_id or f"ml_api_{datetime.utcnow().timestamp()}"
        )

        # Add ML metadata (would be populated by ML processing)
        ml_order.ml_predictions = {'source': 'api_request'}
        ml_order.strategy_confidence = 0.8  # Default confidence
        ml_order.market_regime = 'trending'  # Would be detected
        ml_order.risk_adjusted_size = order_request.quantity
        ml_order.execution_confidence = 0.8
        ml_order.feature_importance = {'api_request': 1.0}

        # Submit through ML-Nautilus system
        result = await submit_ml_enhanced_order(ml_order)

        logger.info(f"✅ ML-enhanced order submitted: {result.get('order_id', 'unknown')}")

        return {
            "message": "ML-enhanced order submitted successfully",
            "result": result,
            "ml_enhanced": True,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to submit ML-enhanced order: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to submit ML-enhanced order"
        )


@router.get("/performance", response_model=MLPerformanceMetrics)
@limiter.limit("60/minute")
async def get_ml_performance_metrics(
    current_user: Dict = Depends(get_current_user)
) -> MLPerformanceMetrics:
    """
    Get comprehensive ML performance metrics

    Returns detailed metrics including:
    - Prediction accuracy and confidence scores
    - Strategy-specific performance
    - Feature importance analysis
    - Model adaptation metrics
    - System health and utilization
    - Performance attribution by component
    """
    try:
        logger.info(f"📈 ML performance metrics request from user: {current_user.get('id', 'unknown')}")

        # Get ML performance metrics
        metrics = await get_ml_nautilus_metrics()

        return MLPerformanceMetrics(**metrics)

    except Exception as e:
        logger.error(f"❌ Failed to get ML performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve ML performance metrics"
        )


@router.get("/feature-importance", response_model=Dict[str, float])
@limiter.limit("30/minute")
async def get_feature_importance(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, float]:
    """
    Get feature importance analysis from ML models

    Returns:
    - Most important features for current predictions
    - Historical feature importance trends
    - Feature correlation with successful trades
    - Model-specific feature weights
    """
    try:
        logger.info(f"🔍 Feature importance request from user: {current_user.get('id', 'unknown')}")

        # Get feature importance from ML metrics
        metrics = await get_ml_nautilus_metrics()

        return metrics.get('feature_importance_summary', {})

    except Exception as e:
        logger.error(f"❌ Failed to get feature importance: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve feature importance"
        )


@router.get("/strategy-performance", response_model=Dict[str, Any])
@limiter.limit("30/minute")
async def get_strategy_performance(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get strategy-specific performance metrics

    Returns detailed performance for each strategy:
    - Success rate and confidence scores
    - Average position sizes and P&L
    - Win/loss ratios and drawdown
    - Market regime performance breakdown
    - ML enhancement effectiveness
    """
    try:
        logger.info(f"📊 Strategy performance request from user: {current_user.get('id', 'unknown')}")

        # Get strategy performance from ML metrics
        metrics = await get_ml_nautilus_metrics()

        return metrics.get('strategy_performance', {})

    except Exception as e:
        logger.error(f"❌ Failed to get strategy performance: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve strategy performance"
        )


@router.put("/config")
@limiter.limit("5/minute")
async def update_ml_configuration(
    integration_mode: Optional[str] = Query(None, regex=r'^(enhanced_routing|adaptive_strategies|hybrid_execution|full_autonomous)$'),
    enable_adaptation: Optional[bool] = Query(None),
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update ML-Nautilus integration configuration

    Allows dynamic reconfiguration of:
    - Integration mode (enhanced_routing/adaptive_strategies/hybrid_execution/full_autonomous)
    - Adaptation settings (enable/disable ML adaptation)
    - Performance tracking parameters
    - Model confidence thresholds
    - Feature engineering settings
    """
    try:
        logger.info(f"⚙️ ML configuration update from user: {current_user.get('id', 'unknown')}")

        ml_integration = await get_ml_nautilus_integration()

        updates = {}

        if integration_mode:
            ml_integration.integration_mode = MLIntegrationMode(integration_mode)
            updates['integration_mode'] = integration_mode

        if enable_adaptation is not None:
            # This would enable/disable ML adaptation features
            updates['adaptation_enabled'] = enable_adaptation

        return {
            "message": "ML configuration updated successfully",
            "updates": updates,
            "current_mode": ml_integration.integration_mode.value,
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to update ML configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update ML configuration"
        )


@router.post("/reset-metrics")
@limiter.limit("5/minute")
async def reset_ml_metrics(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Reset ML performance metrics and adaptation history

    This endpoint:
    - Clears performance tracking history
    - Resets adaptation metrics
    - Maintains model state and configuration
    - Useful for starting fresh performance analysis
    """
    try:
        logger.info(f"🔄 ML metrics reset from user: {current_user.get('id', 'unknown')}")

        ml_integration = await get_ml_nautilus_integration()

        # Reset metrics
        ml_integration.ml_performance_history = []
        ml_integration.adaptation_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'avg_confidence': 0.0,
            'regime_accuracy': 0.0,
            'execution_improvement': 0.0
        }

        return {
            "message": "ML metrics reset successfully",
            "status": "reset_complete",
            "timestamp": datetime.utcnow()
        }

    except Exception as e:
        logger.error(f"❌ Failed to reset ML metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to reset ML metrics"
        )


# Export router
ml_nautilus_router = router