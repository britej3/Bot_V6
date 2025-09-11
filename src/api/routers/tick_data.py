"""
Tick Data API Router

This module provides read-only REST API endpoints for accessing
real-time tick data from cryptocurrency exchanges via CCXT integration.
"""

import logging
import time
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.responses import JSONResponse as SlowAPIJSONResponse

from src.api.models import (
    TickDataResponse,
    TickDataError,
    TickDataConfig,
    TickDataStats
)
from src.api.tick_data_service import TickDataService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Create router
tick_router = APIRouter(
    prefix="/v1/tick-data",
    tags=["tick-data"],
    responses={
        429: {"model": TickDataError, "description": "Rate limit exceeded"},
        500: {"model": TickDataError, "description": "Internal server error"}
    }
)

# Global service instance
_tick_data_service: Optional[TickDataService] = None
_tick_data_config: Optional[TickDataConfig] = None

def get_tick_data_service() -> TickDataService:
    """Dependency to get tick data service instance"""
    if _tick_data_service is None:
        raise HTTPException(
            status_code=503,
            detail="Tick data service not initialized"
        )
    return _tick_data_service

def get_tick_data_config() -> TickDataConfig:
    """Dependency to get tick data configuration"""
    global _tick_data_config
    if _tick_data_config is None:
        _tick_data_config = TickDataConfig()
    return _tick_data_config

async def initialize_tick_data_service():
    """Initialize the tick data service"""
    global _tick_data_service, _tick_data_config

    try:
        _tick_data_config = TickDataConfig()
        _tick_data_service = TickDataService(_tick_data_config)
        await _tick_data_service.start()
        logger.info("Tick data service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize tick data service: {e}")
        raise

@tick_router.on_event("startup")
async def startup_event():
    """Handle router startup"""
    await initialize_tick_data_service()

@tick_router.on_event("shutdown")
async def shutdown_event():
    """Handle router shutdown"""
    global _tick_data_service
    if _tick_data_service:
        await _tick_data_service.stop()

@tick_router.get("/health")
async def health_check():
    """Health check endpoint for tick data service"""
    try:
        service = get_tick_data_service()
        stats = service.get_service_stats()

        return {
            "status": "healthy" if service.is_running else "unhealthy",
            "service": "tick_data_api",
            "timestamp": time.time(),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "tick_data_api",
            "timestamp": time.time(),
            "error": str(e)
        }

@tick_router.get("/symbols", response_model=List[str])
@limiter.limit("30/minute")
async def get_supported_symbols(
    request,
    exchange: Optional[str] = Query(None, description="Filter by exchange")
):
    """Get list of supported trading symbols"""
    try:
        config = get_tick_data_config()

        # Common trading pairs - in production, this would be fetched from exchanges
        common_symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
            "DOT/USDT", "MATIC/USDT", "AVAX/USDT", "LINK/USDT", "UNI/USDT"
        ]

        return common_symbols

    except Exception as e:
        logger.error(f"Error getting supported symbols: {e}")
        raise HTTPException(
            status_code=500,
            detail=TickDataError(
                error_code="INTERNAL_ERROR",
                message="Failed to retrieve supported symbols",
                details={"error": str(e)}
            ).dict()
        )

@tick_router.get("/exchanges", response_model=List[str])
@limiter.limit("30/minute")
async def get_supported_exchanges(request):
    """Get list of supported exchanges"""
    try:
        config = get_tick_data_config()
        return config.supported_exchanges

    except Exception as e:
        logger.error(f"Error getting supported exchanges: {e}")
        raise HTTPException(
            status_code=500,
            detail=TickDataError(
                error_code="INTERNAL_ERROR",
                message="Failed to retrieve supported exchanges",
                details={"error": str(e)}
            ).dict()
        )

@tick_router.get("/{symbol:path}", response_model=TickDataResponse)
@limiter.limit("60/minute")
async def get_tick_data(
    request,
    symbol: str = Query(..., description="Trading pair symbol (e.g., BTC/USDT)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of ticks to retrieve"),
    exchange: Optional[str] = Query(None, description="Preferred exchange"),
    service: TickDataService = Depends(get_tick_data_service)
):
    """
    Get real-time tick data for a trading symbol

    This endpoint provides access to recent tick data from cryptocurrency exchanges.
    Data is cached and refreshed regularly from multiple exchanges for reliability.
    """
    start_time = time.time()

    try:
        # Validate symbol format
        if not _is_valid_symbol(symbol):
            raise HTTPException(
                status_code=400,
                detail=TickDataError(
                    error_code="INVALID_SYMBOL",
                    message=f"Invalid symbol format: {symbol}",
                    details={"expected_format": "BASE/QUOTE (e.g., BTC/USDT)"}
                ).dict()
            )

        # Get tick data from service
        ticks, message = await service.get_tick_data(symbol, limit, exchange)

        # Calculate response time
        response_time = time.time() - start_time

        # Create response
        response = TickDataResponse(
            symbol=symbol,
            limit=limit,
            data=ticks,
            message=message,
            total_count=len(ticks),
            request_timestamp=start_time
        )

        # Log successful request
        logger.info(".3f")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tick data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=TickDataError(
                error_code="INTERNAL_ERROR",
                message="Failed to retrieve tick data",
                details={"symbol": symbol, "error": str(e)}
            ).dict()
        )

@tick_router.get("/stats", response_model=TickDataStats)
@limiter.limit("10/minute")
async def get_service_statistics(
    request,
    service: TickDataService = Depends(get_tick_data_service)
):
    """Get tick data service statistics"""
    try:
        stats = service.get_service_stats()

        # Convert to TickDataStats model
        return TickDataStats(
            total_requests=sum(ex['request_count'] for ex in stats['exchanges'].values()),
            successful_requests=sum(1 for ex in stats['exchanges'].values() if ex['is_connected']),
            failed_requests=sum(ex['error_count'] for ex in stats['exchanges'].values()),
            average_response_time=0.0,  # Would need to track this separately
            last_updated=time.time()
        )

    except Exception as e:
        logger.error(f"Error getting service statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=TickDataError(
                error_code="INTERNAL_ERROR",
                message="Failed to retrieve service statistics",
                details={"error": str(e)}
            ).dict()
        )

@tick_router.delete("/cache/{symbol:path}")
@limiter.limit("5/minute")
async def clear_symbol_cache(
    request,
    symbol: str,
    service: TickDataService = Depends(get_tick_data_service)
):
    """Clear cache for specific symbol (admin operation)"""
    try:
        service.clear_cache(symbol)
        return {
            "message": f"Cache cleared for symbol: {symbol}",
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Error clearing cache for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=TickDataError(
                error_code="INTERNAL_ERROR",
                message="Failed to clear cache",
                details={"symbol": symbol, "error": str(e)}
            ).dict()
        )

@tick_router.get("/config", response_model=TickDataConfig)
@limiter.limit("10/minute")
async def get_service_config(request):
    """Get tick data service configuration"""
    try:
        config = get_tick_data_config()
        return config

    except Exception as e:
        logger.error(f"Error getting service configuration: {e}")
        raise HTTPException(
            status_code=500,
            detail=TickDataError(
                error_code="INTERNAL_ERROR",
                message="Failed to retrieve service configuration",
                details={"error": str(e)}
            ).dict()
        )

def _is_valid_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    if not symbol or '/' not in symbol:
        return False

    parts = symbol.split('/')
    if len(parts) != 2:
        return False

    base, quote = parts
    if not base or not quote:
        return False

    # Basic checks for common crypto symbols
    if len(base) < 2 or len(base) > 10:
        return False
    if len(quote) < 3 or len(quote) > 6:
        return False

    return True

# Note: Exception handlers should be registered at the FastAPI app level
# These handler functions are available for use by the main app

async def rate_limit_handler(request, exc):
    """Handle rate limit exceeded exceptions"""
    return SlowAPIJSONResponse(
        status_code=429,
        content=TickDataError(
            error_code="RATE_LIMIT_EXCEEDED",
            message="Rate limit exceeded. Please try again later.",
            details={"retry_after": exc.retry_after}
        ).dict()
    )

async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception in tick data API: {exc}")
    return JSONResponse(
        status_code=500,
        content=TickDataError(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={"error_type": type(exc).__name__}
        ).dict()
    )