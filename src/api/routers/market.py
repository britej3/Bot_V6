from fastapi import APIRouter
from typing import Dict, Any
from src.api.models import MarketDataResponse, MarketDataPoint
from fastapi.responses import Response

market_router = APIRouter()

@market_router.get("/v1/market-data", response_model=MarketDataResponse)
async def get_market_data(symbol: str = "BTC/USDT", limit: int = 100) -> MarketDataResponse:
    """Get market data endpoint"""
    # For now, return a basic response
    # In real implementation, this would fetch market data
    return MarketDataResponse(
        symbol=symbol,
        limit=limit,
        data=[],
        message="Market data (from market router)"
    )

@market_router.options("/v1/market-data")
async def options_market_data():
    """Handle OPTIONS requests for CORS preflight"""
    return Response(status_code=200)
