from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
from src.api.models import OrderCreate
from src.database.models import Trade, Position
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.dependencies import get_db # Import get_db from dependencies
from src.api.dependencies import get_api_key # Import API key dependency

trading_router = APIRouter()

@trading_router.post("/v1/orders")
async def create_order(order_data: OrderCreate, db: AsyncSession = Depends(get_db)):
    """Create order endpoint"""
    # Pydantic model handles validation automatically
    # In real implementation, this would process the order and interact with exchange

    # Save order to database
    new_trade = Trade(
        symbol=order_data.symbol,
        side=order_data.side,
        quantity=order_data.quantity,
        price=order_data.price if order_data.price is not None else 0.0, # Handle optional price
        exchange="binance", # Placeholder
        commission=0.0, # Placeholder
        pnl=0.0 # Placeholder
    )
    db.add(new_trade)
    await db.flush() # Flush to get the ID if needed, but not commit yet

    return {
        "message": "Order created successfully (from trading router)",
        "order_data": order_data.model_dump()
    }

@trading_router.get("/v1/orders")
async def get_orders(db: AsyncSession = Depends(get_db), api_key: str = Depends(get_api_key)) -> List[Dict]:
    """Get orders endpoint"""
    # For now, fetch all trades from DB
    result = await db.execute(select(Trade))
    trades = result.scalars().all()
    return [trade.__dict__ for trade in trades] # Convert SQLAlchemy objects to dicts

@trading_router.get("/v1/positions")
async def get_positions(db: AsyncSession = Depends(get_db), api_key: str = Depends(get_api_key)) -> List[Dict]:
    """Get positions endpoint"""
    # For now, fetch all positions from DB
    result = await db.execute(select(Position))
    positions = result.scalars().all()
    return [position.__dict__ for position in positions] # Convert SQLAlchemy objects to dicts

@trading_router.get("/v1/trade")
async def get_trading_status(api_key: str = Depends(get_api_key)):
    """Get trading status endpoint"""
    return {"status": "Trading system operational", "message": "Authentication successful"}
