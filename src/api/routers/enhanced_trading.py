"""
Enhanced trading router with self-learning, self-adapting, and self-healing capabilities
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from src.api.models import OrderCreate
from src.learning.learning_manager import LearningManager, LearningConfig

# Initialize learning manager
learning_manager = LearningManager()
learning_manager.start()

# Create router
enhanced_trading_router = APIRouter()

logger = logging.getLogger(__name__)

@enhanced_trading_router.post("/v1/enhanced-orders")
async def create_enhanced_order(order_data: OrderCreate, background_tasks: BackgroundTasks):
    """Create enhanced order with self-learning capabilities"""
    try:
        # Convert order data to dictionary for processing
        order_dict = {
            "symbol": order_data.symbol,
            "side": order_data.side.upper(),
            "quantity": order_data.quantity,
            "price": order_data.price,
            "order_type": order_data.order_type.upper()
        }
        
        # Execute order through enhanced engine
        result = learning_manager.execute_order(order_dict)
        
        # Add experience to learning buffer in background
        if "status" in result and result["status"] == "filled":
            background_tasks.add_task(
                add_order_experience,
                order_dict,
                result
            )
        
        return {
            "message": "Enhanced order processed successfully",
            "order_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing enhanced order: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Order processing failed: {str(e)}")

def add_order_experience(order_dict: Dict, result: Dict) -> None:
    """Add order execution experience to learning buffer"""
    try:
        experience = {
            "symbol": order_dict["symbol"],
            "side": order_dict["side"],
            "quantity": order_dict["quantity"],
            "order_type": order_dict["order_type"],
            "price": order_dict.get("price"),
            "execution_price": result.get("execution_price"),
            "latency_ms": result.get("latency_ms", 0),
            "status": result.get("status"),
            "timestamp": datetime.fromisoformat(result.get("timestamp", datetime.now().isoformat()))
        }
        
        learning_manager.add_trading_experience(experience)
        
    except Exception as e:
        logger.error(f"Error adding order experience: {str(e)}")

@enhanced_trading_router.get("/v1/system-status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        status = learning_manager.get_system_status()
        return {
            "system_status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@enhanced_trading_router.get("/v1/performance-report")
async def get_performance_report():
    """Get detailed performance report"""
    try:
        report = learning_manager.get_performance_report()
        return {
            "performance_report": report,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting performance report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance report: {str(e)}")

@enhanced_trading_router.post("/v1/market-update")
async def update_market_data(market_data: Dict[str, Any]):
    """Update market data for all components"""
    try:
        symbol = market_data.get("symbol", "BTC/USDT")
        price = float(market_data.get("price", 0))
        volume = float(market_data.get("volume", 0))
        
        learning_manager.update_market_data(symbol, price, volume)
        
        return {
            "message": "Market data updated successfully",
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating market data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update market data: {str(e)}")

@enhanced_trading_router.get("/v1/component-status")
async def get_component_status():
    """Get individual component statuses"""
    try:
        status = learning_manager.get_system_status()
        return {
            "component_status": status.get("component_status", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting component status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get component status: {str(e)}")