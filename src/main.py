"""
Main application entry point for CryptoScalp AI
"""

# Import stubs first to provide fallbacks for optional dependencies
from src.trading.stubs import get_dependency_status, log_dependency_status

# Log dependency status
log_dependency_status()

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import logging
from datetime import datetime, timezone

from src.database.manager import DatabaseManager
from src.database.dependencies import get_db, set_database_manager

from src.api.routers.trading import trading_router
from src.api.routers.market import market_router
from src.api.routers.enhanced_trading import enhanced_trading_router
from src.api.routers.tick_data import tick_router
from src.api.routers.nautilus_integration import nautilus_router
from src.api.routers.ml_nautilus_integration import ml_nautilus_router
from src.api.routers.health import health_router, set_health_components
from src.api.routers.backtest import backtest_router
from src.api.routers.metrics import metrics_router

# Import enhanced infrastructure components
from src.database.enhanced_pool_manager import EnhancedDatabaseManager, DatabaseConfig, DatabaseType
from src.database.redis_manager import RedisManager, RedisConfig
from src.monitoring.comprehensive_monitoring import ComprehensiveMonitoringSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CryptoScalp AI",
    description="High-Frequency Cryptocurrency Trading System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Add SlowAPI middleware for rate limiting
try:
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi import Limiter
    app.add_middleware(SlowAPIMiddleware)
except ImportError:
    # SlowAPI not available, skip rate limiting
    pass

# Include API routers
app.include_router(trading_router, prefix="/api")
app.include_router(tick_router, prefix="/api")
app.include_router(market_router, prefix="/api")
app.include_router(enhanced_trading_router, prefix="/api")
app.include_router(nautilus_router, prefix="/api")
app.include_router(ml_nautilus_router, prefix="/api")
app.include_router(health_router, prefix="/api")  # Add health check router
app.include_router(backtest_router, prefix="/api")
app.include_router(metrics_router, prefix="/api")

# Initialize infrastructure components
db_manager = DatabaseManager()

# Initialize enhanced components
enhanced_db_config = DatabaseConfig(
    db_type=DatabaseType.POSTGRESQL,
    host="cryptoscalp-db",
    port=5432,
    database="cryptoscalp_dev",
    username="cryptoscalp",
    password="devpassword",
    pool_size=20,
    max_overflow=30
)

enhanced_db_manager = EnhancedDatabaseManager(enhanced_db_config)

redis_config = RedisConfig(
    host="cryptoscalp-redis",
    port=6379,
    max_memory="512mb",
    max_memory_policy="allkeys-lru"
)

redis_manager = RedisManager(redis_config)

monitoring_system = ComprehensiveMonitoringSystem()

# Set health check components
set_health_components(enhanced_db_manager, redis_manager, monitoring_system)

# Set the database manager for dependencies
set_database_manager(db_manager)

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup event triggered.")
    await db_manager.connect()
    
    # Initialize enhanced components
    try:
        await enhanced_db_manager.connect()
        logger.info("✅ Enhanced database manager connected")
    except Exception as e:
        logger.error(f"❌ Enhanced database manager connection failed: {e}")
    
    try:
        await redis_manager.connect()
        logger.info("✅ Redis manager connected")
    except Exception as e:
        logger.error(f"❌ Redis manager connection failed: {e}")
    
    try:
        await monitoring_system.initialize()
        logger.info("✅ Monitoring system initialized")
    except Exception as e:
        logger.error(f"❌ Monitoring system initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown event triggered.")
    await db_manager.disconnect()
    
    # Shutdown enhanced components
    try:
        await enhanced_db_manager.disconnect()
        logger.info("✅ Enhanced database manager disconnected")
    except Exception as e:
        logger.error(f"❌ Error disconnecting enhanced database manager: {e}")
    
    try:
        await redis_manager.disconnect()
        logger.info("✅ Redis manager disconnected")
    except Exception as e:
        logger.error(f"❌ Error disconnecting Redis manager: {e}")
    
    try:
        await monitoring_system.shutdown()
        logger.info("✅ Monitoring system shutdown")
    except Exception as e:
        logger.error(f"❌ Error shutting down monitoring system: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CryptoScalp AI Trading System",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.websocket("/ws/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for market data"""
    await websocket.accept()
    try:
        # Send initial acknowledgment
        await websocket.send_json({"type": "connected", "message": "WebSocket connected"})
        while True:
            # In real implementation, this would stream market data
            data = await websocket.receive_json()
            if data.get("type") == "subscribe":
                await websocket.send_json({"type": "subscribed", "symbol": data.get("symbol", "BTC/USDT")})
            else:
                await websocket.send_json({"type": "pong", "message": "received"})
    except Exception:
        pass

@app.websocket("/ws/trading")
async def websocket_trading(websocket: WebSocket):
    """WebSocket endpoint for trading"""
    await websocket.accept()
    try:
        while True:
            # In real implementation, this would handle trading commands
            await websocket.receive_text()
            await websocket.send_json({"type": "ping", "message": "pong"})
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
