"""
LearningManager: Central coordinator for self-learning, self-adapting, and self-healing systems
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback
import time

# Import our enhanced components
from src.learning.meta_learning.meta_learning_engine import MetaLearningEngine, MetaLearningConfig
from src.learning.self_adaptation.market_adaptation import AdvancedMarketAdaptation, AdaptationConfig
from src.learning.self_healing.self_healing_engine import EnhancedSelfHealingEngine, HealingConfig
from src.learning.neural_networks.enhanced_neural_network import EnhancedTradingNeuralNetwork, NetworkConfig
from src.trading.hft_engine.ultra_low_latency_engine import UltraLowLatencyTradingEngine, ExecutionConfig

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """Operational modes for the trading system"""
    NORMAL = "normal"
    ADAPTATION = "adaptation"
    HEALING = "healing"
    LEARNING = "learning"
    OPTIMIZATION = "optimization"

@dataclass
class LearningConfig:
    """Configuration for the learning manager"""
    # System modes
    enable_self_learning: bool = True
    enable_self_adaptation: bool = True
    enable_self_healing: bool = True
    
    # Learning parameters
    learning_interval: float = 60.0  # seconds between learning cycles
    adaptation_interval: float = 30.0  # seconds between adaptation checks
    healing_check_interval: float = 5.0  # seconds between healing checks
    
    # Performance targets
    target_sharpe_ratio: float = 2.0
    target_win_rate: float = 0.6
    max_drawdown_threshold: float = 0.1  # 10%
    
    # Integration parameters
    data_buffer_size: int = 10000  # Size of experience buffer

class PerformanceMetrics:
    """Tracks system performance metrics"""
    
    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.current_metrics: Dict[str, float] = {}
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update performance metrics"""
        self.current_metrics.update(metrics)
        self.metrics_history.append({
            "timestamp": datetime.now(),
            "metrics": metrics.copy()
        })
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return self.current_metrics.copy()
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """Get recent performance metrics"""
        if not self.metrics_history:
            return {}
            
        recent_metrics = self.metrics_history[-window:]
        if not recent_metrics:
            return {}
            
        # Average the metrics
        averaged = {}
        metric_keys = recent_metrics[0]["metrics"].keys()
        
        for key in metric_keys:
            values = [m["metrics"].get(key) for m in recent_metrics]
            # Filter out None and non-numeric values before calculating mean
            numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]
            if numeric_values:
                averaged[key] = np.mean(numeric_values)
            else:
                # If no numeric values, keep the last known non-numeric value or set to None
                last_value = next((m["metrics"].get(key) for m in reversed(recent_metrics) if m["metrics"].get(key) is not None), None)
                averaged[key] = last_value
            
        return averaged

class ExperienceBuffer:
    """Stores trading experiences for continual learning"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences: List[Dict] = []
        self.lock = threading.Lock()
        
    def add_experience(self, experience: Dict) -> None:
        """Add a new experience to the buffer"""
        with self.lock:
            self.experiences.append(experience)
            if len(self.experiences) > self.max_size:
                self.experiences = self.experiences[-self.max_size:]
    
    def sample_experiences(self, batch_size: int) -> List[Dict]:
        """Sample a batch of experiences"""
        with self.lock:
            if len(self.experiences) <= batch_size:
                return self.experiences.copy()
            else:
                indices = np.random.choice(len(self.experiences), batch_size, replace=False)
                return [self.experiences[i] for i in indices]
    
    def get_size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.experiences)

class LearningManager:
    """Central coordinator for self-learning, self-adapting, and self-healing systems"""
    
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self.mode = SystemMode.NORMAL
        self.is_running = False
        
        # Initialize performance tracking
        self.performance_tracker = PerformanceMetrics()
        self.experience_buffer = ExperienceBuffer(self.config.data_buffer_size)
        
        # Initialize core components
        self.meta_learning_engine = MetaLearningEngine(MetaLearningConfig())
        self.market_adaptation = AdvancedMarketAdaptation(AdaptationConfig())
        self.self_healing_engine = EnhancedSelfHealingEngine(HealingConfig())
        self.neural_network = EnhancedTradingNeuralNetwork(NetworkConfig())
        self.trading_engine = UltraLowLatencyTradingEngine(ExecutionConfig())
        
        # Threading components
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.learning_thread: Optional[threading.Thread] = None
        self.adaptation_thread: Optional[threading.Thread] = None
        self.healing_thread: Optional[threading.Thread] = None
        
        # Component status tracking
        self.component_status: Dict[str, Dict[str, Any]] = {
            "meta_learning": {"status": "initialized", "last_update": datetime.now()},
            "market_adaptation": {"status": "initialized", "last_update": datetime.now()},
            "self_healing": {"status": "initialized", "last_update": datetime.now()},
            "neural_network": {"status": "initialized", "last_update": datetime.now()},
            "trading_engine": {"status": "initialized", "last_update": datetime.now()}
        }
        
        logger.info("LearningManager initialized")
    
    def start(self) -> None:
        """Start the learning manager"""
        if self.is_running:
            logger.warning("LearningManager is already running")
            return
            
        self.is_running = True
        logger.info("Starting LearningManager")
        
        # Start background threads
        if self.config.enable_self_learning:
            self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
            self.learning_thread.start()
            
        if self.config.enable_self_adaptation:
            self.adaptation_thread = threading.Thread(target=self._adaptation_loop, daemon=True)
            self.adaptation_thread.start()
            
        if self.config.enable_self_healing:
            self.healing_thread = threading.Thread(target=self._healing_loop, daemon=True)
            self.healing_thread.start()
        
        logger.info("LearningManager started successfully")
    
    def stop(self) -> None:
        """Stop the learning manager"""
        self.is_running = False
        logger.info("LearningManager stopped")
    
    def _learning_loop(self) -> None:
        """Background loop for self-learning"""
        while self.is_running:
            try:
                self._perform_learning_cycle()
                time.sleep(self.config.learning_interval)
            except Exception as e:
                logger.error(f"Error in learning loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5)  # Brief pause before retrying
    
    def _adaptation_loop(self) -> None:
        """Background loop for self-adaptation"""
        while self.is_running:
            try:
                self._perform_adaptation_cycle()
                time.sleep(self.config.adaptation_interval)
            except Exception as e:
                logger.error(f"Error in adaptation loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5)
    
    def _healing_loop(self) -> None:
        """Background loop for self-healing"""
        while self.is_running:
            try:
                self._perform_healing_cycle()
                time.sleep(self.config.healing_check_interval)
            except Exception as e:
                logger.error(f"Error in healing loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5)
    
    def _perform_learning_cycle(self) -> None:
        """Perform one cycle of self-learning"""
        try:
            self.mode = SystemMode.LEARNING
            logger.info("Starting learning cycle")
            
            # Sample experiences from buffer
            experiences = self.experience_buffer.sample_experiences(32)
            if not experiences:
                logger.info("No experiences available for learning")
                return
            
            # Update component status
            self.component_status["meta_learning"]["status"] = "learning"
            self.component_status["meta_learning"]["last_update"] = datetime.now()
            
            # In a real implementation, we would train the meta-learning model here
            # For now, we'll just log the activity
            logger.info(f"Processed {len(experiences)} experiences for meta-learning")
            
            # Update performance metrics
            learning_metrics = {
                "experiences_processed": len(experiences),
                "learning_cycle_time": np.random.uniform(0.1, 1.0)  # Simulated time
            }
            self.performance_tracker.update_metrics(learning_metrics)
            
            self.component_status["meta_learning"]["status"] = "idle"
            logger.info("Learning cycle completed")
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {str(e)}")
            self.component_status["meta_learning"]["status"] = "error"
            self.component_status["meta_learning"]["error"] = str(e)
            
        finally:
            self.mode = SystemMode.NORMAL
    
    def _perform_adaptation_cycle(self) -> None:
        """Perform one cycle of self-adaptation"""
        try:
            self.mode = SystemMode.ADAPTATION
            logger.info("Starting adaptation cycle")
            
            # Update component status
            self.component_status["market_adaptation"]["status"] = "adapting"
            self.component_status["market_adaptation"]["last_update"] = datetime.now()
            
            # Get current market condition
            condition = self.market_adaptation.get_current_condition()
            
            # Adapt strategy
            adapted_params = self.market_adaptation.adapt_to_conditions()
            
            # Check for regime change
            regime_change = self.market_adaptation.detect_regime_change()
            
            # Update performance metrics
            adaptation_metrics = {
                "regime": condition.regime.value if condition.regime else "unknown",
                "volatility": condition.volatility if condition.volatility else 0.0,
                "trend_strength": condition.trend_strength if condition.trend_strength else 0.0,
                "regime_change_detected": 1.0 if regime_change else 0.0
            }
            self.performance_tracker.update_metrics(adaptation_metrics)
            
            if adapted_params:
                logger.info(f"Strategy adapted: {len(adapted_params)} parameters updated")
            
            self.component_status["market_adaptation"]["status"] = "idle"
            logger.info("Adaptation cycle completed")
            
        except Exception as e:
            logger.error(f"Error in adaptation cycle: {str(e)}")
            self.component_status["market_adaptation"]["status"] = "error"
            self.component_status["market_adaptation"]["error"] = str(e)
            
        finally:
            self.mode = SystemMode.NORMAL
    
    def _perform_healing_cycle(self) -> None:
        """Perform one cycle of self-healing"""
        try:
            self.mode = SystemMode.HEALING
            logger.info("Starting healing cycle")
            
            # Update component status
            self.component_status["self_healing"]["status"] = "monitoring"
            self.component_status["self_healing"]["last_update"] = datetime.now()
            
            # Check system health
            health = asyncio.run(self.self_healing_engine.check_system_health())
            
            # Predict potential failures
            asyncio.run(self.self_healing_engine.predict_and_prevent_failures(health))
            
            # Update component statuses
            for component, status in self.component_status.items():
                if component != "self_healing":  # Don't update self-healing status here
                    self.self_healing_engine.update_component_status(component, status)
            
            # Update performance metrics
            healing_metrics = {
                "overall_health": health.overall_health,
                "cpu_usage": health.cpu_usage,
                "memory_usage": health.memory_usage,
                "error_rate": health.error_rate
            }
            self.performance_tracker.update_metrics(healing_metrics)
            
            # Check if health is below threshold
            if health.overall_health < 0.5:
                logger.warning(f"System health is low: {health.overall_health:.2f}")
            
            self.component_status["self_healing"]["status"] = "idle"
            logger.info("Healing cycle completed")
            
        except Exception as e:
            logger.error(f"Error in healing cycle: {str(e)}")
            self.component_status["self_healing"]["status"] = "error"
            self.component_status["self_healing"]["error"] = str(e)
            
        finally:
            self.mode = SystemMode.NORMAL
    
    def update_market_data(self, symbol: str, price: float, volume: float) -> None:
        """Update market data for all components"""
        timestamp = datetime.now()
        
        # Update market adaptation system
        self.market_adaptation.update_market_data(price, volume, timestamp)
        
        # Update trading engine market data
        from src.trading.hft_engine.ultra_low_latency_engine import MarketData
        market_data = MarketData(
            symbol=symbol,
            bid_price=price * 0.9999,  # Simulated bid/ask spread
            ask_price=price * 1.0001,
            bid_size=volume * 0.5,
            ask_size=volume * 0.5,
            timestamp=timestamp
        )
        self.trading_engine.update_market_data(market_data)
    
    def add_trading_experience(self, experience: Dict) -> None:
        """Add trading experience to learning buffer"""
        self.experience_buffer.add_experience(experience)
        
        # Update performance metrics
        if "pnl" in experience:
            current_metrics = self.performance_tracker.get_current_metrics()
            trades = current_metrics.get("total_trades", 0) + 1
            total_pnl = current_metrics.get("total_pnl", 0) + experience["pnl"]
            
            experience_metrics = {
                "total_trades": trades,
                "total_pnl": total_pnl,
                "average_pnl": total_pnl / max(1, trades)
            }
            self.performance_tracker.update_metrics(experience_metrics)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "mode": self.mode.value,
            "is_running": self.is_running,
            "performance_metrics": self.performance_tracker.get_current_metrics(),
            "component_status": self.component_status.copy(),
            "experience_buffer_size": self.experience_buffer.get_size(),
            "timestamp": datetime.now()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        recent_performance = self.performance_tracker.get_recent_performance()
        system_status = self.get_system_status()
        
        # Get reports from individual components
        healing_report = {
            "health_history": len(self.self_healing_engine.get_health_history()),
            "failure_history": len(self.self_healing_engine.get_failure_history()),
            "healing_events": len(self.self_healing_engine.get_healing_history())
        }
        
        trading_report = self.trading_engine.get_performance_report()
        
        return {
            "system_status": system_status,
            "recent_performance": recent_performance,
            "healing_report": healing_report,
            "trading_report": trading_report,
            "timestamp": datetime.now()
        }
    
    def execute_order(self, order_dict: Dict) -> Dict:
        """Execute order through ultra-low latency engine"""
        try:
            # Create order object
            from src.trading.hft_engine.ultra_low_latency_engine import Order, OrderSide, OrderType
            
            order = Order(
                order_id=order_dict.get("order_id", f"order_{int(datetime.now().timestamp())}"),
                symbol=order_dict["symbol"],
                side=OrderSide.BUY if order_dict["side"].upper() == "BUY" else OrderSide.SELL,
                order_type=OrderType.MARKET if order_dict.get("order_type", "MARKET").upper() == "MARKET" else OrderType.LIMIT,
                quantity=order_dict["quantity"],
                price=order_dict.get("price")
            )
            
            # Execute order
            executed_order = self.trading_engine.execute_order(order)
            
            # Return execution result
            return {
                "order_id": executed_order.order_id,
                "status": executed_order.status.value,
                "latency_ms": executed_order.latency,
                "exchange": executed_order.exchange,
                "timestamp": executed_order.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize learning manager
    learning_manager = LearningManager()
    
    # Start the system
    learning_manager.start()
    
    # Simulate market data updates
    for i in range(10):
        learning_manager.update_market_data(
            symbol="BTC/USDT",
            price=45000 + np.random.normal(0, 100),  # BTC price with noise
            volume=np.random.exponential(1000)  # Trading volume
        )
        time.sleep(0.1)
    
    # Simulate trading experience
    for i in range(5):
        experience = {
            "symbol": "BTC/USDT",
            "pnl": np.random.normal(0, 10),  # Profit/loss
            "holding_time": np.random.uniform(1, 300),  # Seconds
            "slippage": np.random.uniform(0, 0.1),  # Percentage
            "timestamp": datetime.now()
        }
        learning_manager.add_trading_experience(experience)
    
    # Get system status
    status = learning_manager.get_system_status()
    print(f"System status: {status['mode']}")
    print(f"Components: {len(status['component_status'])} active")
    
    # Get performance report
    report = learning_manager.get_performance_report()
    print(f"Performance metrics: {list(report['recent_performance'].keys())}")
    
    # Execute a sample order
    order_result = learning_manager.execute_order({
        "symbol": "BTC/USDT",
        "side": "BUY",
        "quantity": 0.001,
        "order_type": "MARKET"
    })
    print(f"Order execution: {order_result}")
    
    # Let the system run for a bit
    time.sleep(2)
    
    # Stop the system
    learning_manager.stop()
    
    logger.info("LearningManager test completed successfully")