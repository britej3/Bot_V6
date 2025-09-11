"""
AdvancedMarketAdaptation: Multi-dimensional market condition analysis and adaptation
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classifications"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING = "trending"
    RANGE_BOUND = "range_bound"
    CRASH = "crash"
    BULL_RUN = "bull_run"

@dataclass
class MarketCondition:
    """Represents current market conditions"""
    timestamp: datetime
    volatility: float  # 0-1 normalized volatility
    trend_strength: float  # -1 to 1 (negative = downtrend, positive = uptrend)
    volume_profile: float  # 0-1 normalized volume
    liquidity: float  # 0-1 normalized liquidity
    correlation: float  # Cross-market correlation
    regime: MarketRegime
    confidence: float  # 0-1 confidence in regime classification

@dataclass
class AdaptationConfig:
    """Configuration for market adaptation"""
    # Analysis parameters
    volatility_window: int = 60  # Window for volatility calculation (seconds)
    trend_window: int = 300  # Window for trend analysis (seconds)
    correlation_window: int = 120  # Window for correlation analysis (seconds)
    
    # Thresholds
    high_volatility_threshold: float = 0.02  # 2% volatility threshold
    low_volatility_threshold: float = 0.005  # 0.5% volatility threshold
    strong_trend_threshold: float = 0.01  # 1% trend strength threshold
    
    # Adaptation parameters
    adaptation_frequency: int = 30  # How often to check for adaptation (seconds)
    regime_stability_window: int = 300  # Time to confirm regime change (seconds)
    
    # Performance tracking
    performance_window: int = 100  # Number of trades to analyze

class MarketAnalyzer:
    """Analyzes market conditions from various dimensions"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.recent_volatility = deque(maxlen=config.volatility_window)
        self.recent_prices = deque(maxlen=config.trend_window)
        self.recent_volume = deque(maxlen=config.volatility_window)
        self.recent_correlations = deque(maxlen=config.correlation_window)
        self.regime_history = deque(maxlen=config.regime_stability_window)
        
    def update_market_data(
        self, 
        price: float, 
        volume: float, 
        timestamp: datetime,
        correlated_assets: Optional[List[float]] = None
    ) -> None:
        """Update market data for analysis"""
        self.recent_prices.append((timestamp, price))
        self.recent_volume.append((timestamp, volume))
        self.recent_volatility.append((timestamp, price))
        
        if correlated_assets:
            correlation = self._calculate_correlation(price, correlated_assets)
            self.recent_correlations.append((timestamp, correlation))
    
    def _calculate_volatility(self) -> float:
        """Calculate market volatility"""
        if len(self.recent_volatility) < 2:
            return 0.0
            
        prices = [price for _, price in self.recent_volatility]
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252 * 24 * 60 * 60)  # Annualized
        return min(volatility, 1.0)  # Normalize to 0-1
    
    def _calculate_trend_strength(self) -> float:
        """Calculate trend strength"""
        if len(self.recent_prices) < 10:
            return 0.0
            
        prices = np.array([price for _, price in self.recent_prices])
        time_points = np.array([(t - self.recent_prices[0][0]).total_seconds() 
                               for t, _ in self.recent_prices])
        
        # Linear regression to find trend
        if len(time_points) > 1:
            slope, _ = np.polyfit(time_points, prices, 1)
            # Normalize trend strength
            avg_price = np.mean(prices)
            trend_strength = slope / avg_price if avg_price > 0 else 0.0
            return np.clip(trend_strength, -1.0, 1.0)
        return 0.0
    
    def _calculate_volume_profile(self) -> float:
        """Calculate normalized volume profile"""
        if len(self.recent_volume) < 2:
            return 0.0
            
        volumes = [volume for _, volume in self.recent_volume]
        avg_volume = np.mean(volumes)
        max_volume = np.max(volumes)
        
        if max_volume > 0:
            return avg_volume / max_volume
        return 0.0
    
    def _calculate_correlation(self, price: float, correlated_assets: List[float]) -> float:
        """Calculate cross-market correlation"""
        if not correlated_assets or len(correlated_assets) < 2:
            return 0.0
            
        # Simple correlation calculation
        prices = [price] + correlated_assets[:len(correlated_assets)-1]
        correlations = []
        
        for i in range(1, len(prices)):
            if i < len(correlated_assets):
                corr = np.corrcoef([prices[i-1], prices[i]], 
                                 [correlated_assets[i-1], correlated_assets[i]])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0.0)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _classify_regime(self, volatility: float, trend_strength: float) -> MarketRegime:
        """Classify market regime based on volatility and trend"""
        abs_trend = abs(trend_strength)
        
        if volatility > self.config.high_volatility_threshold:
            if abs_trend > self.config.strong_trend_threshold:
                if trend_strength < 0: # High volatility and strong downtrend
                    return MarketRegime.CRASH
                else: # High volatility and strong uptrend or sideways
                    return MarketRegime.VOLATILE
            else: # High volatility and weak trend
                return MarketRegime.VOLATILE # Still volatile, but not strongly trending
        elif volatility < self.config.low_volatility_threshold:
            if abs_trend > self.config.strong_trend_threshold:
                return MarketRegime.TRENDING
            else:
                return MarketRegime.RANGE_BOUND
        else: # Medium volatility
            if abs_trend > self.config.strong_trend_threshold:
                if trend_strength > 0:
                    return MarketRegime.BULL_RUN
                else:
                    return MarketRegime.CRASH
            else:
                return MarketRegime.NORMAL
    
    def get_current_condition(self) -> MarketCondition:
        """Get current market condition analysis"""
        timestamp = datetime.now()
        volatility = self._calculate_volatility()
        trend_strength = self._calculate_trend_strength()
        volume_profile = self._calculate_volume_profile()
        
        # Liquidity approximation (simplified)
        liquidity = min(volume_profile * 2, 1.0)
        
        # Correlation (if available)
        correlation = (self.recent_correlations[-1][1] 
                      if self.recent_correlations else 0.0)
        
        # Classify regime
        regime = self._classify_regime(volatility, trend_strength)
        
        # Confidence based on data quality and stability
        data_points = len(self.recent_prices)
        confidence = min(data_points / self.config.trend_window, 1.0)
        
        condition = MarketCondition(
            timestamp=timestamp,
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            liquidity=liquidity,
            correlation=correlation,
            regime=regime,
            confidence=confidence
        )
        
        # Store in history
        self.regime_history.append((timestamp, regime))
        
        return condition

class StrategyAdapter:
    """Adapts trading strategies based on market conditions"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.current_strategy_params: Dict[str, Any] = {}
        self.strategy_history: List[Dict] = []
        
    def adapt_strategy(self, condition: MarketCondition) -> Dict[str, Any]:
        """Adapt trading strategy based on market condition"""
        params = self._get_base_parameters()
        
        # Adapt based on regime
        if condition.regime == MarketRegime.VOLATILE:
            params.update({
                "position_size_multiplier": 0.5,  # Reduce position size
                "stop_loss_multiplier": 1.5,      # Wider stop loss
                "take_profit_multiplier": 0.8,    # Reduce take profit
                "frequency_multiplier": 2.0,      # Increase trading frequency
                "risk_per_trade": 0.01            # Reduce risk per trade
            })
        elif condition.regime == MarketRegime.TRENDING:
            params.update({
                "position_size_multiplier": 1.2,  # Increase position size
                "stop_loss_multiplier": 0.8,      # Tighter stop loss
                "take_profit_multiplier": 1.5,    # Increase take profit
                "frequency_multiplier": 0.8,      # Reduce trading frequency
                "risk_per_trade": 0.03            # Increase risk per trade
            })
        elif condition.regime == MarketRegime.RANGE_BOUND:
            params.update({
                "position_size_multiplier": 0.8,  # Reduce position size
                "stop_loss_multiplier": 0.7,      # Tighter stop loss
                "take_profit_multiplier": 0.7,    # Reduce take profit
                "frequency_multiplier": 1.5,      # Increase trading frequency
                "risk_per_trade": 0.015           # Moderate risk per trade
            })
        elif condition.regime == MarketRegime.CRASH:
            params.update({
                "position_size_multiplier": 0.3,  # Significantly reduce position size
                "stop_loss_multiplier": 2.0,      # Much wider stop loss
                "take_profit_multiplier": 0.5,    # Much lower take profit
                "frequency_multiplier": 0.5,      # Significantly reduce trading frequency
                "risk_per_trade": 0.005           # Minimize risk per trade
            })
        elif condition.regime == MarketRegime.BULL_RUN:
            params.update({
                "position_size_multiplier": 1.5,  # Increase position size
                "stop_loss_multiplier": 0.9,      # Slightly tighter stop loss
                "take_profit_multiplier": 1.8,    # Increase take profit
                "frequency_multiplier": 1.2,      # Slightly increase trading frequency
                "risk_per_trade": 0.04            # Increase risk per trade
            })
        
        # Adjust based on volatility
        if condition.volatility > self.config.high_volatility_threshold:
            params["position_size_multiplier"] *= 0.8
            params["risk_per_trade"] *= 0.7
        elif condition.volatility < self.config.low_volatility_threshold:
            params["position_size_multiplier"] *= 1.1
            params["risk_per_trade"] *= 1.2
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + 0.5 * condition.confidence
        for key in params:
            if "multiplier" in key or "risk" in key:
                params[key] *= confidence_multiplier
        
        # Store adapted parameters
        self.current_strategy_params = params.copy()
        
        # Record adaptation
        adaptation_record = {
            "timestamp": condition.timestamp,
            "regime": condition.regime.value,
            "confidence": condition.confidence,
            "params": params.copy()
        }
        self.strategy_history.append(adaptation_record)
        
        logger.info(f"Strategy adapted for {condition.regime.value} regime "
                   f"(confidence: {condition.confidence:.2f})")
        
        return params
    
    def _get_base_parameters(self) -> Dict[str, Any]:
        """Get base trading parameters"""
        return {
            "position_size_multiplier": 1.0,
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 1.0,
            "frequency_multiplier": 1.0,
            "risk_per_trade": 0.02,
            "max_position_size": 0.01,
            "min_position_size": 0.0001,
            "leverage": 10.0
        }
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current adapted parameters"""
        return self.current_strategy_params.copy() if self.current_strategy_params else self._get_base_parameters()

class AdvancedMarketAdaptation:
    """Main class for advanced market adaptation"""
    
    def __init__(self, config: Optional[AdaptationConfig] = None):
        self.config = config or AdaptationConfig()
        self.analyzer = MarketAnalyzer(self.config)
        self.adapter = StrategyAdapter(self.config)
        self.last_adaptation = datetime.min
        logger.info("AdvancedMarketAdaptation initialized")
    
    def update_market_data(
        self, 
        price: float, 
        volume: float, 
        timestamp: Optional[datetime] = None,
        correlated_assets: Optional[List[float]] = None
    ) -> None:
        """Update market data for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.analyzer.update_market_data(price, volume, timestamp, correlated_assets)
    
    def should_adapt(self) -> bool:
        """Check if adaptation should be performed"""
        now = datetime.now()
        time_since_last = (now - self.last_adaptation).total_seconds()
        return time_since_last >= self.config.adaptation_frequency
    
    def adapt_to_conditions(self) -> Optional[Dict[str, Any]]:
        """Perform full market adaptation cycle"""
        if not self.should_adapt():
            return None
            
        # Get current market condition
        condition = self.analyzer.get_current_condition()
        
        # Adapt strategy
        adapted_params = self.adapter.adapt_strategy(condition)
        
        # Update last adaptation time
        self.last_adaptation = datetime.now()
        
        # Log adaptation
        logger.info(f"Market adaptation completed: {condition.regime.value} regime, "
                   f"volatility: {condition.volatility:.4f}, "
                   f"trend: {condition.trend_strength:.4f}")
        
        return adapted_params
    
    def get_current_condition(self) -> MarketCondition:
        """Get current market condition"""
        return self.analyzer.get_current_condition()
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current adapted parameters"""
        return self.adapter.get_current_parameters()
    
    def detect_regime_change(self) -> bool:
        """Detect if market regime has changed"""
        if len(self.analyzer.regime_history) < 2:
            return False
            
        recent_regimes = [regime for _, regime in 
                         list(self.analyzer.regime_history)[-10:]]
        
        if len(recent_regimes) < 2:
            return False
            
        # Check if recent regimes are consistent
        current_regime = recent_regimes[-1]
        previous_regime = recent_regimes[-2]
        
        # Regime change if different and sustained
        if current_regime != previous_regime:
            # Check if new regime is consistent
            recent_consistent = all(r == current_regime 
                                  for r in recent_regimes[-5:])
            if recent_consistent:
                logger.info(f"Market regime change detected: "
                           f"{previous_regime.value} -> {current_regime.value}")
                return True
                
        return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize adaptation system
    adaptation = AdvancedMarketAdaptation()
    
    # Simulate market data updates
    for i in range(100):
        price = 45000 + np.random.normal(0, 100)  # BTC-like price with noise
        volume = np.random.exponential(1000)  # Volume
        timestamp = datetime.now()
        
        # Update market data
        adaptation.update_market_data(price, volume, timestamp)
        
        # Adapt strategy periodically
        if adaptation.should_adapt():
            params = adaptation.adapt_to_conditions()
            if params:
                print(f"Adapted parameters: {params}")
        
        # Check for regime change
        if adaptation.detect_regime_change():
            condition = adaptation.get_current_condition()
            print(f"Regime change to: {condition.regime.value}")
    
    logger.info("AdvancedMarketAdaptation test completed successfully")