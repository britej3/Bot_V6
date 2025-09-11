"""
Advanced Crypto-Specific Feature Engineering Pipeline
====================================================

This module provides high-performance feature engineering specifically designed for
cryptocurrency trading with sub-millisecond processing requirements.

Features:
- Whale activity detection algorithms
- Order flow analysis and imbalance detection  
- Market microstructure indicators
- Volume profile analysis
- Price action pattern recognition
- Market regime classification
- Real-time feature computation optimized for <1ms processing

Performance Targets:
- Feature computation: <1ms per update
- Memory usage: <100MB working set
- Supports 1000+ signals/second throughput
- Integration with JAX/Flax for GPU acceleration
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from functools import partial
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for crypto feature engineering"""
    # Temporal windows for different calculations
    short_window: int = 10      # Short-term (10 ticks)
    medium_window: int = 50     # Medium-term (50 ticks)  
    long_window: int = 200      # Long-term (200 ticks)
    
    # Volume analysis parameters
    volume_profile_bins: int = 20
    whale_threshold_percentile: float = 0.95  # 95th percentile for whale detection
    large_order_min_size: float = 100000      # Minimum size for large order detection
    
    # Market microstructure parameters
    tick_size: float = 0.01
    min_spread_bps: float = 1.0  # Minimum spread in basis points
    
    # Performance parameters
    max_history_length: int = 1000   # Maximum ticks to keep in memory
    enable_gpu_acceleration: bool = True
    batch_size: int = 32
    
    # Feature selection
    include_whale_features: bool = True
    include_order_flow_features: bool = True
    include_microstructure_features: bool = True
    include_regime_features: bool = True

class MarketData(NamedTuple):
    """Market data structure for efficient processing"""
    timestamp: jnp.ndarray  # Unix timestamps
    price: jnp.ndarray      # OHLC prices [Open, High, Low, Close]
    volume: jnp.ndarray     # Volume data
    bid_price: jnp.ndarray  # Best bid prices
    ask_price: jnp.ndarray  # Best ask prices  
    bid_size: jnp.ndarray   # Best bid sizes
    ask_size: jnp.ndarray   # Best ask sizes
    trades: jnp.ndarray     # Individual trade data [price, size, direction]

class CryptoFeatureEngine:
    """High-performance crypto-specific feature engineering engine"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.feature_history = {}
        self.market_data_buffer = None
        self.whale_activity_history = []
        self.order_flow_history = []
        
        # JIT-compile critical functions for performance
        self._setup_jit_functions()
        
        logger.info(f"🚀 CryptoFeatureEngine initialized with config: {config}")
    
    def _setup_jit_functions(self):
        """Setup JIT-compiled functions for maximum performance"""
        
        @jax.jit
        def compute_technical_indicators(prices, volumes):
            """Compute basic technical indicators"""
            # Price-based indicators
            returns = jnp.diff(jnp.log(prices + 1e-8))
            volatility = jnp.std(returns[-self.config.short_window:])
            
            # Volume-based indicators  
            vwap = jnp.sum(prices * volumes) / jnp.sum(volumes + 1e-8)
            volume_ma = jnp.mean(volumes[-self.config.medium_window:])
            
            # Momentum indicators
            rsi = self._compute_rsi(prices)
            macd = self._compute_macd(prices)
            
            return {
                'returns': returns[-1] if len(returns) > 0 else 0.0,
                'volatility': volatility,
                'vwap': vwap,
                'volume_ma': volume_ma,
                'rsi': rsi,
                'macd': macd
            }
        
        @jax.jit  
        def detect_whale_activity(volumes, prices, trades):
            """Detect whale trading activity patterns"""
            if len(volumes) < self.config.short_window:
                return jnp.array([0.0, 0.0, 0.0])  # [whale_score, large_order_count, volume_concentration]
            
            # Volume concentration analysis
            recent_volumes = volumes[-self.config.short_window:]
            volume_threshold = jnp.percentile(volumes[-self.config.long_window:], 
                                            self.config.whale_threshold_percentile)
            
            whale_volumes = recent_volumes > volume_threshold
            whale_score = jnp.mean(whale_volumes.astype(jnp.float32))
            
            # Large order detection
            large_orders = trades[:, 1] > self.config.large_order_min_size  # trade size column
            large_order_count = jnp.sum(large_orders.astype(jnp.float32))
            
            # Volume concentration in recent periods
            total_volume = jnp.sum(recent_volumes)
            max_volume = jnp.max(recent_volumes)
            volume_concentration = max_volume / (total_volume + 1e-8)
            
            return jnp.array([whale_score, large_order_count, volume_concentration])
        
        @jax.jit
        def analyze_order_flow(bid_prices, ask_prices, bid_sizes, ask_sizes, trades):
            """Analyze order flow and market microstructure"""
            if len(bid_prices) < 2:
                return jnp.zeros(8)  # Return zeros for insufficient data
                
            # Spread analysis
            spreads = ask_prices - bid_prices
            avg_spread = jnp.mean(spreads[-self.config.short_window:])
            spread_volatility = jnp.std(spreads[-self.config.short_window:])
            
            # Order book imbalance
            total_bid_size = jnp.sum(bid_sizes[-self.config.short_window:])
            total_ask_size = jnp.sum(ask_sizes[-self.config.short_window:])
            order_imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size + 1e-8)
            
            # Trade direction analysis
            if len(trades) > 0:
                buy_volume = jnp.sum(jnp.where(trades[:, 2] > 0, trades[:, 1], 0))  # direction > 0 = buy
                sell_volume = jnp.sum(jnp.where(trades[:, 2] < 0, trades[:, 1], 0))  # direction < 0 = sell
                trade_direction_ratio = buy_volume / (buy_volume + sell_volume + 1e-8)
                
                # Trade size distribution
                avg_trade_size = jnp.mean(trades[:, 1])
                large_trade_ratio = jnp.mean((trades[:, 1] > avg_trade_size * 2).astype(jnp.float32))
            else:
                trade_direction_ratio = 0.5
                avg_trade_size = 0.0
                large_trade_ratio = 0.0
            
            # Price impact estimation
            if len(trades) > 1:
                price_changes = jnp.diff(trades[:, 0])  # price changes
                volume_weighted_impact = jnp.sum(jnp.abs(price_changes) * trades[1:, 1]) / jnp.sum(trades[1:, 1] + 1e-8)
            else:
                volume_weighted_impact = 0.0
            
            return jnp.array([
                avg_spread,
                spread_volatility, 
                order_imbalance,
                trade_direction_ratio,
                avg_trade_size,
                large_trade_ratio,
                volume_weighted_impact,
                len(trades)  # trade count
            ])
        
        # Store JIT-compiled functions
        self.compute_technical_indicators_jit = compute_technical_indicators
        self.detect_whale_activity_jit = detect_whale_activity
        self.analyze_order_flow_jit = analyze_order_flow
    
    def _compute_rsi(self, prices, period=14):
        """Compute Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
            
        deltas = jnp.diff(prices)
        gains = jnp.where(deltas > 0, deltas, 0)
        losses = jnp.where(deltas < 0, -deltas, 0)
        
        avg_gain = jnp.mean(gains[-period:])
        avg_loss = jnp.mean(losses[-period:])
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _compute_macd(self, prices, fast=12, slow=26, signal=9):
        """Compute MACD indicator"""
        if len(prices) < slow:
            return 0.0
            
        ema_fast = self._compute_ema(prices, fast)
        ema_slow = self._compute_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        return macd_line
    
    def _compute_ema(self, data, period):
        """Compute Exponential Moving Average"""
        if len(data) < period:
            return jnp.mean(data)
            
        alpha = 2.0 / (period + 1)
        weights = jnp.power(1 - alpha, jnp.arange(len(data))[::-1])
        weights = weights / jnp.sum(weights)
        return jnp.sum(data * weights)
    
    def compute_volume_profile(self, prices: jnp.ndarray, volumes: jnp.ndarray) -> jnp.ndarray:
        """Compute volume profile for price levels"""
        if len(prices) < 2:
            return jnp.zeros(self.config.volume_profile_bins)
            
        price_min, price_max = jnp.min(prices), jnp.max(prices)
        if price_max <= price_min:
            return jnp.zeros(self.config.volume_profile_bins)
            
        # Create price bins
        price_bins = jnp.linspace(price_min, price_max, self.config.volume_profile_bins + 1)
        
        # Assign volumes to bins
        volume_profile = jnp.zeros(self.config.volume_profile_bins)
        for i in range(len(prices)):
            bin_idx = jnp.searchsorted(price_bins[1:], prices[i])
            bin_idx = jnp.clip(bin_idx, 0, self.config.volume_profile_bins - 1)
            volume_profile = volume_profile.at[bin_idx].add(volumes[i])
            
        return volume_profile
    
    def detect_market_regime(self, market_data: MarketData) -> Dict[str, float]:
        """Detect current market regime (trend, volatility, liquidity)"""
        prices = market_data.price[:, 3]  # Close prices
        volumes = market_data.volume
        
        if len(prices) < self.config.medium_window:
            return {
                'trend_strength': 0.0,
                'volatility_regime': 0.5,  # Normal
                'liquidity_regime': 0.5,   # Normal
                'momentum': 0.0
            }
        
        # Trend analysis
        short_ma = jnp.mean(prices[-self.config.short_window:])
        long_ma = jnp.mean(prices[-self.config.long_window:])
        trend_strength = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0
        
        # Volatility regime
        returns = jnp.diff(jnp.log(prices + 1e-8))
        current_vol = jnp.std(returns[-self.config.short_window:])
        historical_vol = jnp.std(returns[-self.config.long_window:])
        volatility_regime = current_vol / (historical_vol + 1e-8)
        
        # Liquidity regime (based on volume and spreads)
        current_volume = jnp.mean(volumes[-self.config.short_window:])
        historical_volume = jnp.mean(volumes[-self.config.long_window:])
        liquidity_regime = current_volume / (historical_volume + 1e-8)
        
        # Momentum
        momentum = jnp.sum(jnp.sign(returns[-self.config.short_window:])) / self.config.short_window
        
        return {
            'trend_strength': float(trend_strength),
            'volatility_regime': float(volatility_regime),
            'liquidity_regime': float(liquidity_regime),
            'momentum': float(momentum)
        }
    
    def process_market_update(self, market_data: MarketData) -> Dict[str, Any]:
        """
        Process new market data and extract comprehensive features
        
        Returns:
            Dictionary containing all computed features for model input
        """
        start_time = jax.device_get(jax.device_array(datetime.now().timestamp()))
        
        try:
            features = {}
            
            # Extract basic data
            prices = market_data.price[:, 3]  # Close prices
            volumes = market_data.volume
            
            # 1. Technical indicators
            if len(prices) >= self.config.short_window:
                tech_indicators = self.compute_technical_indicators_jit(prices, volumes)
                features.update(tech_indicators)
            
            # 2. Whale activity detection
            if self.config.include_whale_features and len(volumes) >= self.config.short_window:
                whale_features = self.detect_whale_activity_jit(volumes, prices, market_data.trades)
                features.update({
                    'whale_score': whale_features[0],
                    'large_order_count': whale_features[1], 
                    'volume_concentration': whale_features[2]
                })
            
            # 3. Order flow analysis
            if self.config.include_order_flow_features:
                order_flow_features = self.analyze_order_flow_jit(
                    market_data.bid_price, market_data.ask_price,
                    market_data.bid_size, market_data.ask_size,
                    market_data.trades
                )
                features.update({
                    'avg_spread': order_flow_features[0],
                    'spread_volatility': order_flow_features[1],
                    'order_imbalance': order_flow_features[2],
                    'trade_direction_ratio': order_flow_features[3],
                    'avg_trade_size': order_flow_features[4],
                    'large_trade_ratio': order_flow_features[5],
                    'price_impact': order_flow_features[6],
                    'trade_count': order_flow_features[7]
                })
            
            # 4. Volume profile
            volume_profile = self.compute_volume_profile(prices[-self.config.medium_window:], 
                                                       volumes[-self.config.medium_window:])
            features['volume_profile'] = volume_profile
            
            # 5. Market regime detection
            if self.config.include_regime_features:
                regime_features = self.detect_market_regime(market_data)
                features.update(regime_features)
            
            # 6. Add metadata
            processing_time = datetime.now().timestamp() - start_time
            features.update({
                'processing_time_ms': processing_time * 1000,
                'data_quality_score': self._compute_data_quality_score(market_data),
                'feature_count': len(features),
                'timestamp': market_data.timestamp[-1] if len(market_data.timestamp) > 0 else 0
            })
            
            # Log performance if needed
            if processing_time > 0.001:  # Log if > 1ms
                logger.warning(f"Feature processing took {processing_time*1000:.2f}ms (target: <1ms)")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in feature processing: {e}")
            return self._get_fallback_features()
    
    def _compute_data_quality_score(self, market_data: MarketData) -> float:
        """Compute data quality score based on completeness and consistency"""
        score = 1.0
        
        # Check for missing data
        if len(market_data.price) == 0:
            score *= 0.0
        if len(market_data.volume) == 0:
            score *= 0.5
        if len(market_data.trades) == 0:
            score *= 0.8
            
        # Check for data consistency
        if len(market_data.price) > 0:
            price_range = jnp.max(market_data.price) - jnp.min(market_data.price)
            if price_range <= 0:
                score *= 0.3  # No price movement indicates poor data
                
        return score
    
    def _get_fallback_features(self) -> Dict[str, Any]:
        """Return fallback features when processing fails"""
        return {
            'returns': 0.0,
            'volatility': 0.0, 
            'vwap': 0.0,
            'volume_ma': 0.0,
            'whale_score': 0.0,
            'order_imbalance': 0.0,
            'trend_strength': 0.0,
            'error': True
        }
    
    def get_feature_vector(self, features: Dict[str, Any], target_size: int = 50) -> jnp.ndarray:
        """Convert feature dictionary to fixed-size vector for model input"""
        
        # Define feature order and default values
        feature_keys = [
            'returns', 'volatility', 'vwap', 'volume_ma', 'rsi', 'macd',
            'whale_score', 'large_order_count', 'volume_concentration',
            'avg_spread', 'spread_volatility', 'order_imbalance', 'trade_direction_ratio',
            'avg_trade_size', 'large_trade_ratio', 'price_impact', 'trade_count',
            'trend_strength', 'volatility_regime', 'liquidity_regime', 'momentum'
        ]
        
        # Extract base features
        vector = []
        for key in feature_keys:
            vector.append(features.get(key, 0.0))
        
        # Add volume profile features
        volume_profile = features.get('volume_profile', jnp.zeros(self.config.volume_profile_bins))
        vector.extend(volume_profile.tolist())
        
        # Pad or truncate to target size
        vector = jnp.array(vector)
        if len(vector) < target_size:
            # Pad with zeros
            padding = jnp.zeros(target_size - len(vector))
            vector = jnp.concatenate([vector, padding])
        elif len(vector) > target_size:
            # Truncate to target size
            vector = vector[:target_size]
            
        return vector
    
    def batch_process_features(self, market_data_batch: List[MarketData]) -> jnp.ndarray:
        """Process multiple market data updates in batch for efficiency"""
        
        @jax.vmap
        def process_single(market_data):
            features = self.process_market_update(market_data)
            return self.get_feature_vector(features)
        
        # Convert to batch format and process
        batch_features = []
        for market_data in market_data_batch:
            features = self.process_market_update(market_data)
            feature_vector = self.get_feature_vector(features)
            batch_features.append(feature_vector)
            
        return jnp.stack(batch_features)

# Factory function
def create_crypto_feature_engine(config_dict: Dict[str, Any]) -> CryptoFeatureEngine:
    """Create a CryptoFeatureEngine from configuration dictionary"""
    config = FeatureConfig(
        short_window=config_dict.get('short_window', 10),
        medium_window=config_dict.get('medium_window', 50),
        long_window=config_dict.get('long_window', 200),
        volume_profile_bins=config_dict.get('volume_profile_bins', 20),
        whale_threshold_percentile=config_dict.get('whale_threshold_percentile', 0.95),
        large_order_min_size=config_dict.get('large_order_min_size', 100000),
        include_whale_features=config_dict.get('include_whale_features', True),
        include_order_flow_features=config_dict.get('include_order_flow_features', True),
        include_microstructure_features=config_dict.get('include_microstructure_features', True),
        include_regime_features=config_dict.get('include_regime_features', True)
    )
    
    return CryptoFeatureEngine(config)

if __name__ == "__main__":
    # Example usage and testing
    config = {
        'short_window': 10,
        'medium_window': 50, 
        'long_window': 200,
        'include_whale_features': True,
        'include_order_flow_features': True
    }
    
    # Create feature engine
    feature_engine = create_crypto_feature_engine(config)
    
    # Create sample market data
    sample_data = MarketData(
        timestamp=jnp.arange(100),
        price=jnp.ones((100, 4)) * 50000 + jnp.random.normal(0, 100, (100, 4)),  # OHLC
        volume=jnp.abs(jnp.random.normal(1000, 200, 100)),
        bid_price=jnp.ones(100) * 49990,
        ask_price=jnp.ones(100) * 50010,
        bid_size=jnp.abs(jnp.random.normal(100, 20, 100)),
        ask_size=jnp.abs(jnp.random.normal(100, 20, 100)),
        trades=jnp.column_stack([
            jnp.ones(50) * 50000,  # prices
            jnp.abs(jnp.random.normal(50, 10, 50)),  # sizes
            jnp.random.choice([-1, 1], 50)  # directions
        ])
    )
    
    print("Testing CryptoFeatureEngine...")
    features = feature_engine.process_market_update(sample_data)
    print(f"Extracted {len(features)} features")
    print(f"Processing time: {features.get('processing_time_ms', 0):.3f}ms")
    
    # Test feature vector conversion
    feature_vector = feature_engine.get_feature_vector(features, target_size=50)
    print(f"Feature vector shape: {feature_vector.shape}")
    print("✅ CryptoFeatureEngine test completed")