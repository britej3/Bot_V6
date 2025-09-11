"""
Tick-Level Feature Engine for Advanced Crypto Futures Scalping
Extracts sophisticated features from tick data including FFT, order flow, and microstructure analysis
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
import math

try:
    from scipy.fft import fft, fftfreq
    from scipy.signal import find_peaks
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
except ImportError as e:
    logging.error(f"Missing required dependencies: {e}")
    raise

from src.config.trading_config import AdvancedTradingConfig

logger = logging.getLogger(__name__)


class TickLevelFeatureEngine:
    """Advanced tick-level feature engineering for crypto futures"""

    def __init__(self, config: AdvancedTradingConfig):
        self.config = config

        # Data buffers
        self.tick_buffer = deque(maxlen=config.tick_buffer_size)
        self.price_buffer = deque(maxlen=config.lookback_window)
        self.volume_buffer = deque(maxlen=config.lookback_window)
        self.order_book_buffer = deque(maxlen=100)

        # Scalers for feature normalization
        self.price_scaler = StandardScaler()
        self.volume_scaler = StandardScaler()
        self.fft_scaler = StandardScaler()
        self.pca_scaler = StandardScaler()

        # PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance

        # Feature extraction state
        self.is_fitted = False
        self.feature_stats = {}

        logger.info("🧠 Tick-Level Feature Engine initialized")

    def process_tick_data(self, tick_data: Dict[str, Any]) -> np.ndarray:
        """Process incoming tick data and extract features"""
        try:
            # Store tick data
            self.tick_buffer.append(tick_data)

            # Extract basic data
            timestamp = tick_data.get('timestamp', datetime.utcnow())
            price = tick_data.get('price', 0.0)
            volume = tick_data.get('quantity', 0.0)
            is_buyer_maker = tick_data.get('is_buyer_maker', True)

            # Update buffers
            self.price_buffer.append(price)
            self.volume_buffer.append(volume)

            # Check if we have enough data
            if len(self.price_buffer) < 50:
                return np.array([])

            # Extract comprehensive features
            features = []

            # Basic price and volume features
            price_features = self._extract_price_features()
            volume_features = self._extract_volume_features()
            features.extend(price_features)
            features.extend(volume_features)

            # Technical indicators
            technical_features = self._extract_technical_features()
            features.extend(technical_features)

            # FFT-based features
            fft_features = self._extract_fft_features()
            features.extend(fft_features)

            # Order flow features
            order_flow_features = self._extract_order_flow_features(tick_data)
            features.extend(order_flow_features)

            # Microstructure features
            microstructure_features = self._extract_microstructure_features()
            features.extend(microstructure_features)

            # Cyclical features
            cyclical_features = self._extract_cyclical_features(timestamp)
            features.extend(cyclical_features)

            # Volatility features
            volatility_features = self._extract_volatility_features()
            features.extend(volatility_features)

            # Momentum features
            momentum_features = self._extract_momentum_features()
            features.extend(momentum_features)

            return np.array(features)

        except Exception as e:
            logger.error(f"❌ Failed to process tick data: {e}")
            return np.array([])

    def _extract_price_features(self) -> List[float]:
        """Extract price-based features"""
        prices = np.array(list(self.price_buffer))

        features = [
            prices[-1],  # Current price
            np.mean(prices),  # Mean price
            np.std(prices),  # Price volatility
            np.max(prices) - np.min(prices),  # Price range
            np.percentile(prices, 75) - np.percentile(prices, 25),  # IQR
            self._calculate_price_skewness(prices),  # Price skewness
            self._calculate_price_kurtosis(prices),  # Price kurtosis
        ]

        return features

    def _extract_volume_features(self) -> List[float]:
        """Extract volume-based features"""
        volumes = np.array(list(self.volume_buffer))

        features = [
            volumes[-1],  # Current volume
            np.mean(volumes),  # Mean volume
            np.std(volumes),  # Volume volatility
            np.sum(volumes),  # Total volume
            self._calculate_volume_imbalance(volumes),  # Volume imbalance
        ]

        return features

    def _extract_technical_features(self) -> List[float]:
        """Extract technical indicator features"""
        prices = np.array(list(self.price_buffer))

        if len(prices) < 20:
            return [0.0] * 10

        features = []

        # Simple Moving Averages
        features.append(np.mean(prices[-5:]))  # 5-period SMA
        features.append(np.mean(prices[-10:]))  # 10-period SMA
        features.append(np.mean(prices[-20:]))  # 20-period SMA

        # Exponential Moving Averages
        features.append(self._calculate_ema(prices, 12))  # 12-period EMA
        features.append(self._calculate_ema(prices, 26))  # 26-period EMA

        # RSI
        features.append(self._calculate_rsi(prices, 14))

        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(prices)
        features.extend([macd_line, signal_line, histogram])

        return features

    def _extract_fft_features(self) -> List[float]:
        """Extract FFT-based features for cycle detection"""
        prices = np.array(list(self.price_buffer))

        if len(prices) < self.config.fft_components * 2:
            return [0.0] * self.config.fft_components

        try:
            # Apply FFT
            fft_values = fft(prices)
            fft_freqs = fftfreq(len(prices))

            # Get magnitude and phase
            magnitude = np.abs(fft_values)
            phase = np.angle(fft_values)

            # Extract dominant frequencies
            features = []

            # Sort by magnitude and take top components
            sorted_indices = np.argsort(magnitude)[::-1]
            for i in range(min(self.config.fft_components, len(sorted_indices))):
                idx = sorted_indices[i]
                features.append(magnitude[idx])
                features.append(phase[idx])
                features.append(fft_freqs[idx])

            return features

        except Exception as e:
            logger.warning(f"FFT feature extraction failed: {e}")
            return [0.0] * self.config.fft_components

    def _extract_order_flow_features(self, tick_data: Dict[str, Any]) -> List[float]:
        """
        Extracts enhanced order flow and whale activity features.
        
        - Order Flow Imbalance (OFI): Captures the net difference between buying and selling pressure.
        - Whale Activity: Detects unusually large trades that may signal significant market moves.
        """
        features = []
        
        # --- Enhanced Order Flow Imbalance (OFI) ---
        # This implementation uses a more robust calculation over multiple time windows.
        # For ultra-low latency, these calculations should be optimized, potentially in a compiled language.
        windows = [10, 50, 200]
        full_buffer = list(self.tick_buffer)

        for window in windows:
            if len(full_buffer) >= window:
                recent_ticks = full_buffer[-window:]
                buy_volume = sum(t.get('quantity', 0.0) for t in recent_ticks if t.get('is_buyer_maker', True))
                sell_volume = sum(t.get('quantity', 0.0) for t in recent_ticks if not t.get('is_buyer_maker', True))
                
                ofi = buy_volume - sell_volume
                buy_sell_ratio = buy_volume / (sell_volume + 1e-9)
                
                features.extend([ofi, buy_sell_ratio])
            else:
                # Pad with zeros if not enough data
                features.extend([0.0, 1.0])

        # --- Whale Activity Detection ---
        # Detects trades significantly larger than the recent average.
        quantity = tick_data.get('quantity', 0.0)
        if len(self.volume_buffer) > 20:
            recent_volumes = list(self.volume_buffer)[-20:]
            avg_volume = np.mean(recent_volumes)
            std_volume = np.std(recent_volumes)
            
            # A "whale trade" is defined as a trade several standard deviations above the mean.
            whale_threshold = avg_volume + (self.config.get("whale_detection_std_dev", 3) * std_volume)
            
            is_whale_trade = 1.0 if quantity > whale_threshold else 0.0
            whale_trade_size_diff = max(0, quantity - whale_threshold)
            
            features.extend([is_whale_trade, whale_trade_size_diff])
        else:
            features.extend([0.0, 0.0])
            
        return features

    def _extract_microstructure_features(self) -> List[float]:
        """Extract market microstructure features"""
        features = []

        if len(self.tick_buffer) < 20:
            return [0.0] * 5

        # Price impact analysis
        recent_ticks = list(self.tick_buffer)[-20:]
        price_changes = []
        volume_changes = []

        for i in range(1, len(recent_ticks)):
            prev_price = recent_ticks[i-1].get('price', 0)
            curr_price = recent_ticks[i].get('price', 0)
            if prev_price > 0:
                price_changes.append(abs(curr_price - prev_price) / prev_price)

            prev_volume = recent_ticks[i-1].get('quantity', 0)
            curr_volume = recent_ticks[i].get('quantity', 0)
            volume_changes.append(curr_volume)

        if price_changes and volume_changes:
            # Price impact coefficient
            price_impact = np.corrcoef(price_changes, volume_changes)[0, 1]
            features.append(price_impact if not np.isnan(price_impact) else 0.0)

            # Average price change
            features.append(np.mean(price_changes))

            # Price volatility
            features.append(np.std(price_changes))

            # Volume volatility
            features.append(np.std(volume_changes))

            # Hurst exponent approximation
            hurst = self._calculate_hurst_exponent(list(self.price_buffer))
            features.append(hurst)

        return features

    def _extract_cyclical_features(self, timestamp: datetime) -> List[float]:
        """Extract cyclical time-based features"""
        features = []

        # Time of day features
        hour = timestamp.hour
        minute = timestamp.minute

        features.extend([
            np.sin(2 * np.pi * hour / 24),  # Daily cycle
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * minute / 60),  # Hourly cycle
            np.cos(2 * np.pi * minute / 60),
            np.sin(2 * np.pi * timestamp.second / 60),  # Minute cycle
            np.cos(2 * np.pi * timestamp.second / 60),
        ])

        return features

    def _extract_volatility_features(self) -> List[float]:
        """Extract volatility-based features"""
        prices = np.array(list(self.price_buffer))

        if len(prices) < 20:
            return [0.0] * 5

        # Calculate returns
        returns = np.diff(np.log(prices))

        features = [
            np.std(returns),  # Overall volatility
            np.mean(np.abs(returns)),  # Mean absolute return
            self._calculate_parkinson_volatility(prices),  # Parkinson volatility
            self._calculate_garman_klass_volatility(prices),  # Garman-Klass volatility
            self._calculate_realized_volatility(returns),  # Realized volatility
        ]

        return features

    def _extract_momentum_features(self) -> List[float]:
        """Extract momentum-based features"""
        prices = np.array(list(self.price_buffer))

        if len(prices) < 20:
            return [0.0] * 5

        features = []

        # Price momentum
        for period in [5, 10, 20]:
            if len(prices) >= period:
                momentum = (prices[-1] - prices[-period]) / prices[-period]
                features.append(momentum)
            else:
                features.append(0.0)

        # Rate of change
        for period in [5, 10]:
            if len(prices) >= period:
                roc = (prices[-1] - prices[-period]) / prices[-period] * 100
                features.append(roc)
            else:
                features.append(0.0)

        return features

    def fit_scalers(self, features: np.ndarray):
        """Fit feature scalers"""
        try:
            # Fit scalers
            self.price_scaler.fit(features)
            self.volume_scaler.fit(features)
            self.fft_scaler.fit(features)
            self.pca_scaler.fit(features)

            # Fit PCA
            self.pca.fit(features)

            self.is_fitted = True
            logger.info("✅ Feature scalers fitted")

        except Exception as e:
            logger.error(f"❌ Failed to fit scalers: {e}")

    def transform_features(self, raw_features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scalers"""
        try:
            if not self.is_fitted:
                return raw_features

            # Apply scaling
            scaled_features = self.price_scaler.transform(raw_features.reshape(1, -1))

            # Apply PCA if configured
            if hasattr(self.pca, 'components_'):
                pca_features = self.pca.transform(scaled_features)
                return pca_features.flatten()
            else:
                return scaled_features.flatten()

        except Exception as e:
            logger.error(f"❌ Failed to transform features: {e}")
            return raw_features

    def update_orderbook(self, orderbook_data: Dict[str, Any]):
        """Update order book buffer"""
        self.order_book_buffer.append(orderbook_data)

    def update_trades(self, trade_data: Dict[str, Any]):
        """Update trade buffer"""
        # This is handled in process_tick_data, but can be extended for additional trade analysis
        pass

    # Helper methods for calculations

    def _calculate_price_skewness(self, prices: np.ndarray) -> float:
        """Calculate price distribution skewness"""
        if len(prices) < 3:
            return 0.0
        return float(pd.Series(prices).skew())

    def _calculate_price_kurtosis(self, prices: np.ndarray) -> float:
        """Calculate price distribution kurtosis"""
        if len(prices) < 4:
            return 0.0
        return float(pd.Series(prices).kurtosis())

    def _calculate_volume_imbalance(self, volumes: np.ndarray) -> float:
        """Calculate volume imbalance"""
        if len(volumes) < 2:
            return 0.0
        return float(np.mean(volumes[-10:]) - np.mean(volumes[-20:-10]) if len(volumes) >= 20 else 0.0)

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)

        ema = prices[0]
        multiplier = 2 / (period + 1)

        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0

        returns = np.diff(prices)
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0

        fast_ema = self._calculate_ema(prices, fast)
        slow_ema = self._calculate_ema(prices, slow)
        macd_line = fast_ema - slow_ema

        # For signal line, we'd need historical MACD values
        # This is a simplified version
        signal_line = macd_line  # Placeholder
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_hurst_exponent(self, prices: List[float]) -> float:
        """Calculate Hurst exponent for long-range dependence"""
        if len(prices) < 100:
            return 0.5  # Random walk

        try:
            # Simplified Hurst exponent calculation
            prices = np.array(prices)
            returns = np.diff(np.log(prices))

            # Calculate R/S statistic for different time lags
            rs_values = []
            max_lag = min(20, len(returns) // 10)

            for lag in range(2, max_lag + 1):
                chunks = len(returns) // lag
                if chunks < 2:
                    continue

                rs_chunk = []
                for i in range(chunks):
                    chunk = returns[i*lag:(i+1)*lag]
                    if len(chunk) > 0:
                        mean = np.mean(chunk)
                        deviations = chunk - mean
                        cumulative = np.cumsum(deviations)
                        r = np.max(cumulative) - np.min(cumulative)
                        s = np.std(chunk)
                        if s > 0:
                            rs_chunk.append(r / s)

                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))

            if len(rs_values) >= 2:
                lags = np.arange(2, len(rs_values) + 2)
                log_rs = np.log(rs_values)
                log_lags = np.log(lags)

                # Linear regression for Hurst exponent
                slope = np.polyfit(log_lags, log_rs, 1)[0]
                return slope

        except Exception as e:
            logger.warning(f"Hurst exponent calculation failed: {e}")

        return 0.5

    def _calculate_parkinson_volatility(self, prices: np.ndarray) -> float:
        """Calculate Parkinson volatility"""
        if len(prices) < 2:
            return 0.0

        log_returns = np.diff(np.log(prices))
        return np.sqrt(np.mean(log_returns ** 2))

    def _calculate_garman_klass_volatility(self, prices: np.ndarray) -> float:
        """Calculate Garman-Klass volatility (simplified)"""
        if len(prices) < 2:
            return 0.0

        log_returns = np.diff(np.log(prices))
        return np.sqrt(np.mean(log_returns ** 2))

    def _calculate_realized_volatility(self, returns: np.ndarray) -> float:
        """Calculate realized volatility"""
        if len(returns) == 0:
            return 0.0

        return np.sqrt(np.sum(returns ** 2))

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the feature engineering process"""
        return {
            'is_fitted': self.is_fitted,
            'tick_buffer_size': len(self.tick_buffer),
            'price_buffer_size': len(self.price_buffer),
            'feature_stats': self.feature_stats,
            'config': {
                'lookback_window': self.config.lookback_window,
                'fft_components': self.config.fft_components,
                'order_book_levels': self.config.order_book_levels
            }
        }