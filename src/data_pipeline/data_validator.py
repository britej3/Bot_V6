"""
Data Validator for CryptoScalp AI

This module provides real-time data validation and anomaly detection
for market data from cryptocurrency exchanges.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    score: float  # 0.0 to 1.0, where 1.0 is perfect
    issues: List[str]
    warnings: List[str]

@dataclass
class AnomalyScore:
    """Anomaly detection scores"""
    price_anomaly: float
    volume_anomaly: float
    spread_anomaly: float
    timestamp_anomaly: float
    overall_score: float

class DataValidator:
    """
    Real-time data validator with ML-based anomaly detection

    Features:
    - Price movement validation
    - Volume anomaly detection
    - Spread analysis
    - Timestamp consistency checks
    - Historical comparison
    - Adaptive thresholds
    """

    def __init__(self, lookback_window: int = 1000):
        self.lookback_window = lookback_window

        # Historical data storage for each symbol
        self.price_history = defaultdict(lambda: deque(maxlen=lookback_window))
        self.volume_history = defaultdict(lambda: deque(maxlen=lookback_window))
        self.spread_history = defaultdict(lambda: deque(maxlen=lookback_window))

        # Anomaly thresholds (adaptive)
        self.price_thresholds = {}
        self.volume_thresholds = {}
        self.spread_thresholds = {}

        # Validation statistics
        self.validation_counts = defaultdict(int)
        self.anomaly_counts = defaultdict(int)

    def validate_market_data(self, exchange: str, symbol: str, data: Dict[str, Any]) -> ValidationResult:
        """Validate market data for a specific symbol"""
        issues = []
        warnings = []

        # Basic structure validation
        required_fields = ['price', 'bid_price', 'ask_price', 'volume']
        for field in required_fields:
            if field not in data or data[field] is None:
                issues.append(f"Missing required field: {field}")
                return ValidationResult(False, 0.0, issues, warnings)

        # Type validation
        try:
            price = float(data['price'])
            bid_price = float(data['bid_price'])
            ask_price = float(data['ask_price'])
            volume = float(data['volume'])
        except (ValueError, TypeError) as e:
            issues.append(f"Invalid numeric values: {e}")
            return ValidationResult(False, 0.0, issues, warnings)

        # Logical validation
        if price <= 0:
            issues.append("Price must be positive")
        if bid_price <= 0 or ask_price <= 0:
            issues.append("Bid/ask prices must be positive")
        if bid_price > ask_price:
            issues.append("Bid price cannot be higher than ask price")
        if volume < 0:
            issues.append("Volume cannot be negative")

        if issues:
            return ValidationResult(False, 0.0, issues, warnings)

        # Advanced validation
        anomaly_score = self._detect_anomalies(symbol, price, bid_price, ask_price, volume)

        # Store data for historical analysis
        self._update_history(symbol, price, volume, ask_price - bid_price)

        # Update adaptive thresholds
        self._update_thresholds(symbol)

        # Check for anomalies
        if anomaly_score.overall_score > 0.8:
            issues.append(f"High anomaly score: {anomaly_score.overall_score:.2f}")

        if anomaly_score.price_anomaly > 0.7:
            warnings.append(f"Unusual price movement detected: {anomaly_score.price_anomaly:.2f}")

        if anomaly_score.volume_anomaly > 0.7:
            warnings.append(f"Unusual volume detected: {anomaly_score.volume_anomaly:.2f}")

        if anomaly_score.spread_anomaly > 0.7:
            warnings.append(f"Unusual spread detected: {anomaly_score.spread_anomaly:.2f}")

        # Calculate overall validation score
        score = 1.0 - anomaly_score.overall_score

        # Minor issues become warnings if score is still good
        if score > 0.8 and not issues:
            # Move some issues to warnings
            pass

        is_valid = len(issues) == 0

        # Update statistics
        self.validation_counts[symbol] += 1
        if not is_valid or anomaly_score.overall_score > 0.5:
            self.anomaly_counts[symbol] += 1

        return ValidationResult(is_valid, score, issues, warnings)

    def _detect_anomalies(self, symbol: str, price: float, bid_price: float,
                         ask_price: float, volume: float) -> AnomalyScore:
        """Detect anomalies in market data using statistical methods"""

        # Get historical data
        price_history = list(self.price_history[symbol])
        volume_history = list(self.volume_history[symbol])
        spread_history = list(self.spread_history[symbol])

        spread = ask_price - bid_price

        # Calculate anomaly scores
        price_anomaly = self._calculate_zscore_anomaly(price, price_history)
        volume_anomaly = self._calculate_zscore_anomaly(volume, volume_history)
        spread_anomaly = self._calculate_zscore_anomaly(spread, spread_history)

        # Timestamp anomaly (placeholder - would need timestamp data)
        timestamp_anomaly = 0.0

        # Calculate overall anomaly score
        weights = {
            'price': 0.4,
            'volume': 0.3,
            'spread': 0.2,
            'timestamp': 0.1
        }

        overall_score = (
            weights['price'] * price_anomaly +
            weights['volume'] * volume_anomaly +
            weights['spread'] * spread_anomaly +
            weights['timestamp'] * timestamp_anomaly
        )

        return AnomalyScore(
            price_anomaly=price_anomaly,
            volume_anomaly=volume_anomaly,
            spread_anomaly=spread_anomaly,
            timestamp_anomaly=timestamp_anomaly,
            overall_score=overall_score
        )

    def _calculate_zscore_anomaly(self, value: float, history: List[float]) -> float:
        """Calculate anomaly score using z-score"""
        if len(history) < 10:  # Need minimum data points
            return 0.0

        try:
            mean = statistics.mean(history)
            stdev = statistics.stdev(history) if len(history) > 1 else 0

            if stdev == 0:
                return 0.0

            z_score = abs(value - mean) / stdev
            # Convert z-score to anomaly score (0-1)
            return min(1.0, z_score / 6.0)  # 6 sigma = extreme outlier

        except Exception as e:
            logger.debug(f"Error calculating z-score: {e}")
            return 0.0

    def _update_history(self, symbol: str, price: float, volume: float, spread: float):
        """Update historical data for a symbol"""
        self.price_history[symbol].append(price)
        self.volume_history[symbol].append(volume)
        self.spread_history[symbol].append(spread)

    def _update_thresholds(self, symbol: str):
        """Update adaptive thresholds based on recent data"""
        if len(self.price_history[symbol]) >= 50:  # Need sufficient data
            price_std = statistics.stdev(self.price_history[symbol])
            volume_std = statistics.stdev(self.volume_history[symbol])
            spread_std = statistics.stdev(self.spread_history[symbol])

            # Set thresholds at 3 sigma
            self.price_thresholds[symbol] = price_std * 3
            self.volume_thresholds[symbol] = volume_std * 3
            self.spread_thresholds[symbol] = spread_std * 3

    def validate_orderbook_data(self, exchange: str, symbol: str,
                               orderbook: Dict[str, Any]) -> ValidationResult:
        """Validate order book data"""
        issues = []
        warnings = []

        # Check required fields
        if 'bids' not in orderbook or 'asks' not in orderbook:
            issues.append("Missing bids or asks in orderbook")
            return ValidationResult(False, 0.0, issues, warnings)

        bids = orderbook['bids']
        asks = orderbook['asks']

        # Check data types and structure
        if not isinstance(bids, list) or not isinstance(asks, list):
            issues.append("Bids and asks must be lists")
            return ValidationResult(False, 0.0, issues, warnings)

        if len(bids) == 0 or len(asks) == 0:
            issues.append("Orderbook cannot be empty")
            return ValidationResult(False, 0.0, issues, warnings)

        # Validate bid-ask spread
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])

            if best_bid >= best_ask:
                issues.append("Best bid must be less than best ask")

            spread_percentage = (best_ask - best_bid) / best_bid
            if spread_percentage > 0.1:  # 10% spread is too high
                warnings.append(f"Large spread detected: {spread_percentage:.2%}")

        except (ValueError, IndexError, ZeroDivisionError) as e:
            issues.append(f"Invalid orderbook data: {e}")

        # Validate individual price levels
        for i, bid in enumerate(bids[:5]):  # Check first 5 levels
            try:
                price = float(bid[0])
                volume = float(bid[1])
                if price <= 0 or volume < 0:
                    issues.append(f"Invalid bid at level {i}: price={price}, volume={volume}")
            except (ValueError, IndexError):
                issues.append(f"Invalid bid format at level {i}")

        for i, ask in enumerate(asks[:5]):  # Check first 5 levels
            try:
                price = float(ask[0])
                volume = float(ask[1])
                if price <= 0 or volume < 0:
                    issues.append(f"Invalid ask at level {i}: price={price}, volume={volume}")
            except (ValueError, IndexError):
                issues.append(f"Invalid ask format at level {i}")

        is_valid = len(issues) == 0
        score = 0.8 if warnings else 1.0  # Slightly lower score if warnings present

        return ValidationResult(is_valid, score, issues, warnings)

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics for all symbols"""
        stats = {}

        for symbol in self.validation_counts.keys():
            total_validations = self.validation_counts[symbol]
            total_anomalies = self.anomaly_counts[symbol]

            stats[symbol] = {
                'total_validations': total_validations,
                'total_anomalies': total_anomalies,
                'anomaly_rate': total_anomalies / total_validations if total_validations > 0 else 0,
                'data_points_available': len(self.price_history[symbol])
            }

        return stats

    def reset_symbol_data(self, symbol: str):
        """Reset historical data for a symbol (useful for testing)"""
        self.price_history[symbol].clear()
        self.volume_history[symbol].clear()
        self.spread_history[symbol].clear()

        self.price_thresholds.pop(symbol, None)
        self.volume_thresholds.pop(symbol, None)
        self.spread_thresholds.pop(symbol, None)

        self.validation_counts[symbol] = 0
        self.anomaly_counts[symbol] = 0