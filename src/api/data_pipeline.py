"""
Data Pipeline for Tick Data

This module provides data normalization, validation, and processing pipeline
for tick data from various cryptocurrency exchanges.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from src.api.models import TickDataPoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for data validation issues"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    corrected_value: Optional[Any] = None


@dataclass
class NormalizationResult:
    """Result of data normalization"""
    original_data: Dict[str, Any]
    normalized_data: Dict[str, Any]
    changes_made: List[str]
    warnings: List[str]


class TickDataValidator:
    """Validator for tick data"""

    def __init__(self):
        self.validation_rules = {
            'price': self._validate_price,
            'volume': self._validate_volume,
            'timestamp': self._validate_timestamp,
            'symbol': self._validate_symbol,
            'exchange': self._validate_exchange
        }

    def validate_tick_data(self, tick_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate tick data against all rules"""
        results = []

        for field, validator in self.validation_rules.items():
            if field in tick_data:
                result = validator(tick_data[field], tick_data)
                results.append(result)

        return results

    def _validate_price(self, price: Any, tick_data: Dict[str, Any]) -> ValidationResult:
        """Validate price field"""
        try:
            price_float = float(price)
            if price_float <= 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Price must be positive",
                    details={"price": price}
                )
            elif price_float > 10000000:  # $10M upper bound
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="Price seems unusually high",
                    details={"price": price_float}
                )
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message="Price is valid"
            )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Price must be a valid number",
                details={"price": price}
            )

    def _validate_volume(self, volume: Any, tick_data: Dict[str, Any]) -> ValidationResult:
        """Validate volume field"""
        try:
            volume_float = float(volume)
            if volume_float < 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Volume cannot be negative",
                    details={"volume": volume}
                )
            elif volume_float > 1000000:  # 1M units upper bound
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="Volume seems unusually high",
                    details={"volume": volume_float}
                )
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message="Volume is valid"
            )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Volume must be a valid number",
                details={"volume": volume}
            )

    def _validate_timestamp(self, timestamp: Any, tick_data: Dict[str, Any]) -> ValidationResult:
        """Validate timestamp field"""
        try:
            ts_float = float(timestamp)
            now = time.time()

            # Check if timestamp is in future (allow 1 second tolerance)
            if ts_float > now + 1:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Timestamp cannot be in the future",
                    details={"timestamp": ts_float, "now": now}
                )

            # Check if timestamp is too old (older than 1 day)
            if ts_float < now - 86400:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message="Timestamp is very old",
                    details={"timestamp": ts_float, "now": now}
                )

            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message="Timestamp is valid"
            )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Timestamp must be a valid number",
                details={"timestamp": timestamp}
            )

    def _validate_symbol(self, symbol: Any, tick_data: Dict[str, Any]) -> ValidationResult:
        """Validate symbol field"""
        if not isinstance(symbol, str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Symbol must be a string",
                details={"symbol": symbol}
            )

        if not symbol or len(symbol.strip()) == 0:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Symbol cannot be empty",
                details={"symbol": symbol}
            )

        if len(symbol) > 20:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Symbol is unusually long",
                details={"symbol": symbol}
            )

        # Check format (should be BASE/QUOTE)
        if '/' not in symbol:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Symbol must be in format BASE/QUOTE",
                details={"symbol": symbol}
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.WARNING,
            message="Symbol is valid"
        )

    def _validate_exchange(self, exchange: Any, tick_data: Dict[str, Any]) -> ValidationResult:
        """Validate exchange field"""
        valid_exchanges = {
            'binance', 'okx', 'bybit', 'coinbase', 'kraken',
            'bitfinex', 'huobi', 'kucoin', 'gate'
        }

        if not isinstance(exchange, str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Exchange must be a string",
                details={"exchange": exchange}
            )

        if exchange.lower() not in valid_exchanges:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message="Exchange is not in supported list",
                details={"exchange": exchange, "valid_exchanges": list(valid_exchanges)}
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.WARNING,
            message="Exchange is valid"
        )


class TickDataNormalizer:
    """Normalizer for tick data from different exchanges"""

    def __init__(self):
        self.exchange_normalizers = {
            'binance': self._normalize_binance,
            'okx': self._normalize_okx,
            'bybit': self._normalize_bybit,
            'coinbase': self._normalize_coinbase,
            'kraken': self._normalize_kraken
        }

    def normalize(self, exchange: str, raw_data: Dict[str, Any]) -> NormalizationResult:
        """Normalize tick data from specific exchange"""
        original_data = raw_data.copy()
        changes_made = []
        warnings = []

        # Get appropriate normalizer
        normalizer = self.exchange_normalizers.get(exchange.lower(), self._normalize_generic)

        try:
            normalized_data = normalizer(raw_data)
            changes_made.append(f"Applied {exchange} normalization")
        except Exception as e:
            logger.error(f"Error normalizing data from {exchange}: {e}")
            warnings.append(f"Normalization failed: {str(e)}")
            normalized_data = self._normalize_generic(raw_data)
            changes_made.append("Applied generic normalization due to error")

        return NormalizationResult(
            original_data=original_data,
            normalized_data=normalized_data,
            changes_made=changes_made,
            warnings=warnings
        )

    def _normalize_binance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Binance tick data"""
        normalized = {
            'symbol': data.get('s', data.get('symbol', '')),
            'price': float(data.get('c', data.get('price', 0))),
            'volume': float(data.get('v', data.get('volume', 0))),
            'timestamp': float(data.get('E', data.get('timestamp', time.time()))) / 1000,
            'exchange_timestamp': float(data.get('E', data.get('timestamp', time.time()))),
            'bid_price': float(data.get('b', data.get('bid_price', 0))),
            'ask_price': float(data.get('a', data.get('ask_price', 0)))
        }
        return normalized

    def _normalize_okx(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize OKX tick data"""
        normalized = {
            'symbol': data.get('instId', data.get('symbol', '')),
            'price': float(data.get('last', data.get('price', 0))),
            'volume': float(data.get('vol24h', data.get('volume', 0))),
            'timestamp': time.time(),  # OKX doesn't provide timestamp in ticker
            'exchange_timestamp': time.time(),
            'bid_price': float(data.get('bidPx', data.get('bid_price', 0))),
            'ask_price': float(data.get('askPx', data.get('ask_price', 0)))
        }
        return normalized

    def _normalize_bybit(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Bybit tick data"""
        normalized = {
            'symbol': data.get('symbol', ''),
            'price': float(data.get('lastPrice', data.get('price', 0))),
            'volume': float(data.get('volume24h', data.get('volume', 0))),
            'timestamp': float(data.get('timestampE6', time.time())) / 1000000,
            'exchange_timestamp': float(data.get('timestampE6', time.time())),
            'bid_price': float(data.get('bid1Price', data.get('bid_price', 0))),
            'ask_price': float(data.get('ask1Price', data.get('ask_price', 0)))
        }
        return normalized

    def _normalize_coinbase(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Coinbase tick data"""
        normalized = {
            'symbol': data.get('product_id', data.get('symbol', '')),
            'price': float(data.get('price', 0)),
            'volume': float(data.get('volume_24h', data.get('volume', 0))),
            'timestamp': time.time(),  # Coinbase provides time in different format
            'exchange_timestamp': time.time(),
            'bid_price': float(data.get('best_bid', data.get('bid_price', 0))),
            'ask_price': float(data.get('best_ask', data.get('ask_price', 0)))
        }
        return normalized

    def _normalize_kraken(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Kraken tick data"""
        normalized = {
            'symbol': data.get('pair', data.get('symbol', '')),
            'price': float(data.get('c', data.get('price', 0))),
            'volume': float(data.get('v', data.get('volume', 0))),
            'timestamp': time.time(),  # Kraken timestamp format varies
            'exchange_timestamp': time.time(),
            'bid_price': float(data.get('b', data.get('bid_price', 0))),
            'ask_price': float(data.get('a', data.get('ask_price', 0)))
        }
        return normalized

    def _normalize_generic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic normalization for unknown exchanges"""
        normalized = {
            'symbol': str(data.get('symbol', '')),
            'price': float(data.get('price', 0)),
            'volume': float(data.get('volume', 0)),
            'timestamp': float(data.get('timestamp', time.time())),
            'exchange_timestamp': float(data.get('exchange_timestamp', time.time())),
            'bid_price': float(data.get('bid_price', 0)),
            'ask_price': float(data.get('ask_price', 0))
        }
        return normalized


class DataQualityChecker:
    """Check data quality and detect anomalies"""

    def __init__(self):
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[float]] = {}
        self.history_window = 100  # Keep last 100 data points

    def check_quality(self, tick: TickDataPoint) -> List[str]:
        """Check data quality and return warnings"""
        warnings = []

        # Price anomaly detection
        price_warnings = self._check_price_anomaly(tick)
        warnings.extend(price_warnings)

        # Volume anomaly detection
        volume_warnings = self._check_volume_anomaly(tick)
        warnings.extend(volume_warnings)

        # Update history
        self._update_history(tick)

        return warnings

    def _check_price_anomaly(self, tick: TickDataPoint) -> List[str]:
        """Check for price anomalies"""
        warnings = []
        symbol = tick.symbol
        price = tick.price

        if symbol not in self.price_history:
            return warnings

        prices = self.price_history[symbol]
        if len(prices) < 5:  # Need minimum history
            return warnings

        # Calculate statistics
        mean_price = np.mean(prices)
        std_price = np.std(prices)

        if std_price == 0:
            return warnings

        # Check for price spike (> 3 standard deviations)
        z_score = abs(price - mean_price) / std_price
        if z_score > 3:
            warnings.append(".2f")

        # Check for price drop (> 50% drop)
        if len(prices) > 0:
            last_price = prices[-1]
            price_change_pct = abs(price - last_price) / last_price * 100
            if price_change_pct > 50:
                warnings.append(".1f")

        return warnings

    def _check_volume_anomaly(self, tick: TickDataPoint) -> List[str]:
        """Check for volume anomalies"""
        warnings = []
        symbol = tick.symbol
        volume = tick.volume

        if symbol not in self.volume_history:
            return warnings

        volumes = self.volume_history[symbol]
        if len(volumes) < 5:
            return warnings

        # Calculate statistics
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        if std_volume == 0 or mean_volume == 0:
            return warnings

        # Check for volume spike (> 5 standard deviations)
        z_score = abs(volume - mean_volume) / std_volume
        if z_score > 5:
            warnings.append(".0f")

        # Check for zero volume
        if volume == 0:
            warnings.append(f"Zero volume detected for {symbol}")

        return warnings

    def _update_history(self, tick: TickDataPoint):
        """Update price and volume history"""
        symbol = tick.symbol

        # Update price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(tick.price)
        if len(self.price_history[symbol]) > self.history_window:
            self.price_history[symbol].pop(0)

        # Update volume history
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        self.volume_history[symbol].append(tick.volume)
        if len(self.volume_history[symbol]) > self.history_window:
            self.volume_history[symbol].pop(0)


class TickDataPipeline:
    """Complete data processing pipeline for tick data"""

    def __init__(self):
        self.validator = TickDataValidator()
        self.normalizer = TickDataNormalizer()
        self.quality_checker = DataQualityChecker()

    def process_raw_data(
        self,
        exchange: str,
        raw_data: Dict[str, Any]
    ) -> Tuple[Optional[TickDataPoint], List[str], List[str]]:
        """
        Process raw tick data through the complete pipeline

        Returns:
            Tuple of (processed_tick, warnings, errors)
        """
        warnings = []
        errors = []

        try:
            # Step 1: Normalize data
            normalization_result = self.normalizer.normalize(exchange, raw_data)
            warnings.extend(normalization_result.warnings)

            normalized_data = normalization_result.normalized_data

            # Step 2: Validate data
            validation_results = self.validator.validate_tick_data(normalized_data)

            for result in validation_results:
                if not result.is_valid:
                    if result.severity == ValidationSeverity.ERROR:
                        errors.append(result.message)
                    elif result.severity == ValidationSeverity.WARNING:
                        warnings.append(result.message)
                elif result.severity == ValidationSeverity.WARNING:
                    warnings.append(result.message)

            # Stop processing if critical errors found
            if errors:
                return None, warnings, errors

            # Step 3: Create TickDataPoint
            tick = TickDataPoint(
                timestamp=normalized_data['timestamp'],
                symbol=normalized_data['symbol'],
                price=normalized_data['price'],
                volume=normalized_data['volume'],
                source_exchange=exchange,
                exchange_timestamp=normalized_data.get('exchange_timestamp')
            )

            # Step 4: Quality check
            quality_warnings = self.quality_checker.check_quality(tick)
            warnings.extend(quality_warnings)

            return tick, warnings, errors

        except Exception as e:
            error_msg = f"Pipeline processing error: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None, warnings, errors

    def process_multiple_ticks(
        self,
        exchange: str,
        raw_ticks: List[Dict[str, Any]]
    ) -> Tuple[List[TickDataPoint], List[str], List[str]]:
        """Process multiple ticks"""
        processed_ticks = []
        all_warnings = []
        all_errors = []

        for raw_tick in raw_ticks:
            tick, warnings, errors = self.process_raw_data(exchange, raw_tick)

            if tick:
                processed_ticks.append(tick)
            all_warnings.extend(warnings)
            all_errors.extend(errors)

        return processed_ticks, all_warnings, all_errors

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'symbols_tracked': len(self.quality_checker.price_history),
            'total_price_history_points': sum(len(prices) for prices in self.quality_checker.price_history.values()),
            'total_volume_history_points': sum(len(volumes) for volumes in self.quality_checker.volume_history.values())
        }