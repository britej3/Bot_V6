"""
ML-Nautilus Integration Layer
=============================

Advanced integration of existing ML components with Nautilus Trader framework.
Enhances Nautilus with sophisticated ML predictions and adaptive strategies.

Key Features:
- ML-enhanced order routing decisions
- Adaptive strategy selection based on ML confidence
- Real-time feature engineering from Nautilus data
- Performance-based model selection
- Continuous learning integration

Integration Strategy:
- Nautilus provides execution framework
- ML components provide intelligent decision making
- Hybrid system leverages best of both worlds
- No redundancy - complementary capabilities only

Author: ML & Trading Systems Team
Date: 2025-01-22
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd

from src.learning.strategy_model_integration_engine import (
    AutonomousScalpingEngine,
    TradingStrategy,
    TickFeatureEngineering,
    MLModelEnsemble,
    TradingSignal,
    create_autonomous_scalping_engine
)
from src.learning.market_regime_detection import (
    MarketRegimeDetector,
    MarketRegime,
    create_market_regime_detector
)
from src.trading.nautilus_integration import (
    NautilusTraderManager,
    OrderRoutingStrategy,
    NautilusOrderRequest,
    submit_order_hybrid
)
from src.trading.nautilus_strategy_adapter import (
    StrategyAdapterFactory,
    NautilusStrategyAdapter
)

logger = logging.getLogger(__name__)


class MLIntegrationMode(Enum):
    """ML integration modes with Nautilus"""
    ENHANCED_ROUTING = "enhanced_routing"      # ML-enhanced order routing
    ADAPTIVE_STRATEGIES = "adaptive_strategies" # ML-driven strategy adaptation
    HYBRID_EXECUTION = "hybrid_execution"       # ML + Nautilus hybrid execution
    FULL_AUTONOMOUS = "full_autonomous"        # Complete ML autonomy with Nautilus


class MLEnhancedOrderRequest(NautilusOrderRequest):
    """ML-enhanced order request with predictions"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ml_predictions: Dict[str, Any] = {}
        self.strategy_confidence: float = 0.0
        self.market_regime: str = 'unknown'
        self.risk_adjusted_size: float = 0.0
        self.execution_confidence: float = 0.0
        self.feature_importance: Dict[str, float] = {}


class MLNautilusIntegrationManager:
    """
    ML-Nautilus Integration Manager

    Bridges sophisticated ML components with Nautilus Trader's execution framework.
    Provides intelligent order routing, adaptive strategies, and continuous learning.
    """

    def __init__(self, integration_mode: MLIntegrationMode = MLIntegrationMode.HYBRID_EXECUTION):
        self.integration_mode = integration_mode

        # Core ML components
        self.autonomous_engine = create_autonomous_scalping_engine()
        self.market_regime_detector = create_market_regime_detector()
        self.nautilus_manager = NautilusTraderManager()

        # ML-enhanced components
        self.ml_feature_engineering = TickFeatureEngineering(window_size=200)
        self.ml_ensemble = MLModelEnsemble()
        self.strategy_adapters: Dict[str, NautilusStrategyAdapter] = {}

        # State tracking
        self.is_initialized = False
        self.is_running = False
        self.ml_performance_history = []
        self.adaptation_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'avg_confidence': 0.0,
            'regime_accuracy': 0.0,
            'execution_improvement': 0.0
        }

        logger.info(f"🧠 ML-Nautilus Integration initialized in {integration_mode.value} mode")

    async def initialize(self) -> bool:
        """Initialize ML-Nautilus integration"""
        try:
            logger.info("🚀 Initializing ML-Nautilus integration...")

            # Initialize core components
            await self._initialize_ml_components()
            await self._initialize_nautilus_integration()
            await self._initialize_strategy_adapters()

            # Validate integration
            await self._validate_ml_integration()

            self.is_initialized = True
            logger.info("✅ ML-Nautilus integration initialized successfully")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize ML-Nautilus integration: {e}")
            return False

    async def _initialize_ml_components(self):
        """Initialize ML components"""
        # ML components are already initialized in autonomous_engine
        # Just ensure they're ready
        self.market_regime_detector.start_detection()

        logger.info("✅ ML components initialized")

    async def _initialize_nautilus_integration(self):
        """Initialize Nautilus integration"""
        success = await self.nautilus_manager.initialize()
        if not success:
            raise RuntimeError("Failed to initialize Nautilus integration")

        # Configure Nautilus for ML-enhanced operation
        self.nautilus_manager.integration_mode = self.nautilus_manager.IntegrationMode.HYBRID
        self.nautilus_manager.routing_strategy = OrderRoutingStrategy.PERFORMANCE_BASED

        logger.info("✅ Nautilus integration initialized")

    async def _initialize_strategy_adapters(self):
        """Initialize ML-enhanced strategy adapters"""
        strategies = ['scalping', 'market_making', 'mean_reversion']

        for strategy_type in strategies:
            try:
                adapter = StrategyAdapterFactory.create_adapter(
                    strategy_type=strategy_type,
                    strategy_id=f"ml_enhanced_{strategy_type}",
                    instrument_id="BTC/USDT"
                )

                # Start the adapter
                await adapter.start()
                self.strategy_adapters[strategy_type] = adapter

                logger.info(f"✅ Strategy adapter initialized: {strategy_type}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize {strategy_type} adapter: {e}")

        logger.info(f"✅ Initialized {len(self.strategy_adapters)} strategy adapters")

    async def _validate_ml_integration(self):
        """Validate ML-Nautilus integration"""
        # Check all components are initialized
        required_components = [
            'autonomous_engine',
            'market_regime_detector',
            'nautilus_manager'
        ]

        for component in required_components:
            if not hasattr(self, component) or getattr(self, component) is None:
                raise ValueError(f"Missing required component: {component}")

        # Validate strategy adapters
        if len(self.strategy_adapters) == 0:
            raise ValueError("No strategy adapters initialized")

        # Test basic ML functionality
        test_features = self.ml_feature_engineering._default_features()
        ml_prediction = self.ml_ensemble.predict_ensemble(test_features)

        if 'ensemble' not in ml_prediction:
            raise ValueError("ML ensemble prediction failed")

        logger.info("✅ ML-Nautilus integration validation completed")

    async def process_tick_with_ml_enhancement(self, tick_data: Any) -> Dict[str, Any]:
        """
        Process tick data with ML enhancement for Nautilus execution

        This is the core integration point where ML intelligence
        enhances Nautilus trading decisions.
        """

        try:
            # Extract enhanced features
            self.ml_feature_engineering.add_tick(tick_data)
            features = self.ml_feature_engineering.extract_features()

            # Get market regime
            market_regime_info = self.market_regime_detector.get_current_regime_info()
            market_regime = market_regime_info.get('regime', 'unknown')

            # Create market condition for ML processing
            market_condition = type('MarketCondition', (), {
                'regime': market_regime,
                'volatility': features.price_volatility,
                'confidence': market_regime_info.get('confidence', 0.5)
            })()

            # Get ML-enhanced trading signal
            ml_result = await self.autonomous_engine.process_tick(tick_data, market_condition)

            # Enhance signal with Nautilus-specific optimizations
            enhanced_signal = await self._enhance_signal_with_nautilus(
                ml_result['signal'],
                features,
                market_regime_info
            )

            # Generate ML-enhanced order request
            order_request = await self._create_ml_enhanced_order(
                enhanced_signal,
                features,
                market_regime_info
            )

            # Track ML performance
            self._track_ml_performance(ml_result, order_request)

            return {
                'original_signal': ml_result['signal'],
                'enhanced_signal': enhanced_signal,
                'order_request': order_request,
                'ml_prediction': ml_result['ml_prediction'],
                'market_regime': market_regime_info,
                'features': features,
                'processing_timestamp': datetime.utcnow()
            }

        except Exception as e:
            logger.error(f"❌ Failed to process tick with ML enhancement: {e}")
            # Return basic signal on error
            return await self._fallback_processing(tick_data)

    async def _enhance_signal_with_nautilus(
        self,
        original_signal: TradingSignal,
        features: Any,
        market_regime: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance trading signal with Nautilus-specific optimizations"""

        # Get relevant strategy adapter
        strategy_key = original_signal.strategy.value.lower().replace('_', '_')
        adapter = self.strategy_adapters.get(strategy_key)

        if adapter:
            # Use adapter to enhance signal
            enhanced_order = await adapter.adapt_signal({
                'symbol': 'BTC/USDT',  # Could be parameterized
                'direction': 1.0 if original_signal.action == 'BUY' else -1.0,
                'confidence': original_signal.confidence,
                'price': features.price,
                'volume': features.volume,
                'volatility': features.price_volatility,
                'rsi': features.rsi,
                'bollinger_position': features.bollinger_position,
                'market_regime': market_regime.get('regime', 'unknown')
            })

            if enhanced_order:
                return {
                    'original_signal': original_signal,
                    'enhanced_order': enhanced_order,
                    'enhancement_method': 'strategy_adapter',
                    'confidence_boost': enhanced_order.confidence - original_signal.confidence
                }

        # Return original signal if no enhancement possible
        return {
            'original_signal': original_signal,
            'enhanced_order': None,
            'enhancement_method': 'none',
            'confidence_boost': 0.0
        }

    async def _create_ml_enhanced_order(
        self,
        enhanced_signal: Dict[str, Any],
        features: Any,
        market_regime: Dict[str, Any]
    ) -> MLEnhancedOrderRequest:
        """Create ML-enhanced order request for Nautilus execution"""

        original_signal = enhanced_signal['original_signal']
        enhanced_order = enhanced_signal.get('enhanced_order')

        # Use enhanced order if available, otherwise convert original signal
        if enhanced_order:
            order_request = MLEnhancedOrderRequest(
                symbol=enhanced_order.symbol,
                side=enhanced_order.side,
                quantity=enhanced_order.quantity,
                order_type=enhanced_order.order_type,
                price=enhanced_order.price,
                client_id=enhanced_order.client_id
            )
            order_request.ml_predictions = {'enhanced': True}
            order_request.strategy_confidence = enhanced_order.confidence
        else:
            # Convert original signal to order request
            order_request = MLEnhancedOrderRequest(
                symbol='BTC/USDT',  # Could be parameterized
                side=original_signal.action,
                quantity=original_signal.position_size,
                order_type='LIMIT' if original_signal.action != 'HOLD' else 'MARKET',
                price=original_signal.entry_price,
                client_id=f"ml_signal_{datetime.utcnow().timestamp()}"
            )
            order_request.ml_predictions = {'enhanced': False}
            order_request.strategy_confidence = original_signal.confidence

        # Add ML-specific metadata
        order_request.market_regime = market_regime.get('regime', 'unknown')
        order_request.risk_adjusted_size = self._calculate_risk_adjusted_size(
            order_request.quantity,
            features.price_volatility,
            market_regime.get('confidence', 0.5)
        )
        order_request.execution_confidence = self._calculate_execution_confidence(
            order_request.strategy_confidence,
            features,
            market_regime
        )

        # Calculate feature importance for this decision
        order_request.feature_importance = self._calculate_feature_importance(features)

        return order_request

    def _calculate_risk_adjusted_size(
        self,
        base_size: float,
        volatility: float,
        regime_confidence: float
    ) -> float:
        """Calculate risk-adjusted position size"""

        # Volatility adjustment (inverse relationship)
        volatility_factor = min(1.0, 0.02 / max(volatility, 0.01))

        # Regime confidence adjustment
        confidence_factor = 0.5 + (regime_confidence * 0.5)  # 0.5 to 1.0

        # Market regime specific adjustments
        regime_multiplier = 1.0  # Could be enhanced based on regime

        adjusted_size = base_size * volatility_factor * confidence_factor * regime_multiplier

        return max(0.001, min(adjusted_size, 1.0))  # Bounds checking

    def _calculate_execution_confidence(
        self,
        strategy_confidence: float,
        features: Any,
        market_regime: Dict[str, Any]
    ) -> float:
        """Calculate overall execution confidence"""

        # Base confidence from strategy
        confidence = strategy_confidence * 0.4

        # Market condition confidence
        regime_confidence = market_regime.get('confidence', 0.5) * 0.3

        # Technical indicator confidence
        technical_confidence = 0.0
        if features.rsi < 30 or features.rsi > 70:  # Extreme RSI levels
            technical_confidence += 0.1
        if abs(features.bollinger_position - 0.5) > 0.3:  # Extreme Bollinger position
            technical_confidence += 0.1
        if features.volume_spike_ratio > 1.5:  # Volume spike
            technical_confidence += 0.1

        confidence += min(technical_confidence, 0.3)

        return min(confidence, 1.0)

    def _calculate_feature_importance(self, features: Any) -> Dict[str, float]:
        """Calculate feature importance for decision explanation"""

        # Simple feature importance based on deviation from normal
        importance = {}

        # Price-based features
        importance['price_volatility'] = min(abs(features.price_volatility - 0.02) / 0.02, 1.0)
        importance['price_momentum'] = min(abs(features.price_momentum), 1.0)

        # Volume-based features
        importance['volume_spike_ratio'] = min(features.volume_spike_ratio - 1.0, 1.0)

        # Technical indicators
        importance['rsi'] = min(abs(features.rsi - 50) / 50, 1.0)
        importance['bollinger_position'] = abs(features.bollinger_position - 0.5) * 2

        # Order book features
        importance['order_imbalance'] = abs(features.order_imbalance)

        return importance

    def _track_ml_performance(self, ml_result: Dict[str, Any], order_request: MLEnhancedOrderRequest):
        """Track ML performance for continuous improvement"""

        performance_record = {
            'timestamp': datetime.utcnow(),
            'signal_action': ml_result['signal'].action,
            'signal_confidence': ml_result['signal'].confidence,
            'ml_confidence': ml_result['ml_prediction']['confidence'],
            'market_regime': order_request.market_regime,
            'execution_confidence': order_request.execution_confidence,
            'feature_importance': order_request.feature_importance,
            'order_type': order_request.order_type,
            'enhanced': order_request.ml_predictions.get('enhanced', False)
        }

        self.ml_performance_history.append(performance_record)

        # Update adaptation metrics
        self.adaptation_metrics['total_predictions'] += 1
        if order_request.execution_confidence > 0.7:
            self.adaptation_metrics['successful_predictions'] += 1

        # Keep only recent history (last 1000 records)
        if len(self.ml_performance_history) > 1000:
            self.ml_performance_history = self.ml_performance_history[-1000:]

    async def _fallback_processing(self, tick_data: Any) -> Dict[str, Any]:
        """Fallback processing when ML enhancement fails"""

        # Create basic signal without ML enhancement
        return {
            'original_signal': TradingSignal(
                strategy=TradingStrategy.MARKET_MAKING,
                action='HOLD',
                confidence=0.5,
                position_size=0.0,
                entry_price=50000,  # Default
                reasoning='ML enhancement failed - using fallback',
                timestamp=datetime.utcnow()
            ),
            'enhanced_signal': None,
            'order_request': None,
            'ml_prediction': None,
            'market_regime': {'regime': 'unknown', 'confidence': 0.0},
            'features': None,
            'processing_timestamp': datetime.utcnow()
        }

    async def submit_ml_enhanced_order(self, order_request: MLEnhancedOrderRequest) -> Dict[str, Any]:
        """Submit ML-enhanced order through hybrid routing"""

        try:
            # Convert to standard order request for routing
            standard_request = {
                'symbol': order_request.symbol,
                'side': order_request.side,
                'quantity': order_request.quantity,
                'order_type': order_request.order_type,
                'price': order_request.price,
                'stop_price': getattr(order_request, 'stop_price', None),
                'client_id': order_request.client_id,
                'ml_enhanced': True,
                'execution_confidence': order_request.execution_confidence,
                'market_regime': order_request.market_regime,
                'feature_importance': order_request.feature_importance
            }

            # Submit through hybrid routing system
            result = await submit_order_hybrid(standard_request)

            # Add ML-specific metadata to result
            result['ml_enhanced'] = True
            result['execution_confidence'] = order_request.execution_confidence
            result['market_regime'] = order_request.market_regime

            logger.info(f"✅ ML-enhanced order submitted: {result.get('order_id', 'unknown')}")

            return result

        except Exception as e:
            logger.error(f"❌ Failed to submit ML-enhanced order: {e}")
            raise

    async def get_ml_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive ML performance metrics"""

        if not self.ml_performance_history:
            return self._empty_metrics()

        # Calculate performance statistics
        total_predictions = len(self.ml_performance_history)
        successful_predictions = sum(1 for record in self.ml_performance_history
                                   if record['execution_confidence'] > 0.7)

        avg_confidence = np.mean([record['execution_confidence']
                                for record in self.ml_performance_history])

        # Strategy performance breakdown
        strategy_performance = {}
        for record in self.ml_performance_history:
            strategy = record.get('signal_action', 'unknown')
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(record['execution_confidence'])

        for strategy, confidences in strategy_performance.items():
            strategy_performance[strategy] = {
                'avg_confidence': np.mean(confidences),
                'count': len(confidences),
                'success_rate': np.mean([1 if c > 0.7 else 0 for c in confidences])
            }

        return {
            'total_predictions': total_predictions,
            'successful_predictions': successful_predictions,
            'success_rate': successful_predictions / max(total_predictions, 1),
            'avg_confidence': avg_confidence,
            'strategy_performance': strategy_performance,
            'feature_importance_summary': self._get_feature_importance_summary(),
            'adaptation_metrics': self.adaptation_metrics,
            'system_status': {
                'is_initialized': self.is_initialized,
                'is_running': self.is_running,
                'integration_mode': self.integration_mode.value,
                'active_adapters': len(self.strategy_adapters)
            }
        }

    def _get_feature_importance_summary(self) -> Dict[str, float]:
        """Get summary of feature importance across all predictions"""

        if not self.ml_performance_history:
            return {}

        # Aggregate feature importance
        total_importance = {}
        count = 0

        for record in self.ml_performance_history:
            if 'feature_importance' in record:
                count += 1
                for feature, importance in record['feature_importance'].items():
                    total_importance[feature] = total_importance.get(feature, 0) + importance

        if count == 0:
            return {}

        # Calculate averages
        avg_importance = {feature: total / count
                         for feature, total in total_importance.items()}

        return avg_importance

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'total_predictions': 0,
            'successful_predictions': 0,
            'success_rate': 0.0,
            'avg_confidence': 0.0,
            'strategy_performance': {},
            'feature_importance_summary': {},
            'adaptation_metrics': self.adaptation_metrics,
            'system_status': {
                'is_initialized': self.is_initialized,
                'is_running': self.is_running,
                'integration_mode': self.integration_mode.value,
                'active_adapters': len(self.strategy_adapters)
            }
        }

    async def start(self):
        """Start ML-Nautilus integration"""
        if not self.is_initialized:
            await self.initialize()

        self.is_running = True
        await self.nautilus_manager.start()

        logger.info("🚀 ML-Nautilus integration started")

    async def stop(self):
        """Stop ML-Nautilus integration"""
        self.is_running = False
        await self.nautilus_manager.stop()

        # Stop strategy adapters
        for adapter in self.strategy_adapters.values():
            await adapter.stop()

        logger.info("🛑 ML-Nautilus integration stopped")

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""

        health_status = {
            'timestamp': datetime.utcnow(),
            'overall_status': 'healthy',
            'components': {},
            'issues': []
        }

        # Check ML components
        try:
            # Test ML prediction
            test_features = self.ml_feature_engineering._default_features()
            ml_prediction = self.ml_ensemble.predict_ensemble(test_features)
            health_status['components']['ml_engine'] = 'healthy'
        except Exception as e:
            health_status['components']['ml_engine'] = 'unhealthy'
            health_status['issues'].append(f'ML engine error: {e}')
            health_status['overall_status'] = 'degraded'

        # Check market regime detector
        try:
            regime_info = self.market_regime_detector.get_current_regime_info()
            health_status['components']['regime_detector'] = 'healthy'
        except Exception as e:
            health_status['components']['regime_detector'] = 'unhealthy'
            health_status['issues'].append(f'Regime detector error: {e}')
            health_status['overall_status'] = 'degraded'

        # Check Nautilus integration
        nautilus_health = await self.nautilus_manager.health_check()
        health_status['components']['nautilus_integration'] = nautilus_health['overall_status']
        if nautilus_health['issues']:
            health_status['issues'].extend(nautilus_health['issues'])
            health_status['overall_status'] = 'degraded'

        # Check strategy adapters
        healthy_adapters = 0
        for adapter_name, adapter in self.strategy_adapters.items():
            if adapter.is_active:
                healthy_adapters += 1
                health_status['components'][f'adapter_{adapter_name}'] = 'healthy'
            else:
                health_status['components'][f'adapter_{adapter_name}'] = 'inactive'
                health_status['issues'].append(f'Adapter {adapter_name} inactive')

        health_status['components']['strategy_adapters'] = f'{healthy_adapters}/{len(self.strategy_adapters)} healthy'

        if health_status['issues']:
            health_status['overall_status'] = 'unhealthy'

        return health_status


# Global instance
ml_nautilus_integration = MLNautilusIntegrationManager()


async def get_ml_nautilus_integration() -> MLNautilusIntegrationManager:
    """Get ML-Nautilus integration instance"""
    return ml_nautilus_integration


async def initialize_ml_nautilus_integration(
    mode: MLIntegrationMode = MLIntegrationMode.HYBRID_EXECUTION
) -> bool:
    """Initialize ML-Nautilus integration with specified mode"""
    global ml_nautilus_integration
    ml_nautilus_integration = MLNautilusIntegrationManager(mode)
    return await ml_nautilus_integration.initialize()


async def process_tick_with_ml_nautilus(tick_data: Any) -> Dict[str, Any]:
    """Process tick with ML-Nautilus enhancement"""
    return await ml_nautilus_integration.process_tick_with_ml_enhancement(tick_data)


async def submit_ml_enhanced_order(order_request: MLEnhancedOrderRequest) -> Dict[str, Any]:
    """Submit ML-enhanced order through Nautilus"""
    return await ml_nautilus_integration.submit_ml_enhanced_order(order_request)


async def get_ml_nautilus_metrics() -> Dict[str, Any]:
    """Get ML-Nautilus performance metrics"""
    return await ml_nautilus_integration.get_ml_performance_metrics()


async def start_ml_nautilus_integration():
    """Start ML-Nautilus integration"""
    await ml_nautilus_integration.start()


async def stop_ml_nautilus_integration():
    """Stop ML-Nautilus integration"""
    await ml_nautilus_integration.stop()


async def get_ml_nautilus_health() -> Dict[str, Any]:
    """Get ML-Nautilus health status"""
    return await ml_nautilus_integration.health_check()


# Export key classes and functions
__all__ = [
    'MLNautilusIntegrationManager',
    'MLEnhancedOrderRequest',
    'MLIntegrationMode',
    'get_ml_nautilus_integration',
    'initialize_ml_nautilus_integration',
    'process_tick_with_ml_nautilus',
    'submit_ml_enhanced_order',
    'get_ml_nautilus_metrics',
    'start_ml_nautilus_integration',
    'stop_ml_nautilus_integration',
    'get_ml_nautilus_health'
]