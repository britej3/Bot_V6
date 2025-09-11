"""
Live A/B Testing Manager for CryptoScalp AI

This module implements a sophisticated A/B testing framework that runs multiple models
in parallel, collects comprehensive performance metrics, and enables data-driven
model promotion decisions.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class ABTestConfiguration:
    """Configuration for A/B test execution"""
    test_name: str
    champion_model_id: str
    challenger_model_id: str
    test_duration_hours: int
    capital_allocation: Dict[str, float]  # model_id -> capital percentage
    market_conditions: List[str]  # ['trending', 'ranging', 'high_volatility', 'low_volatility']
    risk_limits: Dict[str, float]
    performance_metrics: List[str] = field(default_factory=lambda: [
        'sharpe_ratio', 'max_drawdown', 'total_return', 'win_rate', 'profit_factor'
    ])
    min_sample_size: int = 100
    confidence_level: float = 0.95


@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance at a point in time"""
    model_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    trade_count: int
    active_positions: int
    capital_allocated: float
    capital_used: float
    pnl_realized: float
    pnl_unrealized: float


@dataclass
class ABTestResults:
    """Results of an A/B test"""
    test_id: str
    configuration: ABTestConfiguration
    start_time: datetime
    end_time: Optional[datetime]
    status: str  # 'running', 'completed', 'failed'
    champion_performance: List[ModelPerformanceSnapshot] = field(default_factory=list)
    challenger_performance: List[ModelPerformanceSnapshot] = field(default_factory=list)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    winner: Optional[str] = None
    confidence_level: float = 0.0


class LiveABTestingManager:
    """
    Manager for live A/B testing of trading models.
    Handles parallel execution, performance monitoring, and statistical analysis.
    """

    def __init__(self, config_path: str = "config/ab_testing_config.json"):
        self.config = self._load_config(config_path)
        self.active_tests: Dict[str, ABTestResults] = {}
        self.completed_tests: Dict[str, ABTestResults] = {}
        self.model_interfaces: Dict[str, Any] = {}
        self.performance_monitor = PerformanceMonitor()
        self.statistical_analyzer = StatisticalAnalyzer()

        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()

        # Background monitoring task
        self._monitoring_task = None
        self._stop_monitoring = False

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load A/B testing configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_concurrent_tests': 3,
            'monitoring_interval_seconds': 60,
            'statistical_significance_threshold': 0.05,
            'risk_management': {
                'max_capital_per_test': 0.2,  # 20% of total capital
                'max_drawdown_limit': 0.05,
                'position_size_limit': 0.02
            }
        }

    def register_model_interface(self, model_id: str, interface: Any):
        """
        Register a model interface for testing.
        Interface should have methods: generate_signal(market_data), execute_trade(signal)
        """
        self.model_interfaces[model_id] = interface
        logger.info(f"Registered model interface: {model_id}")

    async def start_ab_test(self, configuration: ABTestConfiguration) -> str:
        """Start a new A/B test"""
        with self._lock:
            # Check concurrent test limit
            if len(self.active_tests) >= self.config['max_concurrent_tests']:
                raise RuntimeError("Maximum concurrent tests reached")

            # Generate test ID
            test_id = f"ab_test_{configuration.test_name}_{int(time.time())}"

            # Create test results object
            test_results = ABTestResults(
                test_id=test_id,
                configuration=configuration,
                start_time=datetime.now(),
                status='running'
            )

            self.active_tests[test_id] = test_results

        logger.info(f"Started A/B test: {test_id}")

        # Start monitoring task if not already running
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_tests())

        return test_id

    async def stop_ab_test(self, test_id: str) -> bool:
        """Stop an active A/B test"""
        with self._lock:
            if test_id not in self.active_tests:
                return False

            test = self.active_tests[test_id]
            test.end_time = datetime.now()
            test.status = 'completed'

            # Analyze final results
            await self._analyze_test_results(test)

            # Move to completed tests
            self.completed_tests[test_id] = test
            del self.active_tests[test_id]

        logger.info(f"Stopped A/B test: {test_id}")
        return True

    async def _monitor_tests(self):
        """Background task to monitor active tests"""
        logger.info("Starting A/B test monitoring")

        while not self._stop_monitoring:
            try:
                await self._perform_monitoring_cycle()
                await asyncio.sleep(self.config['monitoring_interval_seconds'])
            except Exception as e:
                logger.error(f"Monitoring cycle failed: {e}")
                await asyncio.sleep(60)  # Wait before retry

        logger.info("Stopped A/B test monitoring")

    async def _perform_monitoring_cycle(self):
        """Perform one monitoring cycle for all active tests"""
        for test_id, test_results in list(self.active_tests.items()):
            try:
                await self._monitor_single_test(test_results)
            except Exception as e:
                logger.error(f"Failed to monitor test {test_id}: {e}")

    async def _monitor_single_test(self, test_results: ABTestResults):
        """Monitor a single A/B test"""
        config = test_results.configuration

        # Get current performance snapshots
        champion_snapshot = await self._get_model_performance_snapshot(
            config.champion_model_id
        )
        challenger_snapshot = await self._get_model_performance_snapshot(
            config.challenger_model_id
        )

        # Store snapshots
        test_results.champion_performance.append(champion_snapshot)
        test_results.challenger_performance.append(challenger_snapshot)

        # Check if test duration is exceeded
        elapsed_hours = (datetime.now() - test_results.start_time).total_seconds() / 3600
        if elapsed_hours >= config.test_duration_hours:
            await self.stop_ab_test(test_results.test_id)
            return

        # Check risk limits
        if self._should_stop_for_risk(champion_snapshot, challenger_snapshot, config):
            logger.warning(f"Stopping test {test_results.test_id} due to risk limits")
            await self.stop_ab_test(test_results.test_id)
            return

        # Perform interim analysis if enough data
        if (len(test_results.champion_performance) >= config.min_sample_size and
            len(test_results.challenger_performance) >= config.min_sample_size):
            await self._perform_interim_analysis(test_results)

    async def _get_model_performance_snapshot(self, model_id: str) -> ModelPerformanceSnapshot:
        """Get current performance snapshot for a model"""
        # In practice, this would query the trading system's performance metrics
        # For now, return a mock snapshot
        return ModelPerformanceSnapshot(
            model_id=model_id,
            timestamp=datetime.now(),
            metrics=self.performance_monitor.get_model_metrics(model_id),
            trade_count=np.random.randint(50, 200),
            active_positions=np.random.randint(1, 10),
            capital_allocated=0.1,  # 10%
            capital_used=0.05,      # 5%
            pnl_realized=np.random.normal(0.001, 0.002),
            pnl_unrealized=np.random.normal(0.0005, 0.001)
        )

    def _should_stop_for_risk(self, champion: ModelPerformanceSnapshot,
                            challenger: ModelPerformanceSnapshot,
                            config: ABTestConfiguration) -> bool:
        """Check if test should be stopped due to risk limits"""
        risk_config = self.config['risk_management']

        # Check individual model drawdown
        if (champion.metrics.get('max_drawdown', 0) > risk_config['max_drawdown_limit'] or
            challenger.metrics.get('max_drawdown', 0) > risk_config['max_drawdown_limit']):
            return True

        # Check capital usage
        total_capital_used = champion.capital_used + challenger.capital_used
        if total_capital_used > risk_config['max_capital_per_test']:
            return True

        return False

    async def _perform_interim_analysis(self, test_results: ABTestResults):
        """Perform interim statistical analysis"""
        try:
            significance = self.statistical_analyzer.analyze_performance_difference(
                test_results.champion_performance,
                test_results.challenger_performance,
                test_results.configuration.performance_metrics
            )

            test_results.statistical_significance = significance

            # Check for early stopping criteria
            for metric, p_value in significance.items():
                if p_value < self.config['statistical_significance_threshold']:
                    # Significant difference found
                    if self._is_challenger_better(test_results, metric):
                        logger.info(f"Early stopping: Challenger significantly better in {metric}")
                        test_results.winner = test_results.configuration.challenger_model_id
                        await self.stop_ab_test(test_results.test_id)
                        break

        except Exception as e:
            logger.error(f"Interim analysis failed: {e}")

    def _is_challenger_better(self, test_results: ABTestResults, metric: str) -> bool:
        """Check if challenger is better than champion for a specific metric"""
        # Simple comparison - in practice would be more sophisticated
        champion_latest = test_results.champion_performance[-1].metrics.get(metric, 0)
        challenger_latest = test_results.challenger_performance[-1].metrics.get(metric, 0)

        # For some metrics higher is better, for others lower
        if metric in ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor']:
            return challenger_latest > champion_latest
        elif metric in ['max_drawdown']:
            return challenger_latest < champion_latest

        return False

    async def _analyze_test_results(self, test_results: ABTestResults):
        """Analyze final test results and determine winner"""
        try:
            # Perform comprehensive statistical analysis
            final_significance = self.statistical_analyzer.perform_final_analysis(
                test_results.champion_performance,
                test_results.challenger_performance,
                test_results.configuration.performance_metrics
            )

            test_results.statistical_significance = final_significance

            # Determine winner based on statistical significance and risk-adjusted returns
            champion_score = self._calculate_model_score(test_results.champion_performance)
            challenger_score = self._calculate_model_score(test_results.challenger_performance)

            if challenger_score > champion_score * 1.05:  # 5% improvement threshold
                test_results.winner = test_results.configuration.challenger_model_id
                test_results.confidence_level = self._calculate_confidence_level(
                    final_significance, challenger_score, champion_score
                )
            else:
                test_results.winner = test_results.configuration.champion_model_id
                test_results.confidence_level = 0.8  # Conservative confidence

            logger.info(f"Test {test_results.test_id} completed. Winner: {test_results.winner}")

        except Exception as e:
            logger.error(f"Final analysis failed: {e}")
            test_results.status = 'failed'

    def _calculate_model_score(self, performance_history: List[ModelPerformanceSnapshot]) -> float:
        """Calculate overall model score based on performance history"""
        if not performance_history:
            return 0.0

        # Weighted average of key metrics
        weights = {
            'sharpe_ratio': 0.4,
            'total_return': 0.3,
            'max_drawdown': -0.2,  # Negative weight for drawdown
            'win_rate': 0.1
        }

        latest = performance_history[-1]
        score = 0.0

        for metric, weight in weights.items():
            value = latest.metrics.get(metric, 0)
            if metric == 'max_drawdown':
                # Invert drawdown so lower is better
                value = -value
            score += value * weight

        return score

    def _calculate_confidence_level(self, significance: Dict[str, float],
                                 challenger_score: float, champion_score: float) -> float:
        """Calculate confidence level in the test results"""
        # Average p-values for key metrics
        key_metrics = ['sharpe_ratio', 'total_return']
        avg_p_value = np.mean([significance.get(metric, 1.0) for metric in key_metrics])

        # Convert p-value to confidence (lower p-value = higher confidence)
        confidence_from_stats = 1 - avg_p_value

        # Adjust based on score difference
        score_improvement = (challenger_score - champion_score) / abs(champion_score)
        confidence_from_score = min(score_improvement * 5, 1.0)  # Cap at 1.0

        # Weighted average
        return 0.7 * confidence_from_stats + 0.3 * confidence_from_score

    def get_test_status(self, test_id: str) -> Optional[ABTestResults]:
        """Get status of a specific test"""
        return self.active_tests.get(test_id) or self.completed_tests.get(test_id)

    def get_all_active_tests(self) -> List[ABTestResults]:
        """Get all active tests"""
        return list(self.active_tests.values())

    def get_all_completed_tests(self) -> List[ABTestResults]:
        """Get all completed tests"""
        return list(self.completed_tests.values())

    def save_test_results(self, test_id: str, path: str):
        """Save test results to file"""
        test = self.completed_tests.get(test_id)
        if test is None:
            raise ValueError(f"Test {test_id} not found")

        # Convert to serializable format
        results_dict = {
            'test_id': test.test_id,
            'configuration': {
                'test_name': test.configuration.test_name,
                'champion_model_id': test.configuration.champion_model_id,
                'challenger_model_id': test.configuration.challenger_model_id,
                'test_duration_hours': test.configuration.test_duration_hours,
                'capital_allocation': test.configuration.capital_allocation
            },
            'start_time': test.start_time.isoformat(),
            'end_time': test.end_time.isoformat() if test.end_time else None,
            'status': test.status,
            'winner': test.winner,
            'confidence_level': test.confidence_level,
            'statistical_significance': test.statistical_significance
        }

        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Test results saved to {path}")

    def shutdown(self):
        """Shutdown the A/B testing manager"""
        self._stop_monitoring = True
        self.executor.shutdown(wait=True)
        logger.info("A/B Testing Manager shutdown complete")


class PerformanceMonitor:
    """Monitor model performance in real-time"""

    def __init__(self):
        self.model_metrics = {}

    def get_model_metrics(self, model_id: str) -> Dict[str, float]:
        """Get current metrics for a model"""
        # In practice, this would fetch real metrics from trading system
        # For now, return mock metrics
        return {
            'sharpe_ratio': np.random.normal(1.5, 0.3),
            'max_drawdown': np.random.uniform(0.01, 0.15),
            'total_return': np.random.normal(0.02, 0.01),
            'win_rate': np.random.uniform(0.4, 0.8),
            'profit_factor': np.random.uniform(1.0, 2.0)
        }


class StatisticalAnalyzer:
    """Perform statistical analysis of A/B test results"""

    def analyze_performance_difference(self, champion_perf: List[ModelPerformanceSnapshot],
                                    challenger_perf: List[ModelPerformanceSnapshot],
                                    metrics: List[str]) -> Dict[str, float]:
        """Analyze statistical significance of performance differences"""
        significance = {}

        for metric in metrics:
            champion_values = [snap.metrics.get(metric, 0) for snap in champion_perf]
            challenger_values = [snap.metrics.get(metric, 0) for snap in challenger_perf]

            if len(champion_values) >= 10 and len(challenger_values) >= 10:
                # Perform t-test (simplified)
                mean_diff = abs(np.mean(challenger_values) - np.mean(champion_values))
                std_pooled = np.sqrt((np.var(champion_values) + np.var(challenger_values)) / 2)

                if std_pooled > 0:
                    t_stat = mean_diff / (std_pooled / np.sqrt(len(champion_values)))
                    # Convert to p-value (simplified approximation)
                    p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat / np.sqrt(2))))
                    significance[metric] = p_value
                else:
                    significance[metric] = 1.0
            else:
                significance[metric] = 1.0  # Not enough data

        return significance

    def perform_final_analysis(self, champion_perf: List[ModelPerformanceSnapshot],
                             challenger_perf: List[ModelPerformanceSnapshot],
                             metrics: List[str]) -> Dict[str, float]:
        """Perform comprehensive final statistical analysis"""
        return self.analyze_performance_difference(champion_perf, challenger_perf, metrics)


# Example usage and testing
if __name__ == "__main__":
    async def test_ab_testing():
        # Create A/B testing manager
        manager = LiveABTestingManager()

        # Create test configuration
        config = ABTestConfiguration(
            test_name="btc_usdt_model_comparison",
            champion_model_id="champion_v1",
            challenger_model_id="challenger_v2",
            test_duration_hours=24,
            capital_allocation={"champion_v1": 0.4, "challenger_v2": 0.4},
            market_conditions=["trending", "ranging"],
            risk_limits={"max_drawdown": 0.05, "max_position_size": 0.02}
        )

        # Start test
        test_id = await manager.start_ab_test(config)
        print(f"Started test: {test_id}")

        # Wait a bit
        await asyncio.sleep(5)

        # Stop test
        await manager.stop_ab_test(test_id)

        # Get results
        results = manager.get_test_status(test_id)
        if results:
            print(f"Test completed. Winner: {results.winner}")
            print(f"Confidence level: {results.confidence_level:.2f}")

        manager.shutdown()

    # Run test
    asyncio.run(test_ab_testing())