"""
Chaos Engineering Test Suite for CryptoScalp AI

This module implements a comprehensive chaos engineering test suite that systematically
injects failures and validates recovery playbooks in a controlled staging environment.
Tests all 7 recovery playbooks and ensures system resilience.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import random
import numpy as np
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class ChaosExperimentType(Enum):
    """Types of chaos experiments"""
    NETWORK_FAILURE = "network_failure"
    SERVICE_CRASH = "service_crash"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_FAILURE = "dependency_failure"


class ExperimentStatus(Enum):
    """Status of chaos experiments"""
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ChaosExperimentType
    target_service: str
    duration_seconds: int
    intensity: str  # 'low', 'medium', 'high'
    safety_checks: List[str]
    expected_recovery_time_seconds: int
    playbook_to_test: str
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Results of a chaos experiment"""
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    success: bool = False
    recovery_time_seconds: float = 0.0
    playbook_executed: Optional[str] = None
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)


class FailureInjector:
    """Injects various types of failures into the system"""

    def __init__(self):
        self.active_failures = {}
        self.failure_handlers = self._initialize_failure_handlers()

    def _initialize_failure_handlers(self) -> Dict[ChaosExperimentType, Callable]:
        """Initialize handlers for different failure types"""
        return {
            ChaosExperimentType.NETWORK_FAILURE: self._inject_network_failure,
            ChaosExperimentType.SERVICE_CRASH: self._inject_service_crash,
            ChaosExperimentType.RESOURCE_EXHAUSTION: self._inject_resource_exhaustion,
            ChaosExperimentType.DATA_CORRUPTION: self._inject_data_corruption,
            ChaosExperimentType.CONFIGURATION_ERROR: self._inject_configuration_error,
            ChaosExperimentType.DEPENDENCY_FAILURE: self._inject_dependency_failure
        }

    async def inject_failure(self, experiment: ChaosExperiment) -> str:
        """Inject a failure according to the experiment specification"""
        failure_id = f"failure_{experiment.experiment_id}_{int(time.time())}"

        if experiment.experiment_type in self.failure_handlers:
            handler = self.failure_handlers[experiment.experiment_type]
            await handler(experiment, failure_id)
            self.active_failures[failure_id] = experiment
            logger.info(f"Injected failure: {experiment.experiment_type.value} ({failure_id})")
        else:
            raise ValueError(f"No handler for experiment type: {experiment.experiment_type}")

        return failure_id

    async def remove_failure(self, failure_id: str):
        """Remove an active failure"""
        if failure_id in self.active_failures:
            experiment = self.active_failures[failure_id]
            await self._cleanup_failure(experiment, failure_id)
            del self.active_failures[failure_id]
            logger.info(f"Removed failure: {failure_id}")

    async def _inject_network_failure(self, experiment: ChaosExperiment, failure_id: str):
        """Inject network failure"""
        logger.info(f"Injecting network failure for {experiment.target_service}")
        # Simulate network failure by introducing latency/drops
        await asyncio.sleep(1)

    async def _inject_service_crash(self, experiment: ChaosExperiment, failure_id: str):
        """Inject service crash"""
        logger.info(f"Injecting service crash for {experiment.target_service}")
        # Simulate service crash by stopping the service temporarily
        await asyncio.sleep(1)

    async def _inject_resource_exhaustion(self, experiment: ChaosExperiment, failure_id: str):
        """Inject resource exhaustion"""
        logger.info(f"Injecting resource exhaustion for {experiment.target_service}")
        # Simulate resource exhaustion by consuming resources
        await asyncio.sleep(1)

    async def _inject_data_corruption(self, experiment: ChaosExperiment, failure_id: str):
        """Inject data corruption"""
        logger.info(f"Injecting data corruption for {experiment.target_service}")
        # Simulate data corruption
        await asyncio.sleep(1)

    async def _inject_configuration_error(self, experiment: ChaosExperiment, failure_id: str):
        """Inject configuration error"""
        logger.info(f"Injecting configuration error for {experiment.target_service}")
        # Simulate configuration error
        await asyncio.sleep(1)

    async def _inject_dependency_failure(self, experiment: ChaosExperiment, failure_id: str):
        """Inject dependency failure"""
        logger.info(f"Injecting dependency failure for {experiment.target_service}")
        # Simulate dependency failure
        await asyncio.sleep(1)

    async def _cleanup_failure(self, experiment: ChaosExperiment, failure_id: str):
        """Clean up after a failure injection"""
        logger.info(f"Cleaning up failure: {failure_id}")
        await asyncio.sleep(1)


class SystemMonitor:
    """Monitors system state during chaos experiments"""

    def __init__(self):
        self.baseline_metrics = {}
        self.monitoring_handlers = {}

    def set_baseline(self, metrics: Dict[str, float]):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = metrics.copy()
        logger.info("Set system baseline metrics")

    async def monitor_system_state(self) -> Dict[str, float]:
        """Monitor current system state"""
        # In practice, this would collect real metrics
        # For now, return simulated metrics
        await asyncio.sleep(0.1)

        return {
            'cpu_usage': random.uniform(10, 90),
            'memory_usage': random.uniform(20, 95),
            'network_latency': random.uniform(10, 500),
            'error_rate': random.uniform(0, 0.1),
            'response_time': random.uniform(50, 2000)
        }

    def detect_recovery(self, metrics_before: Dict[str, float],
                       metrics_after: Dict[str, float]) -> bool:
        """Detect if system has recovered based on metrics"""
        # Simple recovery detection based on key metrics
        key_metrics = ['error_rate', 'response_time']

        for metric in key_metrics:
            if metric in metrics_before and metric in metrics_after:
                if metrics_after[metric] > metrics_before[metric] * 1.5:  # 50% degradation
                    return False

        return True

    def calculate_recovery_time(self, start_time: datetime,
                              recovery_time: datetime) -> float:
        """Calculate time taken for recovery"""
        return (recovery_time - start_time).total_seconds()


class SafetyValidator:
    """Ensures chaos experiments don't cause permanent damage"""

    def __init__(self):
        self.safety_limits = {
            'max_cpu_usage': 95.0,
            'max_memory_usage': 90.0,
            'max_response_time_ms': 5000,
            'max_error_rate': 0.5
        }
        self.abort_triggers = []

    def check_safety_limits(self, metrics: Dict[str, float]) -> List[str]:
        """Check if any safety limits have been exceeded"""
        violations = []

        if metrics.get('cpu_usage', 0) > self.safety_limits['max_cpu_usage']:
            violations.append(f"CPU usage too high: {metrics['cpu_usage']:.1f}%")

        if metrics.get('memory_usage', 0) > self.safety_limits['max_memory_usage']:
            violations.append(f"Memory usage too high: {metrics['memory_usage']:.1f}%")

        if metrics.get('response_time', 0) > self.safety_limits['max_response_time_ms']:
            violations.append(f"Response time too high: {metrics['response_time']:.0f}ms")

        if metrics.get('error_rate', 0) > self.safety_limits['max_error_rate']:
            violations.append(f"Error rate too high: {metrics['error_rate']:.2%}")

        return violations

    def should_abort_experiment(self, violations: List[str]) -> bool:
        """Determine if experiment should be aborted due to safety violations"""
        return len(violations) > 2  # Abort if 3+ safety violations


class ChaosEngineeringTestSuite:
    """
    Comprehensive chaos engineering test suite that validates all recovery playbooks
    """

    def __init__(self):
        self.failure_injector = FailureInjector()
        self.system_monitor = SystemMonitor()
        self.safety_validator = SafetyValidator()
        self.playbook_repository = None  # Will be injected

        self.experiments = self._initialize_experiments()
        self.test_results: List[ExperimentResult] = []

        # Thread safety
        self._lock = threading.Lock()

    def _initialize_experiments(self) -> List[ChaosExperiment]:
        """Initialize chaos experiments for testing all 7 playbooks"""

        return [
            # Test Memory Leak Recovery Playbook
            ChaosExperiment(
                experiment_id='exp_memory_leak',
                name='Memory Leak Injection',
                description='Simulate memory leak to test memory leak recovery playbook',
                experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
                target_service='trading_engine',
                duration_seconds=180,
                intensity='medium',
                safety_checks=['memory_usage < 90%', 'no_data_loss'],
                expected_recovery_time_seconds=300,
                playbook_to_test='memory_leak_recovery',
                tags=['memory', 'resource', 'reactive']
            ),

            # Test CPU Overload Recovery Playbook
            ChaosExperiment(
                experiment_id='exp_cpu_overload',
                name='CPU Overload Injection',
                description='Simulate CPU overload to test CPU overload recovery playbook',
                experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
                target_service='model_inference',
                duration_seconds=120,
                intensity='high',
                safety_checks=['cpu_usage < 95%', 'service_responsive'],
                expected_recovery_time_seconds=240,
                playbook_to_test='cpu_overload_recovery',
                tags=['cpu', 'resource', 'reactive']
            ),

            # Test Database Connection Recovery Playbook
            ChaosExperiment(
                experiment_id='exp_db_connection',
                name='Database Connection Failure',
                description='Simulate database connection failure to test database recovery playbook',
                experiment_type=ChaosExperimentType.DEPENDENCY_FAILURE,
                target_service='data_pipeline',
                duration_seconds=60,
                intensity='high',
                safety_checks=['backup_db_available', 'no_data_loss'],
                expected_recovery_time_seconds=180,
                playbook_to_test='database_connection_recovery',
                tags=['database', 'connection', 'reactive']
            ),

            # Test Network Latency Recovery Playbook
            ChaosExperiment(
                experiment_id='exp_network_latency',
                name='Network Latency Injection',
                description='Simulate high network latency to test network recovery playbook',
                experiment_type=ChaosExperimentType.NETWORK_FAILURE,
                target_service='api_gateway',
                duration_seconds=90,
                intensity='medium',
                safety_checks=['latency < 1000ms', 'connection_stable'],
                expected_recovery_time_seconds=120,
                playbook_to_test='network_latency_recovery',
                tags=['network', 'latency', 'reactive']
            ),

            # Test Trading Halt Recovery Playbook
            ChaosExperiment(
                experiment_id='exp_trading_halt',
                name='Trading Halt Simulation',
                description='Simulate trading halt to test trading halt recovery playbook',
                experiment_type=ChaosExperimentType.SERVICE_CRASH,
                target_service='trading_engine',
                duration_seconds=120,
                intensity='critical',
                safety_checks=['no_real_trades_affected', 'emergency_contacts_available'],
                expected_recovery_time_seconds=180,
                playbook_to_test='trading_halt_recovery',
                tags=['trading', 'halt', 'reactive']
            ),

            # Test Memory Usage Prevention Playbook (Proactive)
            ChaosExperiment(
                experiment_id='exp_memory_prevention',
                name='Memory Usage Prevention Test',
                description='Test proactive memory management playbook',
                experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
                target_service='trading_engine',
                duration_seconds=300,
                intensity='low',
                safety_checks=['memory_usage < 75%', 'proactive_scaling_works'],
                expected_recovery_time_seconds=60,
                playbook_to_test='memory_usage_prevention',
                tags=['memory', 'prevention', 'proactive']
            ),

            # Test Performance Optimization Playbook (Proactive)
            ChaosExperiment(
                experiment_id='exp_performance_optimization',
                name='Performance Optimization Test',
                description='Test proactive performance optimization playbook',
                experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
                target_service='model_inference',
                duration_seconds=240,
                intensity='medium',
                safety_checks=['performance_improvement', 'no_service_disruption'],
                expected_recovery_time_seconds=300,
                playbook_to_test='performance_optimization',
                tags=['performance', 'optimization', 'proactive']
            )
        ]

    def set_playbook_repository(self, repository):
        """Set the playbook repository to test against"""
        self.playbook_repository = repository

    async def run_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Run a specific chaos experiment"""
        experiment = next((exp for exp in self.experiments if exp.experiment_id == experiment_id), None)
        if not experiment:
            logger.error(f"Experiment not found: {experiment_id}")
            return None

        result = ExperimentResult(experiment_id=experiment_id, start_time=datetime.now())

        try:
            logger.info(f"Starting chaos experiment: {experiment.name}")

            # Phase 1: Establish baseline
            result.status = ExperimentStatus.RUNNING
            baseline_metrics = await self.system_monitor.monitor_system_state()
            self.system_monitor.set_baseline(baseline_metrics)
            result.metrics_before = baseline_metrics.copy()

            # Phase 2: Inject failure
            failure_id = await self.failure_injector.inject_failure(experiment)
            result.observations.append(f"Injected failure: {experiment.experiment_type.value}")

            # Phase 3: Wait for system to detect and respond
            await asyncio.sleep(min(experiment.duration_seconds, 30))  # Don't wait too long in test

            # Phase 4: Monitor recovery
            result.status = ExperimentStatus.VALIDATING
            recovery_start = datetime.now()

            # Wait for recovery or timeout
            max_wait = experiment.expected_recovery_time_seconds
            wait_interval = 5
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval

                current_metrics = await self.system_monitor.monitor_system_state()

                # Check safety limits
                violations = self.safety_validator.check_safety_limits(current_metrics)
                if violations:
                    result.safety_violations.extend(violations)
                    if self.safety_validator.should_abort_experiment(violations):
                        result.status = ExperimentStatus.ABORTED
                        result.errors.append("Experiment aborted due to safety violations")
                        break

                # Check if recovery occurred
                if self.system_monitor.detect_recovery(baseline_metrics, current_metrics):
                    result.recovery_time_seconds = self.system_monitor.calculate_recovery_time(
                        recovery_start, datetime.now()
                    )
                    result.success = True
                    break

            # Phase 5: Cleanup
            await self.failure_injector.remove_failure(failure_id)
            result.metrics_after = await self.system_monitor.monitor_system_state()

            # Phase 6: Validate results
            if result.status == ExperimentStatus.RUNNING:
                result.status = ExperimentStatus.COMPLETED if result.success else ExperimentStatus.FAILED

            result.end_time = datetime.now()

            # Test the appropriate playbook if available
            if self.playbook_repository and experiment.playbook_to_test:
                await self._test_playbook(experiment.playbook_to_test, result)

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            result.status = ExperimentStatus.FAILED
            result.errors.append(str(e))
            result.end_time = datetime.now()

        with self._lock:
            self.test_results.append(result)

        return result

    async def _test_playbook(self, playbook_id: str, result: ExperimentResult):
        """Test the recovery playbook"""
        if not self.playbook_repository:
            result.errors.append("No playbook repository available for testing")
            return

        try:
            logger.info(f"Testing playbook: {playbook_id}")

            # Execute the playbook
            execution = await self.playbook_repository.execute_playbook(
                playbook_id,
                {'chaos_experiment': True}
            )

            if execution:
                result.playbook_executed = playbook_id
                result.observations.append(f"Playbook execution: {execution.status.value}")

                if execution.status.value == 'completed':
                    result.observations.append("Playbook executed successfully")
                else:
                    result.errors.extend(execution.errors)
                    result.observations.append("Playbook execution had issues")
            else:
                result.errors.append(f"Failed to execute playbook: {playbook_id}")

        except Exception as e:
            result.errors.append(f"Playbook test failed: {str(e)}")

    async def run_full_test_suite(self) -> List[ExperimentResult]:
        """Run the complete chaos engineering test suite"""
        logger.info("Starting full chaos engineering test suite")

        results = []

        for experiment in self.experiments:
            logger.info(f"Running experiment: {experiment.name}")
            result = await self.run_experiment(experiment.experiment_id)
            if result:
                results.append(result)

            # Wait between experiments to allow system recovery
            await asyncio.sleep(60)

        logger.info("Completed full chaos engineering test suite")
        return results

    def get_test_results(self, experiment_id: Optional[str] = None) -> List[ExperimentResult]:
        """Get test results with optional filtering"""
        if experiment_id:
            return [result for result in self.test_results if result.experiment_id == experiment_id]
        return self.test_results.copy()

    def get_test_statistics(self) -> Dict[str, Any]:
        """Get statistics about the chaos engineering tests"""
        total_tests = len(self.test_results)
        if total_tests == 0:
            return {'total_tests': 0}

        successful = sum(1 for result in self.test_results if result.success)
        failed = sum(1 for result in self.test_results if result.status == ExperimentStatus.FAILED)
        aborted = sum(1 for result in self.test_results if result.status == ExperimentStatus.ABORTED)

        success_rate = successful / total_tests * 100

        # Recovery time statistics
        recovery_times = [result.recovery_time_seconds for result in self.test_results
                         if result.success and result.recovery_time_seconds > 0]
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0

        # Playbook effectiveness
        playbook_results = {}
        for result in self.test_results:
            if result.playbook_executed:
                if result.playbook_executed not in playbook_results:
                    playbook_results[result.playbook_executed] = []
                playbook_results[result.playbook_executed].append(result.success)

        playbook_effectiveness = {
            playbook: sum(results) / len(results) * 100
            for playbook, results in playbook_results.items()
        }

        return {
            'total_tests': total_tests,
            'successful_tests': successful,
            'failed_tests': failed,
            'aborted_tests': aborted,
            'success_rate': success_rate,
            'average_recovery_time_seconds': avg_recovery_time,
            'playbook_effectiveness': playbook_effectiveness,
            'safety_violations': sum(len(result.safety_violations) for result in self.test_results)
        }

    def generate_test_report(self, path: str):
        """Generate a comprehensive test report"""
        statistics = self.get_test_statistics()

        report = {
            'test_suite': 'Chaos Engineering Test Suite',
            'timestamp': datetime.now().isoformat(),
            'statistics': statistics,
            'experiments': [
                {
                    'experiment_id': result.experiment_id,
                    'status': result.status.value,
                    'success': result.success,
                    'recovery_time_seconds': result.recovery_time_seconds,
                    'playbook_executed': result.playbook_executed,
                    'safety_violations': len(result.safety_violations),
                    'errors': result.errors,
                    'observations': result.observations
                }
                for result in self.test_results
            ]
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Generated chaos engineering test report: {path}")

    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        statistics = self.get_test_statistics()

        if statistics['success_rate'] < 80:
            recommendations.append("Improve playbook success rate through better error handling")

        if statistics['average_recovery_time_seconds'] > 300:
            recommendations.append("Optimize recovery procedures to reduce mean time to recovery")

        # Playbook-specific recommendations
        for playbook, effectiveness in statistics.get('playbook_effectiveness', {}).items():
            if effectiveness < 80:
                recommendations.append(f"Review and improve {playbook} playbook (current effectiveness: {effectiveness:.1f}%)")

        if statistics.get('safety_violations', 0) > 5:
            recommendations.append("Review safety limits and experiment parameters to reduce safety violations")

        return recommendations


# Example usage and testing
if __name__ == "__main__":
    async def test_chaos_engineering_suite():
        # Create test suite
        suite = ChaosEngineeringTestSuite()

        # Run a single experiment
        print("Running CPU overload experiment...")
        result = await suite.run_experiment('exp_cpu_overload')

        if result:
            print(f"Experiment completed: {result.status.value}")
            print(f"Success: {result.success}")
            print(f"Recovery time: {result.recovery_time_seconds:.1f}s")
            print(f"Errors: {result.errors}")
            print(f"Observations: {result.observations}")

        # Get statistics
        stats = suite.get_test_statistics()
        print(f"\nTest Statistics: {stats}")

        # Generate recommendations
        recommendations = suite.get_recommendations()
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")

        # Generate report
        suite.generate_test_report("chaos_test_report.json")

    # Run test
    asyncio.run(test_chaos_engineering_suite())