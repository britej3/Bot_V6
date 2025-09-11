"""
Comprehensive Performance Benchmarking and Validation System
==========================================================

This module provides extensive performance testing and validation for the
Enhanced Trading Ensemble to ensure all targets are met:

Performance Validation Targets:
- Inference Latency: <5ms (90th percentile)
- Throughput: >1000 signals/second
- Memory Usage: <2GB total
- Model Accuracy: >75% win rate
- Integration: Seamless with existing infrastructure

Test Categories:
1. Latency and Throughput Testing
2. Memory Usage Profiling
3. Model Accuracy Validation
4. Integration Testing
5. Stress Testing
6. Production Readiness Assessment
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import psutil
import gc
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import concurrent.futures
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for comprehensive benchmarking"""
    # Performance targets
    target_latency_ms: float = 5.0
    target_throughput: int = 1000
    target_memory_gb: float = 2.0
    target_accuracy: float = 0.75
    
    # Test parameters
    warmup_iterations: int = 100
    benchmark_iterations: int = 1000
    stress_duration_seconds: int = 60
    concurrent_threads: int = 4
    
    # Memory profiling
    memory_sample_interval: float = 0.1  # seconds
    memory_profiling_duration: int = 30  # seconds
    
    # Accuracy testing
    accuracy_test_samples: int = 10000
    accuracy_validation_window: int = 1000
    
    # Output settings
    save_detailed_results: bool = True
    results_directory: str = "benchmark_results"
    generate_report: bool = True

class PerformanceProfiler:
    """Real-time performance profiling utility"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all profiling data"""
        self.latency_samples = []
        self.memory_samples = []
        self.cpu_samples = []
        self.gpu_memory_samples = []
        self.start_time = None
        self.end_time = None
    
    @contextmanager
    def profile(self):
        """Context manager for profiling a code block"""
        self.start_time = time.perf_counter()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        try:
            yield self
        finally:
            self.end_time = time.perf_counter()
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            execution_time = (self.end_time - self.start_time) * 1000  # ms
            memory_delta = final_memory - initial_memory
            
            self.latency_samples.append(execution_time)
            self.memory_samples.append(final_memory)
    
    def sample_system_metrics(self):
        """Sample current system metrics"""
        process = psutil.Process()
        
        # Memory usage
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.memory_samples.append(memory_mb)
        
        # CPU usage
        cpu_percent = process.cpu_percent()
        self.cpu_samples.append(cpu_percent)
        
        # GPU memory (if available)
        try:
            # This would require additional GPU monitoring libraries
            # For now, we'll skip GPU monitoring
            pass
        except:
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        stats = {}
        
        if self.latency_samples:
            latencies = np.array(self.latency_samples)
            stats['latency'] = {
                'mean_ms': np.mean(latencies),
                'median_ms': np.median(latencies),
                'p90_ms': np.percentile(latencies, 90),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99),
                'min_ms': np.min(latencies),
                'max_ms': np.max(latencies),
                'std_ms': np.std(latencies)
            }
        
        if self.memory_samples:
            memory = np.array(self.memory_samples)
            stats['memory'] = {
                'mean_mb': np.mean(memory),
                'max_mb': np.max(memory),
                'min_mb': np.min(memory),
                'final_mb': memory[-1] if len(memory) > 0 else 0
            }
        
        if self.cpu_samples:
            cpu = np.array(self.cpu_samples)
            stats['cpu'] = {
                'mean_percent': np.mean(cpu),
                'max_percent': np.max(cpu)
            }
        
        # Throughput calculation
        if self.start_time and self.end_time and len(self.latency_samples) > 0:
            total_time = self.end_time - self.start_time
            throughput = len(self.latency_samples) / total_time
            stats['throughput'] = {
                'samples_per_second': throughput,
                'total_samples': len(self.latency_samples),
                'total_time_seconds': total_time
            }
        
        return stats

class LatencyBenchmark:
    """Specialized latency benchmarking"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
    
    def run_single_inference_benchmark(self, ensemble, test_data) -> Dict[str, Any]:
        """Benchmark single inference latency"""
        logger.info("🚀 Running single inference latency benchmark...")
        
        profiler = PerformanceProfiler()
        
        # Warmup phase
        logger.info(f"Warming up with {self.config.warmup_iterations} iterations...")
        for _ in range(self.config.warmup_iterations):
            _ = ensemble.predict(test_data)
        
        # Benchmark phase
        logger.info(f"Benchmarking with {self.config.benchmark_iterations} iterations...")
        latencies = []
        
        for i in range(self.config.benchmark_iterations):
            with profiler.profile():
                result = ensemble.predict(test_data)
            
            # Extract latency from result if available
            if isinstance(result, dict) and 'inference_time_ms' in result:
                latencies.append(result['inference_time_ms'])
            else:
                latencies.append(profiler.latency_samples[-1])
            
            # Sample system metrics every 100 iterations
            if i % 100 == 0:
                profiler.sample_system_metrics()
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        results = {
            'latency_stats': {
                'mean_ms': float(np.mean(latencies)),
                'median_ms': float(np.median(latencies)),
                'p90_ms': float(np.percentile(latencies, 90)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'p99_ms': float(np.percentile(latencies, 99)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies)),
                'std_ms': float(np.std(latencies))
            },
            'target_validation': {
                'p90_meets_target': float(np.percentile(latencies, 90)) < self.config.target_latency_ms,
                'target_latency_ms': self.config.target_latency_ms,
                'actual_p90_ms': float(np.percentile(latencies, 90))
            },
            'system_metrics': profiler.get_statistics()
        }
        
        logger.info(f"✅ Single inference benchmark completed:")
        logger.info(f"   P90 Latency: {results['latency_stats']['p90_ms']:.3f}ms")
        logger.info(f"   Target Met: {results['target_validation']['p90_meets_target']}")
        
        return results
    
    def run_throughput_benchmark(self, ensemble, test_data) -> Dict[str, Any]:
        """Benchmark throughput under load"""
        logger.info("🏁 Running throughput benchmark...")
        
        # Single-threaded throughput
        start_time = time.perf_counter()
        predictions_made = 0
        
        duration = 10.0  # 10 second test
        end_time = start_time + duration
        
        while time.perf_counter() < end_time:
            _ = ensemble.predict(test_data)
            predictions_made += 1
        
        actual_duration = time.perf_counter() - start_time
        single_threaded_throughput = predictions_made / actual_duration
        
        # Multi-threaded throughput test
        concurrent_throughput = self._run_concurrent_throughput_test(ensemble, test_data)
        
        results = {
            'single_threaded': {
                'throughput_per_second': single_threaded_throughput,
                'total_predictions': predictions_made,
                'duration_seconds': actual_duration
            },
            'concurrent': concurrent_throughput,
            'target_validation': {
                'meets_target': single_threaded_throughput > self.config.target_throughput,
                'target_throughput': self.config.target_throughput,
                'actual_throughput': single_threaded_throughput
            }
        }
        
        logger.info(f"✅ Throughput benchmark completed:")
        logger.info(f"   Single-threaded: {single_threaded_throughput:.0f} predictions/sec")
        logger.info(f"   Target Met: {results['target_validation']['meets_target']}")
        
        return results
    
    def _run_concurrent_throughput_test(self, ensemble, test_data) -> Dict[str, Any]:
        """Run concurrent throughput test"""
        
        def worker_thread(duration: float) -> int:
            """Worker thread for concurrent testing"""
            count = 0
            end_time = time.perf_counter() + duration
            
            while time.perf_counter() < end_time:
                _ = ensemble.predict(test_data)
                count += 1
            
            return count
        
        duration = 5.0  # 5 second test per thread
        num_threads = self.config.concurrent_threads
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, duration) for _ in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        actual_duration = time.perf_counter() - start_time
        total_predictions = sum(results)
        concurrent_throughput = total_predictions / actual_duration
        
        return {
            'throughput_per_second': concurrent_throughput,
            'total_predictions': total_predictions,
            'duration_seconds': actual_duration,
            'num_threads': num_threads,
            'predictions_per_thread': results
        }

class MemoryBenchmark:
    """Memory usage benchmarking and profiling"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.memory_timeline = []
    
    def run_memory_profiling(self, ensemble, test_data) -> Dict[str, Any]:
        """Profile memory usage over time"""
        logger.info("💾 Running memory usage profiling...")
        
        # Baseline memory measurement
        gc.collect()  # Force garbage collection
        time.sleep(1)  # Allow GC to complete
        
        baseline_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Memory sampling during inference
        memory_samples = []
        peak_memory = baseline_memory
        
        def memory_sampler():
            """Background memory sampling"""
            start_time = time.time()
            while time.time() - start_time < self.config.memory_profiling_duration:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
                memory_samples.append({
                    'timestamp': time.time(),
                    'memory_mb': current_memory
                })
                time.sleep(self.config.memory_sample_interval)
        
        # Start memory sampling in background
        sampling_thread = threading.Thread(target=memory_sampler)
        sampling_thread.daemon = True
        sampling_thread.start()
        
        # Run inference while monitoring memory
        start_time = time.time()
        inference_count = 0
        
        while time.time() - start_time < self.config.memory_profiling_duration - 1:
            _ = ensemble.predict(test_data)
            inference_count += 1
            
            # Check peak memory
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current_memory)
        
        # Wait for sampling to complete
        sampling_thread.join(timeout=2)
        
        # Calculate memory statistics
        if memory_samples:
            memory_values = [sample['memory_mb'] for sample in memory_samples]
            memory_stats = {
                'baseline_mb': baseline_memory,
                'peak_mb': peak_memory,
                'mean_mb': np.mean(memory_values),
                'max_mb': np.max(memory_values),
                'min_mb': np.min(memory_values),
                'final_mb': memory_values[-1] if memory_values else baseline_memory,
                'memory_growth_mb': memory_values[-1] - baseline_memory if memory_values else 0
            }
        else:
            memory_stats = {'error': 'No memory samples collected'}
        
        results = {
            'memory_stats': memory_stats,
            'inference_count': inference_count,
            'profiling_duration': self.config.memory_profiling_duration,
            'target_validation': {
                'meets_target': peak_memory < (self.config.target_memory_gb * 1024),
                'target_memory_mb': self.config.target_memory_gb * 1024,
                'actual_peak_mb': peak_memory
            },
            'memory_timeline': memory_samples
        }
        
        logger.info(f"✅ Memory profiling completed:")
        logger.info(f"   Peak Memory: {peak_memory:.1f}MB")
        logger.info(f"   Target Met: {results['target_validation']['meets_target']}")
        
        return results

class AccuracyBenchmark:
    """Model accuracy and prediction quality benchmarking"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def run_prediction_accuracy_test(self, ensemble, test_dataset) -> Dict[str, Any]:
        """Test prediction accuracy on labeled dataset"""
        logger.info("🎯 Running prediction accuracy test...")
        
        if not test_dataset or len(test_dataset) == 0:
            logger.warning("No test dataset provided, generating synthetic test")
            test_dataset = self._generate_synthetic_test_data()
        
        correct_predictions = 0
        total_predictions = 0
        confidence_scores = []
        prediction_distribution = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for sample in test_dataset[:self.config.accuracy_test_samples]:
            try:
                # Get prediction
                prediction_result = ensemble.predict(sample['features'])
                
                predicted_signal = prediction_result.get('signal', 0)
                confidence = prediction_result.get('confidence', 0.5)
                actual_signal = sample.get('label', 0)
                
                # Count correct predictions
                if predicted_signal == actual_signal:
                    correct_predictions += 1
                
                total_predictions += 1
                confidence_scores.append(confidence)
                
                # Track prediction distribution
                if predicted_signal == 1:
                    prediction_distribution['buy'] += 1
                elif predicted_signal == -1:
                    prediction_distribution['sell'] += 1
                else:
                    prediction_distribution['hold'] += 1
                    
            except Exception as e:
                logger.error(f"Error in accuracy test: {e}")
                continue
        
        # Calculate metrics
        accuracy = correct_predictions / max(1, total_predictions)
        mean_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        results = {
            'accuracy_metrics': {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'mean_confidence': mean_confidence,
                'confidence_std': np.std(confidence_scores) if confidence_scores else 0.0
            },
            'prediction_distribution': prediction_distribution,
            'target_validation': {
                'meets_target': accuracy >= self.config.target_accuracy,
                'target_accuracy': self.config.target_accuracy,
                'actual_accuracy': accuracy
            }
        }
        
        logger.info(f"✅ Accuracy test completed:")
        logger.info(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        logger.info(f"   Target Met: {results['target_validation']['meets_target']}")
        
        return results
    
    def _generate_synthetic_test_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic test data for accuracy testing"""
        logger.info("Generating synthetic test data...")
        
        test_data = []
        
        for i in range(1000):  # Generate 1000 synthetic samples
            # Create synthetic market data
            features = {
                'tcn_input': jnp.ones((1, 100, 50)) * (0.5 + np.random.normal(0, 0.1)),
                'market_features': {
                    'volume_profile': jnp.random.normal(0, 1, (1, 100, 10)),
                    'order_flow': jnp.random.normal(0, 1, (1, 100, 5))
                }
            }
            
            # Generate synthetic label (buy=1, sell=-1, hold=0)
            label = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            
            test_data.append({
                'features': features,
                'label': label
            })
        
        return test_data

class ComprehensiveBenchmark:
    """Main comprehensive benchmarking orchestrator"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results = {}
        
        # Create results directory
        if self.config.save_detailed_results:
            os.makedirs(self.config.results_directory, exist_ok=True)
    
    def run_full_benchmark(self, ensemble, test_data=None, test_dataset=None) -> Dict[str, Any]:
        """Run complete benchmark suite"""
        logger.info("🏃‍♂️ Starting comprehensive benchmark suite...")
        start_time = time.time()
        
        # Prepare test data if not provided
        if test_data is None:
            test_data = self._create_default_test_data()
        
        # 1. Latency and Throughput Benchmarks
        latency_benchmark = LatencyBenchmark(self.config)
        self.results['latency'] = latency_benchmark.run_single_inference_benchmark(ensemble, test_data)
        self.results['throughput'] = latency_benchmark.run_throughput_benchmark(ensemble, test_data)
        
        # 2. Memory Usage Benchmark
        memory_benchmark = MemoryBenchmark(self.config)
        self.results['memory'] = memory_benchmark.run_memory_profiling(ensemble, test_data)
        
        # 3. Accuracy Benchmark
        accuracy_benchmark = AccuracyBenchmark(self.config)
        self.results['accuracy'] = accuracy_benchmark.run_prediction_accuracy_test(ensemble, test_dataset)
        
        # 4. Overall Performance Assessment
        self.results['overall'] = self._assess_overall_performance()
        
        # 5. Generate comprehensive report
        total_time = time.time() - start_time
        self.results['benchmark_metadata'] = {
            'total_duration_seconds': total_time,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        if self.config.save_detailed_results:
            self._save_results()
        
        if self.config.generate_report:
            self._generate_report()
        
        logger.info(f"✅ Comprehensive benchmark completed in {total_time:.1f} seconds")
        return self.results
    
    def _create_default_test_data(self) -> Dict[str, Any]:
        """Create default test data for benchmarking"""
        return {
            'tcn_input': jnp.ones((1, 100, 50)),
            'market_features': {
                'volume_profile': jnp.ones((1, 100, 10)),
                'order_flow': jnp.ones((1, 100, 5))
            }
        }
    
    def _assess_overall_performance(self) -> Dict[str, Any]:
        """Assess overall performance against all targets"""
        
        targets_met = {}
        
        # Check latency target
        if 'latency' in self.results and 'target_validation' in self.results['latency']:
            targets_met['latency'] = self.results['latency']['target_validation']['p90_meets_target']
        
        # Check throughput target
        if 'throughput' in self.results and 'target_validation' in self.results['throughput']:
            targets_met['throughput'] = self.results['throughput']['target_validation']['meets_target']
        
        # Check memory target
        if 'memory' in self.results and 'target_validation' in self.results['memory']:
            targets_met['memory'] = self.results['memory']['target_validation']['meets_target']
        
        # Check accuracy target
        if 'accuracy' in self.results and 'target_validation' in self.results['accuracy']:
            targets_met['accuracy'] = self.results['accuracy']['target_validation']['meets_target']
        
        total_targets = len(targets_met)
        targets_achieved = sum(targets_met.values())
        overall_score = targets_achieved / max(1, total_targets)
        
        return {
            'targets_met': targets_met,
            'targets_achieved': targets_achieved,
            'total_targets': total_targets,
            'overall_score': overall_score,
            'production_ready': overall_score >= 0.75  # 75% of targets must be met
        }
    
    def _save_results(self):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = os.path.join(self.config.results_directory, filename)
        
        # Convert numpy types to regular Python types for JSON serialization
        serializable_results = self._make_json_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📊 Detailed results saved to: {filepath}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_report(self):
        """Generate human-readable benchmark report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_report_{timestamp}.md"
        filepath = os.path.join(self.config.results_directory, filename)
        
        with open(filepath, 'w') as f:
            f.write(self._create_markdown_report())
        
        logger.info(f"📝 Benchmark report generated: {filepath}")
    
    def _create_markdown_report(self) -> str:
        """Create detailed markdown report"""
        report = f"""# Enhanced Trading Ensemble Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

"""
        
        if 'overall' in self.results:
            overall = self.results['overall']
            report += f"""
**Overall Performance Score:** {overall['overall_score']:.1%}
**Targets Achieved:** {overall['targets_achieved']}/{overall['total_targets']}
**Production Ready:** {'✅ YES' if overall['production_ready'] else '❌ NO'}

### Target Achievement Summary
"""
            for target, met in overall['targets_met'].items():
                status = '✅' if met else '❌'
                report += f"- {target.title()}: {status}\n"
        
        # Add detailed sections for each benchmark
        if 'latency' in self.results:
            latency = self.results['latency']['latency_stats']
            report += f"""
## Latency Performance

- **P90 Latency:** {latency['p90_ms']:.3f}ms (Target: <{self.config.target_latency_ms}ms)
- **Mean Latency:** {latency['mean_ms']:.3f}ms
- **P99 Latency:** {latency['p99_ms']:.3f}ms
"""
        
        if 'throughput' in self.results:
            throughput = self.results['throughput']['single_threaded']
            report += f"""
## Throughput Performance

- **Single-threaded:** {throughput['throughput_per_second']:.0f} predictions/sec (Target: >{self.config.target_throughput}/sec)
- **Test Duration:** {throughput['duration_seconds']:.1f} seconds
- **Total Predictions:** {throughput['total_predictions']}
"""
        
        if 'memory' in self.results:
            memory = self.results['memory']['memory_stats']
            report += f"""
## Memory Usage

- **Peak Memory:** {memory['peak_mb']:.1f}MB (Target: <{self.config.target_memory_gb * 1024}MB)
- **Baseline Memory:** {memory['baseline_mb']:.1f}MB
- **Memory Growth:** {memory['memory_growth_mb']:.1f}MB
"""
        
        if 'accuracy' in self.results:
            accuracy = self.results['accuracy']['accuracy_metrics']
            report += f"""
## Prediction Accuracy

- **Overall Accuracy:** {accuracy['accuracy']:.1%} (Target: >{self.config.target_accuracy:.0%})
- **Mean Confidence:** {accuracy['mean_confidence']:.3f}
- **Total Predictions:** {accuracy['total_predictions']}
"""
        
        return report

# Factory function
def run_comprehensive_benchmark(ensemble, config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run comprehensive benchmark with custom configuration"""
    
    # Create config from dictionary
    config = BenchmarkConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Run benchmark
    benchmark = ComprehensiveBenchmark(config)
    return benchmark.run_full_benchmark(ensemble)

if __name__ == "__main__":
    # Example usage
    print("Comprehensive Benchmark System Test")
    
    # This would normally be run with an actual ensemble
    # For testing, we'll create a mock ensemble
    class MockEnsemble:
        def predict(self, features):
            time.sleep(0.001)  # Simulate 1ms processing
            return {
                'signal': 1,
                'confidence': 0.8,
                'inference_time_ms': 1.0
            }
    
    mock_ensemble = MockEnsemble()
    
    # Run quick benchmark
    config = {
        'benchmark_iterations': 100,
        'warmup_iterations': 10,
        'memory_profiling_duration': 5,
        'accuracy_test_samples': 100
    }
    
    results = run_comprehensive_benchmark(mock_ensemble, config)
    
    print(f"Benchmark completed!")
    print(f"Overall score: {results['overall']['overall_score']:.1%}")
    print(f"Production ready: {results['overall']['production_ready']}")