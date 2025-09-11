#!/usr/bin/env python3
"""
Performance Validation Script for Adaptive Risk Management

This script validates that the adaptive risk management system
meets the performance SLA thresholds defined in config/sla_thresholds.json
"""

import asyncio
import json
import sys
import time
import statistics
import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import psutil
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.learning.adaptive_risk_integration_service import AdaptiveRiskIntegrationService
from src.learning.adaptive_risk_management import create_adaptive_risk_manager


class PerformanceValidator:
    """Validates system performance against SLA thresholds"""
    
    def __init__(self, threshold_file: str = "config/sla_thresholds.json"):
        """Initialize validator with threshold configuration"""
        self.threshold_file = threshold_file
        self.thresholds = self._load_thresholds()
        self.results = {}
        self.passed = True
    
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load SLA thresholds from configuration file"""
        try:
            with open(self.threshold_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Threshold file not found: {self.threshold_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in threshold file: {e}")
            sys.exit(1)
    
    async def validate_risk_assessment_latency(self, service: AdaptiveRiskIntegrationService) -> bool:
        """Validate risk assessment latency meets SLA"""
        print("📊 Validating risk assessment latency...")
        
        thresholds = self.thresholds['adaptive_risk_management']['risk_assessment_latency']
        signals = []
        
        # Generate test signals
        for i in range(100):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy' if i % 2 == 0 else 'sell',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8
            })
        
        latencies = []
        for signal in signals:
            start_time = time.time()
            result = await service.process_trade_signal(signal)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        result = {
            'average_ms': avg_latency,
            'max_ms': max_latency,
            'p95_ms': p95_latency,
            'p99_ms': p99_latency,
            'passed': (
                avg_latency <= thresholds['average_ms'] and
                max_latency <= thresholds['max_ms'] and
                p95_latency <= thresholds['p95_ms'] and
                p99_latency <= thresholds['p99_ms']
            )
        }
        
        self.results['risk_assessment_latency'] = result
        
        if result['passed']:
            print(f"✅ Risk assessment latency: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms, p95={p95_latency:.2f}ms, p99={p99_latency:.2f}ms")
        else:
            print(f"❌ Risk assessment latency FAILED: avg={avg_latency:.2f}ms (threshold: {thresholds['average_ms']}ms), "
                  f"max={max_latency:.2f}ms (threshold: {thresholds['max_ms']}ms)")
            self.passed = False
        
        return result['passed']
    
    async def validate_volatility_calculation_latency(self, service: AdaptiveRiskIntegrationService) -> bool:
        """Validate volatility calculation latency meets SLA"""
        print("📊 Validating volatility calculation latency...")
        
        thresholds = self.thresholds['adaptive_risk_management']['volatility_calculation_latency']
        
        # Generate test data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        # Test different volatility methods
        methods = ['historical', 'ewma']
        all_latencies = []
        
        for method in methods:
            method_latencies = []
            for _ in range(100):
                start_time = time.time()
                result = await service.risk_manager.estimate_volatility(returns, method=method)
                end_time = time.time()
                method_latencies.append((end_time - start_time) * 1000)  # ms
            
            all_latencies.extend(method_latencies)
        
        avg_latency = statistics.mean(all_latencies)
        max_latency = max(all_latencies)
        
        result = {
            'average_ms': avg_latency,
            'max_ms': max_latency,
            'passed': (
                avg_latency <= thresholds['average_ms'] and
                max_latency <= thresholds['max_ms']
            )
        }
        
        self.results['volatility_calculation_latency'] = result
        
        if result['passed']:
            print(f"✅ Volatility calculation latency: avg={avg_latency:.2f}ms, max={max_latency:.2f}ms")
        else:
            print(f"❌ Volatility calculation latency FAILED: avg={avg_latency:.2f}ms (threshold: {thresholds['average_ms']}ms), "
                  f"max={max_latency:.2f}ms (threshold: {thresholds['max_ms']}ms)")
            self.passed = False
        
        return result['passed']
    
    async def validate_throughput(self, service: AdaptiveRiskIntegrationService) -> bool:
        """Validate system throughput meets SLA"""
        print("📊 Validating system throughput...")
        
        thresholds = self.thresholds['adaptive_risk_management']['throughput']
        
        # Test sequential processing
        signals = []
        for i in range(100):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8
            })
        
        start_time = time.time()
        for signal in signals:
            await service.process_trade_signal(signal)
        end_time = time.time()
        
        sequential_throughput = len(signals) / (end_time - start_time)
        
        # Test concurrent processing
        tasks = [service.process_trade_signal(signal) for signal in signals[:50]]
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        concurrent_throughput = len(tasks) / (end_time - start_time)
        
        result = {
            'sequential_signals_per_second': sequential_throughput,
            'concurrent_signals_per_second': concurrent_throughput,
            'passed': (
                sequential_throughput >= thresholds['min_signals_per_second'] and
                concurrent_throughput >= thresholds['min_concurrent_signals']
            )
        }
        
        self.results['throughput'] = result
        
        if result['passed']:
            print(f"✅ Throughput: sequential={sequential_throughput:.2f} signals/sec, "
                  f"concurrent={concurrent_throughput:.2f} signals/sec")
        else:
            print(f"❌ Throughput FAILED: sequential={sequential_throughput:.2f} (threshold: {thresholds['min_signals_per_second']}), "
                  f"concurrent={concurrent_throughput:.2f} (threshold: {thresholds['min_concurrent_signals']})")
            self.passed = False
        
        return result['passed']
    
    async def validate_memory_usage(self, service: AdaptiveRiskIntegrationService) -> bool:
        """Validate memory usage meets SLA"""
        print("📊 Validating memory usage...")
        
        thresholds = self.thresholds['adaptive_risk_management']['memory_usage']
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process signals under load
        signals = []
        for i in range(1000):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8
            })
        
        for signal in signals:
            await service.process_trade_signal(signal)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        result = {
            'memory_increase_mb': memory_increase,
            'passed': memory_increase <= thresholds['max_increase_mb']
        }
        
        self.results['memory_usage'] = result
        
        if result['passed']:
            print(f"✅ Memory usage: {memory_increase:.2f}MB increase")
        else:
            print(f"❌ Memory usage FAILED: {memory_increase:.2f}MB (threshold: {thresholds['max_increase_mb']}MB)")
            self.passed = False
        
        return result['passed']
    
    async def validate_cpu_usage(self, service: AdaptiveRiskIntegrationService) -> bool:
        """Validate CPU usage meets SLA"""
        print("📊 Validating CPU usage...")
        
        thresholds = self.thresholds['adaptive_risk_management']['cpu_usage']
        process = psutil.Process(os.getpid())
        
        # Get baseline CPU usage
        initial_cpu = process.cpu_percent(interval=1.0)
        
        # Process signals under load
        signals = []
        for i in range(500):
            signals.append({
                'symbol': 'BTC/USDT',
                'action': 'buy',
                'quantity': 0.1,
                'price': 50000 + i,
                'confidence': 0.8
            })
        
        start_time = time.time()
        for signal in signals:
            await service.process_trade_signal(signal)
        end_time = time.time()
        
        # Measure CPU usage during processing
        cpu_usage = process.cpu_percent(interval=0.1)
        
        result = {
            'cpu_usage_percentage': cpu_usage,
            'passed': cpu_usage <= thresholds['max_percentage_under_load']
        }
        
        self.results['cpu_usage'] = result
        
        if result['passed']:
            print(f"✅ CPU usage: {cpu_usage:.1f}% under load")
        else:
            print(f"❌ CPU usage FAILED: {cpu_usage:.1f}% (threshold: {thresholds['max_percentage_under_load']}%)")
            self.passed = False
        
        return result['passed']
    
    async def validate_error_rate(self, service: AdaptiveRiskIntegrationService) -> bool:
        """Validate error rate meets SLA"""
        print("📊 Validating error rate...")
        
        thresholds = self.thresholds['adaptive_risk_management']['error_rate']
        
        # Process signals including some invalid ones
        signals = []
        for i in range(1000):
            if i % 50 == 0:  # Add some invalid signals
                signals.append({'invalid': 'data'})
            else:
                signals.append({
                    'symbol': 'BTC/USDT',
                    'action': 'buy',
                    'quantity': 0.1,
                    'price': 50000 + i,
                    'confidence': 0.8
                })
        
        error_count = 0
        for signal in signals:
            result = await service.process_trade_signal(signal)
            if 'error' in result:
                error_count += 1
        
        error_rate = (error_count / len(signals)) * 100
        
        result = {
            'error_rate_percentage': error_rate,
            'passed': error_rate <= thresholds['max_percentage']
        }
        
        self.results['error_rate'] = result
        
        if result['passed']:
            print(f"✅ Error rate: {error_rate:.2f}%")
        else:
            print(f"❌ Error rate FAILED: {error_rate:.2f}% (threshold: {thresholds['max_percentage']}%)")
            self.passed = False
        
        return result['passed']
    
    async def validate_cold_start_performance(self) -> bool:
        """Validate cold start performance meets SLA"""
        print("📊 Validating cold start performance...")
        
        thresholds = self.thresholds['adaptive_risk_management']['cold_start']
        
        # Measure cold start time
        start_time = time.time()
        
        # Create new service instance
        service = AdaptiveRiskIntegrationService()
        await service.initialize()
        await service.start()
        
        cold_start_time = time.time() - start_time
        
        # Process first signal
        signal = {
            'symbol': 'BTC/USDT',
            'action': 'buy',
            'quantity': 0.1,
            'price': 50000,
            'confidence': 0.8
        }
        
        first_signal_start = time.time()
        result = await service.process_trade_signal(signal)
        first_signal_time = time.time() - first_signal_start
        
        # Clean up
        await service.stop()
        
        result = {
            'cold_start_seconds': cold_start_time,
            'first_signal_seconds': first_signal_time,
            'passed': (
                cold_start_time <= thresholds['max_seconds'] and
                first_signal_time <= thresholds['first_signal_max_seconds']
            )
        }
        
        self.results['cold_start'] = result
        
        if result['passed']:
            print(f"✅ Cold start: {cold_start_time:.2f}s, first signal: {first_signal_time:.2f}s")
        else:
            print(f"❌ Cold start FAILED: {cold_start_time:.2f}s (threshold: {thresholds['max_seconds']}s), "
                  f"first signal: {first_signal_time:.2f}s (threshold: {thresholds['first_signal_max_seconds']}s)")
            self.passed = False
        
        return result['passed']
    
    async def validate_sustained_load_performance(self, service: AdaptiveRiskIntegrationService) -> bool:
        """Validate sustained load performance meets SLA"""
        print("📊 Validating sustained load performance...")
        
        thresholds = self.thresholds['adaptive_risk_management']['sustained_load']
        
        # Process signals continuously
        duration = thresholds['duration_seconds']
        start_time = time.time()
        signal_count = 0
        error_count = 0
        
        while time.time() - start_time < duration:
            signal = {
                'symbol': 'BTC/USDT',
                'action': 'buy' if signal_count % 2 == 0 else 'sell',
                'quantity': 0.1,
                'price': 50000 + (signal_count % 100),
                'confidence': 0.8
            }
            
            result = await service.process_trade_signal(signal)
            signal_count += 1
            
            if 'error' in result:
                error_count += 1
            
            # Small delay to simulate real trading
            await asyncio.sleep(0.01)
        
        total_time = time.time() - start_time
        throughput = signal_count / total_time
        error_rate = (error_count / signal_count) * 100
        
        result = {
            'duration_seconds': total_time,
            'signals_processed': signal_count,
            'throughput_signals_per_second': throughput,
            'error_rate_percentage': error_rate,
            'passed': throughput >= thresholds['min_throughput']
        }
        
        self.results['sustained_load'] = result
        
        if result['passed']:
            print(f"✅ Sustained load: {throughput:.2f} signals/sec over {total_time:.1f}s")
        else:
            print(f"❌ Sustained load FAILED: {throughput:.2f} signals/sec (threshold: {thresholds['min_throughput']})")
            self.passed = False
        
        return result['passed']
    
    async def validate_scalability(self, service: AdaptiveRiskIntegrationService) -> bool:
        """Validate system scalability meets SLA"""
        print("📊 Validating system scalability...")
        
        thresholds = self.thresholds['adaptive_risk_management']['scalability']
        
        # Test with different signal volumes
        signal_volumes = [10, 50, 100, 200]
        throughput_results = {}
        
        for volume in signal_volumes:
            signals = []
            for i in range(volume):
                signals.append({
                    'symbol': 'BTC/USDT',
                    'action': 'buy',
                    'quantity': 0.1,
                    'price': 50000 + i,
                    'confidence': 0.8
                })
            
            # Process signals
            start_time = time.time()
            tasks = [service.process_trade_signal(signal) for signal in signals]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = volume / total_time
            throughput_results[volume] = throughput
        
        # Calculate degradation ratio
        base_throughput = throughput_results[10]
        high_volume_throughput = throughput_results[200]
        degradation_ratio = high_volume_throughput / base_throughput
        
        result = {
            'base_throughput': base_throughput,
            'high_volume_throughput': high_volume_throughput,
            'degradation_ratio': degradation_ratio,
            'passed': degradation_ratio >= thresholds['max_degradation_ratio']
        }
        
        self.results['scalability'] = result
        
        if result['passed']:
            print(f"✅ Scalability: degradation ratio {degradation_ratio:.2f}")
        else:
            print(f"❌ Scalability FAILED: degradation ratio {degradation_ratio:.2f} (threshold: {thresholds['max_degradation_ratio']})")
            self.passed = False
        
        return result['passed']
    
    async def run_all_validations(self) -> bool:
        """Run all performance validations"""
        print("🚀 Starting Adaptive Risk Management Performance Validation")
        print("=" * 60)
        
        # Create service instance
        service = AdaptiveRiskIntegrationService()
        await service.initialize()
        await service.start()
        
        try:
            # Run all validations
            await self.validate_risk_assessment_latency(service)
            await self.validate_volatility_calculation_latency(service)
            await self.validate_throughput(service)
            await self.validate_memory_usage(service)
            await self.validate_cpu_usage(service)
            await self.validate_error_rate(service)
            await self.validate_sustained_load_performance(service)
            await self.validate_scalability(service)
            
            # Cold start test needs a fresh instance
            await self.validate_cold_start_performance()
            
        finally:
            await service.stop()
        
        # Print summary
        print("\n" + "=" * 60)
        if self.passed:
            print("🎉 ALL PERFORMANCE VALIDATIONS PASSED!")
        else:
            print("❌ SOME PERFORMANCE VALIDATIONS FAILED!")
        
        print("\n📊 Detailed Results:")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        return self.passed
    
    def save_results(self, output_file: str = "performance_validation_results.json"):
        """Save validation results to file"""
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'thresholds_file': self.threshold_file,
                'passed': self.passed,
                'results': self.results
            }, f, indent=2)
        
        print(f"📄 Results saved to {output_file}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Validate Adaptive Risk Management Performance")
    parser.add_argument(
        "--thresholds",
        default="config/sla_thresholds.json",
        help="Path to SLA thresholds configuration file"
    )
    parser.add_argument(
        "--output",
        default="performance_validation_results.json",
        help="Path to output results file"
    )
    
    args = parser.parse_args()
    
    validator = PerformanceValidator(args.thresholds)
    passed = await validator.run_all_validations()
    validator.save_results(args.output)
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
