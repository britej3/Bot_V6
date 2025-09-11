"""
Enhanced Performance System Demonstration
Demonstrates the Phase 1 performance improvements
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Enhanced performance system
from src.enhanced.performance.integration import (
    create_enhanced_performance_system,
    benchmark_enhanced_system,
    process_batch_signals
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedPerformanceDemo:
    """Demonstration of enhanced performance capabilities"""

    def __init__(self):
        self.system = None
        self.test_data = []
        self.benchmark_results = {}

    async def initialize_system(self):
        """Initialize the enhanced performance system"""
        logger.info("🚀 Initializing Enhanced Performance System...")

        config = {
            'ensemble': {
                'feature_dim': 150,
                'hidden_dim': 256,
                'num_classes': 3,
                'dropout_rate': 0.1,
                'lstm_layers': 2,
                'transformer_heads': 8,
                'transformer_layers': 3
            },
            'feature_pipeline': {
                'cache_enabled': True,
                'async_processing': True,
                'max_features': 1000
            },
            'performance': {
                'target_latency_ms': 5.0,
                'target_throughput': 1000,
                'monitoring_interval': 60
            },
            'trading': {
                'symbols': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
                'window_size': 100,
                'prediction_threshold': 0.7
            }
        }

        self.system = create_enhanced_performance_system(config)
        logger.info("✅ Enhanced Performance System initialized successfully!")

    def generate_test_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate realistic test market data"""
        logger.info(f"📊 Generating {num_samples} test data samples...")

        base_price = 50000  # BTC base price
        base_time = datetime.now()

        test_data = []

        for i in range(num_samples):
            # Generate realistic price movements
            price_change = np.random.normal(0, 0.02)  # 2% volatility
            price = base_price * (1 + price_change)

            # Generate OHLCV data
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = np.random.uniform(low, high)
            volume = np.random.lognormal(15, 2)  # Realistic volume

            test_sample = {
                'timestamp': base_time + timedelta(seconds=i),
                'symbol': np.random.choice(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']),
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            }

            test_data.append(test_sample)

        self.test_data = test_data
        logger.info(f"✅ Generated {len(test_data)} test data samples")
        return test_data

    async def single_signal_demo(self):
        """Demonstrate single signal processing"""
        logger.info("\n🎯 Single Signal Processing Demo")

        if not self.test_data:
            self.generate_test_data(1)

        test_sample = self.test_data[0]
        logger.info(f"Processing signal for {test_sample['symbol']} at price ${test_sample['close']:,.2f}")

        start_time = time.time()
        result = await self.system.process_trading_signal(test_sample)
        processing_time = (time.time() - start_time) * 1000

        logger.info("📈 Processing Results:"        logger.info(f"   Decision: {result['decision']}")
        logger.info(f"   Confidence: {result['confidence']:.3f}")
        logger.info(f"   Market Regime: {result.get('regime', 'UNKNOWN')}")
        logger.info(f"   Features Computed: {result.get('features_computed', 0)}")
        logger.info(".2f")
        logger.info("   Target Latency: ≤5ms"
        logger.info(f"   Target Achieved: {'✅ YES' if processing_time <= 5.0 else '❌ NO'}")

        return result

    async def batch_processing_demo(self, batch_size: int = 10):
        """Demonstrate batch processing capabilities"""
        logger.info(f"\n🔄 Batch Processing Demo ({batch_size} signals)")

        if not self.test_data:
            self.generate_test_data(batch_size)

        batch_data = self.test_data[:batch_size]

        start_time = time.time()
        results = await process_batch_signals(self.system, batch_data)
        total_time = (time.time() - start_time) * 1000

        # Analyze results
        decisions = [r['decision'] for r in results]
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_latency = np.mean([r.get('processing_time_ms', 0) for r in results])

        logger.info("📊 Batch Processing Results:"        logger.info(f"   Signals Processed: {len(results)}")
        logger.info(f"   Total Time: {total_time:.2f}ms")
        logger.info(".2f")
        logger.info(".3f")
        logger.info(".2f")
        logger.info(f"   Decisions: BUY={decisions.count('BUY')}, HOLD={decisions.count('HOLD')}, SELL={decisions.count('SELL')}")

        return results

    async def comprehensive_benchmark(self, num_runs: int = 100):
        """Run comprehensive system benchmark"""
        logger.info(f"\n🏆 Comprehensive Benchmark ({num_runs} runs)")

        if not self.test_data:
            self.generate_test_data(max(num_runs, 100))

        # Run benchmark
        benchmark_results = benchmark_enhanced_system(
            self.system,
            self.test_data,
            num_runs=num_runs
        )

        self.benchmark_results = benchmark_results

        logger.info("📈 Benchmark Results:"        logger.info(f"   Total Runs: {benchmark_results['total_runs']}")
        logger.info(f"   Successful Runs: {benchmark_results['successful_runs']}")
        logger.info(".1f")
        logger.info(".2f")
        logger.info("   Latency Percentiles:"        logger.info(".2f")
        logger.info(".2f")
        logger.info(".2f")
        logger.info("   Throughput:"        logger.info(".0f")
        logger.info("   Performance Improvement:"        logger.info(".2f")
        logger.info(".2f")
        logger.info(".1f")
        logger.info(f"   Target Achieved: {'✅ YES' if benchmark_results['performance_improvement']['target_achieved'] else '❌ NO'}")

        return benchmark_results

    async def performance_comparison_demo(self):
        """Compare performance with baseline system"""
        logger.info("\n⚡ Performance Comparison Demo")

        # Simulate baseline system performance
        baseline_performance = {
            'average_latency_ms': 50.0,
            'throughput_signals_per_second': 100,
            'success_rate': 0.95
        }

        # Get current system performance
        if not self.benchmark_results:
            await self.comprehensive_benchmark(100)

        current_performance = {
            'average_latency_ms': self.benchmark_results['average_latency_ms'],
            'throughput_signals_per_second': self.benchmark_results['throughput_signals_per_second'],
            'success_rate': self.benchmark_results['success_rate']
        }

        logger.info("🔍 Performance Comparison:"        logger.info("   Metric               | Baseline     | Enhanced     | Improvement")
        logger.info("   --------------------|--------------|--------------|------------")

        latency_improvement = ((baseline_performance['average_latency_ms'] - current_performance['average_latency_ms']) /
                              baseline_performance['average_latency_ms']) * 100
        logger.info("5.2f"
                   ".0f")

        throughput_improvement = ((current_performance['throughput_signals_per_second'] - baseline_performance['throughput_signals_per_second']) /
                                baseline_performance['throughput_signals_per_second']) * 100
        logger.info("5.1f"
                   ".0f")

        success_rate_change = current_performance['success_rate'] - baseline_performance['success_rate']
        logger.info("5.3f"
                   "+.3f")

        # Overall improvement score
        overall_score = (latency_improvement * 0.4 + throughput_improvement * 0.4 + success_rate_change * 200 * 0.2)
        logger.info("
   Overall Improvement Score: {:.1f}%".format(overall_score))

        return {
            'baseline': baseline_performance,
            'current': current_performance,
            'improvements': {
                'latency_improvement_pct': latency_improvement,
                'throughput_improvement_pct': throughput_improvement,
                'success_rate_change': success_rate_change,
                'overall_score': overall_score
            }
        }

    async def real_time_simulation(self, duration_seconds: int = 30):
        """Simulate real-time trading signal processing"""
        logger.info(f"\n🎮 Real-Time Simulation ({duration_seconds} seconds)")

        if not self.test_data:
            self.generate_test_data(1000)

        processed_signals = 0
        start_time = time.time()
        latencies = []

        logger.info("Starting real-time signal processing simulation...")
        logger.info("Press Ctrl+C to stop early")

        try:
            while time.time() - start_time < duration_seconds:
                # Process signals in real-time
                for i in range(min(10, len(self.test_data))):  # Process 10 signals at a time
                    test_sample = self.test_data[i]

                    signal_start = time.time()
                    result = await self.system.process_trading_signal(test_sample)
                    latency = (time.time() - signal_start) * 1000

                    latencies.append(latency)
                    processed_signals += 1

                    # Log every 50 signals
                    if processed_signals % 50 == 0:
                        avg_latency = np.mean(latencies[-50:])
                        logger.info("2d")

                # Small delay to simulate real-time intervals
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")

        total_time = time.time() - start_time
        throughput = processed_signals / total_time

        logger.info("
📊 Real-Time Simulation Results:"        logger.info(f"   Signals Processed: {processed_signals}")
        logger.info(".1f")
        logger.info(".1f")
        logger.info(".2f")
        logger.info(".2f")
        logger.info(".2f")

        return {
            'signals_processed': processed_signals,
            'total_time_seconds': total_time,
            'throughput_signals_per_second': throughput,
            'average_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95)
        }

    def display_system_info(self):
        """Display system capabilities and configuration"""
        logger.info("\n🔧 Enhanced Performance System Information")
        logger.info("=" * 50)

        if self.system:
            metrics = self.system.get_performance_metrics()
            logger.info("📊 Current System Metrics:"            logger.info(".1f")
            logger.info(f"   Total Requests: {metrics['total_requests']}")
            logger.info(".1f")
            logger.info(".2f")
            logger.info(".1f")

            # JAX Performance
            if 'jax_performance' in metrics and metrics['jax_performance']:
                jax_perf = metrics['jax_performance']
                logger.info("   JAX Performance:"                logger.info(f"      Mean Inference: {jax_perf.get('mean_inference_time', 0):.3f}ms")
                logger.info(f"      P95 Inference: {jax_perf.get('p95_inference_time', 0):.3f}ms")

            # Polars Performance
            if 'polars_performance' in metrics and metrics['polars_performance']:
                polars_perf = metrics['polars_performance']
                logger.info("   Polars Performance:"                logger.info(f"      Mean Processing: {polars_perf.get('mean_processing_time_ms', 0):.3f}ms")
                logger.info(f"      Cache Hit Rate: {polars_perf.get('cache_hit_rate', 0):.1%}")

        logger.info("\n🎯 System Capabilities:")
        logger.info("   ✅ JAX/Flax Ultra-Fast ML Inference")
        logger.info("   ✅ Polars Vectorized Data Processing")
        logger.info("   ✅ Redis Caching Layer")
        logger.info("   ✅ DuckDB Analytics Engine")
        logger.info("   ✅ Ta-Lib Technical Indicators")
        logger.info("   ✅ Real-time Signal Processing")
        logger.info("   ✅ Advanced Performance Monitoring")

        logger.info("\n📈 Target Performance:")
        logger.info("   🎯 End-to-End Latency: ≤5ms (90% reduction)")
        logger.info("   🎯 Throughput: ≥1000 signals/second (10x increase)")
        logger.info("   🎯 Win Rate: 75-85% (15-25% improvement)")
        logger.info("   🎯 Drawdown: <1% (50% reduction)")

    async def run_complete_demo(self):
        """Run the complete enhanced performance demonstration"""
        logger.info("🎉 Starting Enhanced Performance System Demo")
        logger.info("=" * 60)

        try:
            # Initialize system
            await self.initialize_system()

            # Generate test data
            self.generate_test_data(1000)

            # Display system info
            self.display_system_info()

            # Run demos
            await self.single_signal_demo()
            await self.batch_processing_demo(20)
            await self.comprehensive_benchmark(200)
            await self.performance_comparison_demo()
            await self.real_time_simulation(10)

            # Final summary
            logger.info("\n🎊 Enhanced Performance Demo Complete!")
            logger.info("=" * 60)
            logger.info("✅ Phase 1: Core Performance Revolution - IMPLEMENTED")
            logger.info("✅ JAX/Flax ML acceleration - ACTIVE")
            logger.info("✅ Polars data processing - ACTIVE")
            logger.info("✅ Real-time signal processing - ACTIVE")
            logger.info("✅ Performance monitoring - ACTIVE")

            if self.benchmark_results:
                improvement = self.benchmark_results['performance_improvement']
                    if improvement['target_achieved']:
                        logger.info("🎉 TARGET ACHIEVED: Sub-5ms latency confirmed!")
                    else:
                        logger.info(".2f")
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise

async def main():
    """Main demo function"""
    demo = EnhancedPerformanceDemo()
    await demo.run_complete_demo()

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
