#!/bin/bash
# Performance benchmarking script for CryptoScalp AI

set -e

echo "🔬 Running performance benchmarks for CryptoScalp AI..."

# Check if services are running
echo "📋 Checking service status..."
docker-compose -f docker-compose.prod.yml ps

# Run application benchmarks
echo "🚀 Running application benchmarks..."
if curl -f http://localhost:8000/health/benchmark > /dev/null 2>&1; then
    echo "✅ Application benchmarks completed"
    curl -s http://localhost:8000/health/benchmark | jq .
else
    echo "❌ Could not run application benchmarks"
fi

# Database benchmark
echo "🚀 Running database benchmarks..."
DB_BENCHMARK_OUTPUT=$(docker-compose -f docker-compose.prod.yml exec cryptoscalp-db pgbench -U cryptoscalp -c 10 -j 2 -t 1000 cryptoscalp_prod 2>&1 || true)
echo "$DB_BENCHMARK_OUTPUT"

# Redis benchmark
echo "🚀 Running Redis benchmarks..."
REDIS_BENCHMARK_OUTPUT=$(docker-compose -f docker-compose.prod.yml exec cryptoscalp-redis redis-benchmark -n 10000 -c 50 -q 2>&1 || true)
echo "$REDIS_BENCHMARK_OUTPUT"

# System benchmark
echo "🚀 Running system benchmarks..."
echo "CPU Benchmark:"
sysbench --test=cpu --cpu-max-prime=20000 run 2>/dev/null || echo "Sysbench not available"

echo "Memory Benchmark:"
sysbench --test=memory --memory-block-size=1K --memory-total-size=1G run 2>/dev/null || echo "Sysbench not available"

# Network benchmark (if iperf3 is available)
echo "Network Benchmark:"
iperf3 -c localhost -p 5201 -t 10 2>/dev/null || echo "iperf3 not available or not running"

# Container resource usage
echo "📊 Container resource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" | head -10

echo "🎉 Performance benchmarks completed!"