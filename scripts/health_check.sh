#!/bin/bash
# Health check and monitoring script for CryptoScalp AI

set -e

echo "🩺 Running comprehensive health checks for CryptoScalp AI..."

# Check if services are running
echo "📋 Checking service status..."
docker-compose -f docker-compose.prod.yml ps

# Main application health check
echo "🔧 Checking main application..."
if curl -f http://localhost:8000/health/liveness > /dev/null 2>&1; then
    echo "✅ Main application is alive"
    
    # Check readiness
    if curl -f http://localhost:8000/health/readiness > /dev/null 2>&1; then
        echo "✅ Main application is ready"
    else
        echo "❌ Main application is not ready"
    fi
    
    # Get detailed status
    echo "📊 Getting detailed status..."
    curl -s http://localhost:8000/health/status | jq .
    
else
    echo "❌ Main application is not responding"
    exit 1
fi

# Database health check
echo "🔧 Checking database..."
if docker-compose -f docker-compose.prod.yml exec cryptoscalp-db pg_isready > /dev/null 2>&1; then
    echo "✅ Database is ready"
    
    # Check database metrics
    echo "📊 Getting database metrics..."
    docker-compose -f docker-compose.prod.yml exec cryptoscalp-db psql -U cryptoscalp -c "SELECT COUNT(*) FROM pg_stat_activity;" 2>/dev/null || echo "Could not get database stats"
else
    echo "❌ Database is not ready"
fi

# Redis health check
echo "🔧 Checking Redis..."
if docker-compose -f docker-compose.prod.yml exec cryptoscalp-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
    
    # Check Redis info
    echo "📊 Getting Redis info..."
    docker-compose -f docker-compose.prod.yml exec cryptoscalp-redis redis-cli info | grep -E "used_memory_human|connected_clients|uptime_in_seconds" || echo "Could not get Redis info"
else
    echo "❌ Redis is not ready"
fi

# Check system resources
echo "🔧 Checking system resources..."
echo "CPU Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10

# Check logs for errors
echo "📋 Checking logs for recent errors..."
echo "Recent application errors:"
docker-compose -f docker-compose.prod.yml logs --since 1h cryptoscalp-app | grep -i error | tail -5 || echo "No recent errors found"

echo "Recent database errors:"
docker-compose -f docker-compose.prod.yml logs --since 1h cryptoscalp-db | grep -i error | tail -5 || echo "No recent errors found"

echo "🎉 Health check completed!"