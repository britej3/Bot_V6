#!/bin/bash
# Production deployment script for CryptoScalp AI

set -e  # Exit on any error

echo "🚀 Starting CryptoScalp AI production deployment..."

# Check if running as root (optional, for system-level operations)
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Running as root. This is not recommended for security reasons."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs data models config

# Check dependencies
echo "🔍 Checking dependencies..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Set environment variables if not already set
if [ -z "$DB_PASSWORD" ]; then
    echo "⚠️  DB_PASSWORD not set. Using default password."
    export DB_PASSWORD="prodpassword"
fi

if [ -z "$RABBITMQ_PASSWORD" ]; then
    echo "⚠️  RABBITMQ_PASSWORD not set. Using default password."
    export RABBITMQ_PASSWORD="prodpassword"
fi

if [ -z "$SECRET_KEY" ]; then
    echo "⚠️  SECRET_KEY not set. Generating random key."
    export SECRET_KEY=$(openssl rand -hex 32)
fi

if [ -z "$JWT_SECRET_KEY" ]; then
    echo "⚠️  JWT_SECRET_KEY not set. Generating random key."
    export JWT_SECRET_KEY=$(openssl rand -hex 32)
fi

# Pull latest images
echo "📥 Pulling latest Docker images..."
docker-compose -f docker-compose.prod.yml pull

# Build images
echo "🏗️  Building Docker images..."
docker-compose -f docker-compose.prod.yml build

# Stop existing containers
echo "⏹️  Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down

# Start services
echo "▶️  Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service status
echo "📋 Checking service status..."
docker-compose -f docker-compose.prod.yml ps

# Run health checks
echo "🩺 Running health checks..."
HEALTH_CHECKS_PASSED=true

# Check main application
if curl -f http://localhost:8000/health/liveness > /dev/null 2>&1; then
    echo "✅ Main application is alive"
else
    echo "❌ Main application health check failed"
    HEALTH_CHECKS_PASSED=false
fi

# Check database
if docker-compose -f docker-compose.prod.yml exec cryptoscalp-db pg_isready > /dev/null 2>&1; then
    echo "✅ Database is ready"
else
    echo "❌ Database is not ready"
    HEALTH_CHECKS_PASSED=false
fi

# Check Redis
if docker-compose -f docker-compose.prod.yml exec cryptoscalp-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
    HEALTH_CHECKS_PASSED=false
fi

# Final status
if [ "$HEALTH_CHECKS_PASSED" = true ]; then
    echo "🎉 Deployment completed successfully!"
    echo "📊 Services are now running:"
    docker-compose -f docker-compose.prod.yml ps
    echo ""
    echo "🔗 Access the application at:"
    echo "   API: http://localhost:8000"
    echo "   Dashboard: http://localhost:8501"
    echo "   Grafana: http://localhost:3000"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo "🔐 Default credentials (change in production):"
    echo "   Grafana: admin/admin123"
    echo "   PostgreSQL: cryptoscalp/prodpassword"
else
    echo "❌ Deployment completed with errors!"
    echo "📋 Check the logs for more information:"
    echo "   docker-compose -f docker-compose.prod.yml logs"
    exit 1
fi