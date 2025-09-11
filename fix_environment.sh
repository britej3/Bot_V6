#!/bin/bash

# Advanced Risk Management Environment Fix Script
# ===============================================
# Resolves environment issues preventing full validation of Task Assignment #002
# 
# Issues Fixed:
# 1. NumPy 1.x/2.x compatibility conflict with PyTorch
# 2. Missing psycopg2 for database connectivity  
# 3. Missing dependencies for backtesting and stress testing
#
# Author: Validation Team
# Date: 2025-08-25

set -e  # Exit on any error

echo "🔧 Advanced Risk Management Environment Fix"
echo "==========================================="
echo "Task ID: RISK_STRATEGY_002"
echo "Fixing environment issues preventing full validation..."
echo ""

# Check current environment
echo "📋 Current Environment Status:"
echo "Python Version: $(python --version)"
echo "Platform: $(uname -s) $(uname -m)"
echo ""

# Check current package versions
echo "📦 Current Package Versions:"
python -c "
try:
    import numpy
    print(f'NumPy: {numpy.__version__}')
except ImportError:
    print('NumPy: Not installed')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
except ImportError as e:
    print(f'PyTorch: Import failed - {e}')

try:
    import psycopg2
    print(f'psycopg2: Available')
except ImportError:
    print('psycopg2: Not installed')
"
echo ""

# Fix 1: NumPy Version Compatibility
echo "🔧 Fix 1: Resolving NumPy/PyTorch Compatibility"
echo "------------------------------------------------"
echo "Current issue: NumPy 2.x incompatible with existing PyTorch installation"
echo "Solution: Downgrade NumPy to <2.0 for compatibility"
echo ""

echo "📦 Installing compatible NumPy version..."
pip install "numpy<2.0" --force-reinstall
echo "✅ NumPy compatibility fixed"
echo ""

# Fix 2: Install psycopg2
echo "🔧 Fix 2: Installing Database Connectivity"  
echo "-------------------------------------------"
echo "Current issue: psycopg2 not installed, blocking test infrastructure"
echo "Solution: Install psycopg2-binary for PostgreSQL connectivity"
echo ""

# Check if PostgreSQL is available (needed for psycopg2)
if command -v brew >/dev/null 2>&1; then
    echo "📦 Installing PostgreSQL via Homebrew..."
    brew install postgresql || echo "PostgreSQL may already be installed"
else
    echo "⚠️  Homebrew not found. You may need to install PostgreSQL manually."
fi

echo "📦 Installing psycopg2-binary..."
pip install psycopg2-binary --force-reinstall
echo "✅ Database connectivity installed"
echo ""

# Fix 3: Reinstall PyTorch with correct NumPy
echo "🔧 Fix 3: Reinstalling PyTorch"
echo "-------------------------------"
echo "Current issue: PyTorch compiled against wrong NumPy version"
echo "Solution: Reinstall PyTorch to match NumPy version"
echo ""

echo "📦 Uninstalling existing PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo "📦 Installing PyTorch with NumPy compatibility..."
pip install torch torchvision torchaudio
echo "✅ PyTorch reinstalled"
echo ""

# Fix 4: Install additional testing dependencies
echo "🔧 Fix 4: Installing Testing Dependencies"
echo "------------------------------------------"
echo "Installing additional packages needed for comprehensive testing..."
echo ""

echo "📦 Installing pytest and testing tools..."
pip install pytest pytest-asyncio pytest-mock

echo "📦 Installing scientific computing packages..."
pip install scipy scikit-learn pandas

echo "📦 Installing additional ML packages..."
pip install xgboost

echo "✅ Testing dependencies installed"
echo ""

# Verification
echo "🧪 Verification Tests"
echo "====================="
echo "Running verification tests to ensure environment is working..."
echo ""

echo "✅ Testing Python imports..."
python -c "
import sys
print(f'✅ Python {sys.version}')

# Test NumPy
import numpy as np
print(f'✅ NumPy {np.__version__}')

# Test PyTorch
import torch
print(f'✅ PyTorch {torch.__version__}')

# Test psycopg2
import psycopg2
print('✅ psycopg2 available')

# Test scientific packages
import scipy
print(f'✅ SciPy {scipy.__version__}')

import sklearn
print(f'✅ scikit-learn {sklearn.__version__}')

import pandas as pd
print(f'✅ Pandas {pd.__version__}')

print('\\n🎯 All critical packages imported successfully!')
"

echo ""
echo "✅ Testing Bot_V6 specific imports..."
cd "$(dirname "$0")"
python -c "
import sys
sys.path.append('src')

# Test risk management imports
from learning.adaptive_risk_management import AdaptiveRiskManager, RiskLevel
print('✅ AdaptiveRiskManager imports successfully')

# Test dynamic strategy switching
from learning.dynamic_strategy_switching import DynamicStrategyManager
print('✅ DynamicStrategyManager imports successfully') 

# Test monitoring system
from monitoring.comprehensive_monitoring import ComprehensiveMonitoringSystem
print('✅ ComprehensiveMonitoringSystem imports successfully')

# Test trading engine
from trading.trading_engine import TradingEngine  
print('✅ TradingEngine imports successfully')

print('\\n🎯 All Bot_V6 components import successfully!')
"

echo ""
echo "🎯 ENVIRONMENT FIX COMPLETE!"
echo "============================"
echo ""
echo "✅ All environment issues resolved:"
echo "   - NumPy/PyTorch compatibility fixed"
echo "   - psycopg2 database connectivity installed"  
echo "   - All testing dependencies available"
echo "   - Bot_V6 components importing successfully"
echo ""
echo "🚀 Ready to run complete validation:"
echo "   - Adaptive position sizing backtesting"
echo "   - Market regime detection model training"
echo "   - Stress testing for extreme scenarios"
echo "   - Full integration test suite"
echo ""
echo "Next steps:"
echo "1. Run: python validate_implementations.py"
echo "2. Run: python -m pytest tests/integration/ -v"
echo "3. Run: python -m pytest tests/backtesting/ -v" 
echo "4. Run: python -m pytest tests/stress/ -v"
echo ""
echo "🎉 Task Assignment #002 environment ready for full validation!"