#!/usr/bin/env python3
"""
Quick Import Validation Script
==============================

This script tests if the import issues have been resolved.
"""

import sys
import os
sys.path.append('src')

def test_imports():
    """Test critical imports"""
    print("🔍 Testing critical imports...")
    
    try:
        # Test 1: HighFrequencyTradingEngine (using explicit path)
        print("1. Testing HighFrequencyTradingEngine import...")
        trading_dir = os.path.join('src', 'trading')
        sys.path.insert(0, trading_dir)
        try:
            import hft_engine as hft_module
            HighFrequencyTradingEngine = hft_module.HighFrequencyTradingEngine
        finally:
            if trading_dir in sys.path:
                sys.path.remove(trading_dir)
        print("   ✅ HighFrequencyTradingEngine imported successfully")
        
        # Test 2: NautilusTraderManager
        print("2. Testing NautilusTraderManager import...")
        from src.trading.nautilus_integration import NautilusTraderManager
        print("   ✅ NautilusTraderManager imported successfully")
        
        # Test 3: Adaptive Risk Management
        print("3. Testing AdaptiveRiskManager import...")
        from src.learning.adaptive_risk_management import AdaptiveRiskManager
        print("   ✅ AdaptiveRiskManager imported successfully")
        
        # Test 4: Market Regime Detection
        print("4. Testing MarketRegimeDetector import...")
        from src.learning.market_regime_detection import MarketRegimeDetector
        print("   ✅ MarketRegimeDetector imported successfully")
        
        # Test 5: Basic functionality
        print("5. Testing basic functionality...")
        risk_manager = AdaptiveRiskManager()
        regime_detector = MarketRegimeDetector()
        nautilus_manager = NautilusTraderManager()
        hft_engine = HighFrequencyTradingEngine()
        
        print("   ✅ All components initialized successfully")
        
        print("\n🎉 ALL IMPORTS AND INITIALIZATION SUCCESSFUL!")
        print("✅ The import issues have been resolved!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n🚀 Ready to run validation tests!")
        exit(0)
    else:
        print("\n💥 Import issues need to be resolved first!")
        exit(1)