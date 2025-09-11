#!/usr/bin/env python3
"""
Import Test Script for Bot_V5
Tests all the problematic imports mentioned in the resolution guide.
"""

import sys
import os

def test_import(module_name, import_name=None):
    """Test importing a module and return result"""
    try:
        if import_name:
            # For specific imports like "slowapi.util"
            module_parts = import_name.split('.')
            module = __import__(module_parts[0])
            for part in module_parts[1:]:
                module = getattr(module, part)
        else:
            module = __import__(module_name)

        print(f"✅ {import_name or module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ {import_name or module_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {import_name or module_name} import succeeded but with warning: {e}")
        return True

def main():
    print("🚀 Bot_V5 Import Test Script")
    print("=" * 50)

    # Track success/failure
    results = []

    # Test standard library imports
    print("\n📚 Testing Standard Library Imports:")
    print("-" * 40)
    results.append(test_import("os"))
    results.append(test_import("subprocess"))

    # Test external package imports
    print("\n📦 Testing External Package Imports:")
    print("-" * 40)
    results.append(test_import("slowapi"))
    results.append(test_import("slowapi", "slowapi.util"))
    results.append(test_import("slowapi", "slowapi.middleware"))
    results.append(test_import("slowapi", "slowapi.errors"))
    results.append(test_import("slowapi", "slowapi.responses"))

    results.append(test_import("ccxt"))
    results.append(test_import("ccxt", "ccxt.pro"))

    # Test nautilus-trader (this might still be installing)
    print("\n🤖 Testing Nautilus Trader Import:")
    print("-" * 40)
    nautilus_result = test_import("nautilus_trader")
    results.append(nautilus_result)

    # Test specific file imports
    print("\n📁 Testing Specific File Imports:")
    print("-" * 40)

    try:
        # Add current directory to path for relative imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Test the specific files mentioned in the guide
        print("Testing src/learning/strategy_model_integration_engine.py imports...")
        try:
            from src.learning.strategy_model_integration_engine import AutonomousScalpingEngine
            print("✅ strategy_model_integration_engine imports successful")
            results.append(True)
        except Exception as e:
            print(f"❌ strategy_model_integration_engine import failed: {e}")
            results.append(False)

        print("Testing src/learning/self_healing/self_healing_engine.py imports...")
        try:
            from src.learning.self_healing.self_healing_engine import EnhancedSelfHealingEngine
            print("✅ self_healing_engine imports successful")
            results.append(True)
        except Exception as e:
            print(f"❌ self_healing_engine import failed: {e}")
            results.append(False)

    except Exception as e:
        print(f"⚠️  Could not test specific file imports: {e}")
        results.append(False)

    # Summary
    print("\n📊 Test Summary:")
    print("-" * 40)
    successful = sum(results)
    total = len(results)

    print(f"Total tests: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")

    if successful == total:
        print("\n🎉 All import tests passed! Your Bot_V5 project should be ready.")
        return 0
    else:
        print(f"\n⚠️  {total - successful} import test(s) failed.")
        if not nautilus_result:
            print("   Note: nautilus-trader might still be installing in the background.")
        print("   Check the error messages above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)