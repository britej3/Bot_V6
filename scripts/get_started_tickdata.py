#!/usr/bin/env python3
"""
Getting Started with Tick Data Download
Simple interface to start downloading historical tick data
"""

import os
import sys
from pathlib import Path

def print_banner():
    print("=" * 60)
    print("  🚀 HISTORICAL TICK DATA DOWNLOAD SYSTEM")
    print("  📅 Period: 2024 + 2025 (January - June)")
    print("  💱 Platform: XGBoost Trading System")
    print("=" * 60)

def print_options():
    print("\n📋 QUICK START OPTIONS:")
    print("")
    
    print("1️⃣  DOWNLOAD MAJOR CRYPTO PAIRS (Recommended)")
    print("   python scripts/launch_tickdata_download.py")
    print("   Downloads: BTC, ETH, BNB, XRP, ADA, SOL, DOGE, DOT, AVAX, MATIC")
    print("")
    
    print("2️⃣  DOWNLOAD SPECIFIC SYMBOLS")
    print("   python scripts/launch_tickdata_download.py --symbols BTCUSDT ETHUSDT BNBUSDT")
    print("")
    
    print("3️⃣  RESUME PREVIOUS DOWNLOAD")
    print("   python scripts/launch_tickdata_download.py --resume")
    print("")
    
    print("4️⃣  DOWNLOAD WITH CUSTOM OUTPUT DIRECTORY")
    print("   python scripts/launch_tickdata_download.py --output-dir /path/to/data")
    print("")
    
    print("5️⃣  VALIDATE DOWNLOADED DATA")
    print("   python scripts/validate_tickdata.py")
    print("")

def print_info():
    print("\n📊 WHAT YOU'LL GET:")
    print("• 1-minute resolution tick data")
    print("• 18 months of data (Jan 2024 - June 2025)")
    print("• Multiple formats: Parquet (efficient) + CSV (compatible)")
    print("• Automatic data validation and quality checks")
    print("• Resume capability for interrupted downloads")
    print("• Progress tracking and comprehensive logging")
    print("")
    
    print("💾 STORAGE ESTIMATES:")
    print("• 1 symbol: ~50 MB, ~780k records")
    print("• 10 symbols: ~500 MB, ~7.8M records")
    print("• All major pairs: ~1 GB, ~15M records")
    print("")
    
    print("⏱️  TIME ESTIMATES:")
    print("• 1 symbol: 10-15 minutes")
    print("• 10 symbols: 2-3 hours")
    print("• All major pairs: 4-6 hours")

def print_requirements():
    print("\n✅ REQUIREMENTS:")
    print("• Python 3.8+ with virtual environment")
    print("• Stable internet connection")
    print("• Sufficient disk space (see estimates above)")
    print("• No API keys required for historical data")

def check_setup():
    print("\n🔧 CHECKING SETUP...")
    
    # Check virtual environment
    venv_path = Path("/Users/britebrt/Bot_V5/venv")
    if venv_path.exists():
        print("✅ Virtual environment found")
    else:
        print("❌ Virtual environment not found")
        print("💡 Create with: python -m venv venv && source venv/bin/activate")
        return False
    
    # Check if we can import basic modules
    try:
        import pandas
        import numpy
        print("✅ Core dependencies available")
    except ImportError:
        print("❌ Missing core dependencies")
        print("💡 Install with: pip install -r requirements.txt")
        return False
    
    # Check project structure
    script_path = Path("/Users/britebrt/Bot_V5/scripts/launch_tickdata_download.py")
    if script_path.exists():
        print("✅ Download scripts available")
    else:
        print("❌ Download scripts not found")
        return False
    
    return True

def main():
    print_banner()
    print_options()
    print_info()
    print_requirements()
    
    # Quick setup check
    if check_setup():
        print("\n🎉 SYSTEM READY!")
        print("You can start downloading data using any of the options above.")
    else:
        print("\n⚠️  SETUP REQUIRED")
        print("Please complete the setup steps shown above first.")
        return 1
    
    print("\n" + "=" * 60)
    print("📖 For detailed documentation, see: docs/TICKDATA_DOWNLOAD_README.md")
    print("🆘 For help: python scripts/launch_tickdata_download.py --help")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())