#!/usr/bin/env python3
"""
Cross-Platform Launcher for Historical Tick Data Download
Downloads 1-min tick data for 2024 and 2025 (until June)
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def red(cls, text):
        return f"{cls.RED}{text}{cls.NC}"
    
    @classmethod
    def green(cls, text):
        return f"{cls.GREEN}{text}{cls.NC}"
    
    @classmethod
    def yellow(cls, text):
        return f"{cls.YELLOW}{text}{cls.NC}"
    
    @classmethod
    def blue(cls, text):
        return f"{cls.BLUE}{text}{cls.NC}"
    
    @classmethod
    def cyan(cls, text):
        return f"{cls.CYAN}{text}{cls.NC}"


def print_banner():
    """Print the application banner"""
    print(Colors.blue("=" * 50))
    print(Colors.blue("   Historical Tick Data Downloader"))
    print(Colors.blue("   Period: 2024 + 2025 (Jan-June)"))
    print(Colors.blue("   Platform: XGBoost Trading System"))
    print(Colors.blue("=" * 50))


def check_environment(project_dir: Path) -> bool:
    """Check if the environment is properly set up"""
    print(Colors.cyan("🔧 Checking environment..."))
    
    # Check virtual environment
    venv_path = project_dir / "venv"
    if not venv_path.exists():
        print(Colors.red("❌ Virtual environment not found"))
        print(Colors.yellow("💡 Please create virtual environment first:"))
        print(f"   cd {project_dir}")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("   pip install -r requirements.txt")
        return False
    
    print(Colors.green("✅ Virtual environment found"))
    
    # Check if we can import required modules
    try:
        # Add project to path
        sys.path.insert(0, str(project_dir))
        
        import pandas
        import numpy
        import asyncio
        print(Colors.green("✅ Core dependencies available"))
        
        # Try to import project modules
        from src.config.trading_config import AdvancedTradingConfig
        from src.data_pipeline.binance_data_manager import BinanceDataManager
        print(Colors.green("✅ Project modules accessible"))
        
        return True
        
    except ImportError as e:
        print(Colors.red(f"❌ Missing dependencies: {e}"))
        print(Colors.yellow("💡 Please install requirements:"))
        print("   pip install -r requirements.txt")
        return False


def load_config(project_dir: Path) -> dict:
    """Load download configuration"""
    config_path = project_dir / "config" / "tickdata_download_config.json"
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(Colors.green("✅ Configuration loaded"))
            return config
        except Exception as e:
            print(Colors.yellow(f"⚠️  Error loading config: {e}"))
    
    # Return default config
    return {
        "download_config": {
            "symbols": {
                "major_pairs": [
                    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
                    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT"
                ]
            }
        }
    }


def get_download_estimate(symbols: list, start_date: str, end_date: str) -> dict:
    """Estimate download size and time"""
    from datetime import datetime
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    days = (end_dt - start_dt).days
    
    # Rough estimates based on 1-minute data
    records_per_day = 1440  # 24 * 60 minutes
    total_records = len(symbols) * days * records_per_day
    
    # Estimate file sizes (very rough)
    mb_per_million_records = 50  # Compressed parquet
    estimated_size_mb = (total_records / 1_000_000) * mb_per_million_records
    
    # Estimate time (very rough, depends on API limits)
    estimated_hours = (days * len(symbols)) / 100  # Assuming ~100 symbol-days per hour
    
    return {
        'symbols': len(symbols),
        'days': days,
        'total_records': total_records,
        'estimated_size_mb': estimated_size_mb,
        'estimated_hours': estimated_hours
    }


def confirm_download(estimate: dict) -> bool:
    """Ask user to confirm the download"""
    print(Colors.yellow("\n📊 Download Estimate:"))
    print(f"   🎯 Symbols: {estimate['symbols']}")
    print(f"   📅 Days: {estimate['days']}")
    print(f"   📈 Estimated records: {estimate['total_records']:,}")
    print(f"   💾 Estimated size: {estimate['estimated_size_mb']:.1f} MB")
    print(f"   ⏱️  Estimated time: {estimate['estimated_hours']:.1f} hours")
    
    print(Colors.yellow("\n⚠️  This will download a large amount of data and may take several hours."))
    print(Colors.yellow("   Please ensure you have sufficient disk space and stable internet."))
    
    try:
        response = input("\nDo you want to continue? (y/N): ").strip().lower()
        return response in ['y', 'yes']
    except KeyboardInterrupt:
        print(Colors.yellow("\n🛑 Cancelled by user"))
        return False


def run_download(project_dir: Path, args: argparse.Namespace) -> bool:
    """Run the actual download"""
    print(Colors.green("\n🚀 Starting tick data download..."))
    
    # Build command
    script_path = project_dir / "scripts" / "download_historical_tickdata.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--output-dir", args.output_dir,
        "--symbols"
    ] + args.symbols
    
    if args.resume:
        cmd.append("--resume")
    
    print(Colors.blue(f"📝 Command: {' '.join(cmd)}"))
    
    try:
        # Change to project directory
        os.chdir(project_dir)
        
        # Set PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_dir)
        
        # Run the download
        result = subprocess.run(cmd, env=env)
        
        if result.returncode == 0:
            print(Colors.green("\n🎉 Download completed successfully!"))
            print(Colors.green(f"📁 Data saved to: {args.output_dir}"))
            return True
        else:
            print(Colors.red("\n❌ Download failed or was interrupted"))
            print(Colors.yellow("💡 You can resume with --resume flag"))
            return False
            
    except KeyboardInterrupt:
        print(Colors.yellow("\n⏹️  Download interrupted by user"))
        print(Colors.yellow("💡 You can resume with --resume flag"))
        return False
    except Exception as e:
        print(Colors.red(f"\n❌ Error running download: {e}"))
        return False


def main():
    """Main function"""
    print_banner()
    
    # Get project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Download historical crypto tick data')
    parser.add_argument('--symbols', nargs='+', 
                      default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
                              'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT'],
                      help='Symbols to download')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-06-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default=str(project_dir / 'data' / 'historical'), 
                      help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume previous download')
    parser.add_argument('--skip-checks', action='store_true', help='Skip environment checks')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    print(Colors.cyan(f"\n📋 Configuration:"))
    print(f"   📅 Period: {args.start_date} to {args.end_date}")
    print(f"   🎯 Symbols: {args.symbols}")
    print(f"   📁 Output: {args.output_dir}")
    print(f"   🔄 Resume: {args.resume}")
    
    # Check environment
    if not args.skip_checks:
        if not check_environment(project_dir):
            return 1
    
    # Load configuration
    config = load_config(project_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get download estimate
    estimate = get_download_estimate(args.symbols, args.start_date, args.end_date)
    
    # Confirm download
    if not args.force and not args.resume:
        if not confirm_download(estimate):
            return 0
    
    # Run download
    success = run_download(project_dir, args)
    
    if success:
        print(Colors.green("\n✅ All done! Tick data download completed."))
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())