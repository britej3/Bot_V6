#!/usr/bin/env python3
"""
Historical Tick Data Download Script
Downloads 1-minute tick data for 2024 and 2025 (until June) using BinanceDataManager

Features:
- Batch downloading with rate limiting
- Progress tracking and resumption
- Data validation and storage
- Error handling and retry logic
- CSV and Parquet export options
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
import sys
sys.path.append('/Users/britebrt/Bot_V5')

from src.config.trading_config import AdvancedTradingConfig
from src.data_pipeline.binance_data_manager import BinanceDataManager


class HistoricalDataDownloader:
    """
    Comprehensive historical data downloader for crypto tick data
    """

    def __init__(self, symbols: List[str], output_dir: str = "data/historical"):
        self.symbols = symbols
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure for historical data download
        self.config = AdvancedTradingConfig(
            symbol="BTCUSDT",  # Will be overridden per symbol
            mode="backtest",
            binance_testnet=False,  # Use mainnet for historical data
            historical_data_days=30,
            mlflow_tracking=False,
            redis_ml_enabled=False
        )
        
        # Download parameters
        self.batch_size_days = 7  # Download 7 days at a time
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.retry_attempts = 3
        self.retry_delay = 5.0
        
        # Progress tracking
        self.progress_file = self.output_dir / "download_progress.json"
        self.progress = self._load_progress()
        
        logger.info(f"📥 Historical Data Downloader initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🎯 Symbols: {symbols}")

    def _load_progress(self) -> Dict[str, Any]:
        """Load download progress from file"""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"📊 Loaded progress: {len(progress)} symbols tracked")
                return progress
            else:
                return {}
        except Exception as e:
            logger.error(f"❌ Error loading progress: {e}")
            return {}

    def _save_progress(self):
        """Save download progress to file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Error saving progress: {e}")

    def _get_date_ranges(self, start_date: str, end_date: str) -> List[tuple]:
        """Generate date ranges for batch downloading"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        ranges = []
        current_dt = start_dt
        
        while current_dt < end_dt:
            batch_end = min(current_dt + timedelta(days=self.batch_size_days), end_dt)
            ranges.append((current_dt.strftime('%Y-%m-%d'), batch_end.strftime('%Y-%m-%d')))
            current_dt = batch_end
            
        return ranges

    async def download_symbol_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Download historical data for a specific symbol"""
        try:
            logger.info(f"🚀 Starting download for {symbol}: {start_date} to {end_date}")
            
            # Update config for this symbol
            self.config.symbol = symbol
            
            # Initialize data manager
            data_manager = BinanceDataManager(self.config)
            await data_manager.initialize()
            
            # Create symbol directory
            symbol_dir = self.output_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Get date ranges for batch processing
            date_ranges = self._get_date_ranges(start_date, end_date)
            total_batches = len(date_ranges)
            
            logger.info(f"📊 {symbol}: {total_batches} batches to download")
            
            all_data = []
            completed_batches = 0
            
            for i, (batch_start, batch_end) in enumerate(date_ranges):
                try:
                    # Check if this batch was already completed
                    batch_key = f"{symbol}_{batch_start}_{batch_end}"
                    if batch_key in self.progress.get('completed_batches', []):
                        logger.info(f"⏭️  {symbol}: Batch {i+1}/{total_batches} already completed")
                        completed_batches += 1
                        continue
                    
                    logger.info(f"📥 {symbol}: Downloading batch {i+1}/{total_batches} ({batch_start} to {batch_end})")
                    
                    # Download batch with retry logic
                    batch_data = await self._download_batch_with_retry(
                        data_manager, batch_start, batch_end, symbol
                    )
                    
                    if batch_data is not None and len(batch_data) > 0:
                        all_data.append(batch_data)
                        completed_batches += 1
                        
                        # Mark batch as completed
                        if 'completed_batches' not in self.progress:
                            self.progress['completed_batches'] = []
                        self.progress['completed_batches'].append(batch_key)
                        
                        # Save intermediate progress
                        self._save_progress()
                        
                        # Save batch data immediately
                        batch_file = symbol_dir / f"{symbol}_{batch_start}_{batch_end}.parquet"
                        batch_data.to_parquet(batch_file, compression='gzip')
                        
                        logger.info(f"✅ {symbol}: Batch {i+1}/{total_batches} completed ({len(batch_data)} records)")
                    else:
                        logger.warning(f"⚠️  {symbol}: Batch {i+1}/{total_batches} returned no data")
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"❌ {symbol}: Error in batch {i+1}/{total_batches}: {e}")
                    continue
            
            # Cleanup data manager
            await data_manager.cleanup()
            
            # Combine all data and save final file
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
                
                # Save combined file
                final_file = symbol_dir / f"{symbol}_complete_{start_date}_{end_date}.parquet"
                combined_data.to_parquet(final_file, compression='gzip')
                
                # Also save as CSV for compatibility
                csv_file = symbol_dir / f"{symbol}_complete_{start_date}_{end_date}.csv"
                combined_data.to_csv(csv_file, index=False)
                
                # Update progress
                self.progress[symbol] = {
                    'status': 'completed',
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_records': len(combined_data),
                    'completed_at': datetime.utcnow().isoformat(),
                    'batches_completed': completed_batches,
                    'total_batches': total_batches
                }
                
                self._save_progress()
                
                logger.info(f"🎉 {symbol}: Download completed! {len(combined_data)} total records")
                return True
            else:
                logger.error(f"❌ {symbol}: No data downloaded")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error downloading data for {symbol}: {e}")
            return False

    async def _download_batch_with_retry(self, data_manager: BinanceDataManager, 
                                       start_date: str, end_date: str, symbol: str) -> Optional[pd.DataFrame]:
        """Download a batch with retry logic"""
        for attempt in range(self.retry_attempts):
            try:
                data = await data_manager.download_historical_data(start_date, end_date)
                
                if data is not None and len(data) > 0:
                    return data
                else:
                    logger.warning(f"⚠️  {symbol}: Attempt {attempt+1} returned no data")
                    
            except Exception as e:
                logger.error(f"❌ {symbol}: Attempt {attempt+1} failed: {e}")
                
            if attempt < self.retry_attempts - 1:
                logger.info(f"🔄 {symbol}: Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
        
        logger.error(f"❌ {symbol}: All retry attempts failed")
        return None

    async def download_all_symbols(self, start_date: str, end_date: str):
        """Download data for all symbols"""
        logger.info(f"🚀 Starting bulk download: {start_date} to {end_date}")
        logger.info(f"📊 Symbols to process: {len(self.symbols)}")
        
        start_time = time.time()
        completed = 0
        failed = 0
        
        for i, symbol in enumerate(self.symbols):
            try:
                logger.info(f"📥 Processing symbol {i+1}/{len(self.symbols)}: {symbol}")
                
                # Check if symbol was already completed
                if symbol in self.progress and self.progress[symbol].get('status') == 'completed':
                    logger.info(f"⏭️  {symbol}: Already completed, skipping")
                    completed += 1
                    continue
                
                success = await self.download_symbol_data(symbol, start_date, end_date)
                
                if success:
                    completed += 1
                    logger.info(f"✅ {symbol}: Download successful")
                else:
                    failed += 1
                    logger.error(f"❌ {symbol}: Download failed")
                
                # Progress update
                elapsed_time = time.time() - start_time
                avg_time_per_symbol = elapsed_time / (i + 1)
                remaining_symbols = len(self.symbols) - (i + 1)
                eta = remaining_symbols * avg_time_per_symbol
                
                logger.info(f"📊 Progress: {i+1}/{len(self.symbols)} symbols processed")
                logger.info(f"⏱️  ETA: {eta/3600:.1f} hours")
                
            except Exception as e:
                logger.error(f"❌ Fatal error processing {symbol}: {e}")
                failed += 1
                continue
        
        # Final summary
        total_time = time.time() - start_time
        logger.info(f"🎉 Bulk download completed!")
        logger.info(f"📊 Summary: {completed} successful, {failed} failed")
        logger.info(f"⏱️  Total time: {total_time/3600:.1f} hours")
        
        return completed, failed

    def get_download_summary(self) -> Dict[str, Any]:
        """Get summary of download progress"""
        summary = {
            'total_symbols': len(self.symbols),
            'completed_symbols': 0,
            'failed_symbols': 0,
            'in_progress_symbols': 0,
            'total_records': 0,
            'symbols_status': {}
        }
        
        for symbol in self.symbols:
            if symbol in self.progress:
                status = self.progress[symbol].get('status', 'unknown')
                summary['symbols_status'][symbol] = status
                
                if status == 'completed':
                    summary['completed_symbols'] += 1
                    summary['total_records'] += self.progress[symbol].get('total_records', 0)
                elif status == 'failed':
                    summary['failed_symbols'] += 1
                else:
                    summary['in_progress_symbols'] += 1
            else:
                summary['symbols_status'][symbol] = 'not_started'
        
        return summary


async def main():
    """Main function to run the historical data download"""
    parser = argparse.ArgumentParser(description='Download historical crypto tick data')
    parser.add_argument('--symbols', nargs='+', 
                      default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
                              'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LUNAUSDT'],
                      help='Symbols to download')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-06-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data/historical', help='Output directory')
    parser.add_argument('--resume', action='store_true', help='Resume previous download')
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Historical Tick Data Download")
    logger.info(f"📅 Date range: {args.start_date} to {args.end_date}")
    logger.info(f"🎯 Symbols: {args.symbols}")
    logger.info(f"📁 Output: {args.output_dir}")
    logger.info(f"🔄 Resume mode: {args.resume}")
    
    # Initialize downloader
    downloader = HistoricalDataDownloader(args.symbols, args.output_dir)
    
    # Show current progress if resuming
    if args.resume:
        summary = downloader.get_download_summary()
        logger.info(f"📊 Current progress: {summary['completed_symbols']}/{summary['total_symbols']} symbols completed")
    
    try:
        # Start download
        completed, failed = await downloader.download_all_symbols(args.start_date, args.end_date)
        
        # Final summary
        final_summary = downloader.get_download_summary()
        logger.info(f"🎉 Download session completed!")
        logger.info(f"📊 Final status: {final_summary['completed_symbols']} completed, {final_summary['failed_symbols']} failed")
        logger.info(f"📈 Total records downloaded: {final_summary['total_records']:,}")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Download interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())