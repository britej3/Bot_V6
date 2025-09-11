#!/usr/bin/env python3
"""
Tick Data Validation and Monitoring Script
Validates and monitors the quality of downloaded historical tick data
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TickDataValidator:
    """Validates and monitors tick data quality"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
        
    def validate_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Validate data for a specific symbol"""
        logger.info(f"🔍 Validating data for {symbol}")
        
        symbol_dir = self.data_dir / symbol
        if not symbol_dir.exists():
            return {'status': 'missing', 'error': 'Symbol directory not found'}
        
        results = {
            'symbol': symbol,
            'status': 'unknown',
            'files_found': [],
            'total_records': 0,
            'date_range': {},
            'quality_metrics': {},
            'issues': []
        }
        
        try:
            # Find data files
            parquet_files = list(symbol_dir.glob("*.parquet"))
            csv_files = list(symbol_dir.glob("*.csv"))
            
            results['files_found'] = {
                'parquet': len(parquet_files),
                'csv': len(csv_files),
                'parquet_files': [f.name for f in parquet_files],
                'csv_files': [f.name for f in csv_files]
            }
            
            if not parquet_files and not csv_files:
                results['status'] = 'no_data'
                results['issues'].append('No data files found')
                return results
            
            # Load and validate the main data file
            main_file = None
            for pattern in ['*_complete_*', '*']:
                files = list(symbol_dir.glob(pattern + ".parquet"))
                if files:
                    main_file = files[0]  # Take the first match
                    break
            
            if not main_file:
                # Try CSV files
                for pattern in ['*_complete_*', '*']:
                    files = list(symbol_dir.glob(pattern + ".csv"))
                    if files:
                        main_file = files[0]
                        break
            
            if not main_file:
                results['status'] = 'no_main_file'
                results['issues'].append('No main data file found')
                return results
            
            logger.info(f"📊 Loading data from {main_file.name}")
            
            # Load data
            if main_file.suffix == '.parquet':
                df = pd.read_parquet(main_file)
            else:
                df = pd.read_csv(main_file)
            
            # Convert timestamp if needed
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            results['total_records'] = len(df)
            
            # Date range analysis
            if 'timestamp' in df.columns and len(df) > 0:
                results['date_range'] = {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat(),
                    'days': (df['timestamp'].max() - df['timestamp'].min()).days
                }
            
            # Quality metrics
            quality_metrics = self._calculate_quality_metrics(df)
            results['quality_metrics'] = quality_metrics
            
            # Check for issues
            issues = self._check_data_issues(df)
            results['issues'].extend(issues)
            
            # Determine overall status
            if len(issues) == 0:
                results['status'] = 'excellent'
            elif len(issues) <= 2:
                results['status'] = 'good'
            elif len(issues) <= 5:
                results['status'] = 'fair'
            else:
                results['status'] = 'poor'
            
            logger.info(f"✅ {symbol}: {results['status']} quality, {results['total_records']:,} records")
            
        except Exception as e:
            logger.error(f"❌ Error validating {symbol}: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        metrics = {}
        
        try:
            # Basic statistics
            metrics['record_count'] = len(df)
            metrics['columns'] = list(df.columns)
            
            # Missing data
            metrics['missing_data'] = {
                col: df[col].isnull().sum() for col in df.columns
            }
            
            # Timestamp analysis
            if 'timestamp' in df.columns:
                timestamps = df['timestamp'].dropna()
                if len(timestamps) > 1:
                    time_diffs = timestamps.diff().dropna()
                    metrics['timestamp_analysis'] = {
                        'min_interval_seconds': time_diffs.min().total_seconds(),
                        'max_interval_seconds': time_diffs.max().total_seconds(),
                        'avg_interval_seconds': time_diffs.mean().total_seconds(),
                        'std_interval_seconds': time_diffs.std().total_seconds()
                    }
            
            # Price analysis
            if 'price' in df.columns:
                prices = df['price'].dropna()
                if len(prices) > 0:
                    metrics['price_analysis'] = {
                        'min_price': float(prices.min()),
                        'max_price': float(prices.max()),
                        'avg_price': float(prices.mean()),
                        'std_price': float(prices.std()),
                        'zero_prices': int((prices == 0).sum()),
                        'negative_prices': int((prices < 0).sum())
                    }
            
            # Volume analysis
            if 'quantity' in df.columns:
                volumes = df['quantity'].dropna()
                if len(volumes) > 0:
                    metrics['volume_analysis'] = {
                        'min_volume': float(volumes.min()),
                        'max_volume': float(volumes.max()),
                        'avg_volume': float(volumes.mean()),
                        'zero_volumes': int((volumes == 0).sum()),
                        'negative_volumes': int((volumes < 0).sum())
                    }
            
            # Duplicates
            if 'timestamp' in df.columns:
                metrics['duplicates'] = {
                    'duplicate_timestamps': int(df['timestamp'].duplicated().sum()),
                    'duplicate_rows': int(df.duplicated().sum())
                }
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _check_data_issues(self, df: pd.DataFrame) -> List[str]:
        """Check for common data quality issues"""
        issues = []
        
        try:
            # Check for empty dataset
            if len(df) == 0:
                issues.append("Dataset is empty")
                return issues
            
            # Check required columns
            required_columns = ['timestamp', 'price']
            for col in required_columns:
                if col not in df.columns:
                    issues.append(f"Missing required column: {col}")
            
            # Check for missing data
            for col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                if missing_pct > 5:
                    issues.append(f"High missing data in {col}: {missing_pct:.1f}%")
            
            # Check price data
            if 'price' in df.columns:
                prices = df['price'].dropna()
                if len(prices) > 0:
                    if (prices <= 0).any():
                        issues.append("Found zero or negative prices")
                    
                    # Check for price outliers (beyond 10x standard deviations)
                    mean_price = prices.mean()
                    std_price = prices.std()
                    outliers = prices[(prices > mean_price + 10*std_price) | 
                                    (prices < mean_price - 10*std_price)]
                    if len(outliers) > 0:
                        issues.append(f"Found {len(outliers)} price outliers")
            
            # Check volume data
            if 'quantity' in df.columns:
                volumes = df['quantity'].dropna()
                if len(volumes) > 0:
                    if (volumes < 0).any():
                        issues.append("Found negative volumes")
            
            # Check timestamp data
            if 'timestamp' in df.columns:
                timestamps = df['timestamp'].dropna()
                if len(timestamps) > 1:
                    # Check for large gaps (more than 5 minutes)
                    time_diffs = timestamps.diff().dropna()
                    large_gaps = time_diffs[time_diffs > timedelta(minutes=5)]
                    if len(large_gaps) > 0:
                        issues.append(f"Found {len(large_gaps)} large time gaps (>5 min)")
                    
                    # Check for duplicate timestamps
                    if timestamps.duplicated().any():
                        dup_count = timestamps.duplicated().sum()
                        issues.append(f"Found {dup_count} duplicate timestamps")
            
            # Check data completeness for expected date range
            if 'timestamp' in df.columns and len(df) > 0:
                start_date = df['timestamp'].min()
                end_date = df['timestamp'].max()
                expected_records = (end_date - start_date).total_seconds() / 60  # 1 minute intervals
                actual_records = len(df)
                completeness = (actual_records / expected_records) * 100 if expected_records > 0 else 0
                
                if completeness < 80:
                    issues.append(f"Low data completeness: {completeness:.1f}%")
        
        except Exception as e:
            logger.error(f"Error checking data issues: {e}")
            issues.append(f"Error during validation: {str(e)}")
        
        return issues
    
    def validate_all_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Validate data for all symbols"""
        logger.info(f"🔍 Validating data for {len(symbols)} symbols")
        
        validation_results = {}
        summary = {
            'total_symbols': len(symbols),
            'symbols_with_data': 0,
            'symbols_excellent': 0,
            'symbols_good': 0,
            'symbols_fair': 0,
            'symbols_poor': 0,
            'symbols_missing': 0,
            'total_records': 0
        }
        
        for symbol in symbols:
            result = self.validate_symbol_data(symbol)
            validation_results[symbol] = result
            
            # Update summary
            if result['status'] in ['excellent', 'good', 'fair', 'poor']:
                summary['symbols_with_data'] += 1
                summary['total_records'] += result.get('total_records', 0)
                
                if result['status'] == 'excellent':
                    summary['symbols_excellent'] += 1
                elif result['status'] == 'good':
                    summary['symbols_good'] += 1
                elif result['status'] == 'fair':
                    summary['symbols_fair'] += 1
                elif result['status'] == 'poor':
                    summary['symbols_poor'] += 1
            else:
                summary['symbols_missing'] += 1
        
        return {
            'summary': summary,
            'symbols': validation_results,
            'validation_timestamp': datetime.utcnow().isoformat()
        }
    
    def generate_report(self, validation_results: Dict[str, Any], output_file: Optional[Path] = None) -> str:
        """Generate a human-readable validation report"""
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("TICK DATA VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Summary
        summary = validation_results['summary']
        report_lines.append("SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Total symbols: {summary['total_symbols']}")
        report_lines.append(f"Symbols with data: {summary['symbols_with_data']}")
        report_lines.append(f"Total records: {summary['total_records']:,}")
        report_lines.append("")
        report_lines.append("Quality Distribution:")
        report_lines.append(f"  ✅ Excellent: {summary['symbols_excellent']}")
        report_lines.append(f"  ✅ Good: {summary['symbols_good']}")
        report_lines.append(f"  ⚠️  Fair: {summary['symbols_fair']}")
        report_lines.append(f"  ❌ Poor: {summary['symbols_poor']}")
        report_lines.append(f"  ❓ Missing: {summary['symbols_missing']}")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 30)
        
        for symbol, result in validation_results['symbols'].items():
            status_emoji = {
                'excellent': '✅',
                'good': '✅',
                'fair': '⚠️',
                'poor': '❌',
                'missing': '❓',
                'no_data': '❓',
                'error': '❌'
            }.get(result['status'], '❓')
            
            report_lines.append(f"{status_emoji} {symbol}: {result['status'].upper()}")
            
            if result.get('total_records'):
                report_lines.append(f"    Records: {result['total_records']:,}")
            
            if result.get('date_range'):
                date_range = result['date_range']
                report_lines.append(f"    Date range: {date_range.get('start', 'N/A')} to {date_range.get('end', 'N/A')}")
                report_lines.append(f"    Days: {date_range.get('days', 'N/A')}")
            
            if result.get('issues'):
                report_lines.append(f"    Issues: {len(result['issues'])}")
                for issue in result['issues'][:3]:  # Show first 3 issues
                    report_lines.append(f"      - {issue}")
                if len(result['issues']) > 3:
                    report_lines.append(f"      ... and {len(result['issues']) - 3} more")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"📄 Report saved to {output_file}")
        
        return report


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Validate historical tick data')
    parser.add_argument('--data-dir', default='data/historical', 
                      help='Data directory to validate')
    parser.add_argument('--symbols', nargs='+',
                      default=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
                              'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'MATICUSDT'],
                      help='Symbols to validate')
    parser.add_argument('--output-report', help='Output file for validation report')
    parser.add_argument('--output-json', help='Output file for JSON results')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = TickDataValidator(args.data_dir)
    
    # Run validation
    logger.info("🔍 Starting tick data validation...")
    results = validator.validate_all_symbols(args.symbols)
    
    # Generate report
    report_file = Path(args.output_report) if args.output_report else None
    report = validator.generate_report(results, report_file)
    
    # Print report to console
    print(report)
    
    # Save JSON results
    if args.output_json:
        json_file = Path(args.output_json)
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 JSON results saved to {json_file}")
    
    # Exit with appropriate code
    summary = results['summary']
    if summary['symbols_missing'] > 0 or summary['symbols_poor'] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())