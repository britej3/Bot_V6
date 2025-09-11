# Historical Tick Data Download System

This system provides comprehensive tools for downloading, validating, and managing 1-minute tick data for cryptocurrency trading pairs for the years 2024 and 2025 (until June).

## 🚀 Quick Start

### Option 1: One-Command Launch (Recommended)

```bash
# Download major crypto pairs for 2024-2025
python scripts/launch_tickdata_download.py

# Or with custom symbols
python scripts/launch_tickdata_download.py --symbols BTCUSDT ETHUSDT BNBUSDT

# Resume previous download
python scripts/launch_tickdata_download.py --resume
```

### Option 2: Shell Script (Linux/Mac)

```bash
# Make executable and run
chmod +x scripts/launch_tickdata_download.sh
./scripts/launch_tickdata_download.sh

# With custom options
./scripts/launch_tickdata_download.sh --symbols "BTCUSDT ETHUSDT" --resume
```

### Option 3: Direct Python Script

```bash
# Download with default settings
python scripts/download_historical_tickdata.py \
  --start-date 2024-01-01 \
  --end-date 2025-06-30 \
  --symbols BTCUSDT ETHUSDT BNBUSDT

# Resume previous download
python scripts/download_historical_tickdata.py --resume
```

## 📊 Features

### ✅ **Comprehensive Data Coverage**
- **Period**: January 1, 2024 to June 30, 2025 (18 months)
- **Resolution**: 1-minute tick data
- **Symbols**: Major crypto pairs (BTC, ETH, BNB, XRP, ADA, SOL, etc.)
- **Source**: Binance Futures API

### ✅ **Production-Grade Infrastructure**
- **Batch Processing**: Downloads data in weekly chunks to manage API limits
- **Rate Limiting**: Respects Binance API rate limits (1 request/second)
- **Retry Logic**: Automatic retry with exponential backoff on failures
- **Resume Capability**: Can resume interrupted downloads
- **Progress Tracking**: Persistent progress tracking across sessions

### ✅ **Data Quality & Validation**
- **Real-time Validation**: Validates data during download
- **Quality Metrics**: Comprehensive quality analysis
- **Gap Detection**: Identifies missing data periods
- **Outlier Detection**: Flags unusual price/volume data
- **Completeness Checking**: Ensures data integrity

### ✅ **Multiple Output Formats**
- **Parquet**: Compressed, efficient storage (recommended)
- **CSV**: Human-readable format for compatibility
- **JSON**: Metadata and progress tracking
- **Compression**: GZIP compression for space efficiency

## 📁 File Structure

```
data/historical/
├── BTCUSDT/
│   ├── BTCUSDT_2024-01-01_2024-01-07.parquet
│   ├── BTCUSDT_2024-01-08_2024-01-14.parquet
│   ├── ...
│   ├── BTCUSDT_complete_2024-01-01_2025-06-30.parquet
│   └── BTCUSDT_complete_2024-01-01_2025-06-30.csv
├── ETHUSDT/
│   └── ...
└── download_progress.json
```

## ⚙️ Configuration

### Environment Setup

1. **Virtual Environment** (Required)
```bash
cd /Users/britebrt/Bot_V5
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Binance API Keys** (Optional for historical data)
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET_KEY="your_secret_key"
```

### Configuration File

Edit `config/tickdata_download_config.json` to customize:

```json
{
  "download_config": {
    "symbols": {
      "major_pairs": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
      "defi_tokens": ["UNIUSDT", "LINKUSDT", "AAVEUSDT"],
      "layer1_tokens": ["NEARUSDT", "ATOMUSDT", "ALGOUSDT"]
    },
    "download_settings": {
      "batch_size_days": 7,
      "rate_limit_delay_seconds": 1.0,
      "retry_attempts": 3
    }
  }
}
```

## 🔧 Advanced Usage

### Custom Symbol Lists

```bash
# Major pairs only
python scripts/launch_tickdata_download.py \
  --symbols BTCUSDT ETHUSDT BNBUSDT XRPUSDT ADAUSDT

# DeFi tokens
python scripts/launch_tickdata_download.py \
  --symbols UNIUSDT LINKUSDT AAVEUSDT COMPUSDT SUSHIUSDT

# Layer 1 tokens
python scripts/launch_tickdata_download.py \
  --symbols NEARUSDT ATOMUSDT ALGOUSDT FTMUSDT HBARUSDT
```

### Date Range Customization

```bash
# Download only 2024 data
python scripts/download_historical_tickdata.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --symbols BTCUSDT ETHUSDT

# Download only Q1 2025
python scripts/download_historical_tickdata.py \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --symbols BTCUSDT ETHUSDT
```

### Output Directory

```bash
# Custom output directory
python scripts/launch_tickdata_download.py \
  --output-dir /path/to/custom/directory
```

## 📈 Data Validation

### Automatic Validation

```bash
# Validate all downloaded data
python scripts/validate_tickdata.py

# Validate specific symbols
python scripts/validate_tickdata.py --symbols BTCUSDT ETHUSDT

# Generate detailed report
python scripts/validate_tickdata.py \
  --output-report reports/validation_report.txt \
  --output-json reports/validation_results.json
```

### Quality Metrics

The validation system checks for:
- **Completeness**: Missing data periods
- **Consistency**: Price/volume outliers
- **Accuracy**: Timestamp integrity
- **Format**: Data structure validation

## 📊 Expected Data Volume

### Storage Requirements

| Symbol Count | Time Period | Estimated Size | Records |
|-------------|-------------|----------------|---------|
| 1 symbol | 18 months | ~50 MB | ~780k |
| 10 symbols | 18 months | ~500 MB | ~7.8M |
| 50 symbols | 18 months | ~2.5 GB | ~39M |

### Download Time Estimates

| Symbol Count | Estimated Time | Notes |
|-------------|----------------|--------|
| 1 symbol | 10-15 minutes | Depends on API limits |
| 10 symbols | 2-3 hours | With rate limiting |
| 50 symbols | 8-12 hours | Full download |

## 🛠️ Troubleshooting

### Common Issues

1. **Rate Limit Errors**
   ```bash
   # Increase rate limit delay
   # Edit config: "rate_limit_delay_seconds": 2.0
   ```

2. **Incomplete Downloads**
   ```bash
   # Resume with --resume flag
   python scripts/launch_tickdata_download.py --resume
   ```

3. **Network Timeouts**
   ```bash
   # Check internet connection and retry
   python scripts/launch_tickdata_download.py --resume
   ```

4. **Disk Space**
   ```bash
   # Check available space
   df -h
   # Clean up if needed or use external storage
   ```

### Log Files

- **Download logs**: `data_download.log`
- **Progress tracking**: `data/historical/download_progress.json`
- **Validation reports**: `reports/validation_report.txt`

## 🔍 Data Usage Examples

### Loading Data in Python

```python
import pandas as pd

# Load parquet file (recommended)
df = pd.read_parquet('data/historical/BTCUSDT/BTCUSDT_complete_2024-01-01_2025-06-30.parquet')

# Load CSV file
df = pd.read_csv('data/historical/BTCUSDT/BTCUSDT_complete_2024-01-01_2025-06-30.csv')

# Basic data exploration
print(f"Records: {len(df):,}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Price range: ${df['price'].min():.2f} to ${df['price'].max():.2f}")
```

### Integration with XGBoost Platform

```python
from src.data_pipeline.binance_data_manager import BinanceDataManager
from src.learning.tick_level_feature_engine import TickLevelFeatureEngine

# Load historical data into feature engine
feature_engine = TickLevelFeatureEngine(config)

# Process historical data for training
for _, row in df.iterrows():
    tick_data = {
        'timestamp': row['timestamp'],
        'price': row['price'],
        'quantity': row['quantity'],
        'is_buyer_maker': row.get('is_buyer_maker', True)
    }
    features = feature_engine.process_tick_data(tick_data)
```

## 🚨 Important Notes

### API Limits & Fair Use
- Respects Binance API rate limits
- Uses public historical data endpoints
- No API keys required for historical data
- Please use responsibly and follow Binance terms of service

### Data Quality
- Data quality depends on Binance's historical data availability
- Some periods may have gaps due to maintenance or outages
- Validation tools help identify and report quality issues

### Storage Considerations
- Parquet format recommended for efficient storage
- Compression reduces file sizes by ~60-70%
- Consider using external storage for large datasets

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files for specific error messages
3. Use the validation tools to check data quality
4. Refer to the main project documentation

## 📝 License

This tool is part of the XGBoost-Enhanced Crypto Futures Scalping Platform and follows the same licensing terms as the main project.