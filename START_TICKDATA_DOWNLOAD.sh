#!/bin/bash
# One-line command to start tick data download for 2024-2025
# This downloads major crypto pairs with all the recommended settings

echo "🚀 Starting 1-minute tick data download for 2024-2025..."
echo "📅 Period: January 1, 2024 to June 30, 2025"
echo "🎯 Symbols: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT, DOGEUSDT, DOTUSDT, AVAXUSDT, MATICUSDT"
echo ""

cd /Users/britebrt/Bot_V5
source venv/bin/activate
python scripts/launch_tickdata_download.py