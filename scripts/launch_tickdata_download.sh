#!/bin/bash
"""
Quick Launcher for Historical Tick Data Download
Downloads 1-min tick data for 2024 and 2025 (until June)
"""

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Historical Tick Data Downloader${NC}"
echo -e "${BLUE}  Period: 2024 + 2025 (Jan-June)${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if virtual environment exists
VENV_PATH="$PROJECT_DIR/venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${YELLOW}💡 Please create virtual environment first:${NC}"
    echo -e "   cd $PROJECT_DIR"
    echo -e "   python -m venv venv"
    echo -e "   source venv/bin/activate"
    echo -e "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Check if required packages are installed
echo -e "${BLUE}📦 Checking dependencies...${NC}"
python -c "import pandas, numpy, asyncio, ccxt" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Missing required packages${NC}"
    echo -e "${YELLOW}💡 Installing requirements...${NC}"
    pip install -r "$PROJECT_DIR/requirements.txt"
fi

# Set environment variables
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Default symbols (major crypto pairs)
DEFAULT_SYMBOLS="BTCUSDT ETHUSDT BNBUSDT XRPUSDT ADAUSDT SOLUSDT DOGEUSDT DOTUSDT AVAXUSDT MATICUSDT"

# Parse command line arguments
SYMBOLS="$DEFAULT_SYMBOLS"
OUTPUT_DIR="$PROJECT_DIR/data/historical"
RESUME=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --symbols \"SYM1 SYM2\"   Symbols to download (default: major pairs)"
            echo "  --output-dir DIR        Output directory (default: data/historical)"
            echo "  --resume               Resume previous download"
            echo "  --help                 Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Download major pairs"
            echo "  $0 --symbols \"BTCUSDT ETHUSDT\"      # Download specific symbols"
            echo "  $0 --resume                          # Resume previous download"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Show configuration
echo -e "${GREEN}📋 Download Configuration:${NC}"
echo -e "   📅 Period: 2024-01-01 to 2025-06-30"
echo -e "   🎯 Symbols: $SYMBOLS"
echo -e "   📁 Output: $OUTPUT_DIR"
echo -e "   🔄 Resume: $RESUME"
echo ""

# Ask for confirmation
echo -e "${YELLOW}⚠️  This will download approximately 18 months of 1-minute tick data.${NC}"
echo -e "${YELLOW}   This may take several hours and use significant bandwidth.${NC}"
echo ""
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🛑 Download cancelled by user${NC}"
    exit 0
fi

# Build command arguments
ARGS="--start-date 2024-01-01 --end-date 2025-06-30"
ARGS="$ARGS --output-dir \"$OUTPUT_DIR\""
ARGS="$ARGS --symbols $SYMBOLS"

if [ "$RESUME" = true ]; then
    ARGS="$ARGS --resume"
fi

# Start download
echo -e "${GREEN}🚀 Starting tick data download...${NC}"
echo -e "${BLUE}📊 Command: python scripts/download_historical_tickdata.py $ARGS${NC}"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Execute download with proper argument handling
if [ "$RESUME" = true ]; then
    python scripts/download_historical_tickdata.py \
        --start-date 2024-01-01 \
        --end-date 2025-06-30 \
        --output-dir "$OUTPUT_DIR" \
        --symbols $SYMBOLS \
        --resume
else
    python scripts/download_historical_tickdata.py \
        --start-date 2024-01-01 \
        --end-date 2025-06-30 \
        --output-dir "$OUTPUT_DIR" \
        --symbols $SYMBOLS
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}🎉 Download completed successfully!${NC}"
    echo -e "${GREEN}📁 Data saved to: $OUTPUT_DIR${NC}"
else
    echo -e "${RED}❌ Download failed or was interrupted${NC}"
    echo -e "${YELLOW}💡 You can resume with: $0 --resume${NC}"
fi