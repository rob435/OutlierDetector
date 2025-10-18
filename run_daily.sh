#!/bin/bash
#
# Daily Paper Trading Execution Script
# Runs the stat-arb system with fresh data download and position management
#

# Navigate to project directory
cd /Applications/PairsAlgo

# Activate virtual environment
source venv/bin/activate

# Run the paper trading system
echo "=========================================="
echo "Running PairsAlgo Paper Trading System"
echo "Date: $(date)"
echo "=========================================="

python3 src/main.py

# Optional: Send results to a log file with timestamp
# python3 src/main.py >> logs/paper_trading_$(date +%Y%m%d).log 2>&1

echo ""
echo "Execution complete at $(date)"
echo "=========================================="
