# Daily Paper Trading Execution Script
# Runs the stat-arb system with fresh data download and position management

# Get project root directory (parent of scripts folder)
$ProjectDir = Split-Path -Parent $PSScriptRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running OutlierDetector Paper Trading System" -ForegroundColor Cyan
Write-Host "Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to project root directory
Set-Location $ProjectDir

# Run the paper trading system
python src\main.py

# Optional: Send results to a log file with timestamp
# $LogFile = "logs\paper_trading_$(Get-Date -Format 'yyyyMMdd').log"
# python src\main.py *>&1 | Tee-Object -FilePath $LogFile

Write-Host ""
Write-Host "Execution complete at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

# Keep window open to see results
Read-Host -Prompt "Press Enter to exit"
