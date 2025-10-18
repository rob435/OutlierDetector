@echo off
REM Daily Paper Trading Execution Script
REM Runs the stat-arb system with fresh data download and position management

echo ==========================================
echo Running OutlierDetector Paper Trading System
echo Date: %date% %time%
echo ==========================================
echo.

REM Navigate to project root directory
cd /d "%~dp0.."

REM Run the paper trading system
python src\main.py

REM Optional: Send results to a log file with timestamp
REM python src\main.py >> logs\paper_trading_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log 2>&1

echo.
echo Execution complete at %time%
echo ==========================================
