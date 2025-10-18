@echo off
REM Automated Daily Paper Trading Execution Script
REM For use with Windows Task Scheduler
REM Runs silently without user interaction

REM Navigate to project root directory
cd /d "%~dp0.."

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

REM Run the paper trading system with logging
set LOG_FILE=logs\paper_trading_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.log
echo ========================================== > "%LOG_FILE%"
echo Running OutlierDetector Paper Trading System >> "%LOG_FILE%"
echo Date: %date% %time% >> "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

python src\main.py >> "%LOG_FILE%" 2>&1

echo. >> "%LOG_FILE%"
echo Execution complete at %time% >> "%LOG_FILE%"
echo ========================================== >> "%LOG_FILE%"
