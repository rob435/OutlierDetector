# Automation Scripts

Windows automation scripts for running OutlierDetector paper trading system.

## Quick Start

```cmd
cd scripts
run_daily.bat
```

## Files

- **run_daily.bat** - Manual testing script (shows output)
- **run_daily.ps1** - PowerShell version with enhanced features
- **run_daily_scheduled.bat** - Silent execution for Task Scheduler (auto-logging)
- **README_AUTOMATION.md** - Complete automation guide
- **TASK_SCHEDULER_SETUP.md** - Detailed Task Scheduler setup

## For Automation Setup

See [README_AUTOMATION.md](README_AUTOMATION.md) for complete instructions on:
- Windows Task Scheduler setup
- Running automatically when PC is locked
- Preventing sleep during trading hours
- Logging and monitoring

## Mac/Linux

For Mac, use the `run_daily.sh` script in the project root.
