# OutlierDetector Automation Guide for Windows

## Quick Start

### Test the System Manually
```cmd
run_daily.bat
```

This will run the paper trading system once and show output in the console.

## Files Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `run_daily.bat` | Manual testing | Run from command line to test |
| `run_daily.ps1` | PowerShell version | More features, modern Windows |
| `run_daily_scheduled.bat` | Task Scheduler | Silent execution with auto-logging |
| `TASK_SCHEDULER_SETUP.md` | Setup guide | Complete Task Scheduler instructions |

## Option 1: Manual Execution

### Using Batch File:
```cmd
cd c:\Users\user\Desktop\OutlierDetector\OutlierDetector\scripts
run_daily.bat
```

### Using PowerShell:
```powershell
cd c:\Users\user\Desktop\OutlierDetector\OutlierDetector\scripts
.\run_daily.ps1
```

## Option 2: Windows Task Scheduler (Recommended for Automation)

### Quick Setup (5 minutes):

1. **Open Task Scheduler**: Press `Win + R`, type `taskschd.msc`, press Enter

2. **Create Task**: Click "Create Task" (not "Create Basic Task")

3. **General Tab**:
   - Name: `OutlierDetector Daily Trading`
   - ☑ Run whether user is logged on or not
   - ☑ Run with highest privileges

4. **Triggers Tab**:
   - Click "New"
   - Daily at your preferred time (e.g., 9:00 AM)
   - ☑ Enabled

5. **Actions Tab**:
   - Click "New"
   - Program: `cmd.exe`
   - Arguments: `/c "c:\Users\user\Desktop\OutlierDetector\OutlierDetector\scripts\run_daily_scheduled.bat"`

6. **Conditions Tab**:
   - ☐ UNCHECK "Start only if on AC power"
   - ☑ Wake computer to run this task

7. **Settings Tab**:
   - ☑ Run task as soon as possible after scheduled start is missed

8. **Save**: Enter your Windows password when prompted

### Test Your Scheduled Task:
1. Right-click task → Run
2. Check `logs\` folder for output file
3. Verify "Last Run Result" shows success (0x0)

## Will It Run When...?

| Scenario | Will Run? |
|----------|-----------|
| PC locked | ✅ YES |
| User logged out | ✅ YES |
| Screen off | ✅ YES |
| PC asleep/hibernating | ❌ NO |
| PC shut down | ❌ NO |

## Prevent Sleep During Trading

### Option A: Power Settings (Recommended)
1. `Win + X` → Power Options
2. Additional power settings
3. Change plan settings
4. Put computer to sleep: **Never**

### Option B: Allow Sleep, Wake for Task
1. In Task Scheduler Conditions tab
2. ☑ "Wake the computer to run this task"

**Note**: This works for scheduled sleep, but not if you manually put PC to sleep.

## Logging

The `run_daily_scheduled.bat` automatically logs to:
```
logs\paper_trading_YYYYMMDD_HHMM.log
```

Example: `logs\paper_trading_20251018_0900.log`

### View Logs:
```cmd
cd c:\Users\user\Desktop\OutlierDetector\OutlierDetector\logs
dir /O-D
type paper_trading_20251018_0900.log
```

## Multiple Runs Per Day

For 24/7 crypto markets, you may want to run every few hours:

1. In Task Scheduler Triggers tab, click "New" for each time:
   - 00:00 (midnight)
   - 06:00 (6 AM)
   - 12:00 (noon)
   - 18:00 (6 PM)

**OR** use "Repeat task every" in trigger advanced settings:
- Repeat every: **4 hours**
- For a duration of: **Indefinitely**

## Troubleshooting

### Task runs but nothing happens:
```cmd
REM Test if Python is accessible:
python --version

REM If not found, use full path in scripts:
C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe
```

### Task fails:
1. Check "Last Run Result" in Task Scheduler (should be 0x0)
2. Check logs folder for error messages
3. Run `run_daily.bat` manually first to verify it works

### PC goes to sleep anyway:
1. Check power settings (sleep = Never)
2. Disable "Hybrid Sleep" in advanced power settings
3. Ensure Task Scheduler condition "Wake computer" is checked

## Configuration

### Change Trading System Settings:
Edit `src/main.py` configuration section:
```python
# System Control Configuration
ENABLE_DATA_DOWNLOAD = True  # Set False for faster testing
ENABLE_OUTLIER_DETECTION = True
ENABLE_PAPER_TRADING = True
```

### Change Schedule:
Edit your Task Scheduler triggers to run at different times.

## Security Notes

- Task runs with your Windows user account privileges
- Protect `.env` file containing API keys
- Log files may contain sensitive data - secure the `logs/` directory
- Consider full-disk encryption if storing trading credentials

## Monitoring Your System

### Check Current Positions:
```cmd
python src\view_positions.py
```

### View Latest Log:
```cmd
cd logs
dir /O-D
type [latest_log_file]
```

### Task Scheduler History:
1. Open Task Scheduler
2. Click "View" → "Show Hidden Tasks"
3. Right-click task → "History" tab

## Recommended Setup for Production

1. **Power Settings**: Sleep = Never (or only during non-trading hours)
2. **Task Schedule**: Every 4-6 hours for crypto markets
3. **Logging**: Enabled (use `run_daily_scheduled.bat`)
4. **Monitoring**: Check logs daily for errors
5. **Backup**: Regularly backup `paper_trades/` folder

## Next Steps

1. Test `run_daily.bat` manually
2. Set up Task Scheduler with `run_daily_scheduled.bat`
3. Configure power settings to prevent sleep
4. Monitor first few automated runs via logs
5. Adjust schedule based on market hours and strategy performance

---

**For detailed Task Scheduler setup, see**: `TASK_SCHEDULER_SETUP.md`
