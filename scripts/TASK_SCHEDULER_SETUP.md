# Windows Task Scheduler Setup for Automated Trading

## Overview
This guide shows how to schedule the OutlierDetector paper trading system to run automatically on Windows.

## Files Created
- `run_daily.bat` - Basic Windows batch script
- `run_daily.ps1` - PowerShell script (recommended, more features)
- `run_daily_scheduled.bat` - Silent execution with auto-logging (for Task Scheduler)
- `OutlierDetector_Task.xml` - Pre-configured task definition
- `install_task.bat` - One-click installer

## QUICK INSTALL (Recommended - 30 seconds)

1. **Right-click** `install_task.bat` → **Run as administrator**
2. Done! The task is now scheduled to run daily at 9:00 AM

To change the schedule time, open Task Scheduler (`Win+R` → `taskschd.msc`) and edit the task.

---

## Option 1: Quick Test (Manual Run)

### Test Batch Script:
```cmd
cd c:\Users\user\Desktop\OutlierDetector\OutlierDetector
run_daily.bat
```

### Test PowerShell Script:
```powershell
cd c:\Users\user\Desktop\OutlierDetector\OutlierDetector
.\run_daily.ps1
```

## Option 2: Windows Task Scheduler (Automated)

### Step 1: Open Task Scheduler
1. Press `Win + R`
2. Type: `taskschd.msc`
3. Press Enter

### Step 2: Create New Task
1. Click **"Create Task"** (not "Create Basic Task")
2. Name it: **OutlierDetector Daily Trading**
3. Description: **Runs paper trading system with fresh data**

### Step 3: General Tab Settings
- [x] Run whether user is logged on or not
- [x] Run with highest privileges
- Configure for: **Windows 10**

### Step 4: Triggers Tab
1. Click **"New"**
2. Begin the task: **On a schedule**
3. Settings: **Daily**
4. Start: Choose your preferred time (e.g., 9:00 AM)
5. Recur every: **1 days**
6. [x] Enabled

### Step 5: Actions Tab
1. Click **"New"**
2. Action: **Start a program**

#### For Batch Script:
- Program/script: `cmd.exe`
- Add arguments: `/c "c:\Users\user\Desktop\OutlierDetector\OutlierDetector\scripts\run_daily_scheduled.bat"`

#### For PowerShell Script (Recommended):
- Program/script: `powershell.exe`
- Add arguments: `-ExecutionPolicy Bypass -File "c:\Users\user\Desktop\OutlierDetector\OutlierDetector\scripts\run_daily.ps1"`

### Step 6: Conditions Tab
- [ ] UNCHECK "Start the task only if the computer is on AC power"
- [ ] UNCHECK "Stop if the computer switches to battery power"
- [x] Wake the computer to run this task (optional - keeps PC awake)

### Step 7: Settings Tab
- [x] Allow task to be run on demand
- [x] Run task as soon as possible after a scheduled start is missed
- If task fails, restart every: **5 minutes**
- Attempt to restart up to: **3 times**

### Step 8: Save
- Click **OK**
- Enter your Windows password when prompted

## Will It Run When PC is Locked?

**YES** - Task Scheduler tasks run even when:
- Screen is locked
- User is logged out
- Display is asleep

**BUT WILL NOT RUN IF:**
- PC is in sleep/hibernation mode
- PC is shut down
- PC loses power

## Preventing Sleep During Trading Hours

### Option A: Adjust Power Settings (Recommended)
1. Open **Power Options** (Win + X → Power Options)
2. Click **Additional power settings**
3. Choose **High performance** plan
4. Edit plan settings:
   - Turn off display: **15 minutes**
   - Put computer to sleep: **Never** (or set to after trading hours)

### Option B: Use PowerShell to Prevent Sleep
Modify `run_daily.ps1` to use `Start-Process` with `-NoNewWindow` and prevent sleep during execution.

### Option C: Use Third-Party Tool
Install **Caffeine** or **Don't Sleep** utilities to prevent sleep mode.

## Logging (Optional)

To enable logging, uncomment the logging lines in the scripts:

### In run_daily.bat:
```batch
REM Remove 'REM' from this line:
python src\main.py >> logs\paper_trading_%date:~-4,4%%date:~-10,2%%date:~-7,2%.log 2>&1
```

### In run_daily.ps1:
```powershell
# Remove '#' from these lines:
$LogFile = "logs\paper_trading_$(Get-Date -Format 'yyyyMMdd').log"
python src\main.py *>&1 | Tee-Object -FilePath $LogFile
```

Create the logs directory:
```cmd
mkdir c:\Users\user\Desktop\OutlierDetector\OutlierDetector\logs
```

## Testing Your Scheduled Task

1. Right-click the task in Task Scheduler
2. Click **"Run"**
3. Check **"Last Run Result"** column - should show "The operation completed successfully (0x0)"
4. Check **"Last Run Time"** to confirm it ran

## Troubleshooting

### Task runs but nothing happens:
- Verify Python is in system PATH
- Use full path to Python: `C:\Users\user\AppData\Local\Programs\Python\Python313\python.exe`

### Task fails with error:
- Check "Last Run Result" in Task Scheduler
- Enable logging to see error messages
- Run script manually first to verify it works

### Task doesn't wake computer:
- Ensure "Wake the computer to run this task" is checked in Conditions tab
- Disable hybrid sleep in power settings

## Recommended Schedule

For crypto trading (24/7 markets):
- **Multiple times per day**: Every 4-6 hours (0:00, 06:00, 12:00, 18:00)
- **Or**: Continuous monitoring with shorter intervals

To run multiple times per day:
1. In Triggers tab, click **"New"** for each time
2. Or use **"Repeat task every"** option in advanced settings

## Security Notes

- Task runs with your user privileges
- Ensure API keys in .env are secure
- Logs may contain sensitive data - protect the logs directory
- Consider encrypting the project folder

## Summary

1. Test scripts manually first
2. Create scheduled task in Task Scheduler
3. Adjust power settings to prevent sleep
4. Enable logging for monitoring
5. Test the scheduled task with "Run" button

The system will now run automatically even when you're not logged in!
