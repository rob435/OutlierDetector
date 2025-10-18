@echo off
echo ==========================================
echo Installing OutlierDetector Scheduled Task
echo ==========================================
echo.

REM Import the task into Task Scheduler
schtasks /create /tn "OutlierDetector Daily Trading" /xml "%~dp0OutlierDetector_Task.xml" /f

if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! Task installed successfully.
    echo.
    echo The system will now run daily at 9:00 AM.
    echo.
    echo To modify the schedule:
    echo 1. Press Win + R
    echo 2. Type: taskschd.msc
    echo 3. Find "OutlierDetector Daily Trading"
    echo 4. Right-click and select "Properties"
    echo.
    echo To test the task now:
    echo 1. Open Task Scheduler
    echo 2. Right-click "OutlierDetector Daily Trading"
    echo 3. Click "Run"
) else (
    echo.
    echo ERROR! Task installation failed.
    echo Please run this script as Administrator.
    echo.
    echo Right-click install_task.bat and select "Run as administrator"
)

echo.
pause
