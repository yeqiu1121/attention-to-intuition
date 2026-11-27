@echo off
cd /d "%~dp0"
python run_all.py
if errorlevel 1 (
  echo Experiment failed. Read the failure and test logs in the newest runs directory.
  pause
  exit /b 1
)
echo Experiment completed. Open README.md and REPORT.md for details.
pause
