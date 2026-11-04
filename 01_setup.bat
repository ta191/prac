@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Mode setup
if errorlevel 1 echo Setup failed. Please read the error above and logs\setup.log.
pause
