@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Mode models
if errorlevel 1 echo Model preparation failed. Please read logs\models.log.
pause
