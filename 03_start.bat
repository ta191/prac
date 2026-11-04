@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv-ui\Scripts\python.exe" (
  echo Run 01_setup.bat first.
  pause
  exit /b 1
)
set "PYTHONUTF8=1"
set "VOICEAI_ROOT=%~dp0"
".venv-ui\Scripts\python.exe" run.py
pause
