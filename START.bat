@echo off
setlocal
cd /d "%~dp0"
title VoiceAI Local
echo VoiceAI Local - first-time setup and start
echo.
if not exist "tools\setup-complete.flag" goto setup
if not exist ".venv-ui\Scripts\python.exe" goto setup
if not exist ".venv-tts\Scripts\python.exe" goto setup
if not exist ".venv-vc\Scripts\python.exe" goto setup
if not exist "tools\ffmpeg.exe" goto setup
goto models

:setup
echo [1/3] Preparing Python and dependencies. Internet access is required.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Mode setup
if errorlevel 1 goto failed

:models
if not exist "models\tts-ready.json" goto prepare_models
if not exist "models\vc-ready.json" goto prepare_models
goto start_app

:prepare_models
echo [2/3] Downloading AI models and checking GPU loading. Please wait.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Mode models
if errorlevel 1 goto failed

:start_app
echo [3/3] Opening the app. Keep this window open while using VoiceAI.
call "%~dp003_start.bat"
exit /b %errorlevel%

:failed
echo.
echo Preparation stopped. Please share the error above or the log in logs.
echo You can run START.bat again after correcting the problem.
pause
exit /b 1
