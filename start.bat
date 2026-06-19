@echo off
title HackathonBot
cd /d "%~dp0"

echo.
echo  HackathonBot starting...
echo  Press Ctrl+C to stop.
echo.

:: Open browser after a 2-second delay (runs in background while server starts)
start /b cmd /c "timeout /t 2 /nobreak > nul && start http://localhost:8000"

python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

pause
