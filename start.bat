@echo off
title Heretic - Starting...
echo ========================================
echo   HERETIC - Uncensored Mistral 7B
echo ========================================
echo.

set PYTHONPATH=D:\heretic\python-libs
set HF_HOME=D:\heretic\hf-cache
set PYTHONUNBUFFERED=1

echo [1/2] Starting API server on port 8000...
start "Heretic API" cmd /k "set PYTHONPATH=D:\heretic\python-libs && set HF_HOME=D:\heretic\hf-cache && set PYTHONUNBUFFERED=1 && python %~dp0scripts\heretic_api.py"

echo [2/2] Starting Chat UI on port 3333...
cd /d %~dp0chat-app
start "Heretic Chat" cmd /k "node server.js"

echo.
echo Waiting for API to load (~90 seconds)...
timeout /t 90 /nobreak >nul

echo.
echo ========================================
echo   READY! Open http://localhost:3333
echo ========================================
start http://localhost:3333
pause
