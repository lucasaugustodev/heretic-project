@echo off
echo Stopping Heretic processes...
taskkill /FI "WINDOWTITLE eq Heretic API" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Heretic Chat" /F >nul 2>&1
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV /NH ^| findstr heretic_api') do taskkill /PID %%a /F >nul 2>&1
echo Done.
pause
