@echo off
title Heretic - First Time Setup
echo ========================================
echo   HERETIC - First Time Setup
echo ========================================
echo.

echo [1/4] Creating directories...
mkdir D:\heretic\python-libs 2>nul
mkdir D:\heretic\hf-cache 2>nul
mkdir D:\heretic\tmp 2>nul

echo [2/4] Installing Python dependencies to D:\heretic\python-libs...
set TMPDIR=D:\heretic\tmp
set TEMP=D:\heretic\tmp
set TMP=D:\heretic\tmp
pip install --target D:\heretic\python-libs --cache-dir D:\heretic\pip-cache torch --pre --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --target D:\heretic\python-libs --cache-dir D:\heretic\pip-cache transformers accelerate bitsandbytes peft sentencepiece protobuf heretic-llm

echo [3/4] Installing Node.js dependencies...
cd /d %~dp0chat-app
npm install

echo [4/4] Pre-downloading model...
set PYTHONPATH=D:\heretic\python-libs
set HF_HOME=D:\heretic\hf-cache
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')"

echo.
echo ========================================
echo   Setup complete! Run start.bat
echo ========================================
pause
