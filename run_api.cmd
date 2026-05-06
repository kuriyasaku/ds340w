@echo off
setlocal
cd /d "%~dp0"
python -m uvicorn src.api.api_server:app --reload
pause
