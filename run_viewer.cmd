@echo off
setlocal
cd /d "%~dp0"
python -m streamlit run src/ui/viewer_app.py
pause
