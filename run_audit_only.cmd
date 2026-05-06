@echo off
setlocal
cd /d "%~dp0"
echo ================================================================
echo AUDIT ONLY MODE
echo ================================================================
python run_pipeline_debug.py --skip-preprocess --skip-features --skip-train --skip-ranking --preview
pause
