@echo off
setlocal
cd /d "%~dp0"
echo ================================================================
echo CAMELYON16 PATHOLOGY AUDIT DEBUG PIPELINE
echo ================================================================
python run_pipeline_debug.py --preview
pause
