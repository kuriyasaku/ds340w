@echo off
setlocal
cd /d "%~dp0"
echo ================================================================
echo CAMELYON16 PATHOLOGY AUDIT DEBUG PIPELINE WITHOUT TRAINING
echo ================================================================
python run_pipeline_debug.py --skip-train --preview
pause
