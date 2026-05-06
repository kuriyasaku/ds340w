@echo off
setlocal
cd /d "%~dp0"
echo ================================================================
echo GENERATING PRESENTATION DEMO RESULTS
echo ================================================================
python generate_fake_results_presentation.py
echo ================================================================
echo STARTING STREAMLIT VIEWER
echo ================================================================
python -m streamlit run src/ui/viewer_app.py
pause
