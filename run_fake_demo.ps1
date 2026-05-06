Set-Location $PSScriptRoot
Write-Host "================================================================"
Write-Host "GENERATING PRESENTATION DEMO RESULTS"
Write-Host "================================================================"
python generate_fake_results_presentation.py
Write-Host "================================================================"
Write-Host "STARTING STREAMLIT VIEWER"
Write-Host "================================================================"
python -m streamlit run src/ui/viewer_app.py
