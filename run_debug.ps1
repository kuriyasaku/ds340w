Set-Location $PSScriptRoot
Write-Host "================================================================"
Write-Host "CAMELYON16 PATHOLOGY AUDIT DEBUG PIPELINE"
Write-Host "================================================================"
python run_pipeline_debug.py --preview
