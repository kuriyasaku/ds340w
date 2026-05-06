#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "================================================================"
echo "GENERATING PRESENTATION DEMO RESULTS"
echo "================================================================"
python generate_fake_results_presentation.py
echo "================================================================"
echo "STARTING STREAMLIT VIEWER"
echo "================================================================"
python -m streamlit run src/ui/viewer_app.py
