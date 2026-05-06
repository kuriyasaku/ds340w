#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "================================================================"
echo "CAMELYON16 PATHOLOGY AUDIT DEBUG PIPELINE"
echo "================================================================"
python run_pipeline_debug.py --preview
