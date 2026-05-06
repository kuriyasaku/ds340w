# CAMELYON16 Pathology Audit Project

This project is a modular scaffold for a CAMELYON16 pathology AI audit workflow.

Main stages:
- Whole-slide preprocessing for 269 processed diagnostic WSI cases
- Frozen feature extraction with a Phikon-like encoder wrapper
- Attention-based MIL training for slide-level classification
- Ranked patch export with attention, probability, coordinates, and metadata
- Controlled perturbation audit across multiple perturbation families
- Heatmap generation and explanation stability analysis
- Streamlit audit viewer UI
- FastAPI serving layer for programmatic access

Recommended layout:

```text
camelyon16_audit_project_v2/
  requirements.txt
  README.md
  run_pipeline.py
  src/
    config/
      settings.py
      paths.py
    utils/
      io_utils.py
      logging_utils.py
      seed_utils.py
    data/
      slide_registry.py
      wsi_dataset.py
      wsi_preprocess.py
    features/
      phikon_wrapper.py
    models/
      mil_backbone.py
      attention_mil.py
    train/
      losses.py
      train_mil.py
    infer/
      rank_export.py
    audit/
      perturbations.py
      metrics.py
      heatmaps.py
      audit_runner.py
    ui/
      viewer_app.py
    api/
      api_server.py
```

Example local folders:

```text
project_root/
  data/
    camelyon16/
      slides/
      masks/
      metadata.csv
  outputs/
    tiles/
    bags/
    checkpoints/
    rankings/
    audit/
    heatmaps/
```

Typical commands:

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run src/ui/viewer_app.py
uvicorn src.api.api_server:app --reload
```
