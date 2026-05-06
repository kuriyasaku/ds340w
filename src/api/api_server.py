from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from src.config.paths import AUDIT_DIR, CHECKPOINTS_DIR, RANKINGS_DIR


app = FastAPI(title="Pathology Audit API")


@app.get("/")
def root():
    return {"message": "Pathology Audit API running"}


@app.get("/audit/files")
def audit_files():
    files = [str(p) for p in AUDIT_DIR.rglob("*_audit.csv")]
    return {"count": len(files), "files": files}


@app.get("/audit/summary")
def audit_summary():
    summaries = []
    for path in AUDIT_DIR.rglob("*_summary.csv"):
        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()
        row["file"] = str(path)
        summaries.append(row)
    return {"count": len(summaries), "rows": summaries}


@app.get("/rankings/files")
def rankings_files():
    files = [str(p) for p in RANKINGS_DIR.glob("*_rankings.csv")]
    return {"count": len(files), "files": files}


@app.get("/train/summary")
def train_summary():
    path = CHECKPOINTS_DIR / "training_summary.json"
    if not path.exists():
        return {"exists": False}
    return {"exists": True, "path": str(path), "text": path.read_text(encoding="utf-8")}
