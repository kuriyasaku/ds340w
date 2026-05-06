from pathlib import Path

import pandas as pd

from generate_fake_results import build


if __name__ == "__main__":
    build(root=".", slides=8, patches_per_slide=32, seed=123)
    root = Path(".")
    for path in (root / "outputs" / "audit").rglob("*_audit.csv"):
        df = pd.read_csv(path)
        df["confidence_drop"] = df["confidence_drop"].clip(-0.06, 0.02)
        df["perturbed_prob_tumor"] = (df["original_prob_tumor"] + df["confidence_drop"]).clip(0.01, 0.999)
        df["perturbed_pred"] = df["original_pred"]
        df["prediction_flip"] = 0
        df["explanation_shift"] = df["explanation_shift"].clip(0.02, 0.12)
        df.to_csv(path, index=False)
