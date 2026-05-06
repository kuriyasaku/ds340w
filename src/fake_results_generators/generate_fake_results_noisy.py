from pathlib import Path

import numpy as np
import pandas as pd

from generate_fake_results import build


if __name__ == "__main__":
    build(root=".", slides=10, patches_per_slide=40, seed=99)
    root = Path(".")
    for path in (root / "outputs" / "audit").rglob("*_audit.csv"):
        df = pd.read_csv(path)
        rng = np.random.default_rng(99)
        df["confidence_drop"] = df["confidence_drop"] + rng.normal(0, 0.045, len(df))
        df["perturbed_prob_tumor"] = (df["original_prob_tumor"] + df["confidence_drop"]).clip(0.01, 0.999)
        df["perturbed_pred"] = (df["perturbed_prob_tumor"] >= 0.5).astype(int)
        df["prediction_flip"] = (df["original_pred"] != df["perturbed_pred"]).astype(int)
        df["explanation_shift"] = (df["explanation_shift"] + rng.normal(0, 0.04, len(df))).clip(0, 1)
        df.to_csv(path, index=False)
