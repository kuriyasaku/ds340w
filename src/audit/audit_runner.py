from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from src.audit.heatmaps import save_heatmap_bundle
from src.audit.metrics import explanation_shift, summarize_audit_rows
from src.audit.perturbations import apply_named_perturbation
from src.config.paths import AUDIT_DIR, HEATMAPS_DIR, RANKINGS_DIR
from src.config.settings import CONFIG
from src.utils.logging_utils import get_logger


logger = get_logger("audit_runner")


class TinyPatchClassifier(torch.nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = torch.flatten(h, 1)
        return self.classifier(h)


class AuditRunner:
    def __init__(self):
        self.device = torch.device(CONFIG.runtime.device if torch.cuda.is_available() else "cpu")
        self.model = TinyPatchClassifier().to(self.device).eval()
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def _load_image(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        return self.transform(Image.fromarray(image)).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _predict(self, image: np.ndarray) -> Dict:
        x = self._to_tensor(image)
        logits = self.model(x)
        prob = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        pred = int(prob.argmax())
        confidence = float(prob[pred])
        feat = self.model.features(x)
        cam = feat.mean(dim=1)[0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return {
            "pred": pred,
            "prob_normal": float(prob[0]),
            "prob_tumor": float(prob[1]),
            "confidence": confidence,
            "cam": cam,
        }

    def run_for_csv(self, rankings_csv: Path, perturbation_name: str) -> Path:
        df = pd.read_csv(rankings_csv)
        df = df[df["selected_for_audit"] == 1].reset_index(drop=True)
        rows: List[Dict] = []
        for row in tqdm(df.to_dict(orient="records"), leave=False):
            tile_path = row["tile_path"]
            if not tile_path or not Path(tile_path).exists():
                continue
            original = self._load_image(tile_path)
            perturbed = apply_named_perturbation(original, perturbation_name, {
                "kernel_size": CONFIG.audit.blur_kernel,
                "gamma": CONFIG.audit.brightness_gamma,
                "reduction": CONFIG.audit.stain_reduction,
                "area_ratio": CONFIG.audit.artifact_ratio,
            })
            pred_a = self._predict(original)
            pred_b = self._predict(perturbed)
            slide_id = row["slide_id"]
            patch_rank = int(row["rank"])
            prefix_a = HEATMAPS_DIR / slide_id / f"rank_{patch_rank:05d}_original"
            prefix_b = HEATMAPS_DIR / slide_id / f"rank_{patch_rank:05d}_{perturbation_name}"
            heat_a, overlay_a = save_heatmap_bundle(original, pred_a["cam"], prefix_a)
            heat_b, overlay_b = save_heatmap_bundle(perturbed, pred_b["cam"], prefix_b)
            rows.append({
                "slide_id": slide_id,
                "tile_path": tile_path,
                "rank": patch_rank,
                "perturbation": perturbation_name,
                "original_prob_tumor": pred_a["prob_tumor"],
                "perturbed_prob_tumor": pred_b["prob_tumor"],
                "original_pred": pred_a["pred"],
                "perturbed_pred": pred_b["pred"],
                "confidence_drop": pred_b["prob_tumor"] - pred_a["prob_tumor"],
                "prediction_flip": int(pred_a["pred"] != pred_b["pred"]),
                "explanation_shift": explanation_shift(pred_a["cam"], pred_b["cam"]),
                "original_heatmap": str(heat_a),
                "perturbed_heatmap": str(heat_b),
                "original_overlay": str(overlay_a),
                "perturbed_overlay": str(overlay_b),
            })
        out_df = pd.DataFrame(rows)
        out_dir = AUDIT_DIR / perturbation_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{rankings_csv.stem}_audit.csv"
        out_df.to_csv(out_path, index=False)
        summary = summarize_audit_rows(rows)
        pd.DataFrame([summary]).to_csv(out_dir / f"{rankings_csv.stem}_summary.csv", index=False)
        return out_path

    def run_all(self) -> List[Path]:
        csvs = sorted(RANKINGS_DIR.glob("*_rankings.csv"))
        outputs = []
        for csv_path in csvs:
            logger.info(f"auditing {csv_path.name}")
            for perturbation_name in CONFIG.audit.perturbations:
                outputs.append(self.run_for_csv(csv_path, perturbation_name))
        return outputs


if __name__ == "__main__":
    AuditRunner().run_all()
