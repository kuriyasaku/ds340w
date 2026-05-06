from pathlib import Path
from typing import List

import pandas as pd
import torch
from tqdm import tqdm

from src.config.paths import BAGS_DIR, CHECKPOINTS_DIR, RANKINGS_DIR
from src.config.settings import CONFIG
from src.data.slide_registry import SlideRegistry
from src.models.attention_mil import AttentionMIL
from src.utils.logging_utils import get_logger


logger = get_logger("rank_export")


class RankExporter:
    def __init__(self):
        self.device = torch.device(CONFIG.runtime.device if torch.cuda.is_available() else "cpu")
        self.model = AttentionMIL(
            input_dim=CONFIG.mil.input_dim,
            hidden_dim=CONFIG.mil.hidden_dim,
            attention_dim=CONFIG.mil.attention_dim,
            num_classes=CONFIG.mil.num_classes,
            dropout=CONFIG.mil.dropout,
        ).to(self.device)
        ckpt = torch.load(CHECKPOINTS_DIR / "best_mil.pt", map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def export_slide(self, slide_id: str) -> Path:
        payload = torch.load(BAGS_DIR / f"{slide_id}.pt", map_location="cpu")
        feats = payload["features"].float().to(self.device)
        coords = payload["coords"].cpu().numpy()
        tile_paths = payload.get("tile_paths", [])
        out = self.model(feats)
        slide_prob = out["prob"].cpu().numpy()[0]
        inst_prob = out["instance_prob"].cpu().numpy()
        attention = out["attention"].cpu().numpy()
        rows = []
        for i in range(len(attention)):
            rows.append({
                "slide_id": slide_id,
                "patch_index": i,
                "x": int(coords[i][0]),
                "y": int(coords[i][1]),
                "tile_path": tile_paths[i] if i < len(tile_paths) else "",
                "attention": float(attention[i]),
                "instance_prob_normal": float(inst_prob[i][0]),
                "instance_prob_tumor": float(inst_prob[i][1]),
                "slide_prob_normal": float(slide_prob[0]),
                "slide_prob_tumor": float(slide_prob[1]),
            })
        df = pd.DataFrame(rows).sort_values("attention", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        top_n = max(1, int(len(df) * CONFIG.mil.top_k_ratio))
        df["selected_for_audit"] = 0
        df.loc[: top_n - 1, "selected_for_audit"] = 1
        path = RANKINGS_DIR / f"{slide_id}_rankings.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def run(self) -> List[Path]:
        registry = SlideRegistry()
        outputs = []
        for slide_id in tqdm(registry.all_slide_ids()):
            logger.info(f"exporting rankings for {slide_id}")
            outputs.append(self.export_slide(slide_id))
        return outputs


if __name__ == "__main__":
    RankExporter().run()
