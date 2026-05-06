from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config.paths import BAGS_DIR


class SlideBagDataset(Dataset):
    def __init__(self, split_frame: pd.DataFrame, bags_dir: Path = BAGS_DIR):
        self.frame = split_frame.reset_index(drop=True)
        self.bags_dir = bags_dir

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict:
        row = self.frame.iloc[idx]
        slide_id = row["slide_id"]
        label = int(row["label"])
        bag_path = self.bags_dir / f"{slide_id}.pt"
        payload = torch.load(bag_path, map_location="cpu")
        features = payload["features"].float()
        coords = payload["coords"].long()
        return {
            "slide_id": slide_id,
            "label": torch.tensor(label).long(),
            "features": features,
            "coords": coords,
            "tile_paths": payload.get("tile_paths", []),
        }


class PatchAuditDataset(Dataset):
    def __init__(self, rankings_csv: Path):
        self.frame = pd.read_csv(rankings_csv)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict:
        row = self.frame.iloc[idx].to_dict()
        return row
