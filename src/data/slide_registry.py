from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config.paths import METADATA_FILE, SLIDES_DIR


class SlideRegistry:
    def __init__(self, metadata_file: Path = METADATA_FILE, slides_dir: Path = SLIDES_DIR):
        self.metadata_file = metadata_file
        self.slides_dir = slides_dir
        self.frame = self._load()

    def _load(self) -> pd.DataFrame:
        if self.metadata_file.exists():
            frame = pd.read_csv(self.metadata_file)
        else:
            slide_files = sorted(self.slides_dir.glob("*.tif"))
            rows = []
            for path in slide_files:
                label = 1 if "tumor" in path.stem.lower() else 0
                split = "train"
                rows.append({"slide_id": path.stem, "slide_path": str(path), "label": label, "split": split})
            frame = pd.DataFrame(rows)
        if "slide_path" not in frame.columns:
            frame["slide_path"] = frame["slide_id"].apply(lambda x: str(self.slides_dir / f"{x}.tif"))
        return frame

    def get_split(self, split: str) -> pd.DataFrame:
        return self.frame[self.frame["split"] == split].reset_index(drop=True)

    def get_slide(self, slide_id: str) -> Dict:
        row = self.frame[self.frame["slide_id"] == slide_id].iloc[0]
        return row.to_dict()

    def all_slide_ids(self) -> List[str]:
        return self.frame["slide_id"].tolist()
