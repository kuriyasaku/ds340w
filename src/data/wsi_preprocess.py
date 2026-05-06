from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import openslide
except Exception:
    openslide = None

from src.config.paths import TILES_DIR
from src.config.settings import CONFIG
from src.data.slide_registry import SlideRegistry
from src.utils.logging_utils import get_logger


logger = get_logger("wsi_preprocess")


class WSIPreprocessor:
    def __init__(self):
        self.tile_size = CONFIG.data.tile_size
        self.stride = CONFIG.data.stride
        self.tissue_threshold = CONFIG.data.tissue_threshold
        self.max_tiles_per_slide = CONFIG.data.max_tiles_per_slide

    def _read_slide(self, slide_path: Path):
        if openslide is not None:
            return openslide.OpenSlide(str(slide_path))
        image = Image.open(slide_path).convert("RGB")
        return image

    def _slide_dims(self, slide) -> Tuple[int, int]:
        if openslide is not None and hasattr(slide, "dimensions"):
            return slide.dimensions
        return slide.size

    def _read_region(self, slide, x: int, y: int, size: int) -> np.ndarray:
        if openslide is not None and hasattr(slide, "read_region"):
            region = slide.read_region((x, y), 0, (size, size)).convert("RGB")
            return np.array(region)
        region = slide.crop((x, y, x + size, y + size))
        return np.array(region)

    def _tissue_score(self, tile: np.ndarray) -> float:
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        val = hsv[:, :, 2].astype(np.float32) / 255.0
        mask = (sat > 0.08) & (val < 0.95)
        return float(mask.mean())

    def process_slide(self, slide_id: str, slide_path: Path) -> List[Path]:
        output_dir = TILES_DIR / slide_id
        output_dir.mkdir(parents=True, exist_ok=True)
        slide = self._read_slide(slide_path)
        width, height = self._slide_dims(slide)
        saved = []
        count = 0
        for y in range(0, max(1, height - self.tile_size), self.stride):
            for x in range(0, max(1, width - self.tile_size), self.stride):
                tile = self._read_region(slide, x, y, self.tile_size)
                if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                    continue
                score = self._tissue_score(tile)
                if score < self.tissue_threshold:
                    continue
                name = f"{slide_id}__x_{x}__y_{y}.png"
                path = output_dir / name
                Image.fromarray(tile).save(path)
                saved.append(path)
                count += 1
                if count >= self.max_tiles_per_slide:
                    return saved
        return saved

    def run(self) -> None:
        registry = SlideRegistry()
        for row in tqdm(registry.frame.to_dict(orient="records")):
            slide_id = row["slide_id"]
            slide_path = Path(row["slide_path"])
            logger.info(f"processing slide {slide_id}")
            self.process_slide(slide_id, slide_path)
