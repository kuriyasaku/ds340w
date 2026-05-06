from pathlib import Path
from typing import List, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.config.paths import BAGS_DIR, TILES_DIR
from src.config.settings import CONFIG
from src.data.slide_registry import SlideRegistry
from src.utils.logging_utils import get_logger


logger = get_logger("phikon_wrapper")


class TileDataset(Dataset):
    def __init__(self, tile_paths: List[Path]):
        self.tile_paths = tile_paths
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx: int):
        path = self.tile_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), str(path)


class DummyPhikonEncoder(torch.nn.Module):
    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class FeatureExtractor:
    def __init__(self):
        self.device = torch.device(CONFIG.runtime.device if torch.cuda.is_available() else "cpu")
        self.batch_size = CONFIG.features.batch_size
        self.feature_dim = CONFIG.features.feature_dim
        self.model = DummyPhikonEncoder(self.feature_dim).to(self.device).eval()

    def _parse_coord(self, path: Path) -> Tuple[int, int]:
        parts = path.stem.split("__")
        x = int(parts[-2].replace("x_", ""))
        y = int(parts[-1].replace("y_", ""))
        return x, y

    def extract_slide(self, slide_id: str) -> None:
        tile_dir = TILES_DIR / slide_id
        tile_paths = sorted(tile_dir.glob("*.png"))
        if len(tile_paths) == 0:
            return
        dataset = TileDataset(tile_paths)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=CONFIG.runtime.num_workers)
        feats = []
        paths = []
        with torch.no_grad():
            for batch, batch_paths in tqdm(loader, leave=False):
                batch = batch.to(self.device)
                out = self.model(batch)
                feats.append(out.cpu())
                paths.extend(batch_paths)
        features = torch.cat(feats, dim=0)
        coords = torch.tensor([self._parse_coord(Path(p)) for p in paths]).long()
        payload = {"features": features, "coords": coords, "tile_paths": paths}
        torch.save(payload, BAGS_DIR / f"{slide_id}.pt")

    def run(self) -> None:
        registry = SlideRegistry()
        BAGS_DIR.mkdir(parents=True, exist_ok=True)
        for slide_id in registry.all_slide_ids():
            logger.info(f"extracting features for {slide_id}")
            self.extract_slide(slide_id)
