from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data"
CAMELYON16_ROOT = DATA_ROOT / "camelyon16"
SLIDES_DIR = CAMELYON16_ROOT / "slides"
MASKS_DIR = CAMELYON16_ROOT / "masks"
METADATA_FILE = CAMELYON16_ROOT / "metadata.csv"
OUTPUT_ROOT = ROOT / "outputs"
TILES_DIR = OUTPUT_ROOT / "tiles"
BAGS_DIR = OUTPUT_ROOT / "bags"
CHECKPOINTS_DIR = OUTPUT_ROOT / "checkpoints"
RANKINGS_DIR = OUTPUT_ROOT / "rankings"
AUDIT_DIR = OUTPUT_ROOT / "audit"
HEATMAPS_DIR = OUTPUT_ROOT / "heatmaps"
LOGS_DIR = OUTPUT_ROOT / "logs"


ALL_PATHS = [
    DATA_ROOT,
    CAMELYON16_ROOT,
    SLIDES_DIR,
    MASKS_DIR,
    OUTPUT_ROOT,
    TILES_DIR,
    BAGS_DIR,
    CHECKPOINTS_DIR,
    RANKINGS_DIR,
    AUDIT_DIR,
    HEATMAPS_DIR,
    LOGS_DIR,
]


def ensure_paths() -> None:
    for p in ALL_PATHS:
        p.mkdir(parents=True, exist_ok=True)
