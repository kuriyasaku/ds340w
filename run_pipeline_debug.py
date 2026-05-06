import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch

from src.audit.audit_runner import AuditRunner
from src.config.paths import BAGS_DIR, CHECKPOINTS_DIR, HEATMAPS_DIR, LOGS_DIR, OUTPUT_ROOT, RANKINGS_DIR, SLIDES_DIR, TILES_DIR, ensure_paths
from src.data.slide_registry import SlideRegistry
from src.data.wsi_preprocess import WSIPreprocessor
from src.features.phikon_wrapper import FeatureExtractor
from src.infer.rank_export import RankExporter
from src.train.train_mil import MILTrainer
from src.utils.seed_utils import seed_everything


class StageTimer:
    def __init__(self):
        self.records = []

    def run(self, name, fn, skip=False):
        started = time.time()
        record = {
            "stage": name,
            "status": "skipped" if skip else "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": 0.0,
            "error": "",
        }
        print_line()
        print(f"[PIPELINE] stage={name} status={record['status']}")
        if skip:
            self.records.append(record)
            return None
        try:
            result = fn()
            record["status"] = "completed"
            return result
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
            print(f"[ERROR] {name} failed")
            print(traceback.format_exc())
            raise
        finally:
            record["duration_seconds"] = round(time.time() - started, 3)
            print(f"[PIPELINE] stage={name} status={record['status']} duration={record['duration_seconds']}s")
            self.records.append(record)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.records, indent=2), encoding="utf-8")


def print_line():
    print("=" * 96)


def command_exists(cmd):
    return shutil.which(cmd) is not None


def count_files(path, pattern):
    if not Path(path).exists():
        return 0
    return len(list(Path(path).rglob(pattern)))


def system_snapshot():
    gpu_available = torch.cuda.is_available()
    snapshot = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd()),
        "cuda_available": bool(gpu_available),
        "cuda_device_count": int(torch.cuda.device_count()) if gpu_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if gpu_available else "cpu",
        "torch_version": torch.__version__,
        "openslide_available": command_exists("openslide-show-properties") or command_exists("openslide-write-png"),
    }
    return snapshot


def write_snapshot():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    snap = system_snapshot()
    path = LOGS_DIR / "debug_environment.json"
    path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print_line()
    print("[ENVIRONMENT]")
    for k, v in snap.items():
        print(f"{k}: {v}")
    return snap


def preflight_checks(strict=False):
    ensure_paths()
    write_snapshot()
    slide_count = count_files(SLIDES_DIR, "*.tif")
    bag_count = count_files(BAGS_DIR, "*.pt")
    ranking_count = count_files(RANKINGS_DIR, "*_rankings.csv")
    audit_count = count_files(OUTPUT_ROOT / "audit", "*_audit.csv")
    heatmap_count = count_files(HEATMAPS_DIR, "*.png")
    print_line()
    print("[PREFLIGHT]")
    print(f"slides_dir={SLIDES_DIR}")
    print(f"slides_tif_count={slide_count}")
    print(f"bags_count={bag_count}")
    print(f"rankings_count={ranking_count}")
    print(f"audit_csv_count={audit_count}")
    print(f"heatmap_png_count={heatmap_count}")
    if strict and slide_count == 0:
        raise FileNotFoundError(f"No .tif WSI files found in {SLIDES_DIR}")
    return {
        "slides": slide_count,
        "bags": bag_count,
        "rankings": ranking_count,
        "audit": audit_count,
        "heatmaps": heatmap_count,
    }


def postflight_report():
    report = {
        "tiles": count_files(TILES_DIR, "*.png"),
        "bags": count_files(BAGS_DIR, "*.pt"),
        "checkpoints": count_files(CHECKPOINTS_DIR, "*.pt"),
        "rankings": count_files(RANKINGS_DIR, "*_rankings.csv"),
        "audit_csv": count_files(OUTPUT_ROOT / "audit", "*_audit.csv"),
        "heatmaps": count_files(HEATMAPS_DIR, "*.png"),
    }
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = LOGS_DIR / "postflight_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_line()
    print("[POSTFLIGHT]")
    for k, v in report.items():
        print(f"{k}: {v}")
    return report


def preview_registry():
    registry = SlideRegistry()
    print_line()
    print("[SLIDE REGISTRY]")
    print(f"registered_slides={len(registry.frame)}")
    if len(registry.frame) > 0:
        print(registry.frame.head(10).to_string(index=False))
    return registry


def preview_outputs():
    print_line()
    print("[OUTPUT PREVIEW]")
    for path in sorted(RANKINGS_DIR.glob("*_rankings.csv"))[:3]:
        df = pd.read_csv(path)
        print(f"ranking_file={path.name} rows={len(df)}")
        print(df.head(3).to_string(index=False))
    for path in sorted((OUTPUT_ROOT / "audit").rglob("*_audit.csv"))[:3]:
        df = pd.read_csv(path)
        print(f"audit_file={path.name} rows={len(df)}")
        print(df.head(3).to_string(index=False))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-features", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-ranking", action="store_true")
    p.add_argument("--skip-audit", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preview", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    ensure_paths()
    seed_everything(args.seed)
    timer = StageTimer()
    timer.run("preflight", lambda: preflight_checks(strict=args.strict))
    timer.run("registry", preview_registry)
    if args.dry_run:
        print_line()
        print("[DRY RUN] no compute stages were executed")
        timer.save(LOGS_DIR / "debug_stage_timing.json")
        return
    timer.run("preprocessing", lambda: WSIPreprocessor().run(), skip=args.skip_preprocess)
    timer.run("feature_extraction", lambda: FeatureExtractor().run(), skip=args.skip_features)
    timer.run("mil_training", lambda: MILTrainer().fit(), skip=args.skip_train)
    timer.run("ranking_export", lambda: RankExporter().run(), skip=args.skip_ranking)
    timer.run("audit_runner", lambda: AuditRunner().run_all(), skip=args.skip_audit)
    timer.run("postflight", postflight_report)
    if args.preview:
        timer.run("preview_outputs", preview_outputs)
    timer.save(LOGS_DIR / "debug_stage_timing.json")
    print_line()
    print("[DONE] debug pipeline completed")


if __name__ == "__main__":
    main()
