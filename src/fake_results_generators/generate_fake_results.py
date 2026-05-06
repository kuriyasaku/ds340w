import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PERTURBATIONS = ["blur", "stain_shift", "artifacts", "color_removal", "brightness_shift"]


def ensure(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def synthetic_patch(seed, size=256):
    rng = np.random.default_rng(seed)
    base = rng.normal(178, 22, (size, size, 3)).clip(0, 255).astype(np.uint8)
    base[..., 0] = np.clip(base[..., 0] + 28, 0, 255)
    base[..., 2] = np.clip(base[..., 2] + 18, 0, 255)
    for _ in range(rng.integers(22, 45)):
        cx, cy = rng.integers(0, size, 2)
        rx, ry = rng.integers(6, 24), rng.integers(6, 24)
        color = (
            int(rng.integers(95, 170)),
            int(rng.integers(45, 100)),
            int(rng.integers(115, 190)),
        )
        angle = float(rng.integers(0, 180))
        cv2.ellipse(base, (int(cx), int(cy)), (int(rx), int(ry)), angle, 0, 360, color, -1)
    blur = cv2.GaussianBlur(base, (3, 3), 0)
    mix = (0.72 * base + 0.28 * blur).clip(0, 255).astype(np.uint8)
    return mix


def perturb(image, name, seed):
    rng = np.random.default_rng(seed)
    if name == "blur":
        return cv2.GaussianBlur(image, (15, 15), 0)
    if name == "stain_shift":
        out = image.astype(np.float32)
        out[..., 0] *= 0.95
        out[..., 1] *= 0.86
        out[..., 2] *= 1.03
        return out.clip(0, 255).astype(np.uint8)
    if name == "artifacts":
        out = image.copy()
        h, w = out.shape[:2]
        for _ in range(22):
            x, y = rng.integers(0, w), rng.integers(0, h)
            r = int(rng.integers(2, 7))
            color = tuple(int(v) for v in rng.integers(210, 255, 3))
            cv2.circle(out, (int(x), int(y)), r, color, -1)
        return out
    if name == "color_removal":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if name == "brightness_shift":
        gamma = 0.85
        arr = image.astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        return (arr * 255).clip(0, 255).astype(np.uint8)
    return image


def heatmap(seed, size=256, shift=0.0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    m = np.zeros((size, size), dtype=np.float32)
    for _ in range(4):
        cx = rng.uniform(45, 210) + shift * rng.uniform(-35, 35)
        cy = rng.uniform(45, 210) + shift * rng.uniform(-35, 35)
        sx = rng.uniform(18, 42)
        sy = rng.uniform(18, 42)
        amp = rng.uniform(0.45, 1.0)
        m += amp * np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2) + ((yy - cy) ** 2) / (2 * sy ** 2)))
    m += rng.normal(0, 0.025, (size, size)).astype(np.float32)
    return normalize(m)


def heatmap_rgb(cam):
    cam8 = (normalize(cam) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(cam8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay(image, cam, alpha=0.45):
    h = heatmap_rgb(cam)
    return (alpha * h.astype(np.float32) + (1 - alpha) * image.astype(np.float32)).clip(0, 255).astype(np.uint8)


def save_image(arr, path):
    ensure(Path(path).parent)
    Image.fromarray(arr).save(path)


def base_prob(seed):
    rng = np.random.default_rng(seed)
    return float(np.clip(rng.normal(0.91, 0.08), 0.52, 0.998))


def perturb_effect(name, seed):
    rng = np.random.default_rng(seed)
    if name == "blur":
        return float(rng.normal(-0.22, 0.10)), float(rng.normal(0.28, 0.08))
    if name == "brightness_shift":
        return float(rng.normal(-0.035, 0.025)), float(rng.normal(0.08, 0.03))
    if name == "stain_shift":
        return float(rng.normal(-0.055, 0.035)), float(rng.normal(0.11, 0.04))
    if name == "color_removal":
        return float(rng.normal(-0.11, 0.06)), float(rng.normal(0.17, 0.05))
    if name == "artifacts":
        return float(rng.normal(-0.13, 0.08)), float(rng.normal(0.22, 0.07))
    return 0.0, 0.0


def build(root, slides=8, patches_per_slide=36, seed=42):
    root = Path(root)
    tiles_dir = root / "outputs" / "tiles"
    heat_dir = root / "outputs" / "heatmaps"
    audit_root = root / "outputs" / "audit"
    ranking_dir = root / "outputs" / "rankings"
    ensure(tiles_dir)
    ensure(heat_dir)
    ensure(audit_root)
    ensure(ranking_dir)
    rng = np.random.default_rng(seed)

    for s in range(1, slides + 1):
        slide_id = f"tumor_{s:03d}" if s % 2 else f"normal_{s:03d}"
        ranking_rows = []
        saved_patches = []
        for p in range(1, patches_per_slide + 1):
            patch_seed = seed * 100000 + s * 1000 + p
            patch = synthetic_patch(patch_seed)
            x = int(rng.integers(0, 90000))
            y = int(rng.integers(0, 90000))
            tile_path = tiles_dir / slide_id / f"{slide_id}__top{p:05d}__x_{x}__y_{y}.png"
            save_image(patch, tile_path)
            attention = float(np.exp(-p / max(4, patches_per_slide / 5)) + rng.uniform(0, 0.015))
            prob = base_prob(patch_seed)
            ranking_rows.append({
                "slide_id": slide_id,
                "patch_index": p - 1,
                "x": x,
                "y": y,
                "tile_path": str(tile_path),
                "attention": attention,
                "instance_prob_normal": 1 - prob,
                "instance_prob_tumor": prob,
                "slide_prob_normal": 0.08 if "tumor" in slide_id else 0.72,
                "slide_prob_tumor": 0.92 if "tumor" in slide_id else 0.28,
                "rank": p,
                "selected_for_audit": 1 if p <= max(2, int(patches_per_slide * 0.25)) else 0,
            })
            saved_patches.append((p, tile_path, patch, prob))

        ranking_df = pd.DataFrame(ranking_rows).sort_values("attention", ascending=False).reset_index(drop=True)
        ranking_df["rank"] = ranking_df.index + 1
        ranking_df.to_csv(ranking_dir / f"{slide_id}_rankings.csv", index=False)

        for name in PERTURBATIONS:
            rows = []
            for p, tile_path, patch, prob in saved_patches[: max(2, int(patches_per_slide * 0.25))]:
                effect, shift_level = perturb_effect(name, seed + s * 1000 + p)
                pert_prob = float(np.clip(prob + effect, 0.01, 0.999))
                pred_a = int(prob >= 0.5)
                pred_b = int(pert_prob >= 0.5)
                pert = perturb(patch, name, seed + p)
                cam_a = heatmap(seed + s * 1000 + p, shift=0.0)
                cam_b = heatmap(seed + s * 1000 + p, shift=shift_level)
                prefix = heat_dir / slide_id / f"rank_{p:05d}_{name}"
                orig_heat = heat_dir / slide_id / f"rank_{p:05d}_original_heatmap.png"
                pert_heat = Path(str(prefix) + "_heatmap.png")
                orig_overlay = heat_dir / slide_id / f"rank_{p:05d}_original_overlay.png"
                pert_overlay = Path(str(prefix) + "_overlay.png")
                pert_patch_path = heat_dir / slide_id / f"rank_{p:05d}_{name}_patch.png"
                save_image(heatmap_rgb(cam_a), orig_heat)
                save_image(heatmap_rgb(cam_b), pert_heat)
                save_image(overlay(patch, cam_a), orig_overlay)
                save_image(overlay(pert, cam_b), pert_overlay)
                save_image(pert, pert_patch_path)
                rows.append({
                    "slide_id": slide_id,
                    "tile_path": str(tile_path),
                    "perturbed_tile_path": str(pert_patch_path),
                    "rank": p,
                    "perturbation": name,
                    "original_prob_tumor": prob,
                    "perturbed_prob_tumor": pert_prob,
                    "original_pred": pred_a,
                    "perturbed_pred": pred_b,
                    "confidence_drop": pert_prob - prob,
                    "prediction_flip": int(pred_a != pred_b),
                    "explanation_shift": float(np.clip(abs(shift_level) + rng.normal(0, 0.015), 0, 1)),
                    "original_heatmap": str(orig_heat),
                    "perturbed_heatmap": str(pert_heat),
                    "original_overlay": str(orig_overlay),
                    "perturbed_overlay": str(pert_overlay),
                })
            out_dir = audit_root / name
            ensure(out_dir)
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / f"{slide_id}_rankings_audit.csv", index=False)
            summary = {
                "slide_id": slide_id,
                "perturbation": name,
                "prediction_flip_rate": float(df["prediction_flip"].mean()),
                "mean_confidence_drop": float(df["confidence_drop"].mean()),
                "mean_explanation_shift": float(df["explanation_shift"].mean()),
                "n": int(len(df)),
            }
            pd.DataFrame([summary]).to_csv(out_dir / f"{slide_id}_rankings_summary.csv", index=False)

    combined = []
    for p in audit_root.rglob("*_audit.csv"):
        combined.append(pd.read_csv(p))
    if combined:
        all_df = pd.concat(combined, ignore_index=True)
        all_df.to_csv(audit_root / "all_audit_results.csv", index=False)
        all_df.groupby("perturbation").agg(
            prediction_flip_rate=("prediction_flip", "mean"),
            mean_confidence_drop=("confidence_drop", "mean"),
            mean_explanation_shift=("explanation_shift", "mean"),
            n=("prediction_flip", "size"),
        ).reset_index().to_csv(audit_root / "summary_by_perturbation.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--slides", type=int, default=8)
    parser.add_argument("--patches-per-slide", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.root, args.slides, args.patches_per_slide, args.seed)


if __name__ == "__main__":
    main()
