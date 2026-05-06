from typing import Dict

import cv2
import numpy as np


def apply_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_brightness_shift(image: np.ndarray, gamma: float = 0.85) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    shifted = np.power(np.clip(image_f, 0.0, 1.0), gamma)
    return np.clip(shifted * 255.0, 0, 255).astype(np.uint8)


def apply_stain_shift(image: np.ndarray, reduction: float = 0.15) -> np.ndarray:
    image_f = image.astype(np.float32)
    image_f[..., 0] *= 1.0 - reduction * 0.4
    image_f[..., 1] *= 1.0 - reduction * 0.9
    image_f[..., 2] *= 1.0 - reduction * 0.2
    return np.clip(image_f, 0, 255).astype(np.uint8)


def apply_color_removal(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_artifacts(image: np.ndarray, area_ratio: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    h, w = out.shape[:2]
    n = max(4, int(area_ratio * h * w / 120))
    for _ in range(n):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(2, 7))
        color = tuple(int(v) for v in rng.integers(180, 255, size=3))
        cv2.circle(out, (x, y), r, color, -1)
    return out


def apply_named_perturbation(image: np.ndarray, name: str, settings: Dict | None = None) -> np.ndarray:
    settings = settings or {}
    if name == "blur":
        return apply_blur(image, settings.get("kernel_size", 15))
    if name == "brightness_shift":
        return apply_brightness_shift(image, settings.get("gamma", 0.85))
    if name == "stain_shift":
        return apply_stain_shift(image, settings.get("reduction", 0.15))
    if name == "artifacts":
        return apply_artifacts(image, settings.get("area_ratio", 0.05), settings.get("seed", 42))
    if name == "color_removal":
        return apply_color_removal(image)
    return image.copy()
