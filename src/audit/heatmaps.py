from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SimpleGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)
        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            spatial = image_tensor.shape[-2:]
            return np.ones(spatial, dtype=np.float32)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def heatmap_to_rgb(cam: np.ndarray) -> np.ndarray:
    heat = np.uint8(np.clip(cam, 0.0, 1.0) * 255)
    return cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = heatmap_to_rgb(cam)
    if heat.shape[:2] != image.shape[:2]:
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]))
    mix = (alpha * heat.astype(np.float32) + (1 - alpha) * image.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return mix


def save_heatmap_bundle(image: np.ndarray, cam: np.ndarray, out_prefix: Path) -> Tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    heat = heatmap_to_rgb(cam)
    overlay = overlay_heatmap(image, cam)
    heat_path = Path(str(out_prefix) + "_heatmap.png")
    overlay_path = Path(str(out_prefix) + "_overlay.png")
    Image.fromarray(heat).save(heat_path)
    Image.fromarray(overlay).save(overlay_path)
    return heat_path, overlay_path
