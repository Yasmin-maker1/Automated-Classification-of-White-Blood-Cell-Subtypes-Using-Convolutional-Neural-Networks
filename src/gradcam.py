"""Grad-CAM: which image regions pushed the network towards its prediction.

Selvaraju et al. (2017). We take the activations of the last convolutional stage, weight each channel by the average
gradient of the class score, sum, keep the positive part, and upsample to the image size.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402


def target_layer_for(model: nn.Module) -> nn.Module:
    """Last convolutional stage of each model we use."""
    if hasattr(model, "layer4"):        # torchvision ResNet
        return model.layer4[-1]
    if hasattr(model, "features"):      # our SmallCNN
        return model.features[-1]
    raise ValueError("Do not know which layer to use for Grad-CAM on this model.")


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()
        # tensor hook instead of a module backward hook: works with in-place ReLUs
        output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1, 3, H, W). Returns (cam in [0, 1] with shape (H, W), logits)."""
        with torch.enable_grad():
            self.model.zero_grad(set_to_none=True)
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), logits.detach()


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet heatmap over the ORIGINAL image (cam is resized to its width and height)."""
    image = image.convert("RGB")
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    base = np.asarray(image).astype(np.float32) / 255.0
    return Image.fromarray((((1 - alpha) * base + alpha * heat) * 255).astype(np.uint8))
