"""Load a trained checkpoint once and classify single images (used by the demo)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image

from data import build_transforms
from gradcam import GradCAM, overlay_cam, target_layer_for
from model import build_model
from utils import get_device


class Predictor:
    def __init__(self, checkpoint: str, device: torch.device | None = None):
        self.device = device or get_device()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        self.model_name = ckpt["model_name"]
        self.model = build_model(self.model_name, len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.transform = build_transforms(ckpt["args"]["img_size"], train=False)
        self.cam = GradCAM(self.model, target_layer_for(self.model))

    def predict(self, image: Image.Image) -> dict:
        """Returns probabilities per class, the predicted class, its confidence and a Grad-CAM overlay."""
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        cam, logits = self.cam(x)  # one forward + backward pass gives both the scores and the heatmap
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "probs": dict(zip(self.classes, probs)),
            "pred": self.classes[best],
            "confidence": probs[best],
            "overlay": overlay_cam(image, cam),
        }
