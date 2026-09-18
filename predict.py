"""Classify one (or several) blood cell images.

    python src/predict.py --checkpoint outputs/<run-name>/best.pt path/to/cell.jpeg

Prints the predicted white-cell type and the top-k class probabilities. This is
the "program that takes one blood-cell photograph and names the white-cell type"
from Section 5 of the proposal (research prototype, not a clinical tool).
"""
from __future__ import annotations

import argparse

import torch
from PIL import Image

from data import build_transforms
from model import build_model
from utils import get_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--top-k", type=int, default=4)
    a = p.parse_args(argv)

    device = get_device()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = build_model(ckpt["model_name"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tf = build_transforms(ckpt["args"]["img_size"], train=False)

    for path in a.images:
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu()
        top = torch.topk(probs, min(a.top_k, len(classes)))
        print(f"\n{path}\n  Prediction: {classes[top.indices[0]].upper()}  ({top.values[0]:.1%})")
        for v, i in zip(top.values, top.indices):
            print(f"    {classes[i]:<12s} {v:.1%}")


if __name__ == "__main__":
    main()
