"""Create a tiny FAKE dataset with the same folder layout as the Kaggle one.

Use it to check that your environment and the whole pipeline work before (or
without) downloading the real data. The images are coloured blobs, NOT cells -
never report results from this data.

    python tests/make_fake_data.py --out data_fake
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
COLORS = {"EOSINOPHIL": (200, 60, 120), "LYMPHOCYTE": (60, 60, 170),
          "MONOCYTE": (120, 80, 180), "NEUTROPHIL": (160, 100, 150)}
RADIUS = {"EOSINOPHIL": 55, "LYMPHOCYTE": 40, "MONOCYTE": 70, "NEUTROPHIL": 50}


def make_image(cls: str, rng: np.random.Generator) -> Image.Image:
    img = Image.new("RGB", (320, 240), (235, 205, 205))
    d = ImageDraw.Draw(img)
    cx, cy = 160 + rng.integers(-20, 20), 120 + rng.integers(-15, 15)
    r = RADIUS[cls] + rng.integers(-8, 8)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS[cls])
    arr = np.asarray(img).astype(np.int16) + rng.integers(-12, 12, size=(240, 320, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_fake")
    ap.add_argument("--n-train", type=int, default=40, help="images per class in TRAIN")
    ap.add_argument("--n-test", type=int, default=12, help="images per class in TEST")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.out) / "dataset2-master" / "images"
    for split, n in (("TRAIN", a.n_train), ("TEST", a.n_test)):
        for cls in CLASSES:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                make_image(cls, rng).save(d / f"_{i}_{rng.integers(1, 10**7)}.jpeg")
    # a decoy folder that the loader must ignore
    (root / "TEST_SIMPLE" / "EOSINOPHIL").mkdir(parents=True, exist_ok=True)
    make_image("EOSINOPHIL", rng).save(root / "TEST_SIMPLE" / "EOSINOPHIL" / "_0_1.jpeg")
    print(f"Fake dataset written to {root}")


if __name__ == "__main__":
    main()
