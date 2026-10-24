"""Create a tiny FAKE dataset with the same folder layout as the Kaggle one.

Use it to check that your environment and the whole pipeline work before (or
without) downloading the real data. The images are coloured blobs, NOT cells -
never report results from this data.

Like the real data, every TRAIN/TEST image is a rotated / flipped / shifted copy of an
"original photo" (dataset-master/JPEGImages, with labels.csv). The true source of every
copy is written to <out>/fake_sources.csv so tests/check_group_split.py can verify that
src/group_split.py recovers it. Per class: 6 photos only in TRAIN, 1 photo only in TEST,
1 photo in both (so the test overlap detection is exercised too).

    python tests/make_fake_data.py --out data_fake
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
COLORS = {"EOSINOPHIL": (200, 60, 120), "LYMPHOCYTE": (60, 60, 170),
          "MONOCYTE": (120, 80, 180), "NEUTROPHIL": (160, 100, 150)}
RADIUS = {"EOSINOPHIL": 55, "LYMPHOCYTE": 40, "MONOCYTE": 70, "NEUTROPHIL": 50}


def make_original(cls: str, rng: np.random.Generator) -> Image.Image:
    """640x480 'photo': red-cell rings at random places (unique layout per photo) + one white cell."""
    img = Image.new("RGB", (640, 480), (235, 205, 205))
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(70, 100)):  # densely packed red cells, like a real smear
        x, y, r = rng.integers(0, 640), rng.integers(0, 480), int(rng.integers(18, 30))
        tint = tuple(int(c) for c in rng.integers(150, 230, size=3))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(225, 175, 180), outline=tint, width=7)
    cx, cy = 320 + rng.integers(-110, 110), 240 + rng.integers(-80, 80)
    r = RADIUS[cls] + rng.integers(-8, 8)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS[cls])
    arr = np.asarray(img).astype(np.int16) + rng.integers(-10, 10, size=(480, 640, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def augment(photo: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Like the Kaggle copies: 320x240, random flip, any rotation (black corners), small shift."""
    img = photo.resize((320, 240), Image.BILINEAR)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(float(rng.uniform(0, 360)), resample=Image.BILINEAR, fillcolor=(0, 0, 0),
                     translate=(int(rng.integers(-8, 9)), int(rng.integers(-8, 9))))
    arr = np.asarray(img).astype(np.int16) + rng.integers(-6, 6, size=(240, 320, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_fake")
    ap.add_argument("--copies", type=int, default=7, help="copies per photo in each folder it appears in")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    root = out / "dataset2-master" / "images"
    orig_dir = out / "dataset-master" / "dataset-master" / "JPEGImages"
    orig_dir.mkdir(parents=True, exist_ok=True)

    plan = [("TRAIN",)] * 6 + [("TEST",), ("TRAIN", "TEST")]  # per class: where each photo's copies go
    labels, truth, num = [], [], 0
    for cls in CLASSES:
        for j, folders in enumerate(plan):
            photo = make_original(cls, rng)
            name = f"BloodImage_{num:05d}"
            photo.save(orig_dir / f"{name}.jpg", quality=95)
            labels.append((num, num, cls))
            for split in folders:
                d = root / split / cls
                d.mkdir(parents=True, exist_ok=True)
                n = a.copies if len(folders) == 1 else max(2, a.copies // 2)
                for _ in range(n):
                    fname = f"_{j}_{rng.integers(1, 10**7)}.jpeg"   # Kaggle-style: _<source index>_<random>
                    augment(photo, rng).save(d / fname)
                    truth.append((f"{split}/{cls}/{fname}", name))
            num += 1
    with open(orig_dir.parent / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "Image", "Category"])
        w.writerows(labels)
    with open(out / "fake_sources.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "source"])
        w.writerows(truth)
    # a decoy folder that the loader must ignore
    (root / "TEST_SIMPLE" / "EOSINOPHIL").mkdir(parents=True, exist_ok=True)
    augment(make_original("EOSINOPHIL", rng), rng).save(root / "TEST_SIMPLE" / "EOSINOPHIL" / "_0_1.jpeg")
    print(f"Fake dataset written to {root} ({len(truth)} images from {num} photos in {orig_dir})")


if __name__ == "__main__":
    main()
