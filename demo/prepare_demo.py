"""Prepare the demo: example images + a static showcase figure (the fallback if the live demo fails).

    python demo/prepare_demo.py --data-root data --checkpoint <run>/best.pt --split-file group_split.json

With --split-file (recommended, final setup):
  * picks N images per class from the CLEAN TEST subset: test images whose source photo is not in train and whose
    photo match is sure (see src/group_split.py), each from a DIFFERENT photo, so the demo shows cells the model has
    truly never seen. The test set was already evaluated once by evaluate.py; picking demo images changes no decision.
Without --split-file (midterm behaviour): picks images from the random 10% validation split of the TRAIN folder.

Copies them to demo/examples/ and saves results/figures/demo_showcase.png: image, prediction, confidence, Grad-CAM.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data import find_split_dirs, split_indices  # noqa: E402
from predictor import Predictor  # noqa: E402


def clean_test_examples(test_dir: Path, split_file: str, per_class: int, rng):
    """(class, path) pairs: clean test images, one per source photo, per_class photos per class."""
    spec = json.loads(Path(split_file).read_text())["images"]
    ds = datasets.ImageFolder(test_dir)
    classes = [c.lower() for c in ds.classes]
    chosen = []
    for ci, cname in enumerate(classes):
        by_photo = {}
        for path, y in ds.samples:
            if y != ci:
                continue
            info = spec.get(f"TEST/{Path(os.path.relpath(path, test_dir)).as_posix()}")
            if info and not info["source_in_train"]:
                by_photo.setdefault(info["source"], []).append(path)
        photos = sorted(by_photo)
        for photo in rng.choice(photos, size=min(per_class, len(photos)), replace=False):
            chosen.append((cname, Path(rng.choice(by_photo[photo]))))
    return classes, chosen


def val_examples(train_dir: Path, val_frac: float, seed: int, per_class: int, rng):
    ds = datasets.ImageFolder(train_dir)
    _, val_idx = split_indices(ds.targets, val_frac, seed)
    classes = [c.lower() for c in ds.classes]
    chosen = []
    for ci, cname in enumerate(classes):
        pool = [i for i in val_idx if ds.targets[i] == ci]
        for i in rng.choice(pool, size=min(per_class, len(pool)), replace=False):
            chosen.append((cname, Path(ds.samples[i][0])))
    return classes, chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-file", default=None, help="group split from src/group_split.py: use clean test images")
    ap.add_argument("--out-dir", default="demo/examples")
    ap.add_argument("--showcase", default="results/figures/demo_showcase.png")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    train_dir, test_dir = find_split_dirs(a.data_root)
    if a.split_file:
        classes, picked = clean_test_examples(test_dir, a.split_file, a.per_class, rng)
        source = "Clean test images (source photo never seen in training)"
    else:
        classes, picked = val_examples(train_dir, a.val_frac, a.seed, a.per_class, rng)
        source = "Validation images (random split of the TRAIN folder)"

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        old.unlink()
    chosen, counter = [], {}
    for cname, src in picked:
        counter[cname] = counter.get(cname, 0) + 1
        dst = out / f"{cname}_{counter[cname]}{src.suffix.lower()}"
        shutil.copy(src, dst)
        chosen.append((cname, dst))

    predictor = Predictor(a.checkpoint)
    results = []
    for true_cls, path in chosen:
        r = predictor.predict(Image.open(path))
        results.append((true_cls, path, r))
        flag = "OK   " if r["pred"] == true_cls else "WRONG"
        print(f"{flag} {path.name:18s} true={true_cls:11s} pred={r['pred']:11s} conf={r['confidence']:.3f}")
    n_ok = sum(r["pred"] == t for t, _, r in results)
    print(f"{n_ok}/{len(results)} correct  (a handful of examples, not a result: see eval/metrics.json for that)")

    per = a.per_class
    fig, axes = plt.subplots(len(classes), per * 2, figsize=(3.4 * per * 2, 2.9 * len(classes)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for true_cls, path, r in results:
        row = classes.index(true_cls)
        col = int(path.stem.rsplit("_", 1)[1]) - 1
        ax_img, ax_cam = axes[row, col * 2], axes[row, col * 2 + 1]
        ax_img.imshow(Image.open(path).convert("RGB"))
        ax_cam.imshow(r["overlay"])
        ok = r["pred"] == true_cls
        ax_img.set_title(f"true: {true_cls}", fontsize=10)
        ax_cam.set_title(f"pred: {r['pred']} ({r['confidence']:.0%})", fontsize=10, color="#1a7f37" if ok else "#c62828")
    fig.suptitle(f"{source}: image, prediction and Grad-CAM heatmap ({predictor.model_name})", fontsize=12)
    fig.tight_layout()
    Path(a.showcase).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.showcase, dpi=140)
    plt.close(fig)
    print(f"\nExamples in {out}/  ·  showcase figure: {a.showcase}")


if __name__ == "__main__":
    main()
