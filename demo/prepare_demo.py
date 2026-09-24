"""Prepare the demo: example images + a static showcase figure (the fallback if the live demo fails).

    python demo/prepare_demo.py --data-root data --checkpoint outputs/resnet50_v1/best.pt

  * picks N images per class from the VALIDATION split (held out from training; the test folder is never touched)
    and copies them to demo/examples/
  * classifies them and saves results/figures/demo_showcase.png: image, prediction, confidence and Grad-CAM heatmap
"""
from __future__ import annotations

import argparse
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="demo/examples")
    ap.add_argument("--showcase", default="results/figures/demo_showcase.png")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    train_dir, _ = find_split_dirs(a.data_root)          # only the TRAIN folder is read, never TEST
    ds = datasets.ImageFolder(train_dir)
    _, val_idx = split_indices(ds.targets, a.val_frac, a.seed)
    classes = [c.lower() for c in ds.classes]
    rng = np.random.default_rng(a.seed)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        old.unlink()

    chosen = []                                           # (true_class, path)
    for ci, cname in enumerate(classes):
        pool = [i for i in val_idx if ds.targets[i] == ci]
        for k, i in enumerate(rng.choice(pool, size=min(a.per_class, len(pool)), replace=False), start=1):
            src = Path(ds.samples[i][0])
            dst = out / f"{cname}_{k}{src.suffix.lower()}"
            shutil.copy(src, dst)
            chosen.append((cname, dst))

    predictor = Predictor(a.checkpoint)
    results = []
    for true_cls, path in chosen:
        r = predictor.predict(Image.open(path))
        results.append((true_cls, path, r))
        flag = "OK " if r["pred"] == true_cls else "WRONG"
        print(f"{flag} {path.name:26s} true={true_cls:11s} pred={r['pred']:11s} conf={r['confidence']:.3f}")

    per = a.per_class
    fig, axes = plt.subplots(len(classes), per * 2, figsize=(3.4 * per * 2, 2.9 * len(classes)))
    axes = np.atleast_2d(axes)
    for n, (true_cls, path, r) in enumerate(results):
        row, col = divmod(n, per)
        ax_img, ax_cam = axes[row, col * 2], axes[row, col * 2 + 1]
        ax_img.imshow(Image.open(path).convert("RGB"))
        ax_cam.imshow(r["overlay"])
        ok = r["pred"] == true_cls
        ax_img.set_title(f"true: {true_cls}", fontsize=10)
        ax_cam.set_title(f"pred: {r['pred']} ({r['confidence']:.0%})", fontsize=10, color="#1a7f37" if ok else "#c62828")
        ax_img.axis("off")
        ax_cam.axis("off")
    fig.suptitle("Validation examples (not used for training): image, prediction and Grad-CAM heatmap", fontsize=12)
    fig.tight_layout()
    Path(a.showcase).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.showcase, dpi=140)
    plt.close(fig)
    print(f"\nExamples in {out}/  ·  showcase figure: {a.showcase}")


if __name__ == "__main__":
    main()
