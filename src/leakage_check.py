"""Look for near-duplicate images across splits (augmentation leakage).

Why: the Kaggle collection is ~12,500 augmented copies generated from only a few
hundred original photos. If the official train/test split was made AFTER the
augmentation, a flipped/rotated copy of a training cell can sit in the test folder
and inflate test accuracy. The proposal keeps the official test folder sealed, but
that only protects us from leakage WE create, not from leakage already in the data.

Method: embed every image with an ImageNet ResNet-50 (2048-d, L2-normalised) and,
for each held-out image, find its most similar image in the training pool (cosine
similarity). This is a heuristic - not proof - so ALWAYS look at the saved pair
images before drawing conclusions.

    python src/leakage_check.py --data-root data

Outputs to outputs/leakage/ :  summary.json, top_pairs.png, random_pairs.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from data import build_transforms, find_split_dirs, split_indices
from model import build_model
from utils import get_device, save_json, set_seed

THRESHOLDS = (0.90, 0.95, 0.98)


@torch.no_grad()
def embed_folder(folder, model, device, img_size, batch_size, num_workers):
    ds = datasets.ImageFolder(folder, transform=build_transforms(img_size, train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = []
    for x, _ in loader:
        feats.append(F.normalize(model(x.to(device)), dim=1).cpu())
    return torch.cat(feats), ds


def nearest(query, ref, chunk=1024):
    """For each query vector, the highest cosine similarity in ref and its index."""
    best_sim, best_idx = [], []
    for i in range(0, len(query), chunk):
        sim = query[i:i + chunk] @ ref.T
        s, j = sim.max(dim=1)
        best_sim.append(s)
        best_idx.append(j)
    return torch.cat(best_sim).numpy(), torch.cat(best_idx).numpy()


def summarise(sims):
    return {"mean": float(sims.mean()), "median": float(np.median(sims)), "p95": float(np.percentile(sims, 95)),
            "max": float(sims.max()),
            **{f"frac_ge_{t:.2f}": float((sims >= t).mean()) for t in THRESHOLDS}}


def plot_pairs(pairs, out_path, title):
    """pairs: list of (query_path, ref_path, sim)."""
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.1 * n))
    axes = np.atleast_2d(axes)
    for row, (q, r, s) in zip(axes, pairs):
        row[0].imshow(Image.open(q).convert("RGB"))
        row[1].imshow(Image.open(r).convert("RGB"))
        row[0].set_title("held-out", fontsize=7)
        row[1].set_title(f"nearest train image (sim {s:.3f})", fontsize=7)
        for ax in row:
            ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="outputs/leakage")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (only for smoke tests)")
    ap.add_argument("--n-pairs", type=int, default=8)
    a = ap.parse_args(argv)

    set_seed(a.seed)
    device = get_device()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model("resnet50", 4, pretrained=not a.no_pretrained)
    model.fc = nn.Identity()
    model.to(device).eval()

    train_dir, test_dir = find_split_dirs(a.data_root)
    print("Embedding TRAIN folder ...")
    e_train, ds_train = embed_folder(train_dir, model, device, a.img_size, a.batch_size, a.num_workers)
    print("Embedding TEST folder ...")
    e_test, ds_test = embed_folder(test_dir, model, device, a.img_size, a.batch_size, a.num_workers)

    train_idx, val_idx = split_indices(ds_train.targets, a.val_frac, a.seed)
    train_paths = [s[0] for s in ds_train.samples]
    test_paths = [s[0] for s in ds_test.samples]

    results, pair_sets = {}, {}
    # 1) validation (carved from TRAIN) vs the training part of TRAIN
    sim_v, j_v = nearest(e_train[val_idx], e_train[train_idx])
    results["val_vs_train"] = summarise(sim_v)
    pair_sets["val_vs_train"] = (sim_v, [train_paths[i] for i in val_idx], [train_paths[train_idx[j]] for j in j_v])
    # 2) official TEST vs the training part of TRAIN (the one that matters for the final number)
    sim_t, j_t = nearest(e_test, e_train[train_idx])
    results["test_vs_train"] = summarise(sim_t)
    pair_sets["test_vs_train"] = (sim_t, test_paths, [train_paths[train_idx[j]] for j in j_t])

    print("\nNearest-neighbour cosine similarity (higher = more alike):")
    for name, r in results.items():
        print(f"  {name:14s} median {r['median']:.3f}  p95 {r['p95']:.3f}  max {r['max']:.3f}  "
              + "  ".join(f">={t:.2f}: {r[f'frac_ge_{t:.2f}']:.1%}" for t in THRESHOLDS))

    sims, qp, rp = pair_sets["test_vs_train"]
    order = np.argsort(-sims)[:a.n_pairs]
    plot_pairs([(qp[i], rp[i], sims[i]) for i in order], out / "top_pairs.png",
               "TEST images with the MOST similar training image")
    rng = np.random.default_rng(a.seed)
    rand = rng.choice(len(sims), size=min(a.n_pairs, len(sims)), replace=False)
    plot_pairs([(qp[i], rp[i], sims[i]) for i in rand], out / "random_pairs.png",
               "Random TEST images (what a typical neighbour looks like)")

    save_json({"results": results, "pretrained": not a.no_pretrained,
               "note": "Heuristic only: inspect top_pairs.png before concluding anything."}, out / "summary.json")
    print(f"\nSaved to {out}/  -> open top_pairs.png and random_pairs.png and LOOK at them.")


if __name__ == "__main__":
    main()
