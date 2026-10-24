"""Group-aware split: find the ORIGINAL photo behind every augmented image, then split by photo.

Why: the ~12,500 Kaggle images (dataset2-master) are rotated / flipped / shifted copies of a few hundred
original microscope photos (dataset-master/JPEGImages, shipped in the same download). A random split puts
copies of the same photo on both sides, so validation (and possibly test) accuracy is optimistic. The
midterm check compared images to each other without rotation, so rotated copies could be missed.

Method (rotation- and flip-aware):
  1. Describe every augmented image by WHAT IS WHERE inside the central circle of the frame (the part a
     rotation about the centre never cuts off, so black corners do not matter). Two feature types:
       resnet50 - ImageNet ResNet-50 last feature map, one descriptor per grid cell (layout-sensitive)
       pixels   - model-free: 48x48 grey thumbnail, band-pass filtered (red-cell outlines)
     resnet50 decides (default). pixels can be added as a cross-check; on the real data it agreed with
     resnet50 on only ~25% of images while resnet50's matches were visibly correct, so it is off by default.
  2. Describe every original photo the same way under 2 flips x (360 / angle-step) rotations, with black
     corners exactly like the augmentation produces them (default: every 5 degrees -> 144 views per photo).
  3. Each augmented image is assigned to the original whose best view is most similar (cosine = average
     agreement over the grid cells). Assigning to a known source avoids the "chaining" problem of
     clustering images with a similarity threshold.
     Safety net: images whose best photo is not clearly better than the runner-up (margin < --min-margin)
     are UNSURE. They are left out of train AND val (so a doubtful match cannot leak a photo), and an unsure
     test image never counts as "clean".
  4. TRAIN folder -> train / val with StratifiedGroupKFold, groups = source photo, so no photo is shared.
  5. TEST images whose source photo is also in train are flagged; evaluate.py then reports accuracy on
     the full test folder AND on the source-independent ("clean") subset.

Checks printed (and saved) so you can trust or reject the result:
  * margin: how much better the chosen photo fits than the runner-up (small margins = unsure);
  * label agreement: does the matched original's label (labels.csv) contain the image's class folder?
    (weak on its own: a wrong photo of the same class also "agrees")
  * label agreement split by margin band: where the wrong matches live, which justifies --min-margin;
  * similarity / margin distribution and side-by-side pictures of the weakest and random matches: LOOK.

    python src/group_split.py --data-root data --out splits/group_split.json

Then train with:  python src/train.py ... --split-file splits/group_split.json
Figures and tables go to --fig-dir (default: outputs/group_split/).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from data import IMAGENET_MEAN, IMAGENET_STD, build_transforms, find_split_dirs, split_indices
from model import build_model
from utils import get_device, save_json, set_seed

AUG_SIZE = (320, 240)  # (W, H) of the augmented Kaggle images; originals are scaled to this first


def central_mask(n: int, shrink: float = 0.9) -> torch.Tensor:
    """Boolean n x n grid: the inscribed circle of the 320x240 frame (rotation-safe), after the
    square resize it becomes an ellipse (semi-axes 0.375 and 0.5 of the side). `shrink` leaves room
    for the small shifts the augmentation also applies."""
    c = (torch.arange(n) + 0.5) / n - 0.5
    yy, xx = torch.meshgrid(c, c, indexing="ij")
    ax, ay = 0.5 * AUG_SIZE[1] / AUG_SIZE[0] * shrink, 0.5 * shrink
    m = (xx / ax) ** 2 + (yy / ay) ** 2 <= 1
    return m if m.any() else torch.ones(n, n, dtype=torch.bool)


class PixelFeatures(nn.Module):
    """Model-free: grey 48x48 thumbnail, minus a 13-px blur (removes slow brightness changes), lightly
    smoothed (tolerates the 2.5-degree / few-pixel misalignment left by the view grid), central ellipse only.
    Settings chosen on the fake data in tests/, where the true source of every image is known."""

    def __init__(self, size: int = 48, highpass: int = 13, smooth: int = 3):
        super().__init__()
        self.size, self.highpass, self.smooth = size, highpass, smooth
        self.register_buffer("mask", central_mask(size))

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x.float().mean(dim=1, keepdim=True), self.size)
        hp = g - F.avg_pool2d(g, self.highpass, 1, self.highpass // 2, count_include_pad=False)
        hp = F.avg_pool2d(hp, self.smooth, 1, self.smooth // 2, count_include_pad=False)
        v = hp[:, 0][:, self.mask]
        return v - v.mean(dim=1, keepdim=True)


class ResNetGrid(nn.Module):
    """ImageNet ResNet-50 last feature map (7x7 at 224 px); every cell inside the central ellipse is
    randomly projected 2048 -> 64 and L2-normalised, so cosine = mean agreement over cells."""

    def __init__(self, pretrained: bool, img_size: int, dim: int = 64):
        super().__init__()
        net = build_model("resnet50", 4, pretrained=pretrained)
        self.body = nn.Sequential(*list(net.children())[:-2])
        gen = torch.Generator().manual_seed(0)
        self.register_buffer("proj", torch.randn(2048, dim, generator=gen) / dim ** 0.5)

    def forward(self, x):
        f = self.body(x).float()                                   # (B, 2048, h, w), h = w = 7 at 224 px
        mask = central_mask(f.shape[-1]).flatten().to(f.device)
        f = f.flatten(2).transpose(1, 2)[:, mask]                  # (B, cells inside, 2048)
        return F.normalize(f @ self.proj, dim=2).flatten(1)


def make_extractor(kind: str, pretrained: bool, img_size: int) -> nn.Module:
    return PixelFeatures() if kind == "pixels" else ResNetGrid(pretrained, img_size)


# --------------------------------------------------------------------------- #
# Finding the original photos and their labels
# --------------------------------------------------------------------------- #
def find_originals(data_root: str | Path, originals_dir: str | None = None):
    """Return (list of original image paths, path to labels.csv or None)."""
    if originals_dir:
        jpeg_dir = Path(originals_dir)
    else:
        jpeg_dir = None
        for dirpath, _, filenames in os.walk(data_root):
            if Path(dirpath).name.lower() == "jpegimages" and any(f.lower().endswith((".jpg", ".jpeg"))
                                                                  for f in filenames):
                jpeg_dir = Path(dirpath)
                break
        if jpeg_dir is None:
            raise FileNotFoundError(
                f"No JPEGImages folder found under '{data_root}'. It is part of the Kaggle download "
                f"(dataset-master/dataset-master/JPEGImages). Pass --originals-dir if it is elsewhere.")
    paths = sorted(p for p in jpeg_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    labels = jpeg_dir.parent / "labels.csv"
    return paths, (labels if labels.exists() else None)


def read_labels(labels_csv: Path | None) -> dict[int, set[str]]:
    """labels.csv -> {image number: {'NEUTROPHIL', ...}} (a photo can carry several labels or none)."""
    if labels_csv is None:
        return {}
    df = pd.read_csv(labels_csv)
    cols = {c.strip().lower(): c for c in df.columns}
    if "image" not in cols or "category" not in cols:
        print(f"  (labels.csv has columns {list(df.columns)}; expected Image and Category - label check skipped)")
        return {}
    out = {}
    for img, cat in zip(df[cols["image"]], df[cols["category"]]):
        try:
            num = int(img)
        except (TypeError, ValueError):
            continue
        if isinstance(cat, str) and cat.strip():
            out[num] = {c.strip().upper() for c in cat.split(",") if c.strip()}
    return out


def photo_number(path: Path) -> int | None:
    m = re.findall(r"\d+", path.stem)
    return int(m[-1]) if m else None


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class OriginalsDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize(AUG_SIZE, Image.BILINEAR)
        return TF.to_tensor(img)  # 0..1, not normalised yet: rotation must fill with BLACK


def normalise(x):
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def embed(model, x, use_amp):
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=use_amp):
        return F.normalize(model(x).float(), dim=1)


@torch.no_grad()
def embed_folder(folder, model, device, img_size, batch_size, num_workers, use_amp):
    ds = datasets.ImageFolder(folder, transform=build_transforms(img_size, train=False))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = [embed(model, x.to(device), use_amp).cpu() for x, _ in loader]
    return torch.cat(feats), ds


@torch.no_grad()
def embed_original_views(paths, model, device, img_size, angles, batch_size, num_workers, use_amp):
    """Returns (n_originals, n_views, D) and the list of (flip, angle) for each view."""
    views = [(flip, ang) for flip in (False, True) for ang in angles]
    loader = DataLoader(OriginalsDataset(paths), batch_size=max(1, batch_size // 8), shuffle=False,
                        num_workers=num_workers)
    out = []
    for x in loader:
        x = x.to(device)
        per_view = []
        for flip, ang in views:
            v = TF.hflip(x) if flip else x
            if ang:
                v = TF.rotate(v, float(ang), interpolation=TF.InterpolationMode.BILINEAR, fill=[0.0, 0.0, 0.0])
            v = TF.resize(v, [img_size, img_size], antialias=True)
            per_view.append(embed(model, normalise(v), use_amp))
        out.append(torch.stack(per_view, dim=1).cpu())  # (b, V, D)
    return torch.cat(out), views


def match_to_originals(aug, orig_views, device, chunk=256):
    """For every augmented image: best original, its best view, similarity, and margin to the runner-up."""
    n_orig, n_views, dim = orig_views.shape
    ref = orig_views.reshape(n_orig * n_views, dim).to(device)
    best, best_view, sim1, sim2 = [], [], [], []
    for i in range(0, len(aug), chunk):
        s = (aug[i:i + chunk].to(device) @ ref.T).view(-1, n_orig, n_views)
        s_orig, v_idx = s.max(dim=2)                                    # best view per original
        top = s_orig.topk(min(2, n_orig), dim=1)
        best.append(top.indices[:, 0].cpu())
        sim1.append(top.values[:, 0].cpu())
        sim2.append((top.values[:, 1] if n_orig > 1 else torch.zeros_like(top.values[:, 0])).cpu())
        best_view.append(v_idx.gather(1, top.indices[:, :1]).squeeze(1).cpu())
    return (torch.cat(best).numpy(), torch.cat(best_view).numpy(),
            torch.cat(sim1).numpy(), torch.cat(sim2).numpy())


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def group_train_val(classes, groups, val_frac, seed):
    """StratifiedGroupKFold; keep the fold closest to val_frac with the most similar class mix."""
    n_splits = max(2, round(1 / val_frac))
    classes, groups = np.asarray(classes), np.asarray(groups)
    overall = np.bincount(classes, minlength=classes.max() + 1) / len(classes)
    best, best_score = None, np.inf
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in sgkf.split(np.zeros(len(classes)), classes, groups):
        mix = np.bincount(classes[va], minlength=len(overall)) / len(va)
        score = abs(len(va) / len(classes) - val_frac) + np.abs(mix - overall).sum()
        if score < best_score:
            best, best_score = (np.sort(tr), np.sort(va)), score
    return best


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def rotated_original(path, flip, angle):
    img = Image.open(path).convert("RGB").resize(AUG_SIZE, Image.BILINEAR)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0)) if angle else img


def plot_matches(rows, out_path, title):
    """rows: list of (aug_path, orig_path, flip, angle, sim, margin, label_ok)."""
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.4, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for ax, (ap, op, flip, ang, s, m, ok) in zip(axes, rows):
        ax[0].imshow(Image.open(ap).convert("RGB"))
        ax[1].imshow(rotated_original(op, flip, ang))
        ax[0].set_title(f"{Path(ap).parent.name.lower()} / {Path(ap).name}", fontsize=6.5)
        lab = "" if ok is None else (" label ok" if ok else " LABEL WRONG")
        ax[1].set_title(f"{Path(op).stem} {'flip ' if flip else ''}rot {ang:g}°  sim {s:.3f} "
                        f"(+{m:.3f}){lab}", fontsize=6.5)
        for a in ax:
            a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_hist(sim, margin, out_path, min_margin=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(sim, bins=60, color="#2a6fb0")
    ax[0].set(title="Similarity to the matched original", xlabel="cosine similarity", ylabel="images")
    ax[1].hist(margin, bins=60, color="#d9822b")
    ax[1].set(title="Margin over the second-best original", xlabel="similarity difference", ylabel="images")
    if min_margin:
        ax[1].axvline(min_margin, color="black", ls="--", lw=1)
        ax[1].text(min_margin, ax[1].get_ylim()[1] * 0.95, f"  unsure < {min_margin}", fontsize=8, va="top")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--originals-dir", default=None, help="defaults to the JPEGImages folder under --data-root")
    ap.add_argument("--out", default="splits/group_split.json", help="split file used by train.py --split-file")
    ap.add_argument("--fig-dir", default="outputs/group_split")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--angle-step", type=float, default=5.0, help="degrees between rotated views of each original")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--features", nargs="+", default=["resnet50"], choices=["resnet50", "pixels"],
                    help="first one decides the split, any others are only reported as cross-checks")
    ap.add_argument("--min-margin", type=float, default=0.01,
                    help="images whose best photo beats the runner-up by less than this are 'unsure' and are "
                         "excluded from train/val and from the clean test subset")
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (only for smoke tests)")
    ap.add_argument("--n-pairs", type=int, default=8)
    a = ap.parse_args(argv)

    set_seed(a.split_seed)
    device = get_device()
    use_amp = device.type == "cuda"
    fig_dir = Path(a.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = find_split_dirs(a.data_root)
    orig_paths, labels_csv = find_originals(a.data_root, a.originals_dir)
    labels = read_labels(labels_csv)
    angles = list(np.arange(0.0, 360.0, a.angle_step))
    print(f"TRAIN: {train_dir}\nTEST : {test_dir}\nOriginals: {len(orig_paths)} photos in {orig_paths[0].parent}"
          f"  (labels for {len(labels)})\nViews per original: 2 flips x {len(angles)} rotations")

    matches = {}
    for kind in a.features:
        print(f"\n[{kind}]{' (random weights - smoke test only)' if a.no_pretrained and kind == 'resnet50' else ''}")
        model = make_extractor(kind, not a.no_pretrained, a.img_size).to(device).eval()
        print("  embedding TRAIN ...")
        e_train, ds_train = embed_folder(train_dir, model, device, a.img_size, a.batch_size, a.num_workers, use_amp)
        print("  embedding TEST ...")
        e_test, ds_test = embed_folder(test_dir, model, device, a.img_size, a.batch_size, a.num_workers, use_amp)
        print("  embedding rotated/flipped views of the originals ...")
        e_orig, views = embed_original_views(orig_paths, model, device, a.img_size, angles, a.batch_size,
                                             a.num_workers, use_amp)
        res = [match_to_originals(e, e_orig, device) for e in (e_train, e_test)]
        matches[kind] = [np.concatenate(parts) for parts in zip(*res)]   # src, view, sim1, sim2
        del e_train, e_test, e_orig, model
    primary = a.features[0]
    src, view, s1, s2 = matches[primary]

    classes = [c.upper() for c in ds_train.classes]
    rows, k = [], 0
    for folder, root, ds in (("TRAIN", train_dir, ds_train), ("TEST", test_dir, ds_test)):
        for path, y in ds.samples:
            o, v = src[k], view[k]
            lab = labels.get(photo_number(orig_paths[o]))
            rows.append({"key": f"{folder}/{Path(os.path.relpath(path, root)).as_posix()}", "folder": folder,
                         "path": path, "class": classes[y], "y": y, "source": orig_paths[o].stem,
                         "source_path": str(orig_paths[o]), "flip": views[v][0], "angle": float(views[v][1]),
                         "sim": float(s1[k]), "margin": float(s1[k] - s2[k]),
                         "label_ok": None if lab is None else classes[y] in lab,
                         **{f"source_{kd}": orig_paths[m[0][k]].stem for kd, m in matches.items() if kd != primary}})
            k += 1
    df = pd.DataFrame(rows)

    df["group"] = df.source
    df["unsure"] = df.margin < a.min_margin

    # ---- train / val split by source photo (unsure images are left out) ----
    tr_mask = (df.folder == "TRAIN").to_numpy()
    tr_rows = df[tr_mask]
    sure = ~tr_rows.unsure.to_numpy()
    sure_pos = np.where(sure)[0]
    tr_i, va_i = group_train_val(tr_rows.y.to_numpy()[sure], tr_rows.source.to_numpy()[sure], a.val_frac, a.split_seed)
    tr_idx, va_idx = sure_pos[tr_i], sure_pos[va_i]          # positions within the TRAIN folder
    split = np.array(["test"] * len(df), dtype=object)
    tr_pos = np.where(tr_mask)[0]
    split[tr_pos] = "excluded"
    split[tr_pos[tr_idx]] = "train"
    split[tr_pos[va_idx]] = "val"
    df["split"] = split
    train_sources = set(df.source[df.split == "train"])
    # "source_in_train" = NOT proven independent of train: photo in train, or the match itself is unsure
    df["source_in_train"] = df.source.isin(train_sources) | df.unsure
    assert not (set(df.source[df.split == "val"]) & train_sources), "val shares a photo with train"

    # ---- sanity: refuse to write a degenerate split ----
    n_tr, n_va = int((df.split == "train").sum()), int((df.split == "val").sum())
    val_share = n_va / max(n_tr + n_va, 1)
    if n_tr == 0 or n_va == 0 or not (0.4 * a.val_frac <= val_share <= 2.5 * a.val_frac):
        raise SystemExit(f"!! Refusing to write the split: train={n_tr}, val={n_va} (val share {val_share:.1%}). "
                         f"Send this output.")

    # ---- diagnostics ----
    summary = {"features": a.features, "min_margin": a.min_margin}
    for kd in a.features[1:]:
        summary[f"agreement_{primary}_vs_{kd}"] = float((df.source == df[f"source_{kd}"]).mean())
    lab = df.label_ok.dropna().astype(bool)
    summary["label_agreement"] = float(lab.mean()) if len(lab) else None
    summary["label_agreement_by_class"] = (df.dropna(subset=["label_ok"]).assign(ok=lambda d: d.label_ok.astype(bool))
                                           .groupby("class").ok.mean().round(4).to_dict())
    bands = [0, 0.005, 0.01, 0.02, 0.05, 1.0]
    band = pd.cut(df.margin, bands, right=False, labels=[f"{lo:.3f}-{hi:.3f}" for lo, hi in zip(bands, bands[1:])])
    by_band = (df.assign(band=band, ok=df.label_ok.astype(float)).groupby("band", observed=False)
               .agg(images=("ok", "size"), label_agreement=("ok", "mean")))
    summary["by_margin_band"] = {str(k): {"images": int(v.images), "label_agreement": round(float(v.label_agreement), 4)}
                                 for k, v in by_band.iterrows() if v.images}
    summary["similarity"] = {"median": float(df.sim.median()), "p01": float(df.sim.quantile(0.01)),
                             "min": float(df.sim.min())}
    summary["margin"] = {"median": float(df.margin.median()), "p01": float(df.margin.quantile(0.01))}
    summary["unsure"] = {"train_folder_excluded": int((df.split == "excluded").sum()),
                         "test_unsure": int((df.unsure & (df.split == "test")).sum()),
                         "photos_used": int(df.source.nunique()), "photos_available": len(orig_paths)}

    per_class = []
    for c in classes:
        d = df[df["class"] == c]
        s_tr, s_va, s_te = (set(d.source[d.split == k]) for k in ("train", "val", "test"))
        per_class.append({"class": c.lower(), "train imgs": int((d.split == "train").sum()),
                          "train photos": len(s_tr), "val imgs": int((d.split == "val").sum()),
                          "val photos": len(s_va), "excluded (unsure)": int((d.split == "excluded").sum()),
                          "test imgs": int((d.split == "test").sum()), "test photos": len(s_te),
                          "test clean": int(((d.split == "test") & ~d.source_in_train).sum())})
    per_class = pd.DataFrame(per_class)
    per_class.to_csv(fig_dir / "per_class_sources.csv", index=False)

    test = df[df.split == "test"]
    summary["test"] = {"n": int(len(test)), "n_photos": int(test.source.nunique()),
                       "n_photo_in_train": int((test.source.isin(train_sources)).sum()),
                       "n_unsure": int(test.unsure.sum()),
                       "n_clean": int((~test.source_in_train).sum()),
                       "photos_shared_with_train": int(len(set(test.source) & train_sources))}

    # before / after: share of validation images whose photo is also in train
    old_tr, old_va = split_indices(ds_train.targets, a.val_frac, a.split_seed)
    src_tr = tr_rows.source.to_numpy()
    summary["val_sharing_photo_with_train"] = {"random_split": float(np.isin(src_tr[old_va], src_tr[old_tr]).mean()),
                                               "group_split": float(np.isin(src_tr[va_idx], src_tr[tr_idx]).mean())}

    # ---- outputs ----
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = {"method": "assign each augmented image to its most similar original photo (rotation + flip views), "
                      "StratifiedGroupKFold by photo for train/val, unsure matches excluded",
            "params": {**vars(a), "pretrained": not a.no_pretrained, "n_originals": len(orig_paths)},
            "summary": summary,
            "images": {r.key: {"split": r.split, "source": r.source, "group": r.group, "sim": round(r.sim, 4),
                               "margin": round(r.margin, 4), "unsure": bool(r.unsure),
                               "source_in_train": bool(r.source_in_train)}
                       for r in df.itertuples()}}
    out.write_text(json.dumps(spec, indent=1))
    df.drop(columns=["source_path"]).to_csv(fig_dir / "assignments.csv", index=False)
    save_json({**summary, "per_class": per_class.to_dict(orient="records")}, fig_dir / "summary.json")

    def mk(sub):
        return [(r.path, r.source_path, r.flip, r.angle, r.sim, r.margin, r.label_ok) for r in sub.itertuples()]
    plot_hist(df.sim.to_numpy(), df.margin.to_numpy(), fig_dir / "match_quality.png", a.min_margin)
    plot_matches(mk(df.nsmallest(a.n_pairs, "margin")), fig_dir / "weakest_matches.png",
                 "Weakest matches (smallest margin) - these are excluded as unsure")
    sure_rows = df[~df.unsure]
    plot_matches(mk(sure_rows.nsmallest(a.n_pairs, "margin")), fig_dir / "weakest_kept_matches.png",
                 f"Weakest matches that are KEPT (margin just above {a.min_margin}): same photo?")
    plot_matches(mk(sure_rows.sample(min(a.n_pairs, len(sure_rows)), random_state=a.split_seed)),
                 fig_dir / "random_matches.png", "Random kept images and the original photo they were matched to")
    overlap = test[test.source.isin(train_sources) & ~test.unsure]
    if len(overlap):
        plot_matches(mk(overlap.sample(min(a.n_pairs, len(overlap)), random_state=a.split_seed)),
                     fig_dir / "test_overlap_examples.png", "TEST images whose source photo is also in train")

    # ---- report ----
    print(f"\n=== Matching quality ({primary} decides) ===")
    for kd in a.features[1:]:
        print(f"Cross-check: {primary} vs {kd} chose the same photo for "
              f"{summary[f'agreement_{primary}_vs_{kd}']:.1%} of images")
    if summary["label_agreement"] is not None:
        print(f"Label agreement (matched photo's label contains the image's class): {summary['label_agreement']:.1%}")
        print(f"  by class: {summary['label_agreement_by_class']}")
    print(f"Similarity to matched photo: median {summary['similarity']['median']:.3f}, "
          f"1st percentile {summary['similarity']['p01']:.3f}")
    print("By margin (how much the chosen photo beats the runner-up):")
    print(by_band.assign(label_agreement=lambda d: d.label_agreement.map("{:.1%}".format)).to_string())
    u = summary["unsure"]
    print(f"Unsure (margin < {a.min_margin}): {u['train_folder_excluded']} TRAIN-folder images excluded, "
          f"{u['test_unsure']} TEST images not counted as clean.  Photos used: {u['photos_used']} of "
          f"{u['photos_available']}")
    print("\n=== Source photos per split ===")
    print(per_class.to_string(index=False))
    t = summary["test"]
    print(f"\nTEST: {t['n']} images from {t['n_photos']} photos; {t['n_photo_in_train']} images "
          f"({t['n_photo_in_train'] / max(t['n'], 1):.1%}) come from {t['photos_shared_with_train']} photos that are "
          f"also in train; {t['n_unsure']} are unsure.  Clean (source-independent) test subset: {t['n_clean']} images.")
    v = summary["val_sharing_photo_with_train"]
    print(f"Validation images whose photo is also in train: random split {v['random_split']:.1%}  ->  "
          f"group split {v['group_split']:.1%}")
    if summary["label_agreement"] is not None and summary["label_agreement"] < 0.95:
        print("\n!! WARNING: label agreement is below 95%, so the matching may be unreliable. "
              "Look at the figures before using this split.")
    print(f"\nSplit file: {out}\nFigures/tables: {fig_dir}/  -> LOOK at weakest_kept_matches.png and random_matches.png")


if __name__ == "__main__":
    main()
