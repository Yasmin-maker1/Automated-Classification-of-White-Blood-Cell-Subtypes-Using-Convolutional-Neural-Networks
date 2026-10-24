"""Dataset handling for the Kaggle "Blood Cell Images" collection (Mooney, 2018).

Layout we expect somewhere under --data-root (this is how Kaggle ships it):

    dataset2-master/images/TRAIN/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg
    dataset2-master/images/TEST/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg

The folder TEST_SIMPLE and the 410-image raw set (dataset-master/) are ignored on
purpose, as stated in the proposal.

Split policy (from the proposal):
  * The official TEST folder is used ONLY for the final evaluation.
  * 10% of the official TRAIN folder is held out (stratified) for validation.
  * With a split file from group_split.py, train/val are split by SOURCE PHOTO instead (no photo on both
    sides), and the test images whose source photo is in train are marked so the "clean" subset can be
    reported separately.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from utils import seed_worker

EXPECTED_CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Finding the folders
# --------------------------------------------------------------------------- #
def find_split_dirs(data_root: str | Path) -> tuple[Path, Path]:
    """Return (train_dir, test_dir) by searching under data_root.

    A valid split folder is named TRAIN or TEST (any case) and directly contains
    the four class folders. TEST_SIMPLE is skipped because its name differs.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Data root '{root}' does not exist. Download the dataset first "
            f"(see README, 'Getting the data')."
        )
    found: dict[str, Path] = {}
    for dirpath, dirnames, _ in os.walk(root):
        name = Path(dirpath).name.upper()
        if name in ("TRAIN", "TEST") and name not in found:
            if set(EXPECTED_CLASSES).issubset({d.upper() for d in dirnames}):
                found[name] = Path(dirpath)
    if "TRAIN" not in found or "TEST" not in found:
        raise FileNotFoundError(
            f"Could not find TRAIN and TEST folders containing {EXPECTED_CLASSES} "
            f"under '{root}'. Found: {sorted(found)}."
        )
    return found["TRAIN"], found["TEST"]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transforms(img_size: int = 224, train: bool = False) -> transforms.Compose:
    """Bilinear resize (320x240 -> img_size x img_size), then ImageNet normalisation.

    Training adds the light augmentation described in the proposal: horizontal
    flip, small rotation, mild colour jitter. The dataset is already heavily
    augmented, so we deliberately do not stack more on top.
    """
    resize = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)
    normalise = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return transforms.Compose([resize, *normalise])
    return transforms.Compose(
        [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            *normalise,
        ]
    )


# --------------------------------------------------------------------------- #
# Datasets and loaders
# --------------------------------------------------------------------------- #
def split_indices(targets, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val indices over the official TRAIN folder."""
    targets = np.asarray(targets)
    idx = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_frac, stratify=targets, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def split_from_file(split_file, train_samples, test_samples, train_dir, test_dir):
    """Read a group_split.py file -> (train_idx, val_idx, clean_test_idx)."""
    spec = json.loads(Path(split_file).read_text())["images"]

    def keys(prefix, root, samples):
        return [f"{prefix}/{Path(os.path.relpath(p, root)).as_posix()}" for p, _ in samples]

    k_train, k_test = keys("TRAIN", train_dir, train_samples), keys("TEST", test_dir, test_samples)
    missing = [k for k in k_train + k_test if k not in spec]
    if missing:
        raise RuntimeError(f"{len(missing)} images are not in the split file {split_file} (e.g. {missing[:3]}). "
                           f"It was made for different data: rerun src/group_split.py.")
    split = np.array([spec[k]["split"] for k in k_train])
    if set(split) - {"train", "val", "excluded"}:   # excluded = source photo too uncertain to place safely
        raise RuntimeError(f"Unexpected split labels in {split_file}: {sorted(set(split))}")
    train_idx, val_idx = np.where(split == "train")[0], np.where(split == "val")[0]
    clean_test_idx = np.array([i for i, k in enumerate(k_test) if not spec[k]["source_in_train"]], dtype=int)
    return train_idx, val_idx, clean_test_idx


def build_datasets(data_root: str | Path, img_size: int = 224, val_frac: float = 0.10, seed: int = 42,
                   split_file: str | Path | None = None):
    """Return a dict with train/val/test datasets, class names and split indices.

    split_file=None -> stratified random split of the TRAIN folder (midterm behaviour).
    split_file=...  -> group split from group_split.py; also returns clean_test_idx.
    """
    train_dir, test_dir = find_split_dirs(data_root)

    train_full_aug = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, train=True))
    train_full_eval = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, train=False))
    test_ds = datasets.ImageFolder(test_dir, transform=build_transforms(img_size, train=False))

    if train_full_aug.classes != test_ds.classes:
        raise RuntimeError(f"TRAIN classes {train_full_aug.classes} != TEST classes {test_ds.classes}")
    if [c.upper() for c in train_full_aug.classes] != EXPECTED_CLASSES:
        raise RuntimeError(f"Unexpected class folders: {train_full_aug.classes}")

    clean_test_idx = None
    if split_file:
        train_idx, val_idx, clean_test_idx = split_from_file(split_file, train_full_aug.samples, test_ds.samples,
                                                             train_dir, test_dir)
    else:
        train_idx, val_idx = split_indices(train_full_aug.targets, val_frac, seed)
    return {
        "train": Subset(train_full_aug, train_idx),   # augmented
        "val": Subset(train_full_eval, val_idx),      # same files, NO augmentation
        "test": test_ds,
        "classes": [c.lower() for c in train_full_aug.classes],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "clean_test_idx": clean_test_idx,   # None without a split file
        "split_file": str(split_file) if split_file else None,
        "train_dir": train_dir,
        "test_dir": test_dir,
    }


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
    )
