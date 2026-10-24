"""Check src/group_split.py on the FAKE data, where the true source of every image is known.

    python tests/check_group_split.py --data-root data_fake --split-file outputs_smoke/group_split.json

Fails (exit code 1) unless
  * at least --min-accuracy of the images are matched to their true photo,
  * no photo group is on both sides of train/val, and test images are flagged exactly when their group is
    in train (the split logic itself),
  * merge_groups() joins photos the way it should (unit test).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group_split import merge_groups  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data_fake")
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--min-accuracy", type=float, default=0.97)
    a = ap.parse_args()
    problems = []

    # unit test of the safety net: a->b disagreement and b->c disagreement join a, b, c; d stays alone
    g = merge_groups(["a", "b", "d", "a"], [["b", "c", "d", "a"]])
    if not (g["a"] == g["b"] == g["c"] and g["d"] != g["a"]):
        problems.append(f"merge_groups gave {g}")

    truth = pd.read_csv(Path(a.data_root) / "fake_sources.csv").set_index("key").source
    got = pd.DataFrame.from_dict(json.loads(Path(a.split_file).read_text())["images"], orient="index")
    if set(got.index) != set(truth.index):
        problems.append(f"split file covers {len(got)} images, truth has {len(truth)}")
    got = got.loc[got.index.intersection(truth.index)]

    acc = (got.source == truth[got.index]).mean()
    print(f"Matched to the true source photo: {acc:.1%} of {len(got)} images (required: {a.min_accuracy:.0%})")
    if acc < a.min_accuracy:
        problems.append(f"only {acc:.1%} matched to the true photo")

    shared = set(got.group[got.split == "train"]) & set(got.group[got.split == "val"])
    if shared:
        problems.append(f"groups in both train and val: {sorted(shared)[:3]}")
    if (got.split == "val").sum() == 0:
        problems.append("validation set is empty")
    test = got[got.split == "test"]
    expected_flag = test.group.isin(set(got.group[got.split == "train"]))
    if not (test.source_in_train == expected_flag).all():
        problems.append("test overlap flags are inconsistent with the train groups")

    true_src = truth[got.index]
    leak = set(true_src[got.split == "train"]) & set(true_src[got.split == "val"])
    leak_imgs = int((got.split.eq("val") & true_src.isin(set(true_src[got.split == "train"]))).sum())
    print(f"True photos on both sides of train/val: {len(leak)} ({leak_imgs} val images) - caused only by wrong matches")
    print(f"Test images flagged as sharing a photo with train: {int(test.source_in_train.sum())} "
          f"(truly sharing: {int(true_src[test.index].isin(set(true_src[got.split == 'train'])).sum())})")

    if problems:
        print("GROUP SPLIT CHECK FAILED:\n  - " + "\n  - ".join(problems))
        sys.exit(1)
    print("GROUP SPLIT CHECK PASSED")


if __name__ == "__main__":
    main()
