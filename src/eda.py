"""Exploratory data analysis (Weeks 1-2 in the proposal: 'print a per-folder count').

    python src/eda.py --data-root data

Outputs to outputs/eda/ :
    class_counts.csv / class_counts.png   images per class per split (compare with Table 1)
    sample_grid.png                       4 random images per class
    summary.json                          image sizes, corrupt files, mismatches vs Table 1
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from data import EXPECTED_CLASSES, find_split_dirs
from utils import save_json

IMG_EXT = {".jpg", ".jpeg", ".png"}

# Numbers stated in Table 1 of the proposal
TABLE1 = {
    "TRAIN": {"EOSINOPHIL": 2497, "LYMPHOCYTE": 2483, "MONOCYTE": 2487, "NEUTROPHIL": 2499},
    "TEST": {"EOSINOPHIL": 574, "LYMPHOCYTE": 620, "MONOCYTE": 620, "NEUTROPHIL": 616},
}


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="outputs/eda")
    ap.add_argument("--skip-verify", action="store_true", help="skip the (slow-ish) corrupt-file check")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    random.seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir = find_split_dirs(a.data_root)
    splits = {"TRAIN": train_dir, "TEST": test_dir}
    print(f"TRAIN: {train_dir}\nTEST : {test_dir}\n")

    files: dict[tuple[str, str], list[Path]] = {}
    for split, d in splits.items():
        for cdir in sorted(x for x in d.iterdir() if x.is_dir()):
            files[(split, cdir.name.upper())] = list_images(cdir)

    rows = []
    for (split, cls), fl in files.items():
        expected = TABLE1.get(split, {}).get(cls)
        rows.append({"split": split, "class": cls, "count": len(fl), "table1": expected,
                     "matches_table1": expected == len(fl)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "class_counts.csv", index=False)
    print(df.to_string(index=False))
    tot = df.groupby("split")["count"].sum()
    print(f"\nTotals: {tot.to_dict()}  (proposal / Praveen et al.: 9,966 train + 2,430 test = 12,396)")

    # bar chart
    piv = df.pivot(index="class", columns="split", values="count")
    ax = piv.plot(kind="bar", figsize=(7, 4), rot=20)
    ax.set(title="Images per class", ylabel="count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "class_counts.png", dpi=150)
    plt.close()

    # image sizes and modes (sampled), corrupt-file check (all)
    sizes, modes = Counter(), Counter()
    for fl in files.values():
        for f in random.sample(fl, min(len(fl), 100)):
            with Image.open(f) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
    corrupt = []
    if not a.skip_verify:
        for fl in files.values():
            for f in fl:
                try:
                    with Image.open(f) as im:
                        im.verify()
                except Exception as e:  # noqa: BLE001
                    corrupt.append(f"{f}: {e}")
    print(f"\nImage sizes (sampled, W x H): {dict(sizes.most_common(5))}")
    print(f"Colour modes (sampled): {dict(modes)}")
    print(f"Corrupt files: {len(corrupt)}" + (" (check skipped)" if a.skip_verify else ""))

    # sample grid
    fig, axes = plt.subplots(len(EXPECTED_CLASSES), 4, figsize=(11, 2.8 * len(EXPECTED_CLASSES)))
    for r, cls in enumerate(EXPECTED_CLASSES):
        pool = files.get(("TRAIN", cls), [])
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if pool:
                f = random.choice(pool)
                ax.imshow(Image.open(f).convert("RGB"))
            if c == 0:
                ax.set_title(cls.lower(), loc="left", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "sample_grid.png", dpi=130)
    plt.close()

    save_json({"totals": tot.to_dict(), "image_sizes_sampled": {f"{w}x{h}": n for (w, h), n in sizes.items()},
               "modes_sampled": dict(modes), "corrupt_files": corrupt,
               "all_counts_match_table1": bool(df["matches_table1"].all())}, out / "summary.json")
    print(f"\nSaved to {out}/")
    if not df["matches_table1"].all():
        print("NOTE: some counts differ from Table 1 - record the real numbers in the paper (proposal, Sec. 4).")


if __name__ == "__main__":
    main()
