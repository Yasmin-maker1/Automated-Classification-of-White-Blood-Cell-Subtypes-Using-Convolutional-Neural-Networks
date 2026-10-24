"""Error analysis on the TEST folder, after evaluate.py: per-photo results and Grad-CAM examples.

    python src/error_analysis.py --exp-dir outputs/experiments --split-file splits/group_split.json \
        --runs smallcnn_scratch_s42 densenet169_ft_s42

Needs <exp-dir>/<run>/eval/predictions.csv (written by evaluate.py). Uses no new test-set decisions: it only
re-reads predictions that were already made, plus a few forward/backward passes for the heatmaps.

Why per photo: the ~2,500 test images come from only ~113 original photos (group_split.py), and a model tends to get
all rotated copies of one photo right or all of them wrong. So the honest unit of evidence is the photo, not the image.

Writes to <exp-dir>/<run>/eval/:
    per_photo.csv            one row per (class, source photo): copies, share correct
    gradcam_examples.png     top: image, bottom: Grad-CAM overlay; 1 correct example per class, then the most
                             confident errors, each from a DIFFERENT photo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from predictor import Predictor


def test_key(path: str) -> str:
    """The split file's key for a test image: TEST/<CLASS FOLDER>/<file name>."""
    p = Path(path)
    return f"TEST/{p.parent.name}/{p.name}"


def per_photo_table(pred: pd.DataFrame, spec: dict) -> pd.DataFrame:
    pred = pred.assign(photo=pred.path.map(lambda p: spec[test_key(p)]["source"]), ok=pred.true == pred.pred)
    return (pred.groupby(["true", "photo"]).agg(copies=("ok", "size"), share_correct=("ok", "mean"))
            .reset_index().rename(columns={"true": "class"}))


def photo_summary(per_photo: pd.DataFrame) -> pd.DataFrame:
    g = per_photo.groupby("class")
    return pd.DataFrame({"test photos": g.size(),
                         "all copies right": g.share_correct.apply(lambda s: int((s == 1).sum())),
                         "all copies wrong": g.share_correct.apply(lambda s: int((s == 0).sum())),
                         "mixed": g.share_correct.apply(lambda s: int(((s > 0) & (s < 1)).sum()))})


def gradcam_grid(run_dir: Path, pred: pd.DataFrame, spec: dict, n_errors: int, out_path: Path):
    pred = pred.assign(photo=pred.path.map(lambda p: spec[test_key(p)]["source"]), ok=pred.true == pred.pred)
    right = pred[pred.ok].groupby("true").head(1)
    wrong = pred[~pred.ok].sort_values("confidence", ascending=False).drop_duplicates("photo").head(n_errors)
    rows = pd.concat([right, wrong])
    predictor = Predictor(str(run_dir / "best.pt"))
    fig, axes = plt.subplots(2, len(rows), figsize=(2.6 * len(rows), 5.6), squeeze=False)
    for j, r in enumerate(rows.itertuples()):
        img = Image.open(r.path).convert("RGB")
        out = predictor.predict(img)
        axes[0, j].imshow(img)
        axes[1, j].imshow(out["overlay"])
        axes[0, j].set_title(f"true: {r.true}\npred: {out['pred']} ({out['confidence']:.2f})", fontsize=8,
                             color="green" if out["pred"] == r.true else "red")
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM, {run_dir.name}: {len(right)} correct (one per class), "
                 f"{len(wrong)} most confident errors (different photos)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-errors", type=int, default=4)
    ap.add_argument("--no-gradcam", action="store_true")
    a = ap.parse_args(argv)

    spec = json.loads(Path(a.split_file).read_text())["images"]
    for run in a.runs:
        run_dir = Path(a.exp_dir) / run
        pred = pd.read_csv(run_dir / "eval" / "predictions.csv")
        per_photo = per_photo_table(pred, spec)
        per_photo.to_csv(run_dir / "eval" / "per_photo.csv", index=False)
        print(f"===== {run}")
        print(photo_summary(per_photo).to_string())
        if not a.no_gradcam:
            out = run_dir / "eval" / "gradcam_examples.png"
            gradcam_grid(run_dir, pred, spec, a.n_errors, out)
            print(f"Grad-CAM figure: {out}")
        print()


if __name__ == "__main__":
    main()
