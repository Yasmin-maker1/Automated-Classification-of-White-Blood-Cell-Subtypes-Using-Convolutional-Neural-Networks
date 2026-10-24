"""Run the model comparison: every backbone under the SAME recipe, several seeds each.

    python src/run_experiments.py --data-root data --amp --num-workers 2

Default grid (3 seeds each -> 15 runs):
    resnet50, efficientnet_b0, densenet169   fine-tuned  (2 head-only epochs, then everything)
    resnet50                                 frozen backbone (head only for all epochs)
    smallcnn                                 from-scratch baseline

Only the backbone (and, for the frozen run, --freeze-epochs) changes between runs. The train/val
split is fixed by --split-seed, so every run sees exactly the same images. --seed changes only
weight init of the new head, data order and augmentation.

Runs that already finished (history.csv exists) are skipped, so after a Colab disconnect just run
the same command again. Point --out-dir at Google Drive to keep results through a disconnect.

Run names: <model>_<ft|frozen|scratch>_s<seed>, e.g. densenet169_ft_s43.
Afterwards: python src/aggregate_runs.py --out-dir <same out dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="outputs/experiments")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="42 matches the existing midterm runs")
    p.add_argument("--finetune-models", nargs="*", default=["resnet50", "efficientnet_b0", "densenet169"])
    p.add_argument("--frozen-models", nargs="*", default=["resnet50"],
                   help="models that also get a frozen-backbone run")
    p.add_argument("--no-baseline", action="store_true", help="skip the from-scratch smallcnn runs")
    p.add_argument("--split-file", default=None, help="group split from src/group_split.py (recommended)")
    # the shared recipe (same values as the midterm resnet50_v1 run)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--baseline-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true", help="only for the smoke test (no internet)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    return p.parse_args(argv)


def build_grid(a):
    """List of (run_name, model, mode, seed, epochs, freeze_epochs)."""
    grid = []
    for seed in a.seeds:
        for m in a.finetune_models:
            grid.append((f"{m}_ft_s{seed}", m, "ft", seed, a.epochs, a.freeze_epochs))
        for m in a.frozen_models:
            grid.append((f"{m}_frozen_s{seed}", m, "frozen", seed, a.epochs, a.epochs))
        if not a.no_baseline:
            grid.append((f"smallcnn_scratch_s{seed}", "smallcnn", "scratch", seed, a.baseline_epochs, 0))
    return grid


def main(argv=None):
    a = parse_args(argv)
    out_dir = Path(a.out_dir)
    grid = build_grid(a)
    print(f"{len(grid)} runs -> {out_dir}/")
    failed = []
    for i, (name, model, mode, seed, epochs, freeze) in enumerate(grid, 1):
        if (out_dir / name / "history.csv").exists():
            done_split = json.loads((out_dir / name / "config.json").read_text())["args"].get("split_file")
            if done_split != a.split_file:
                sys.exit(f"{name} in {out_dir} was trained with split_file={done_split}, but this call uses "
                         f"{a.split_file}. Use a different --out-dir so the two splits are not mixed.")
            print(f"[{i}/{len(grid)}] {name}: already done, skipping")
            continue
        cmd = [sys.executable, str(SRC / "train.py"),
               "--data-root", a.data_root, "--out-dir", str(out_dir), "--run-name", name,
               "--model", model, "--seed", str(seed), "--split-seed", str(a.split_seed),
               "--epochs", str(epochs), "--freeze-epochs", str(freeze),
               "--batch-size", str(a.batch_size), "--lr", str(a.lr),
               "--backbone-lr-mult", str(a.backbone_lr_mult), "--patience", str(a.patience),
               "--img-size", str(a.img_size), "--num-workers", str(a.num_workers)]
        if a.amp:
            cmd.append("--amp")
        if a.no_pretrained:
            cmd.append("--no-pretrained")
        if a.split_file:
            cmd += ["--split-file", a.split_file]
        print(f"\n[{i}/{len(grid)}] {name}\n  " + " ".join(cmd[1:]))
        if a.dry_run:
            continue
        t0 = time.time()
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
            print(f"  !! {name} failed, continuing with the next run")
        else:
            print(f"  {name} finished in {(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"\nFailed runs (rerun the same command to retry): {failed}")
        sys.exit(1)
    print(f"\nAll runs done. Next: python src/aggregate_runs.py --out-dir {out_dir}")


if __name__ == "__main__":
    main()
