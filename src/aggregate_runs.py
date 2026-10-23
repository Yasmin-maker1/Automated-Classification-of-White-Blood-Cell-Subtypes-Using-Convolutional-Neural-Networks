"""Summarise many runs as mean ± std over seeds, one row per configuration.

    python src/aggregate_runs.py --out-dir outputs/experiments

Reads every <out-dir>/<run>/config.json + history.csv (+ eval/metrics.json if evaluate.py was run).
Validation numbers come from the checkpoint epoch (lowest validation loss), which is what
evaluate.py tests. Test columns stay empty until evaluate.py has been run on those checkpoints.

Writes <out-dir>/summary_runs.csv (one row per run) and summary.csv / summary.md (per configuration).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def config_name(args: dict) -> str:
    if args["model"] == "smallcnn":
        return "smallcnn (scratch)"
    mode = "frozen" if args["freeze_epochs"] >= args["epochs"] else "fine-tuned"
    return f"{args['model']} ({mode})"


def load_runs(out_dir: Path) -> pd.DataFrame:
    rows = []
    for cfg_path in sorted(out_dir.glob("*/config.json")):
        run_dir = cfg_path.parent
        if not (run_dir / "history.csv").exists():
            continue  # unfinished
        cfg = json.loads(cfg_path.read_text())
        hist = pd.read_csv(run_dir / "history.csv")
        best = hist.loc[hist["val_loss"].idxmin()]
        row = {"run": run_dir.name, "config": config_name(cfg["args"]), "seed": cfg["args"]["seed"],
               "params_M": cfg.get("params", {}).get("total", float("nan")) / 1e6,
               "best_epoch": int(best["epoch"]), "epochs_run": len(hist),
               "val_loss": best["val_loss"], "val_acc": best["val_acc"],
               "minutes": hist["seconds"].sum() / 60}
        m_path = run_dir / "eval" / "metrics.json"
        if m_path.exists():
            m = json.loads(m_path.read_text())
            row.update(test_acc=m["accuracy"], test_macro_f1=m["macro_f1"])
        rows.append(row)
    return pd.DataFrame(rows)


def mean_std(s: pd.Series, digits: int) -> str:
    s = s.dropna()
    if s.empty:
        return ""
    if len(s) == 1:
        return f"{s.iloc[0]:.{digits}f}"
    return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="outputs/experiments")
    a = p.parse_args(argv)
    out_dir = Path(a.out_dir)

    runs = load_runs(out_dir)
    if runs.empty:
        raise SystemExit(f"No finished runs found in {out_dir}/")
    for col in ("test_acc", "test_macro_f1"):
        if col not in runs:
            runs[col] = float("nan")
    runs.sort_values(["config", "seed"]).to_csv(out_dir / "summary_runs.csv", index=False)

    summary = []
    for cfg, g in runs.groupby("config", sort=False):
        summary.append({"configuration": cfg, "seeds": len(g), "params (M)": f"{g['params_M'].iloc[0]:.1f}",
                        "val loss": mean_std(g["val_loss"], 4), "val acc": mean_std(g["val_acc"], 4),
                        "test acc": mean_std(g["test_acc"], 4), "test macro-F1": mean_std(g["test_macro_f1"], 4),
                        "best epoch": mean_std(g["best_epoch"].astype(float), 1),
                        "min / run": mean_std(g["minutes"], 1)})
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "summary.csv", index=False)

    header = "| " + " | ".join(summary.columns) + " |"
    lines = [header, "|" + "---|" * len(summary.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in summary.itertuples(index=False)]
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(md + "\n")
    print(md)
    print(f"\nmean ± sample std over seeds. Saved summary.csv, summary.md, summary_runs.csv in {out_dir}/")


if __name__ == "__main__":
    main()
