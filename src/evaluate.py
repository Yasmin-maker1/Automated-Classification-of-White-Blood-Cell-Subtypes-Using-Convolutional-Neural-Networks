"""Evaluate a trained checkpoint on the official TEST folder.

    python src/evaluate.py --checkpoint outputs/<run-name>/best.pt

Run this ONCE per final model. Tuning hyper-parameters against the test set would
invalidate the comparison with Praveen et al. (2021), so use validation numbers
(from train.py) while you experiment.

Writes to outputs/<run-name>/eval/ :
    metrics.json, per_class.csv, classification_report.txt,
    confusion_matrix.png, predictions.csv, misclassified.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from data import build_datasets, make_loader
from model import build_model
from utils import get_device, save_json

PRAVEEN_BASELINE = 0.90  # 90% subtype classification, Praveen et al. (2021)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def plot_confusion(cm, classes, path, title):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "counts"), (axes[1], cm_norm, ".2f", "row-normalised (recall)")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set(xlabel="predicted", ylabel="true", title=f"{title} - {sub}")
        thresh = data.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(paths, y_true, y_pred, conf, classes, out_path, max_n=16):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return 0
    # show the most confident mistakes first: those are the most informative
    wrong = wrong[np.argsort(-conf[wrong])][:max_n]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.set_title(f"true: {classes[y_true[i]]}\npred: {classes[y_pred[i]]} ({conf[i]:.2f})", fontsize=8)
    fig.suptitle("Most confident test errors", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return len(wrong)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None, help="defaults to the one used for training")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default=None, help="defaults to <checkpoint folder>/eval")
    a = p.parse_args(argv)

    device = get_device()
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    classes = ckpt["classes"]
    out_dir = Path(a.out_dir) if a.out_dir else Path(a.checkpoint).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_datasets(a.data_root or targs["data_root"], targs["img_size"], targs["val_frac"], targs["seed"])
    if data["classes"] != classes:
        raise RuntimeError("Class order in the data differs from the checkpoint.")
    test_loader = make_loader(data["test"], a.batch_size, False, a.num_workers, targs["seed"])

    model = build_model(ckpt["model_name"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    probs, y_true = predict_loader(model, test_loader, device)
    y_pred = probs.argmax(1)
    conf = probs.max(1)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    macro_f1 = float(np.mean(f1))
    top2 = float(np.mean([y_true[i] in np.argsort(-probs[i])[:2] for i in range(len(y_true))]))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    # most common confusions (off-diagonal cells), incl. the pair the proposal cares about
    confusions = [(classes[i], classes[j], int(cm[i, j])) for i in range(len(classes))
                  for j in range(len(classes)) if i != j and cm[i, j] > 0]
    confusions.sort(key=lambda t: -t[2])
    mono, lymph = classes.index("monocyte"), classes.index("lymphocyte")
    mono_lymph = int(cm[mono, lymph] + cm[lymph, mono])

    per_class = pd.DataFrame({"class": classes, "precision": prec, "recall": rec, "f1": f1, "support": support})
    per_class.to_csv(out_dir / "per_class.csv", index=False)
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))
    paths = [s[0] for s in data["test"].samples]
    pd.DataFrame({"path": paths, "true": [classes[i] for i in y_true], "pred": [classes[i] for i in y_pred],
                  "confidence": conf, **{f"p_{c}": probs[:, k] for k, c in enumerate(classes)}}
                 ).to_csv(out_dir / "predictions.csv", index=False)
    plot_confusion(cm, classes, out_dir / "confusion_matrix.png", ckpt["model_name"])
    n_err = plot_errors(paths, y_true, y_pred, conf, classes, out_dir / "misclassified.png")

    metrics = {"model": ckpt["model_name"], "checkpoint": str(a.checkpoint), "n_test": int(len(y_true)),
               "accuracy": acc, "macro_f1": macro_f1, "top2_accuracy": top2,
               "monocyte_lymphocyte_confusions": mono_lymph,
               "top_confusions": confusions[:5],
               "baseline_praveen_2021": PRAVEEN_BASELINE, "meets_baseline": bool(acc >= PRAVEEN_BASELINE)}
    save_json(metrics, out_dir / "metrics.json")

    print(f"\nTest accuracy : {acc:.4f}   (baseline to match: {PRAVEEN_BASELINE:.2f} -> "
          f"{'MET' if acc >= PRAVEEN_BASELINE else 'not met'})")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Top-2 accuracy: {top2:.4f}\n")
    print(per_class.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nMonocyte<->lymphocyte confusions: {mono_lymph}")
    print(f"Most common confusions (true -> predicted): {confusions[:3]}")
    print(f"\nSaved to {out_dir}/  ({n_err} errors shown in misclassified.png)")


if __name__ == "__main__":
    main()
