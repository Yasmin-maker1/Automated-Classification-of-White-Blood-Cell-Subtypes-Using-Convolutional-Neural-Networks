"""Train the white blood cell classifier.

Examples
--------
# Main model (ResNet-50, ImageNet weights, 2 epochs head-only then full fine-tune)
python src/train.py --data-root data --model resnet50 --epochs 15 --freeze-epochs 2

# Weak from-scratch baseline
python src/train.py --data-root data --model smallcnn --epochs 20 --run-name smallcnn_baseline

Everything is written to outputs/<run-name>/ :
    best.pt          checkpoint with the lowest validation loss
    history.csv      per-epoch loss / accuracy
    curves.png       training curves
    config.json      arguments, dataset sizes, environment
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import build_datasets, make_loader
from model import (build_model, count_parameters, freeze_batchnorm,
                   set_backbone_trainable, split_param_groups)
from utils import get_device, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data", help="folder that contains the unzipped Kaggle dataset")
    p.add_argument("--model", default="resnet50", choices=["resnet50", "smallcnn"])
    p.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--freeze-epochs", type=int, default=2,
                   help="epochs at the start where only the new FC head is trained (resnet50 only)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate for the classification head")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                   help="backbone lr = lr * this, once unfrozen")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="none", choices=["none", "plateau"])
    p.add_argument("--patience", type=int, default=3, help="early stopping patience on validation loss")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only, faster + less memory)")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              use_amp=False, freeze_bn=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        freeze_batchnorm(model)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def plot_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    ax[1].plot(epochs, [h["val_acc"] for h in history], marker="o", label="validation")
    ax[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device()
    use_amp = args.amp and device.type == "cuda"
    if args.model == "smallcnn":
        args.freeze_epochs = 0

    run_name = args.run_name or f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data = build_datasets(args.data_root, args.img_size, args.val_frac, args.seed)
    classes = data["classes"]
    train_loader = make_loader(data["train"], args.batch_size, True, args.num_workers, args.seed)
    val_loader = make_loader(data["val"], args.batch_size, False, args.num_workers, args.seed)
    print(f"device={device}  amp={use_amp}  classes={classes}")
    print(f"train={len(data['train'])}  val={len(data['val'])}  test(not touched)={len(data['test'])}")

    model = build_model(args.model, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        split_param_groups(model, args.lr, args.backbone_lr_mult), weight_decay=args.weight_decay
    )
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
                 if args.scheduler == "plateau" else None)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    save_json(
        {"args": vars(args), "device": str(device), "torch": torch.__version__,
         "n_train": len(data["train"]), "n_val": len(data["val"]), "n_test": len(data["test"]),
         "classes": classes},
        run_dir / "config.json",
    )

    history, best_val, bad_epochs = [], float("inf"), 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        if epoch == 1 or epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, not frozen)
            print(f"-- epoch {epoch}: backbone {'FROZEN' if frozen else 'TRAINABLE'} "
                  f"({count_parameters(model)['trainable']:,} trainable params)")

        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp, frozen)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        if scheduler:
            scheduler.step(va_loss)

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": time.time() - t0})
        print(f"epoch {epoch:02d}/{args.epochs}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}  ({time.time() - t0:.0f}s)")

        if va_loss < best_val:
            best_val, bad_epochs = va_loss, 0
            torch.save({"model_state": model.state_dict(), "model_name": args.model, "classes": classes,
                        "args": vars(args), "epoch": epoch, "val_loss": va_loss, "val_acc": va_acc},
                       run_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: validation loss has not improved for {args.patience} epochs.")
                break

    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    plot_curves(history, run_dir / "curves.png")
    best = min(history, key=lambda h: h["val_loss"])
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Best epoch {best['epoch']}: "
          f"val loss {best['val_loss']:.4f}, val acc {best['val_acc']:.4f}")
    print(f"Saved to {run_dir}/   ->  next: python src/evaluate.py --checkpoint {run_dir / 'best.pt'}")
    return run_dir


if __name__ == "__main__":
    main()
