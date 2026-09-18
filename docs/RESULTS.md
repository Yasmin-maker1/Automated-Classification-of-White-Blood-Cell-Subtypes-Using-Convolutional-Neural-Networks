# Results so far (validation only)

Status on Sep 18, 2026. **The official test folder has not been used.** All numbers below are validation numbers from
one run per model (seed 42, Colab T4 GPU, mixed precision). Figures are in `results/figures/`, per-epoch logs in
`results/logs/`. Trained checkpoints (`best.pt`, ~95 MB for ResNet-50) are **not** in the repo; they are on Google Drive
(`MyDrive/wbc_project/outputs/`).

## Data as loaded (differs slightly from Table 1 of the proposal)

| Class | Train folder | Test folder |
|---|---|---|
| Eosinophil | 2,497 | 623 (proposal: 574) |
| Lymphocyte | 2,483 | 620 |
| Monocyte | 2,478 (proposal: 2,487) | 620 |
| Neutrophil | 2,499 | 624 (proposal: 616) |
| **Total** | **9,957** (proposal: 9,966) | **2,487** (proposal: 2,430) |

All images are 320×240 RGB JPEG; no corrupt files. Split used for training: 8,961 train / 996 validation (stratified 10%
of the train folder) / 2,487 test (sealed). That is 72.0% / 8.0% / 20.0% of 12,444 images.

## Runs

| Run | Command (abridged) | Stopped | Best epoch | Val loss | Val acc | Time |
|---|---|---|---|---|---|---|
| `smallcnn_baseline` | `--model smallcnn --epochs 20 --amp` | epoch 14 (early stopping) | 11 | 0.1656 | 0.9498 | 16.2 min |
| `resnet50_v1` | `--model resnet50 --epochs 15 --freeze-epochs 2 --amp` | epoch 7 (early stopping) | 4 | 0.0066 | 1.0000 | 8.4 min |

Other ResNet-50 numbers: head-only epochs 1–2 reached validation accuracy 0.680 and 0.756 (first two epochs of the same
run, not a separate run). After unfreezing, validation accuracy stayed between 0.990 and 1.000 (0.995 at epoch 7).

## Findings

1. **Transfer learning helps.** Fine-tuned ResNet-50 reaches ~0.996 validation accuracy within three epochs; the
   from-scratch CNN plateaus around 0.91–0.95 and stays noisy.
2. **Validation accuracy is saturated,** so it cannot separate configurations. The test folder must decide, and it is
   used once, after the configuration is frozen.
3. **Possible augmentation overlap between splits** (`results/logs/leakage_summary.json`). Nearest-neighbour cosine
   similarity of ImageNet ResNet-50 features: 25.4% of validation images have a training image at ≥ 0.95, versus 0.5% of
   test images; the highest test similarity is 0.956 and none reaches 0.98. No near-identical pairs were found by eye
   (`leakage_top_pairs.png`), but the measure is not rotation-invariant, so rotated copies are not ruled out. Treat as
   preliminary; validation accuracy may be optimistic.
4. **Images are wide fields, not tight crops.** Each shows many red blood cells and one white cell, sometimes small or
   partly cut off. The proposal describes them as pre-cropped single cells; correct this wording in the paper.

## Differences from the proposal (record in the paper)

- Loaded counts differ from Table 1 (see above).
- Augmentation: horizontal flip, rotation ±10°, mild color jitter (brightness/contrast/saturation 0.1, hue 0.02).
- Optimiser: Adam, head learning rate 1e-3, backbone 1e-4 once unfrozen, weight decay 1e-4, batch size 64; two head-only
  epochs, then full fine-tuning; early stopping on validation loss (patience 3).
- No class weights (classes are balanced).

## Not done yet

- Test-set evaluation (`src/evaluate.py`), once, for the final configuration.
- More seeds; a full frozen-backbone run; a rotation-aware overlap check.
- Per-class precision/recall and confusion matrix (produced by `evaluate.py`).
