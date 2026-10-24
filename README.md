# Automated Classification of White Blood Cell Subtypes Using CNNs

CSCI 7090 – Advanced Computer Vision and Deep Learning (Georgia Southern University, Fall 2026)
Authors: Yasmin Rocio Orduz Landazabal, Matthew Ragsdale · Instructor: Dr. Meenalosini Vimal Cruz

Four-class classifier (eosinophil, lymphocyte, monocyte, neutrophil) for single microscope images from the Kaggle
*Blood Cell Images* dataset (Mooney, 2018). The final version compares a small CNN trained from scratch with three
ImageNet-pretrained networks (ResNet-50, EfficientNet-B0, DenseNet-169) under one training recipe, 3 seeds each, on a
**leak-free split by source photo**, and reports per-class precision/recall, macro-F1, confusion matrices, error
analysis and Grad-CAM on the held-out test folder.

**Full walkthrough with code, tables and figures: [`docs/walkthrough/WALKTHROUGH.md`](docs/walkthrough/WALKTHROUGH.md).**
Result tables only: [`docs/RESULTS.md`](docs/RESULTS.md).

## Main results

The 12,444 Kaggle images are rotated/flipped copies of 366 original photos. With the random split used at the
midterm, **99.9%** of validation images came from a photo that was also in training, which is why validation accuracy
reached 100%. `src/group_split.py` matches every image back to its original photo and splits by photo: **0.0%**
overlap. It also found that 6.8% of the official test images share a photo with training, so every test result is
also reported on the **clean subset** (2,205 test images whose photo was never in training).

| Model (3 seeds, mean ± std) | Params | Val accuracy | Test macro-F1 | Clean-test macro-F1 |
|---|---|---|---|---|
| **SmallCNN, from scratch (selected by validation loss)** | 0.4 M | 0.917 ± 0.008 | 0.865 ± 0.015 | 0.851 ± 0.017 |
| DenseNet-169, fine-tuned | 12.5 M | 0.907 ± 0.005 | 0.870 ± 0.011 | 0.856 ± 0.013 |
| EfficientNet-B0, fine-tuned | 4.0 M | 0.900 ± 0.016 | 0.861 ± 0.010 | 0.846 ± 0.011 |
| ResNet-50, fine-tuned | 23.5 M | 0.913 ± 0.012 | 0.847 ± 0.011 | 0.836 ± 0.011 |
| ResNet-50, frozen backbone | 23.5 M | 0.713 ± 0.004 | 0.583 ± 0.009 | 0.579 ± 0.010 |
| SmallCNN, augmentation off (ablation) | 0.4 M | 0.859 ± 0.014 | 0.827 ± 0.053 | 0.812 ± 0.057 |
| SmallCNN, pad to square (ablation) | 0.4 M | 0.938 ± 0.014 | 0.789 ± 0.061 | 0.780 ± 0.060 |

* Real performance is ~85–87% macro-F1, not ~100%. The top four models are within one standard deviation of each other.
* Frozen ImageNet features are not enough; augmentation helps; padding instead of squashing hurts.
* Main confusions: neutrophil ↔ eosinophil and monocyte → neutrophil. Errors come in whole photos (all rotated copies of
  a photo right or all wrong), and the test set holds only ~113 independent photos.
* Praveen et al. (2021) report 90% on this dataset with a random split; not reached on a leak-free split.

## Repository layout

```
src/
  data.py              finds TRAIN/TEST, transforms (resize: squash/pad/crop, augmentation on/off), split from file
  group_split.py       matches every image to its original photo (rotation/flip-aware) -> group split file
  model.py             SmallCNN, ResNet-50, EfficientNet-B0, DenseNet-169; freeze/unfreeze the backbone
  train.py             training loop, two-stage fine-tuning, early stopping, checkpoint, curves
  run_experiments.py   the comparison grid (models x seeds, frozen run, ablations); skips finished runs
  aggregate_runs.py    mean ± std per configuration -> summary.csv / summary.md
  evaluate.py          test evaluation (run once per model): metrics, clean-subset metrics, confusion matrix, errors
  error_analysis.py    per-photo results and Grad-CAM example grids
  gradcam.py           Grad-CAM;  predictor.py: load a checkpoint and classify one image;  predict.py: CLI version
  eda.py               dataset counts, image sizes, sample grid
  leakage_check.py     the midterm near-duplicate check (not rotation-aware; kept for the before/after comparison)
  utils.py             seeds, device, JSON helper
notebooks/
  colab_quickstart.ipynb   run everything on a free Colab T4 (split, 21 training runs, evaluation, analysis)
  final_defense.ipynb      the 10-minute code walkthrough: shows the code from src/ and the saved results
  demo_colab.ipynb         start the live demo in Colab
demo/                  live web demo (app.py) and example/showcase preparation (prepare_demo.py)
tests/                 fake dataset with known source photos + a check that group_split.py recovers them
scripts/smoke_test.sh  end-to-end test of the whole pipeline on fake data (CPU, ~6 min, no download)
docs/                  WALKTHROUGH.md (code + results), RESULTS.md (tables), PLAN.md (original schedule)
results/               midterm figures and logs (random split, superseded; kept for the before/after story)
proposal/              project proposal (LaTeX + PDF)
Midterm_Presentation.* midterm slides (PDF and PowerPoint)
```

Data and trained checkpoints are not in the repo (see `.gitignore`); the checkpoints and all run outputs are on the
project Google Drive (`MyDrive/wbc_project/`).

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/smoke_test.sh                              # ~6 min on CPU; proves the pipeline works
```

Training was done on Google Colab (T4 GPU): open `notebooks/colab_quickstart.ipynb` from GitHub in Colab and run the
cells top to bottom.

## 2. Getting the data

Dataset: <https://www.kaggle.com/datasets/paultimothymooney/blood-cells> (MIT license).

```bash
pip install kaggle          # needs an API token: Kaggle -> Settings -> Create New Token -> ~/.kaggle/kaggle.json
kaggle datasets download -d paultimothymooney/blood-cells -p data --unzip
```

The code searches under `--data-root` for `TRAIN/` and `TEST/` folders that contain the four class folders, and for
the original photos in `dataset-master/dataset-master/JPEGImages` (used by `group_split.py`). `TEST_SIMPLE/` is ignored.

## 3. Workflow (final)

```bash
# a) Split by source photo (once; ~3 min on a T4)
python src/group_split.py --data-root data --out splits/group_split.json --min-margin 0.02 --val-frac 0.2

# b) Model comparison: 3 backbones fine-tuned + ResNet-50 frozen + SmallCNN, seeds 42/43/44 (15 runs, ~2.5 h on a T4)
python src/run_experiments.py --data-root data --out-dir outputs/experiments --split-file splits/group_split.json --amp

# c) Ablations: SmallCNN without augmentation, and with pad-to-square (6 runs)
python src/run_experiments.py --data-root data --out-dir outputs/experiments --split-file splits/group_split.json --amp --only-ablations

# d) Test evaluation, once per saved model (model selection uses validation only)
python src/evaluate.py --checkpoint outputs/experiments/<run>/best.pt
python src/aggregate_runs.py --out-dir outputs/experiments

# e) Error analysis and Grad-CAM
python src/error_analysis.py --exp-dir outputs/experiments --split-file splits/group_split.json \
    --runs smallcnn_scratch_s42 densenet169_ft_s42
```

Rules we kept: the test folder is never used for any decision; every configuration uses the same recipe and split;
the selection rule (lowest mean validation loss) was fixed before looking at test results.

## 4. Live demo

Upload or pick a cell image and see the predicted type, the class probabilities and a Grad-CAM heatmap. The page
header shows that checkpoint's test results. Research prototype, not a clinical tool.

```bash
pip install -r requirements-demo.txt
python demo/prepare_demo.py --data-root data --checkpoint outputs/experiments/smallcnn_scratch_s42/best.pt \
    --split-file splits/group_split.json     # 2 clean test images per class + results/figures/demo_showcase.png
python demo/app.py --checkpoint outputs/experiments/smallcnn_scratch_s42/best.pt   # add --share for a public link
```

Easiest in class: open `notebooks/demo_colab.ipynb` in Colab (the checkpoint and split file are loaded from Google
Drive). `results/figures/demo_showcase.png` is a static backup in case the live demo fails.

## Limitations

* Validation holds few photos per class (4 lymphocyte, 6 monocyte), so it is ~5 points more optimistic than test.
* ± is seed-to-seed spread on one split; with ~113 independent test photos, per-class numbers are less certain.
* 783 training-folder images whose source photo could not be identified with confidence were left out.
* One public dataset; not a clinical tool.

## References

Mooney (2018), Blood Cell Images, Kaggle · Praveen et al. (2021), arXiv:2108.04614 · He et al. (2016), ResNet ·
Tan & Le (2019), EfficientNet · Huang et al. (2017), DenseNet · Selvaraju et al. (2017), Grad-CAM. Full list in the
proposal and the midterm slides.
