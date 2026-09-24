# Automated Classification of White Blood Cell Subtypes Using CNNs

CSCI 7090 – Advanced Computer Vision and Deep Learning (Georgia Southern University, Fall 2026)
Authors: Yasmin Rocio Orduz Landazabal, Matthew Ragsdale · Instructor: Dr. Meenalosini Vimal Cruz

Four-class classifier (eosinophil, lymphocyte, monocyte, neutrophil) built by fine-tuning an
ImageNet-pretrained **ResNet-50** on the Kaggle *Blood Cell Images* dataset (Mooney, 2018).
Target from the proposal: **≥ 90 % test accuracy** (Praveen et al., 2021 report 90 %), reported
together with per-class precision/recall, macro-F1 and a confusion matrix. The full proposal is in
[`proposal/`](proposal/proposal.pdf); the schedule is in [`docs/PLAN.md`](docs/PLAN.md).

## Repository layout

```
proposal/            proposal (LaTeX + PDF)
docs/PLAN.md         schedule mapped to the course calendar, task split, checklists
docs/RESULTS.md      results so far (validation only), findings, differences from the proposal
results/             figures and per-epoch logs from the Colab runs (checkpoints are NOT in the repo)
src/
  data.py            find TRAIN/TEST folders, transforms, stratified 90/10 train/val split
  model.py           ResNet-50 (transfer learning) + small from-scratch CNN baseline
  train.py           training loop, early stopping, checkpoints, curves
  evaluate.py        FINAL test-set evaluation: metrics, confusion matrix, error gallery
  predict.py         classify single images (the "demo")
  eda.py             class counts vs Table 1, image sizes, corrupt files, sample grid
  leakage_check.py   near-duplicate search between splits (augmentation leakage)
  gradcam.py         Grad-CAM heatmaps
  predictor.py       load a checkpoint and classify one image (used by the demo)
  utils.py
tests/make_fake_data.py   synthetic dataset in the Kaggle layout (for testing only)
scripts/smoke_test.sh     end-to-end check on fake data, no download needed
notebooks/colab_quickstart.ipynb   run everything on a free Colab GPU
notebooks/demo_colab.ipynb         start the live demo in Colab
demo/                web demo (app.py) and example/showcase preparation (prepare_demo.py)
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/smoke_test.sh                              # ~3 min on CPU; proves the pipeline works
```

`torch` picks CPU/GPU automatically. For an NVIDIA GPU, install the CUDA build of PyTorch from
<https://pytorch.org/get-started/locally/> first if `torch.cuda.is_available()` is `False`.

## 2. Getting the data

Dataset: <https://www.kaggle.com/datasets/paultimothymooney/blood-cells> (MIT license).

```bash
pip install kaggle          # needs an API token: Kaggle -> Settings -> Create New Token -> ~/.kaggle/kaggle.json
kaggle datasets download -d paultimothymooney/blood-cells -p data --unzip
```

Or download the zip in the browser and unzip it into `data/`. The code searches under `--data-root`
for `TRAIN/` and `TEST/` folders that contain the four class folders, so the exact nesting does not
matter (normally `data/dataset2-master/dataset2-master/images/{TRAIN,TEST}/...`). `TEST_SIMPLE/` and
the 410-image raw set are ignored, as stated in the proposal.

## 3. Workflow

```bash
# a) Look at the data (do this first; compares real counts with Table 1 of the proposal)
python src/eda.py --data-root data

# b) Check for augmentation leakage between splits, then LOOK at outputs/leakage/*.png
python src/leakage_check.py --data-root data

# c) Weak from-scratch baseline (shows whether transfer learning helps)
python src/train.py --data-root data --model smallcnn --epochs 20 --run-name smallcnn_baseline

# d) Main model: 2 epochs head-only, then whole network at 10x smaller LR for the backbone
python src/train.py --data-root data --model resnet50 --epochs 15 --freeze-epochs 2 --amp --run-name resnet50_v1

# e) FINAL model only -- touches the sealed official test folder
python src/evaluate.py --checkpoint outputs/resnet50_v1/best.pt

# f) Demo
python src/predict.py --checkpoint outputs/resnet50_v1/best.pt path/to/cell.jpeg
```

Tips
* CUDA out of memory → `--batch-size 32`. No GPU → expect hours per run; use `--freeze-epochs 5 --epochs 5`
  (frozen backbone) or run on Colab / the university cluster.
* Compare experiments by their **validation** numbers (`outputs/<run>/history.csv`, `curves.png`).
  Run `evaluate.py` on the test folder only for the final chosen configuration.
* Runs are seeded (`--seed 42`); GPU non-determinism can still cause tiny differences.

## 4. Results (validation only so far)

The official test folder has **not** been used yet; details, figures and caveats are in
[`docs/RESULTS.md`](docs/RESULTS.md) and `results/`.

| Model | Val acc (best epoch) | Val loss | Test acc | Macro-F1 | Notes |
|---|---|---|---|---|---|
| Small CNN (scratch) | 0.950 (ep. 11) | 0.166 | not run | not run | stopped at epoch 14, 16.2 min |
| ResNet-50, backbone frozen | 0.756 (ep. 2) | 0.752 | not run | not run | first two epochs of the fine-tuning run |
| ResNet-50, fine-tuned | 1.000 (ep. 4) | 0.0066 | not run | not run | 0.995 at epoch 7; stopped by early stopping, 8.4 min |
| Praveen et al. (2021) | – | – | 0.90 | – | published baseline, YOLOv3 pipeline (test-set number) |

Validation accuracy is saturated near 100% and validation may be optimistic (possible augmentation overlap between
splits), so the single test-set evaluation will decide the final result.

## 5. Live demo

Upload or pick a cell image and see the predicted type, the class probabilities and a Grad-CAM heatmap showing where
the network looks. Research prototype, not a clinical tool.

```bash
pip install -r requirements-demo.txt
python demo/prepare_demo.py --data-root data --checkpoint outputs/resnet50_v1/best.pt   # examples + static showcase figure
python demo/app.py --checkpoint outputs/resnet50_v1/best.pt                              # add --share for a public link
```

Easiest on presentation day: open `notebooks/demo_colab.ipynb` in Colab (the trained checkpoint is loaded from Google
Drive). The examples come from the validation split; the test folder is not used. `results/figures/demo_showcase.png`
is a static backup of the same examples in case the live demo fails.
