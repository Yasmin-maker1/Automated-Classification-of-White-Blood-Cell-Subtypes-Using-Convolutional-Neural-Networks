# Results (final, group-aware split)

Status on Oct 24, 2026. Colab T4, mixed precision, 3 seeds per configuration (42, 43, 44), mean ± sample std.
Run outputs (checkpoints, curves, eval folders) are on Google Drive: `MyDrive/wbc_project/experiments_groupsplit/`.
The midterm numbers (random split) are kept at the end for comparison.

## Split

`src/group_split.py` matches each of the 12,444 images to the original photo it was augmented from (366 photos in
`dataset-master/JPEGImages`, 144 rotated/flipped views each), then splits train/val by photo (StratifiedGroupKFold,
20% val). Matches with margin < 0.02 over the second-best photo are left out (label agreement drops to 55–81% below
that, versus 97.5–100% above).

| | Images |
|---|---|
| Train / validation / excluded (unsure) | 7,436 / 1,738 / 783 |
| Test folder | 2,487 (113 photos) |
| Test images whose photo is also in train | 170 (6.8%, 37 photos) |
| Clean test subset (photo not in train, match sure) | 2,205 |
| Validation images sharing a photo with train | 0.0% (99.9% with the midterm random split) |

## Model comparison (same recipe for every model)

Selection rule fixed in advance: lowest mean validation loss. Every model was evaluated once on the test folder.

| Configuration | Params (M) | Val loss | Val acc | Test macro-F1 | Clean-test macro-F1 |
|---|---|---|---|---|---|
| SmallCNN, from scratch (selected) | 0.4 | 0.225 ± 0.011 | 0.917 ± 0.008 | 0.865 ± 0.015 | 0.851 ± 0.017 |
| DenseNet-169, fine-tuned | 12.5 | 0.275 ± 0.025 | 0.907 ± 0.005 | 0.870 ± 0.011 | 0.856 ± 0.013 |
| EfficientNet-B0, fine-tuned | 4.0 | 0.262 ± 0.030 | 0.900 ± 0.016 | 0.861 ± 0.010 | 0.846 ± 0.011 |
| ResNet-50, fine-tuned | 23.5 | 0.293 ± 0.016 | 0.913 ± 0.012 | 0.847 ± 0.011 | 0.836 ± 0.011 |
| ResNet-50, frozen backbone | 23.5 | 0.654 ± 0.006 | 0.713 ± 0.004 | 0.583 ± 0.009 | 0.579 ± 0.010 |

Ablations (SmallCNN, same recipe, 3 seeds; only one setting changed):

| Configuration | Val loss | Val acc | Test macro-F1 | Clean-test macro-F1 |
|---|---|---|---|---|
| Baseline: augmentation on, squash resize | 0.225 ± 0.011 | 0.917 ± 0.008 | 0.865 ± 0.015 | 0.851 ± 0.017 |
| Augmentation off | 0.395 ± 0.116 | 0.859 ± 0.014 | 0.827 ± 0.053 | 0.812 ± 0.057 |
| Pad to square | 0.277 ± 0.069 | 0.938 ± 0.014 | 0.789 ± 0.061 | 0.780 ± 0.060 |

Both ablations are worse and less stable than the baseline, so augmentation and the squash resize stay. Pad-to-square
has the best validation accuracy but the worst test score: validation accuracy alone is not a reliable guide here.

## Findings

1. Without the leak, performance is ~85% macro-F1, not 99–100%. The midterm "transfer learning wins" result came from
   the leaky validation split.
2. The four non-frozen models are within one standard deviation of each other on test.
3. Frozen ImageNet features are not enough (58% macro-F1): the network has to adapt to these images.
4. Main confusions: neutrophil ↔ eosinophil (both granulocytes) and monocyte → neutrophil. Lymphocytes are near 100%.
5. Errors come in whole photos: per test photo, both models get all copies right or all wrong for every monocyte photo
   (13 right, 3 wrong). The test set has ~113 independent photos, so per-class numbers are less certain than the ±.
6. Grad-CAM: SmallCNN attends to the white cell even when wrong; DenseNet's attention stays central when the white cell
   sits near the image edge.
7. Praveen et al. (2021) report 90% on this dataset with a random split; not reached on a leak-free split.

## Midterm numbers (random split, 1 seed) — superseded

| Run | Best epoch | Val loss | Val acc |
|---|---|---|---|
| `smallcnn_baseline` | 11 | 0.1656 | 0.9498 |
| `resnet50_v1` | 4 | 0.0066 | 1.0000 |
