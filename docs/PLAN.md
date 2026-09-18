# Project plan (mapped to the real course calendar)

Written Friday **Sep 18, 2026** (end of Week 5).

## Important: the calendar is tighter than the proposal timeline

The proposal assumes "ten working weeks". The instructor's schedule has **midterm presentations in Weeks 7–9
(Wednesdays Sep 30, Oct 7, Oct 14)** and **final presentations in Weeks 12–14 (Wednesdays Nov 4, 11, 18)**. We do not
yet know which slot we get, so plan for the earliest: **Wed Sep 30 – about 12 days from now.**

| Week | Dates (Tue–Thu) | Wed agenda | Our target |
|---|---|---|---|
| 5 | Sep 15–17 | Module 3 | *(past)* |
| 6 | Sep 22–24 | Module 4 | Repo set up, data downloaded, EDA + leakage check done, first ResNet-50 run started |
| **7** | **Sep 29–Oct 1** | **Midterm presentations (from)**, Module 5 | **Midterm slides ready by Mon Sep 28** |
| 8 | Oct 6–8 | Midterm presentations, Module 6 | Hyper-parameter pass on validation set |
| 9 | Oct 13–15 | Midterm presentations | Last midterm slot; choose final configuration |
| 10 | Oct 20–22 | Guest lecture, Module 7 | Final training runs; paper skeleton |
| 11 | Oct 27–29 | Module 8 | **Single evaluation on the test folder**; confusion matrix, error analysis |
| 12 | Nov 3–5 | Final presentations (from), Module 9 | Final slides ready by Mon Nov 2 (worst case: present Nov 4) |
| 13 | Nov 10–12 | Final presentations, Module 10 | Paper draft complete, peer/instructor read-through |
| 14 | Nov 17–19 | Final presentations | Paper + code repo final |
| — | Nov 30–Dec 5 | FINAL EXAM | — |

**Ask Dr. Cruz:** (1) what the midterm presentation must contain and how long it is, (2) the due date of the written
paper and the code, (3) which presentation slot we have.

## Next 12 days (to the earliest midterm)

| By | Task | Command / output |
|---|---|---|
| Sat Sep 19 | Both: clone repo, copy these files in, `pip install -r requirements.txt`, run `bash scripts/smoke_test.sh` | "SMOKE TEST PASSED" |
| Sun Sep 20 | Download the Kaggle data; run EDA | `python src/eda.py --data-root data` → `outputs/eda/` |
| Mon Sep 21 | Leakage check; **look at** `top_pairs.png` | `python src/leakage_check.py --data-root data` |
| Tue Sep 22 | Start ResNet-50 baseline on a GPU (Colab or cluster) | `train.py --model resnet50 --amp` |
| Wed Sep 23 | Small-CNN baseline; compare validation curves | `train.py --model smallcnn` |
| Thu–Fri Sep 24–25 | Frozen-backbone run (`--freeze-epochs 15`) for the ablation; check curves for over-fitting | `history.csv`, `curves.png` |
| Sat–Sun Sep 26–27 | Build midterm slides | see checklist below |
| Mon Sep 28 | Rehearse; fix gaps | — |

### Suggested split (adjust as you like)
* **Yasmin:** data side – EDA, leakage check, dataset slides, README results table.
* **Matthew:** model side – training runs, baselines, curves, GPU access.
* **Both:** midterm slides, review each other's code, paper writing (split by section).

## Midterm presentation – suggested content (confirm requirements with the instructor)
1. Problem and why it matters (technician fatigue; monocyte vs. lymphocyte example from the proposal).
2. Related work in one slide (Praveen 90 % baseline; why per-class reporting is our angle).
3. Dataset: real class counts vs. Table 1, sample grid, augmentation/leakage finding.
4. Approach: ResNet-50 fine-tuning, split, augmentation, metrics.
5. **Preliminary results on validation data only** (curves, small CNN vs. ResNet-50). Do not show test numbers yet.
6. Risks and remaining plan.

## Rules we set for ourselves
* The official `TEST` folder is touched **once**, for the final model (`evaluate.py`). All tuning uses the validation split.
* The problem statement and dataset do not change (course rule). Implementation details may, and get documented.
* Record every real number that differs from Table 1 of the proposal (image counts, sizes) and state it in the paper.
* Keep every run's `outputs/<run>/config.json`; that is our experiment log. Summarise runs in the README table.

## Final deliverables checklist
- [ ] Code repository (this repo) runs from a clean clone with the README instructions
- [ ] Results table + confusion matrix + per-class precision/recall + error examples
- [ ] Comparison with Praveen et al. (2021), 90 %, with the caveats below
- [ ] Discussion of monocyte ↔ lymphocyte confusion (from Section 1 of the proposal)
- [ ] Limitations: leakage finding, stain/resolution drift, not a clinical tool
- [ ] Conference-style paper (template/length to be confirmed)
- [ ] Final presentation slides

## Issue to investigate early: augmentation leakage
The proposal notes that the ~12,500 images are augmented versions of a much smaller raw set. If the official
TRAIN/TEST split was made *after* augmentation, flipped/rotated copies of the same cell can appear on both sides, which
would push test accuracy above what a new patient's slide would give. `leakage_check.py` searches for this. It is a
heuristic (embedding similarity), so judge by eye. Whatever it shows, report it: a high score with an honest limitation
is better than a high score with an unexplained one. If leakage looks real, discuss with the instructor whether to add a
near-duplicate-aware validation split (allowed: it changes the split method, not the dataset or the problem).
