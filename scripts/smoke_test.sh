#!/usr/bin/env bash
# End-to-end check on FAKE data (~2-4 min on CPU). Run from the repo root:
#     bash scripts/smoke_test.sh
# If this finishes, your environment and the whole pipeline work.
set -euo pipefail

FAKE=data_fake
OUT=outputs_smoke
IMG=112   # small images so this runs quickly on CPU

python tests/make_fake_data.py --out "$FAKE"
python src/eda.py --data-root "$FAKE" --out-dir "$OUT/eda"

# tiny runs, random init (no internet needed)
python src/train.py --data-root "$FAKE" --model resnet50 --no-pretrained --img-size $IMG \
    --epochs 2 --freeze-epochs 1 --batch-size 16 --num-workers 0 \
    --out-dir "$OUT" --run-name smoke_resnet50
python src/train.py --data-root "$FAKE" --model smallcnn --img-size $IMG \
    --epochs 2 --batch-size 16 --num-workers 0 \
    --out-dir "$OUT" --run-name smoke_smallcnn

# group-aware split by source photo, checked against the known fake sources
# (pixel features only: the ResNet features need ImageNet weights, which a smoke test cannot download)
python src/group_split.py --data-root "$FAKE" --out "$OUT/group_split.json" --fig-dir "$OUT/group_split" \
    --features pixels --img-size $IMG --num-workers 0 --batch-size 32
python tests/check_group_split.py --data-root "$FAKE" --split-file "$OUT/group_split.json"
python src/run_experiments.py --data-root "$FAKE" --out-dir "$OUT/experiments_group" --no-pretrained \
    --img-size $IMG --epochs 2 --freeze-epochs 1 --baseline-epochs 2 --batch-size 16 --num-workers 0 \
    --seeds 1 --finetune-models efficientnet_b0 --frozen-models --no-baseline --split-file "$OUT/group_split.json"
python src/evaluate.py --checkpoint "$OUT/experiments_group/efficientnet_b0_ft_s1/best.pt" --num-workers 0
python src/aggregate_runs.py --out-dir "$OUT/experiments_group"
# mixing splits in one results folder must be refused
if python src/run_experiments.py --data-root "$FAKE" --out-dir "$OUT/experiments_group" --no-pretrained \
    --seeds 1 --finetune-models efficientnet_b0 --frozen-models --no-baseline --dry-run 2>/dev/null; then
    echo "ERROR: run_experiments.py accepted a results folder made with a different split"; exit 1
fi

# the comparison runner: every backbone, fine-tuned + frozen, 2 seeds, tiny settings
python src/run_experiments.py --data-root "$FAKE" --out-dir "$OUT/experiments" --no-pretrained \
    --img-size $IMG --epochs 2 --freeze-epochs 1 --baseline-epochs 2 --batch-size 16 --num-workers 0 \
    --seeds 1 2 --frozen-models resnet50 efficientnet_b0 densenet169
python src/run_experiments.py --data-root "$FAKE" --out-dir "$OUT/experiments" --no-pretrained \
    --img-size $IMG --epochs 2 --freeze-epochs 1 --baseline-epochs 2 --batch-size 16 --num-workers 0 \
    --seeds 1 2 --frozen-models resnet50 efficientnet_b0 densenet169   # second call must skip everything
for m in efficientnet_b0 densenet169; do
    python src/evaluate.py --checkpoint "$OUT/experiments/${m}_ft_s1/best.pt" --num-workers 0
    python src/predict.py --checkpoint "$OUT/experiments/${m}_ft_s1/best.pt" \
        "$FAKE/dataset2-master/images/TEST/MONOCYTE/$(ls "$FAKE/dataset2-master/images/TEST/MONOCYTE" | head -1)"
done
python src/aggregate_runs.py --out-dir "$OUT/experiments"

python src/evaluate.py --checkpoint "$OUT/smoke_resnet50/best.pt" --num-workers 0
python src/evaluate.py --checkpoint "$OUT/smoke_smallcnn/best.pt" --num-workers 0
python src/predict.py --checkpoint "$OUT/smoke_smallcnn/best.pt" \
    "$FAKE/dataset2-master/images/TEST/MONOCYTE/$(ls "$FAKE/dataset2-master/images/TEST/MONOCYTE" | head -1)"
python src/leakage_check.py --data-root "$FAKE" --out-dir "$OUT/leakage" --img-size $IMG \
    --no-pretrained --num-workers 0

echo; echo "SMOKE TEST PASSED. Delete $FAKE and $OUT when you are done."
