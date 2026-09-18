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

python src/evaluate.py --checkpoint "$OUT/smoke_resnet50/best.pt" --num-workers 0
python src/evaluate.py --checkpoint "$OUT/smoke_smallcnn/best.pt" --num-workers 0
python src/predict.py --checkpoint "$OUT/smoke_smallcnn/best.pt" \
    "$FAKE/dataset2-master/images/TEST/MONOCYTE/$(ls "$FAKE/dataset2-master/images/TEST/MONOCYTE" | head -1)"
python src/leakage_check.py --data-root "$FAKE" --out-dir "$OUT/leakage" --img-size $IMG \
    --no-pretrained --num-workers 0

echo; echo "SMOKE TEST PASSED. Delete $FAKE and $OUT when you are done."
