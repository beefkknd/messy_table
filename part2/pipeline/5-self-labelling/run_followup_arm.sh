#!/bin/bash
set -eo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1
python=../../engineering/.venv/bin/python
trusted=../../data/yolo/supervised-frames.json
metadata=../data/cohorts/supplemental240.json
case "$1" in
    T) mode=replay; supplemental=();;
    S) mode=single; supplemental=(--metadata "$metadata" --supplement teacher/pseudo1000/frames-single.json);;
    L) mode=ladder; supplemental=(--metadata "$metadata" --supplement teacher/pseudo1000/frames-ladder.json);;
    O) mode=oracle; supplemental=(--metadata "$metadata" --supplement "$2");;
    *) echo "Expected T, S, L, or O" >&2; exit 2;;
esac
arm=$1
mkdir -p runs
"$python" continue_training.py --arm "$mode" --trusted "$trusted" "${supplemental[@]}" \
    --out "runs/train/$arm" --steps 300 > "runs/train-$arm.log" 2>&1
tail -n 1 "runs/train-$arm.log"
for cohort in calibration30 development100 original100; do
    "$python" run_prediction.py --manifest "../../data/cohorts/$cohort.json" \
        --weights "runs/train/$arm/checkpoint.pt" --out "runs/inference/$arm/$cohort" \
        --imgsz 768 --conf .001 --max-det 300 --head one2many --device mps \
        > "runs/inference-$arm-$cohort.log" 2>&1
    tail -n 1 "runs/inference-$arm-$cohort.log"
done
