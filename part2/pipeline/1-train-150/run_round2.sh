#!/bin/bash
set -eo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1
python=.venv/bin/python
weights=../../../everyday-identity/assets/models/yolo26m.pt
trusted=../data/yolo/supervised-frames.json
supp_b=teacher/matched-pseudo180/frames-B-matched.json
supp_c=teacher/pseudo180/frames-C.json
supp_d=teacher/pseudo180/frames-D.json
mkdir -p round2
"$python" - "$trusted" "$supp_b" "$supp_c" "$supp_d" <<'PY'
import json
import sys
from pathlib import Path
from common import sha256, write_json

files = [Path(name) for name in sys.argv[1:]]
teacher = Path("teacher/pseudo180")
teacher_freeze = json.loads((teacher / "freeze.json").read_text())
for name, expected in teacher_freeze.items():
    if sha256(teacher / name) != expected:
        raise ValueError(f"Teacher artifact changed: {name}")
matched_freeze = json.loads(Path("teacher/matched-pseudo180/freeze.json").read_text())
if matched_freeze["matched_b_sha256"] != sha256(files[1]):
    raise ValueError("Matched B manifest changed")
if matched_freeze["upstream_teacher_freeze_sha256"] != sha256(teacher / "freeze.json"):
    raise ValueError("Teacher freeze changed after the crop slot match")
data = Path("../data")
data_freeze = json.loads((data / "freeze.json").read_text())
for name, expected in data_freeze.items():
    if sha256(data / name) != expected:
        raise ValueError(f"Data artifact changed: {name}")
write_json("round2/freeze-plan.json", {
    "arms": {"A": "trusted-only replay", "B": "matched crop slots with original positives",
             "C": "matched crop slots with original and supplemental positives",
             "D": "same C positives and ordered slots on full RGB frames"},
    "manifests_sha256": {str(file): sha256(file) for file in files},
    "code_sha256": {name: sha256(name) for name in (
        "run_round2.sh", "train_detector.py", "predict_detector.py", "score_detector.py", "partial_loss.py",
    )},
    "seed": 923, "optimizer_steps": 300, "batch": 4, "supplement_slots": 1, "train_imgsz": 768,
    "train_frozen_layers": 10, "lr": 0.0003, "device": "mps", "train_head_mix": [0.8, 0.2],
    "predict_imgsz": 768, "predict_head": "one2many", "nms_iou": 0.7, "max_det": 300,
    "fixed_confidence": 0.25, "prediction_floor": 0.001,
    "calibration_reference": "A at fixed confidence .25",
    "calibration_rule": "Most permissive grid threshold satisfying BOTH reference prediction and unmapped count ceilings",
    "confidence_grid": [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
                        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                        0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995, 1.00],
    "cohort_manifests_sha256": {
        name: sha256(data / "cohorts" / f"{name}.json")
        for name in ("calibration30", "development100", "original100")
    },
})
PY
for arm in A B C D; do
    supplemental=()
    case "$arm" in
        B) supplemental=(--supplement "$supp_b");;
        C) supplemental=(--supplement "$supp_c");;
        D) supplemental=(--supplement "$supp_d");;
    esac
    echo "Training arm $arm"
    "$python" train_detector.py --trusted "$trusted" "${supplemental[@]}" \
        --out "round2/train/$arm" --steps 300 --batch 4 --supplement-slots 1 \
        --imgsz 768 --device mps --freeze-layers 10 --lr .0003 --seed 923 --inference-head one2one \
        > "round2/train-$arm.log" 2>&1
    tail -n 1 "round2/train-$arm.log"
done
for arm in A B C D; do
    for cohort in calibration30 development100 original100; do
        echo "Predicting $cohort with arm $arm"
        "$python" predict_detector.py --weights "round2/train/$arm/checkpoint.pt" \
            --manifest "../data/cohorts/$cohort.json" --out "round2/inference/$arm/$cohort" \
            --imgsz 768 --head one2many --device mps --conf .001 --max-det 300 \
            > "round2/predict-$arm-$cohort.log" 2>&1
        tail -n 1 "round2/predict-$arm-$cohort.log"
    done
done
