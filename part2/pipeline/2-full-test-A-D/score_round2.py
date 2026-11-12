"""Apply frozen calibration rules, then report paired detection effects by setup."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import HERE, PROGRAM, read_text, sha256, write_json

ARMS = ("pretrained768", "A", "B", "C", "D")
METRICS = {
    "both_view_coverage": ("shared_localized_both", "shared_objects"),
    "localization_recall": ("localized", "ground_truth"),
    "localization_precision": ("localized", "detections"),
    "unmapped_per_image": ("unmapped", "images"),
}


def read(path):
    return json.loads(read_text(path))


def score_cohort(root, arm, cohort, thresholds):
    predictions = HERE / "inference" / arm / cohort if arm == "pretrained768" else root / "inference" / arm / cohort
    output = root / "scores" / cohort / f"{arm}.json"
    command = [
        sys.executable, str(HERE / "score_detector.py"),
        "--predictions", str(predictions),
        "--manifest", str(PROGRAM / "data/cohorts" / f"{cohort}.json"),
        "--truth", str(PROGRAM / "data/scoring" / f"{cohort}.json"),
        "--out", str(output),
        "--thresholds", *[str(t) for t in thresholds],
    ]
    subprocess.run(command, check=True)
    return read(output)["thresholds"]


def calibrate(root, plan):
    grid = plan["confidence_grid"]
    fixed = plan["fixed_confidence"]
    result = {
        arm: score_cohort(root, arm, "calibration30", grid if arm in "BCD" else [fixed])
        for arm in ARMS
    }
    reference = result["A"][str(fixed)]["counts"]
    ceilings = {key: reference[key] for key in ("detections", "unmapped")}
    selected = {"A": fixed, "pretrained768": fixed}
    for arm in "BCD":
        selected[arm] = next(
            threshold for threshold in sorted(grid)
            if all(result[arm][str(threshold)]["counts"][key] <= limit for key, limit in ceilings.items())
        )
    write_json(root / "calibrated-thresholds.json", {
        "thresholds": selected,
        "reference": "A at 0.25",
        "ceilings": ceilings,
        "selected_calibration_counts": {
            arm: {key: result[arm][str(threshold)]["counts"][key] for key in ceilings}
            for arm, threshold in selected.items()
        },
        "calibration_scores_sha256": {
            arm: sha256(root / "scores/calibration30" / f"{arm}.json") for arm in ARMS
        },
        "frozen_plan_sha256": sha256(root / "freeze-plan.json"),
    })
    print(json.dumps({"calibrated": selected, "ceilings": ceilings}, indent=2))


def paired_delta(candidate, reference):
    left, right = candidate["setup_counts"], reference["setup_counts"]
    if left.keys() != right.keys():
        raise ValueError("Setup IDs differ across a paired contrast")
    group_ids = sorted(left)
    index = np.random.default_rng(5110).integers(len(group_ids), size=(10000, len(group_ids)))
    effects = {}
    for name, (num, den) in METRICS.items():
        arrays = [
            np.array([[counts[group].get(num, 0), counts[group].get(den, 0)] for group in group_ids])
            for counts in (left, right)
        ]
        bootstrap = [array[index].sum(axis=1) for array in arrays]
        valid = (bootstrap[0][:, 1] > 0) & (bootstrap[1][:, 1] > 0)
        samples = bootstrap[0][valid, 0] / bootstrap[0][valid, 1] - bootstrap[1][valid, 0] / bootstrap[1][valid, 1]
        observed = (
            candidate["metrics"][name] - reference["metrics"][name]
            if candidate["metrics"][name] is not None and reference["metrics"][name] is not None
            else None
        )
        effects[name] = {
            "delta": observed,
            "paired_setup_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist() if samples.size else None,
        }
    return {"setup_groups": len(group_ids), "effects": effects}


def finish(root, plan):
    selection = read(root / "calibrated-thresholds.json")
    if selection["frozen_plan_sha256"] != sha256(root / "freeze-plan.json"):
        raise ValueError("Calibration selection used a different freeze plan")
    for arm, expected in selection["calibration_scores_sha256"].items():
        if sha256(root / "scores/calibration30" / f"{arm}.json") != expected:
            raise ValueError("Calibration scores changed after threshold selection")
    fixed = plan["fixed_confidence"]
    selected = selection["thresholds"]
    output = {}
    contrasts = [(arm, "pretrained768") for arm in "ABCD"] + [
        ("B", "A"), ("C", "A"), ("C", "B"), ("C", "D"),
    ]
    for cohort in ("calibration30", "development100", "original100"):
        scores = {
            arm: (
                read(root / "scores/calibration30" / f"{arm}.json")["thresholds"]
                if cohort == "calibration30"
                else score_cohort(root, arm, cohort, sorted({fixed, selected[arm]}))
            )
            for arm in ARMS
        }
        output[cohort] = {}
        for label, thresholds in (
            ("fixed", dict.fromkeys(ARMS, fixed)), ("calibrated_against_trusted", selected)
        ):
            at_threshold = {arm: scores[arm][str(thresholds[arm])] for arm in ARMS}
            output[cohort][label] = {
                "arms": {
                    arm: {
                        "threshold": thresholds[arm], "counts": score["counts"], "metrics": score["metrics"],
                    }
                    for arm, score in at_threshold.items()
                },
                "paired_differences": {
                    f"{candidate}_minus_{reference}": paired_delta(at_threshold[candidate], at_threshold[reference])
                    for candidate, reference in contrasts
                },
            }
    write_json(root / "scoring-summary.json", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibration", "finish"))
    args = parser.parse_args()
    root = HERE / "round2"
    plan = read(root / "freeze-plan.json")
    if sha256(HERE / "score_detector.py") != plan["code_sha256"]["score_detector.py"]:
        raise ValueError("Detector scorer changed after freeze")
    if args.phase == "calibration":
        calibrate(root, plan)
    else:
        finish(root, plan)


if __name__ == "__main__":
    main()
