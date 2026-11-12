"""Verify the continuation experiment and apply its frozen detector operating rules."""
import argparse
import subprocess
import sys

import numpy as np

from common import A_CHECKPOINT, HERE, PREVIOUS, PROGRAM, sha256, write_json
from build_stronger_teacher import read

RUNS = HERE / "runs"
ARMS = ("T", "S", "L", "O")
METRICS = {
    "both_view_coverage": ("shared_localized_both", "shared_objects"),
    "localization_recall": ("localized", "ground_truth"),
    "localization_precision": ("localized", "detections"),
    "unmapped_per_image": ("unmapped", "images"),
}


def verify():
    protocol_sha = sha256(PROGRAM / "PROTOCOL.json")
    audit = read(HERE / "audits/replay-cpu/run-plan.json")
    results = {}
    schedules = {}
    for arm in ARMS:
        run = RUNS / "train" / arm
        plan = read(run / "run-plan.json")
        performance = read(run / "training.json")
        if plan["protocol_sha256"] != protocol_sha or plan["audit_only"] or plan["restoration_device"] != "mps":
            raise ValueError(f"Wrong run protocol/device for {arm}")
        for key in ("incoming_checkpoint_sha256", "incoming_model_state_sha256", "incoming_optimizer_state_sha256",
                    "previous_optimizer_steps", "optimizer_groups", "optimizer_state_layout"):
            if plan[key] != audit[key]:
                raise ValueError(f"Restored {key} differs on {arm}")
        if plan["incoming_frozen_blocks_sha256"] != performance["outgoing_frozen_blocks_sha256"]:
            raise ValueError(f"Frozen blocks changed on {arm}")
        if performance["checkpoint_sha256"] != sha256(run / "checkpoint.pt"):
            raise ValueError(f"New checkpoint hash mismatch on {arm}")
        if performance["completed_optimizer_steps"] != 300 or performance["total_optimizer_steps"] != 600:
            raise ValueError(f"Wrong step budget for {arm}")
        if [step["step"] for step in performance["per_step"]] != list(range(301, 601)):
            raise ValueError(f"Noncontinuous steps for {arm}")
        slots = [batch[-1] for batch in plan["schedule"]]
        if plan["supplemental_offset"] != (300 if arm == "T" else 0):
            raise ValueError(f"Wrong supplemental stream offset for {arm}")
        if arm != "T" and len({row["row_id"] for row in slots[:240]}) != 240:
            raise ValueError(f"Not all frozen supplemental frames were trained in {arm}")
        for cohort in ("calibration30", "development100", "original100"):
            inference = read(RUNS / "inference" / arm / cohort / "inference-plan.json")
            if (
                inference["weights_sha256"] != performance["checkpoint_sha256"] or inference["imgsz"] != 768
                or inference["head"] != "one2many" or inference["nms_iou"] != 0.7
                or inference["max_det"] != 300 or inference["confidence_floor"] != 0.001
            ):
                raise ValueError(f"Wrong detector inference configuration for {arm}/{cohort}")
        schedules[arm] = plan["schedule"]
        results[arm] = {
            "checkpoint_sha256": performance["checkpoint_sha256"],
            "train_plan_sha256": sha256(run / "run-plan.json"),
            "training_sha256": sha256(run / "training.json"),
            "unique_supplemental_frames": len({row["row_id"] for row in slots}),
            "supplemental_positive_exposures": sum(row["boxes"] for row in slots),
            "empty_supplemental_exposures": sum(not row["boxes"] for row in slots),
            "wall_seconds": performance["wall_seconds"],
            "peak_sampled_mps_driver_bytes": performance["peak_sampled_mps_driver_bytes"],
        }
    for t, s, l, o in zip(*(schedules[arm] for arm in ARMS), strict=True):
        if not t[:3] == s[:3] == l[:3] == o[:3] or not all(len(batch) == 4 for batch in (t, s, l, o)):
            raise ValueError("Common trusted schedule differs")
        if any(not s[-1][key] == l[-1][key] == o[-1][key] for key in ("row_id", "image", "setup")):
            raise ValueError("Supplemental row schedules differ")
        if any(row["scope"] != scope for row, scope in (
            (s[-1], "partial_frame"), (l[-1], "partial_frame"), (o[-1], "complete_frame")
        )):
            raise ValueError("Expected oracle complete and pseudo partial training scopes")
    write_json(RUNS / "verified-training.json", {
        "protocol_sha256": protocol_sha,
        "initial_checkpoint_sha256": sha256(A_CHECKPOINT),
        "initial_model_state_sha256": audit["incoming_model_state_sha256"],
        "initial_optimizer_state_sha256": audit["incoming_optimizer_state_sha256"],
        "frozen_block_digests_verified": True,
        "common_trusted_and_matched_supplemental_schedules_verified": True,
        "arms": results,
    })


def score(arm, cohort, thresholds):
    destination = RUNS / "scores" / cohort / f"{arm}.json"
    subprocess.run([
        sys.executable, str(HERE / "run_scoring.py"),
        "--predictions", str(RUNS / "inference" / arm / cohort),
        "--manifest", str(PREVIOUS.parent / "data/cohorts" / f"{cohort}.json"),
        "--truth", str(PREVIOUS.parent / "data/scoring" / f"{cohort}.json"),
        "--out", str(destination), "--thresholds", *map(str, thresholds),
    ], check=True)
    return read(destination)["thresholds"]


def paired(candidate, control, seed):
    left, right = candidate["setup_counts"], control["setup_counts"]
    if left.keys() != right.keys():
        raise ValueError("Mismatched setup groups")
    groups = sorted(left)
    draw = np.random.default_rng(seed).integers(len(groups), size=(20000, len(groups)))
    result = {}
    for metric, (numerator, denominator) in METRICS.items():
        arrs = [
            np.array([[tally[group].get(numerator, 0), tally[group].get(denominator, 0)] for group in groups])
            for tally in (left, right)
        ]
        a, b = (arr[draw].sum(axis=1) for arr in arrs)
        valid = (a[:, 1] > 0) & (b[:, 1] > 0)
        diff = a[valid, 0] / a[valid, 1] - b[valid, 0] / b[valid, 1]
        point_a, point_b = candidate["metrics"][metric], control["metrics"][metric]
        result[metric] = {
            "delta": point_a - point_b if point_a is not None and point_b is not None else None,
            "paired_setup_bootstrap_ci95": np.quantile(diff, [0.025, 0.975]).tolist() if len(diff) else None,
        }
    return {"setup_groups": len(groups), "bootstrap_draws": len(draw), "effects": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    verification = RUNS / "verified-training.json"
    if args.verify_only or not verification.exists():
        verify()
    if args.verify_only:
        return
    protocol = read(PROGRAM / "PROTOCOL.json")
    amendment = read(PROGRAM / "AMENDMENT_01.json")
    evaluation = protocol["evaluation"]
    fixed = evaluation["fixed_detection_cutoff"]
    grid = evaluation["calibration_grid"]
    cal = {arm: score(arm, "calibration30", [fixed] if arm == "T" else grid) for arm in ARMS}
    ceilings = {key: cal["T"][str(fixed)]["counts"][key] for key in ("detections", "unmapped")}
    selected = {"T": fixed, "A_parent": fixed}
    for arm in "SLO":
        selected[arm] = next(
            threshold for threshold in sorted(grid)
            if all(cal[arm][str(threshold)]["counts"][key] <= limit for key, limit in ceilings.items())
        )
    calibration_path = RUNS / "calibrated-thresholds.json"
    write_json(calibration_path, {
        "thresholds": selected, "reference": "T at 0.25", "ceilings": ceilings,
        "chosen_calibration_counts": {
            arm: {key: cal[arm][str(selected[arm])]["counts"][key] for key in ceilings} for arm in ARMS
        },
        "calibration_score_sha256": {arm: sha256(RUNS / "scores/calibration30" / f"{arm}.json") for arm in ARMS},
    })
    results = {}
    comparisons = (("T", "A_parent"), ("S", "T"), ("L", "T"), ("O", "T"), ("L", "S"), ("O", "L"))
    for cohort in evaluation["cohorts"]:
        scored = dict(cal) if cohort == "calibration30" else {
            arm: score(arm, cohort, sorted({fixed, selected[arm]})) for arm in ARMS
        }
        scored["A_parent"] = read(PREVIOUS / "round2/scores" / cohort / "A.json")["thresholds"]
        results[cohort] = {}
        for regime, limits in (("fixed", dict.fromkeys(scored, fixed)), ("calibrated", selected)):
            observations = {arm: scores[str(limits[arm])] for arm, scores in scored.items()}
            results[cohort][regime] = {
                "arms": {
                    arm: {"threshold": limits[arm], "counts": obs["counts"], "metrics": obs["metrics"]}
                    for arm, obs in observations.items()
                },
                "paired_differences": {
                    f"{candidate}_minus_{control}": paired(
                        observations[candidate], observations[control], amendment["bootstrap_seed"]
                    )
                    for candidate, control in comparisons
                },
            }
    write_json(RUNS / "scoring-summary.json", {
        "protocol_sha256": sha256(PROGRAM / "PROTOCOL.json"),
        "amendment_sha256": sha256(PROGRAM / "AMENDMENT_01.json"),
        "calibrated_thresholds_sha256": sha256(calibration_path),
        "cohorts": results,
    })


if __name__ == "__main__":
    main()
