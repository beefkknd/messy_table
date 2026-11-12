"""Score frozen single-pass outputs with the existing MessyTable box mapper."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from common import ROOT, initialize_runtime, read_text, sha256, write_json

initialize_runtime()
sys.path.insert(0, str(ROOT / "evaluation"))
import numpy as np
from scipy.optimize import linear_sum_assignment
from run_full_scene import map_detections

SIZE_KEYS = ("small", "medium", "large")


def xyxy_to_xywh(box):
    left, top, right, bottom = box
    return [left, top, right - left, bottom - top]


def localization_diagnostics(detections, truth, mapped_predictions):
    predicted = np.array([row["box_xyxy"] for row in detections], dtype=float).reshape(-1, 4)
    targets = np.array(list(truth.values()), dtype=float).reshape(-1, 4)
    targets[:, 2:] += targets[:, :2]
    overlap = np.maximum(
        np.minimum(predicted[:, None, 2:], targets[None, :, 2:])
        - np.maximum(predicted[:, None, :2], targets[None, :, :2]),
        0,
    ).prod(axis=2)
    target_area = (targets[:, 2:] - targets[:, :2]).prod(axis=1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).prod(axis=1)
    union = predicted_area[:, None] + target_area[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    rows, cols = linear_sum_assignment(np.where(iou >= 0.75, iou, 0), maximize=True)
    unmapped = sorted(set(range(len(detections))) - mapped_predictions)
    duplicate_or_conflict = (iou[unmapped] >= 0.50).any(axis=1)
    coverage = np.divide(overlap, target_area[None, :], out=np.zeros_like(overlap), where=target_area[None, :] > 0)
    return {
        "localized_at_iou75": int((iou[rows, cols] >= 0.75).sum()),
        "unmapped_duplicate_or_assignment_conflict": int(duplicate_or_conflict.sum()),
        "group_box_candidates": int(((coverage >= 0.80).sum(axis=1) >= 2).sum()),
    }


def view_score(raw, truth_view, cohort_view, threshold):
    if raw["sha256"] != cohort_view["sha256"]:
        raise ValueError("Predictions use a different source image from the frozen cohort")
    detections = [d for d in raw["detections"] if d["confidence"] > threshold]
    truth = truth_view.get("objects_xywh_historical")
    if truth is None:
        truth = {oid: xyxy_to_xywh(box) for oid, box in truth_view["objects_xyxy"].items()}
    mapping = map_detections(detections, truth)
    matched = set(mapping.values())
    count = Counter(ground_truth=len(truth), detections=len(detections), localized=len(matched))
    count["unmapped"] = len(detections) - len(matched)
    count.update(localization_diagnostics(detections, truth, set(mapping)))
    image_area = cohort_view["width"] * cohort_view["height"]
    for oid, (_, _, width, height) in truth.items():
        area = width * height / image_area
        bucket = "small" if area < 0.01 else "medium" if area < 0.05 else "large"
        count[f"ground_truth_{bucket}"] += 1
        count[f"localized_{bucket}"] += int(oid) in matched
    return dict(count), matched, set(map(int, truth))


def score_at_threshold(predictions, truths, cohorts, threshold):
    cases = []
    total = Counter()
    setup_counts = {}
    for pred in predictions:
        case_id = pred["id"]
        cohort = cohorts[case_id]
        truth = truths[case_id]
        setup = pred["setup"]
        if setup != cohort["setup"] or setup != truth["setup"]:
            raise ValueError(f"Inconsistent setup for {case_id}")
        views = {view["id"]: view for view in cohort["views"]}
        if set(pred["views"]) != set(truth["views"]) or set(pred["views"]) != set(views):
            raise ValueError(f"Inconsistent views for {case_id}")
        if set(views) != {"A", "B"}:
            raise ValueError(f"Expected the primary A/B pair for {case_id}")
        tally = Counter(pairs=1, images=2)
        matched_by_view, annotated_by_view, view_counts = {}, {}, {}
        for view_id, raw in pred["views"].items():
            counts, matched, annotated = view_score(raw, truth["views"][view_id], views[view_id], threshold)
            tally.update(counts)
            matched_by_view[view_id] = matched
            annotated_by_view[view_id] = annotated
            view_counts[view_id] = {**counts, "localized_ids": sorted(matched)}
        shared = annotated_by_view["A"] & annotated_by_view["B"]
        if "shared_object_ids" in truth and set(map(int, truth["shared_object_ids"])) != shared:
            raise ValueError(f"Truth shared identities disagree with per-view labels in {case_id}")
        both = shared & matched_by_view["A"] & matched_by_view["B"]
        one = shared & (matched_by_view["A"] ^ matched_by_view["B"])
        tally.update(
            shared_objects=len(shared),
            shared_localized_both=len(both),
            shared_localized_one=len(one),
            shared_localized_neither=len(shared) - len(both) - len(one),
        )
        total.update(tally)
        setup_counts.setdefault(setup, Counter()).update(tally)
        cases.append({
            "id": case_id, "setup": setup, "counts": dict(tally), "views": view_counts,
            "shared_ids_localized_both": sorted(both), "shared_ids_localized_one": sorted(one),
        })
    return {"counts": dict(total), "metrics": ratios(total), "setup_counts": setup_counts, "cases": cases}


def ratios(total):
    def ratio(a, b):
        return total[a] / total[b] if total[b] else None

    return {
        "localization_precision": ratio("localized", "detections"),
        "localization_recall": ratio("localized", "ground_truth"),
        "localization_recall_at_iou75": ratio("localized_at_iou75", "ground_truth"),
        "both_view_coverage": ratio("shared_localized_both", "shared_objects"),
        "unmapped_per_image": ratio("unmapped", "images"),
        "recall_by_area_fraction": {key: ratio(f"localized_{key}", f"ground_truth_{key}") for key in SIZE_KEYS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction run directory")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    args = parser.parse_args()
    freeze = json.loads(read_text(args.predictions / "freeze.json"))
    if set(freeze) != {"inference-plan.json", "predictions.json"}:
        raise ValueError("Incomplete prediction freeze")
    for name, expected in freeze.items():
        if sha256(args.predictions / name) != expected:
            raise ValueError(f"Frozen prediction artifact changed: {name}")
    plan = json.loads(read_text(args.predictions / "inference-plan.json"))
    if plan["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("Scoring manifest differs from the prediction run")
    if any(t < plan["confidence_floor"] or t > 1 for t in args.thresholds):
        raise ValueError("Scoring thresholds must lie between the detector confidence floor and 1")
    predictions = json.loads(read_text(args.predictions / "predictions.json"))["scenes"]
    truth_document = json.loads(read_text(args.truth))
    if truth_document.get("cohort_sha256") != plan["manifest_sha256"]:
        raise ValueError("Truth belongs to a different frozen cohort")
    truth_rows = truth_document["scenes"]
    cohort_rows = json.loads(read_text(args.manifest))["scenes"]
    truths = {row["id"]: row for row in truth_rows}
    cohorts = {row["id"]: row for row in cohort_rows}
    if not (len(predictions) == len(truths) == len(cohorts) == len(cohort_rows) == len(truth_rows)):
        raise ValueError("Incomplete or duplicate cohort/truth rows")
    if {row["id"] for row in predictions} != set(truths) or set(truths) != set(cohorts):
        raise ValueError("Predictions, truth, and cohort IDs differ")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, {
        "prediction_sha256": freeze["predictions.json"],
        "truth_sha256": sha256(args.truth),
        "mapping_sha256": sha256(ROOT / "evaluation/run_full_scene.py"),
        "scoring_code_sha256": sha256(__file__),
        "thresholds": {
            str(threshold): score_at_threshold(predictions, truths, cohorts, threshold)
            for threshold in args.thresholds
        },
    })


if __name__ == "__main__":
    main()
