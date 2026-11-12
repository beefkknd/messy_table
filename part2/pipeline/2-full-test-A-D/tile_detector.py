"""Add fixed overlapping native-pixel tiles to frozen A full-frame predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

from common import A_CHECKPOINT, HERE, PREVIOUS, PROGRAM, initialize_runtime, local_path, pin_ultralytics, read_text, sha256, write_json

initialize_runtime()
sys.path.append(str(PREVIOUS))

import cv2
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

from build_teacher import iou
from predict_detector import predict_one

COHORTS = ("calibration30", "development100", "original100")
TILE = 768
STRIDE = 576
EDGE_MARGIN = 4
OVERLAP_IOU = 0.55
CONFIDENCE = 0.25
RAW_FULL_FRAME_FLOOR = 0.001
MAX_DET = 300
REPLAY_PAIRS = 3
REPLAY_CONFIDENCE_ATOL = 1e-6


def read(path):
    return json.loads(read_text(path))


def axis_starts(length):
    if length <= 0:
        raise ValueError("Image dimensions must be positive")
    stop = max(length - TILE, 0)
    starts = list(range(0, stop + 1, STRIDE))
    if starts[-1] != stop:
        starts.append(stop)
    return starts


def windows(width, height):
    return [
        (left, top, min(left + TILE, width), min(top + TILE, height))
        for top in axis_starts(height) for left in axis_starts(width)
    ]


def project(det, tile, width, height):
    left, top, right, bottom = tile
    x0, y0, x1, y1 = det["box_xyxy"]
    tw, th = right - left, bottom - top
    clipped = [min(tw, max(0, x0)), min(th, max(0, y0)), min(tw, max(0, x1)), min(th, max(0, y1))]
    u0, v0, u1, v1 = clipped
    mapped = {**det, "box_xyxy": [left + u0, top + v0, left + u1, top + v1]}
    if u0 >= u1 or v0 >= v1:
        return mapped, "invalid"
    at_artificial_edge = (
        (left > 0 and u0 <= EDGE_MARGIN)
        or (top > 0 and v0 <= EDGE_MARGIN)
        or (right < width and tw - u1 <= EDGE_MARGIN)
        or (bottom < height and th - v1 <= EDGE_MARGIN)
    )
    return mapped, "artificial_edge" if at_artificial_edge else None


def merge(baseline, tiles):
    retained = list(baseline)
    usable = [
        (tile_index, detection_index, row)
        for tile_index, tile in enumerate(tiles)
        for detection_index, row in enumerate(tile["detections"])
        if row["decision"] is None
    ]
    usable.sort(key=lambda entry: (-entry[2]["mapped"]["confidence"], entry[0], entry[1]))
    for _, _, row in usable:
        det = row["mapped"]
        matching = next(
            (index for index, prev in enumerate(retained) if iou(det["box_xyxy"], prev["box_xyxy"]) >= OVERLAP_IOU),
            None,
        )
        if matching is None:
            row["decision"] = "added"
            row["retained_index"] = len(retained)
            retained.append(det)
        else:
            row["decision"] = "duplicate_full" if matching < len(baseline) else "duplicate_tile"
            row["retained_index"] = matching
    return retained


def frozen_full_frame(cohort, source_sha):
    source = PREVIOUS / "round2/inference/A" / cohort
    manifest_path = PREVIOUS.parent / "data/cohorts" / f"{cohort}.json"
    freeze = read(source / "freeze.json")
    if set(freeze) != {"inference-plan.json", "predictions.json"}:
        raise ValueError(f"Unexpected cached prediction freeze for {cohort}")
    if any(sha256(source / name) != digest for name, digest in freeze.items()):
        raise ValueError(f"Cached full-frame predictions changed for {cohort}")
    plan = read(source / "inference-plan.json")
    required = {
        "manifest_sha256": sha256(manifest_path), "weights_sha256": source_sha,
        "imgsz": TILE, "head": "one2many", "nms_iou": 0.7,
        "confidence_floor": RAW_FULL_FRAME_FLOOR, "max_det": MAX_DET,
        "device": "mps", "torch": str(torch.__version__), "ultralytics": ultralytics_version,
    }
    if any(plan.get(key) != value for key, value in required.items()):
        raise ValueError(f"Cached full-frame controls differ for {cohort}")
    scenes = read(manifest_path)["scenes"]
    cached = read(source / "predictions.json")["scenes"]
    for scene, old in zip(scenes, cached, strict=True):
        if scene["id"] != old["id"] or scene["setup"] != old["setup"]:
            raise ValueError(f"Cached scene mismatch for {cohort}")
        if {view["id"]: view["sha256"] for view in scene["views"]} != {
            key: view["sha256"] for key, view in old["views"].items()
        }:
            raise ValueError(f"Cached source view mismatch for {cohort}/{scene['id']}")
    return manifest_path, scenes, cached, sha256(source / "freeze.json")


def selected(rows):
    return [row for row in rows if row["confidence"] > CONFIDENCE]


def source_image(view):
    path = local_path(view["path"])
    if sha256(path) != view["sha256"]:
        raise ValueError(f"Source image differs from frozen bytes: {path}")
    return path


def replay(model, inputs):
    records = []
    for cohort, (_, scenes, cached, _) in inputs.items():
        for scene, old in zip(scenes[:REPLAY_PAIRS], cached[:REPLAY_PAIRS], strict=True):
            for view in scene["views"]:
                path = source_image(view)
                started = time.monotonic()
                actual = predict_one(model, path, TILE, RAW_FULL_FRAME_FLOOR, "mps", MAX_DET, "one2many")
                elapsed = time.monotonic() - started
                expected = old["views"][view["id"]]["detections"]
                geometry = [(det["box_xyxy"], det["class"]) for det in actual] == [
                    (det["box_xyxy"], det["class"]) for det in expected
                ]
                drift = max(
                    [abs(a["confidence"] - b["confidence"]) for a, b in zip(actual, expected, strict=True)],
                    default=0.0,
                ) if geometry else None
                records.append({
                    "cohort": cohort, "scene": scene["id"], "view": view["id"], "image_sha256": view["sha256"],
                    "expected_count": len(expected), "actual_count": len(actual),
                    "expected_above_cutoff": len(selected(expected)), "actual_above_cutoff": len(selected(actual)),
                    "same_class_box_order": geometry, "max_abs_confidence_drift": drift, "seconds": elapsed,
                    "passed": geometry and drift <= REPLAY_CONFIDENCE_ATOL,
                })
    return records


def run_view(model, view, full):
    path = source_image(view)
    began = time.monotonic()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    height, width = view["height"], view["width"]
    if image is None or image.shape[:2] != (height, width):
        raise ValueError(f"Native dimensions differ from frozen manifest: {path}")
    tiles = []
    for left, top, right, bottom in windows(width, height):
        region = left, top, right, bottom
        pixels = image[top:bottom, left:right]
        tile_start = time.monotonic()
        predictions = selected(predict_one(model, pixels, TILE, CONFIDENCE, "mps", MAX_DET, "one2many"))
        elapsed = time.monotonic() - tile_start
        detections = []
        for det in predictions:
            mapped, decision = project(det, region, width, height)
            detections.append({"local": det, "mapped": mapped, "decision": decision})
        tiles.append({"xyxy": region, "seconds": elapsed, "detections": detections})
    baseline = selected(full["detections"])
    merged = merge(baseline, tiles)
    pipeline = time.monotonic() - began
    tally = Counter(row["decision"] for tile in tiles for row in tile["detections"])
    return {
        "sha256": view["sha256"],
        "detections": merged,
        "seconds": pipeline + full["seconds"],
        "cached_full_frame_seconds": full["seconds"],
        "tile_seconds": sum(tile["seconds"] for tile in tiles),
        "tile_pipeline_seconds": pipeline,
        "tile_passes": len(tiles),
        "added_detections": len(merged) - len(baseline),
        "tile_decisions": dict(tally),
    }, tiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    if sha256(args.protocol) != args.protocol_sha256:
        raise ValueError("Frozen protocol digest mismatch")
    protocol = read(args.protocol)
    pin_ultralytics()
    if not torch.backends.mps.is_available():
        raise RuntimeError("Expected MPS")
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    source_sha = sha256(A_CHECKPOINT)
    if source_sha != read(A_CHECKPOINT.parent / "training.json")["checkpoint_sha256"] or source_sha != protocol["weights_sha256"]:
        raise ValueError("Frozen supervised A checkpoint digest mismatch")
    inputs = {cohort: frozen_full_frame(cohort, source_sha) for cohort in COHORTS}
    detector = {
        "full_frame_resolution": TILE, "tile_inference_resolution": TILE, "confidence_cutoff_exclusive": CONFIDENCE,
        "head": "one2many", "nms_iou": 0.7, "max_per_pass": MAX_DET,
    }
    tiling = {
        "native_tile_side": TILE, "stride": STRIDE, "interior_edge_margin_px": EDGE_MARGIN,
        "merge_iou": OVERLAP_IOU, "boosting": False, "global_box_cap": None,
    }
    if (
        protocol["training"] is not False
        or protocol["cohorts"] != {cohort: sha256(path) for cohort, (path, _, _, _) in inputs.items()}
        or any(protocol["detector"][key] != value for key, value in detector.items())
        or any(protocol["tiling"][key] != value for key, value in tiling.items())
    ):
        raise ValueError("Implementation and frozen protocol differ")
    root = HERE / "inference"
    root.mkdir(exist_ok=False)
    raw = HERE / "raw"
    raw.mkdir(exist_ok=False)
    write_json(root / "pre-inference-lock.json", {
        "protocol_sha256": args.protocol_sha256,
        "weights_sha256": source_sha,
        "code_sha256": {str(path): sha256(path) for path in (
            Path(__file__).resolve(), HERE / "common.py", PREVIOUS / "predict_detector.py", PREVIOUS / "build_teacher.py"
        )},
        "input_freezes": {cohort: frozen for cohort, (_, _, _, frozen) in inputs.items()},
        "tile_size_px": TILE, "stride_px": STRIDE, "artificial_edge_rejection_inclusive_px": EDGE_MARGIN,
        "cross_branch_dedup_iou": OVERLAP_IOU, "confidence_strictly_above": CONFIDENCE,
        "replay_first_pairs_per_cohort": REPLAY_PAIRS, "replay_abs_confidence_tolerance": REPLAY_CONFIDENCE_ATOL,
    })
    model = YOLO(str(A_CHECKPOINT))
    control = replay(model, inputs)
    write_json(root / "replay.json", {"passed": all(row["passed"] for row in control), "frames": control})
    if not all(row["passed"] for row in control):
        raise ValueError("Cached detector controls did not replay at the frozen tolerance")
    print(f"Historical replay passed on {len(control)} frames.", flush=True)
    for cohort, (manifest_path, scenes, cached, full_freeze) in inputs.items():
        dest = root / cohort
        dest.mkdir(exist_ok=False)
        write_json(dest / "inference-plan.json", {
            "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "weights": str(A_CHECKPOINT), "weights_sha256": source_sha, "protocol_sha256": args.protocol_sha256,
            "code_sha256": sha256(__file__), "cached_full_frame_freeze_sha256": full_freeze,
            "imgsz": TILE, "head": "one2many", "nms_iou": 0.7, "confidence_floor": CONFIDENCE,
            "max_det": MAX_DET, "device": "mps", "torch": str(torch.__version__), "ultralytics": ultralytics_version,
            "mode": "Frozen full-frame A at 768 + original RGB native tiles",
            "tile_stride_px": STRIDE, "artificial_edge_rejection_inclusive_px": EDGE_MARGIN,
            "greedy_dedup_iou_inclusive": OVERLAP_IOU,
            "coordinate_convention": "Original image pixels, integer xyxy from local rounding and origin offsets",
            "timing": "Per-view seconds combines measured tile pipeline and cached full-frame latency",
        })
        output, tile_evidence, flat_views = [], [], []
        started = time.monotonic()
        for index, (scene, old) in enumerate(zip(scenes, cached, strict=True), 1):
            views, evidence = {}, {}
            for view in scene["views"]:
                view_id = view["id"]
                views[view_id], evidence[view_id] = run_view(model, view, old["views"][view_id])
                flat_views.append(views[view_id])
            record = {"id": scene["id"], "setup": scene["setup"]}
            output.append({**record, "views": views})
            tile_evidence.append({**record, "views": evidence})
            if index % 20 == 0 or index == len(scenes):
                print(f"{cohort}: {index}/{len(scenes)} pairs, {time.monotonic() - started:.1f}s.", flush=True)
        write_json(dest / "predictions.json", {
            "scenes": output, "wall_seconds": time.monotonic() - started,
            "frames": len(flat_views), "tile_passes": sum(view["tile_passes"] for view in flat_views),
            "total_detector_passes_conceptual": sum(view["tile_passes"] + 1 for view in flat_views),
            "added_detections": sum(view["added_detections"] for view in flat_views),
            "median_cached_full_frame_seconds": statistics.median(view["cached_full_frame_seconds"] for view in flat_views),
            "median_tile_pipeline_seconds": statistics.median(view["tile_pipeline_seconds"] for view in flat_views),
            "median_combined_seconds": statistics.median(view["seconds"] for view in flat_views),
        })
        write_json(raw / f"{cohort}.json", {"scenes": tile_evidence})
        write_json(dest / "freeze.json", {name: sha256(dest / name) for name in ("inference-plan.json", "predictions.json")})
    write_json(root / "freeze.json", {
        str(path.relative_to(HERE)): sha256(path)
        for path in [
            root / "pre-inference-lock.json", root / "replay.json",
            *(root / cohort / "freeze.json" for cohort in COHORTS),
            *(raw / f"{cohort}.json" for cohort in COHORTS),
        ]
    })


if __name__ == "__main__":
    main()
