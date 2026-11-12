"""Freeze teacher detections and generate matched pseudo-label crop/full-frame manifests."""
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from common import ROOT, WEIGHTS, WEIGHTS_SHA, initialize_runtime, local_path, pin_ultralytics, read_text, sha256, write_json

initialize_runtime()

import cv2
import torch
from ultralytics import YOLO, __version__ as ultralytics_version

from predict_detector import predict_one

BRANCHES = {"rgb_1536": 1536, "rgb_768": 768, "gray_768": 768}
ITEM_PROXY = {
    "apple", "backpack", "banana", "book", "bottle", "bowl", "broccoli", "cake", "carrot",
    "cell phone", "clock", "cup", "donut", "fork", "handbag", "hot dog", "keyboard", "knife",
    "laptop", "mouse", "orange", "pizza", "remote", "sandwich", "scissors", "spoon", "sports ball",
    "teddy bear", "toothbrush", "vase", "wine glass",
}
CONFIDENCE_FLOOR = 0.25
SEED_CONFIDENCE = 0.80
OVERLAP = 0.55
MAX_DET = 300
CAP = 12
MIN_SIDE = 12
MAX_AREA_FRACTION = 0.30
CROP_PAD = 0.20


def iou(a, b):
    overlap = max(min(a[2], b[2]) - max(a[0], b[0]), 0) * max(min(a[3], b[3]) - max(a[1], b[1]), 0)
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


def box_qualifies(row, width, height):
    left, top, right, bottom = row["box_xyxy"]
    span_x, span_y = right - left, bottom - top
    return (
        row["class"] in ITEM_PROXY and 0 <= left < right <= width and 0 <= top < bottom <= height
        and min(span_x, span_y) >= MIN_SIDE and span_x * span_y <= MAX_AREA_FRACTION * width * height
    )


def candidate_inventory(branches, width, height):
    eligible = {
        branch: [
            {**row, "source_branch": branch, "source_index": index}
            for index, row in enumerate(values)
            if box_qualifies(row, width, height)
        ]
        for branch, values in branches.items()
    }
    baseline = []
    for row in sorted(eligible["rgb_1536"], key=lambda r: -r["confidence"]):
        if row["confidence"] < SEED_CONFIDENCE or len(baseline) == CAP:
            break
        if all(iou(row["box_xyxy"], item["box_xyxy"]) < OVERLAP for item in baseline):
            baseline.append({**row, "admission": "original_seed"})
    supplemental = []
    alternate = eligible["rgb_768"] + eligible["gray_768"]
    for row in sorted(alternate, key=lambda r: -r["confidence"]):
        if row["confidence"] < SEED_CONFIDENCE or len(baseline) + len(supplemental) == CAP:
            break
        box = row["box_xyxy"]
        if any(iou(box, item["box_xyxy"]) >= OVERLAP for item in baseline + supplemental):
            continue
        other = "gray_768" if row["source_branch"] == "rgb_768" else "rgb_768"
        supports = [
            (iou(box, support["box_xyxy"]), support)
            for support in eligible[other]
            if iou(box, support["box_xyxy"]) >= OVERLAP
        ]
        if supports:
            best_iou, best = max(supports, key=lambda pair: (pair[0], pair[1]["confidence"]))
            supplemental.append({
                **row, "admission": "supported_supplement", "support": best, "support_iou": best_iou,
            })
    return baseline, supplemental


def bounds(box, width, height):
    left, top, right, bottom = box
    dx, dy = (right - left) * CROP_PAD, (bottom - top) * CROP_PAD
    return [max(0, math.floor(left - dx)), max(0, math.floor(top - dy)),
            min(width, math.ceil(right + dx)), min(height, math.ceil(bottom + dy))]


def fully_inside(box, region):
    return region[0] <= box[0] and region[1] <= box[1] and box[2] <= region[2] and box[3] <= region[3]


def export_frames(output, scene, view, image, seeds):
    results = []
    for seed in seeds:
        region = bounds(seed["box_xyxy"], view["width"], view["height"])
        left, top, right, bottom = region
        key = f"{scene['id']}-{view['id']}-" + "-".join(map(str, region))
        crop_path = output / "crops" / f"{key}.png"
        if not crop_path.exists() and not cv2.imwrite(str(crop_path), image[top:bottom, left:right]):
            raise OSError(f"Failed to save crop {crop_path}")
        contained = [label for label in seeds if fully_inside(label["box_xyxy"], region)]
        boxes = [label["box_xyxy"] for label in contained]
        remapped = [[x0 - left, y0 - top, x1 - left, y1 - top] for x0, y0, x1, y1 in boxes]
        row = {
            "partition": "train",
            "setup": scene["setup"],
            "scene": scene["id"],
            "view": view["id"],
            "row_id": f"{scene['id']}-{view['id']}-{seed['source_branch']}-{seed['source_index']}",
            "source_image": view["path"],
            "source_image_sha256": view["sha256"],
            "crop_xyxy": region,
            "anchor": seed,
            "source_labels": contained,
        }
        results.append((
            {**row, "scope": "partial_crop", "image": str(crop_path.relative_to(ROOT)),
             "image_sha256": sha256(crop_path), "boxes_xyxy": remapped},
            {**row, "scope": "partial_frame", "image": view["path"], "image_sha256": view["sha256"],
             "boxes_xyxy": boxes},
        ))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    pin_ultralytics()
    if sha256(WEIGHTS) != WEIGHTS_SHA:
        raise ValueError("Unexpected teacher checkpoint hash")
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    scenes = json.loads(read_text(args.manifest))["scenes"]
    if not scenes or any(scene["native_split"] != "train" for scene in scenes):
        raise ValueError("Teacher inputs must all belong to native train")
    output = args.out.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Teacher output must be within workspace")
    (output / "crops").mkdir(parents=True, exist_ok=False)
    plan = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "source_weights_sha256": WEIGHTS_SHA,
        "code_sha256": {str(path.name): sha256(path) for path in (
            Path(__file__), Path(__file__).with_name("predict_detector.py"), Path(__file__).with_name("common.py"),
        )},
        "torch": str(torch.__version__), "ultralytics": ultralytics_version,
        "device": args.device, "inference_head": "one2many", "nms_iou": 0.7,
        "branches": BRANCHES, "raw_confidence_floor": CONFIDENCE_FLOOR, "raw_max_detections": MAX_DET,
        "class_proxy_allowlist": sorted(ITEM_PROXY),
        "min_side_px": MIN_SIDE, "max_original_image_area_fraction": MAX_AREA_FRACTION,
        "minimum_seed_confidence": SEED_CONFIDENCE, "min_support_iou": OVERLAP, "dedup_iou": OVERLAP,
        "seed_cap_per_frame": CAP, "crop_padding_each_side": CROP_PAD,
        "B": "High confidence original-RGB1536 seed with class proxy, geometry and greedy class-agnostic dedup",
        "C": "Retain B, then append qualifying RGB768 or gray768 seed supported by opposite 768 branch above raw floor, with same class proxy and geometry on both boxes and class-agnostic dedup",
        "D": "Same C training rows and same annotated positives in original full-frame coordinates",
        "cropped_positives": "Keep all admitted boxes fully contained in crop, mask all unknown anchor negatives",
    }
    write_json(output / "teacher-plan.json", plan)
    model = YOLO(str(WEIGHTS))
    raw_rows = []
    admission_rows = []
    training = {arm: [] for arm in "BCD"}
    count = Counter()
    started = time.monotonic()
    for index, scene in enumerate(scenes, 1):
        for view in scene["views"]:
            source = local_path(view["path"])
            if sha256(source) != view["sha256"]:
                raise ValueError(f"Image changed after manifest freeze: {source}")
            image = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None or image.shape[:2] != (view["height"], view["width"]):
                raise ValueError(f"Unexpected image dimensions for {source}")
            gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            branches = {}
            timings = {}
            for branch, imgsz in BRANCHES.items():
                before = time.monotonic()
                branches[branch] = predict_one(
                    model, gray if branch == "gray_768" else image, imgsz, CONFIDENCE_FLOOR,
                    args.device, MAX_DET, "one2many",
                )
                timings[branch] = time.monotonic() - before
            record = {"scene": scene["id"], "setup": scene["setup"], "view": view}
            raw_rows.append({**record, "branches": branches, "seconds": timings})
            baseline, additions = candidate_inventory(branches, view["width"], view["height"])
            admission_rows.append({**record, "original": baseline, "supplements": additions})
            crop_baseline = export_frames(output, scene, view, image, baseline)
            crop_supplement = export_frames(output, scene, view, image, baseline + additions)
            training["B"].extend(crop for crop, _ in crop_baseline)
            training["C"].extend(crop for crop, _ in crop_supplement)
            training["D"].extend(frame for _, frame in crop_supplement)
            count.update(images=1, original_seeds=len(baseline), supplemental_seeds=len(additions),
                         frames_with_original_seed=bool(baseline), frames_with_supplement=bool(additions))
        if index % 10 == 0 or index == len(scenes):
            print(f"Teacher {index}/{len(scenes)} scenes | B={count['original_seeds']} C additions={count['supplemental_seeds']} | {time.monotonic() - started:.1f}s", flush=True)
    write_json(output / "raw-predictions.json", {"frames": raw_rows})
    write_json(output / "admitted.json", {
        "frames": admission_rows, "counts": dict(count), "wall_seconds": time.monotonic() - started,
    })
    names = ["teacher-plan.json", "raw-predictions.json", "admitted.json"]
    for arm, rows in training.items():
        filename = f"frames-{arm}.json"
        write_json(output / filename, {"frames": rows})
        names.append(filename)
    write_json(output / "freeze.json", {name: sha256(output / name) for name in names})


if __name__ == "__main__":
    main()
