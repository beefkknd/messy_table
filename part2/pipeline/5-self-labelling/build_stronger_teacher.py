"""Run the frozen supervised A teacher on the label-hidden native train pool."""
import argparse
from collections import Counter
import json
from pathlib import Path
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

BRANCHES = {"rgb_768": 768, "rgb_1536": 1536, "gray_768": 768}
RAW_FLOOR = 0.25
SEED_FLOOR = 0.80
SUPPORT_FLOOR = 0.45
IOU = 0.55
MIN_SIDE = 12
MAX_AREA_FRACTION = 0.30
CAP = 64
MAX_DET = 300


def read(path):
    return json.loads(read_text(path))


def view_rows(path):
    scenes = read(path)["scenes"]
    rows = []
    identities = set()
    for scene in scenes:
        if scene["native_split"] != "train":
            raise ValueError(f"Teacher input outside native training: {scene['id']}")
        for view in scene["views"]:
            row_id = f"{scene['id']}-{view['id']}"
            if row_id in identities:
                raise ValueError(f"Duplicate row ID {row_id}")
            identities.add(row_id)
            rows.append((row_id, scene, view))
    return rows


def qualifies(row, width, height):
    left, top, right, bottom = row["box_xyxy"]
    return (
        row["class"] == "tabletop_item" and 0 <= left < right <= width and 0 <= top < bottom <= height
        and min(right - left, bottom - top) >= MIN_SIDE
        and (right - left) * (bottom - top) <= MAX_AREA_FRACTION * width * height
    )


def admits(branches, width, height):
    eligible = {
        branch: [
            {**row, "source_branch": branch, "source_index": index}
            for index, row in enumerate(detections) if qualifies(row, width, height)
        ]
        for branch, detections in branches.items()
    }
    primary = []
    for row in sorted(eligible["rgb_768"], key=lambda item: -item["confidence"]):
        if row["confidence"] < SEED_FLOOR or len(primary) == CAP:
            break
        if all(iou(row["box_xyxy"], item["box_xyxy"]) < IOU for item in primary):
            primary.append({**row, "admission": "one_pass_seed"})
    additions = []
    alternates = eligible["rgb_1536"] + eligible["gray_768"]
    for row in sorted(alternates, key=lambda item: -item["confidence"]):
        if row["confidence"] < SEED_FLOOR or len(primary) + len(additions) == CAP:
            break
        box = row["box_xyxy"]
        if any(iou(box, item["box_xyxy"]) >= IOU for item in primary + additions):
            continue
        supports = []
        for branch, detections in eligible.items():
            if branch == row["source_branch"]:
                continue
            for support in detections:
                overlap = iou(box, support["box_xyxy"])
                if support["confidence"] > SUPPORT_FLOOR and overlap >= IOU:
                    supports.append((overlap, support))
        if supports:
            overlap, support = max(supports, key=lambda pair: (pair[0], pair[1]["confidence"]))
            additions.append({
                **row, "admission": "supported_supplement", "support": support, "support_iou": overlap,
            })
    return primary, additions


def export_row(row_id, scene, view, labels):
    return {
        "row_id": row_id, "scene": scene["id"], "setup": scene["setup"], "view": view["id"],
        "partition": "train", "scope": "partial_frame", "image": view["path"], "image_sha256": view["sha256"],
        "boxes_xyxy": [label["box_xyxy"] for label in labels], "allow_empty_partial": not labels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROGRAM / "PROTOCOL.json")
    parser.add_argument("--amendment", type=Path, default=PROGRAM / "AMENDMENT_01.json")
    args = parser.parse_args()
    pin_ultralytics()
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS required for the frozen teacher run")
    source_sha = sha256(A_CHECKPOINT)
    if source_sha != read(A_CHECKPOINT.parent / "training.json")["checkpoint_sha256"]:
        raise ValueError("Frozen supervised teacher checkpoint digest mismatch")
    protocol = read(args.protocol)
    amendment = read(args.amendment)
    if (
        amendment["parent_protocol_sha256"] != sha256(args.protocol)
        or amendment["support_confidence_operator"] != "strict >0.45; seed>=0.80;raw floor>0.25"
    ):
        raise ValueError("Frozen amendment differs from the implemented confidence comparisons")
    settings = {
        "head": "one2many", "nms_iou": 0.7, "raw_max_det": MAX_DET, "branches": BRANCHES,
        "raw_confidence_floor": RAW_FLOOR, "seed_confidence": SEED_FLOOR, "support_confidence": SUPPORT_FLOOR,
        "support_iou": IOU, "dedup_iou": IOU, "min_side_px": MIN_SIDE, "max_frame_area_fraction": MAX_AREA_FRACTION,
        "accepted_cap_per_frame": CAP,
    }
    if any(protocol["teacher"][key] != value for key, value in settings.items()):
        raise ValueError("Teacher implementation differs from the frozen protocol")
    if (
        protocol["parent_checkpoint_sha256"] != source_sha
        or protocol["teacher_pool_sha256"] != sha256(args.pool)
        or protocol["supplemental_sha256"] != sha256(args.supplemental)
    ):
        raise ValueError("Teacher input digests differ from the frozen protocol")
    pool = view_rows(args.pool)
    supplemental = view_rows(args.supplemental)
    by_id = {row_id: (scene["setup"], view) for row_id, scene, view in pool}
    if len(pool) != 1000 or len(supplemental) != 240:
        raise ValueError("Expected frozen Pseudo1000 and Supplemental240 frame counts")
    for row_id, scene, view in supplemental:
        if by_id.get(row_id) != (scene["setup"], view):
            raise ValueError(f"Supplemental view differs from the unlabeled pool: {row_id}")
    if len({scene["setup"] for _, scene, _ in supplemental}) != 91:
        raise ValueError("Supplemental240 must represent every 91-setup pool group")
    out = args.out.resolve()
    if not out.is_relative_to(HERE):
        raise ValueError("Teacher output must be inside follow-up engineering")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "teacher-plan.json", {
        "checkpoint": str(A_CHECKPOINT),
        "protocol_sha256": sha256(args.protocol),
        "amendment_sha256": sha256(args.amendment),
        "source_weights_sha256": source_sha,
        "weights_role": "Frozen supervised A from Round 2, after 300 trusted-only optimizer updates",
        "pool": str(args.pool.resolve()), "pool_sha256": sha256(args.pool),
        "supplemental": str(args.supplemental.resolve()), "supplemental_sha256": sha256(args.supplemental),
        "code_sha256": {
            str(path): sha256(path) for path in (
                Path(__file__).resolve(), HERE / "common.py", PREVIOUS / "predict_detector.py", PREVIOUS / "build_teacher.py",
            )
        },
        "torch": str(torch.__version__), "ultralytics": ultralytics_version, "device": "mps",
        "inference_head": "one2many", "nms_iou": 0.7, "max_det": MAX_DET,
        "branches": BRANCHES, "confidence_floor": RAW_FLOOR, "seed_confidence": SEED_FLOOR,
        "minimum_alternate_support_confidence_exclusive": SUPPORT_FLOOR, "dedup_iou": IOU, "support_iou": IOU,
        "min_side_px": MIN_SIDE, "max_frame_area_fraction": MAX_AREA_FRACTION,
        "max_accepted_per_frame": CAP, "class": "tabletop_item",
        "single_policy": "RGB768 geometry-qualified seeds at >=0.80 in descending confidence",
        "ladder_policy": "Keep every single-pass seed, then append other-branch seeds at >=0.80 supported by a different branch strictly above 0.45",
        "supplemental_inputs": "Original RGB full frames in exact Supplemental240 order, including empty partial slots",
    })
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    model = YOLO(str(A_CHECKPOINT))
    if model.model.names != {0: "tabletop_item"}:
        raise ValueError("Incorrect one-class teacher taxonomy")
    counts = Counter()
    raw, admitted, training = [], [], {}
    requested_ids = {row_id for row_id, _, _ in supplemental}
    started = time.monotonic()
    for index, (row_id, scene, view) in enumerate(pool, 1):
        path = local_path(view["path"])
        if sha256(path) != view["sha256"]:
            raise ValueError(f"Image bytes differ from the frozen manifest: {path}")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if bgr is None or bgr.shape[:2] != (view["height"], view["width"]):
            raise ValueError(f"Unexpected source image dimensions: {path}")
        gray = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        branches, timings = {}, {}
        for branch, size in BRANCHES.items():
            branch_start = time.monotonic()
            branches[branch] = predict_one(
                model, gray if branch == "gray_768" else bgr, size, RAW_FLOOR, "mps", MAX_DET, "one2many"
            )
            timings[branch] = time.monotonic() - branch_start
        primary, extras = admits(branches, view["width"], view["height"])
        record = {"row_id": row_id, "scene": scene["id"], "setup": scene["setup"], "view": view}
        raw.append({**record, "branches": branches, "seconds": timings})
        admitted.append({**record, "original": primary, "supplements": extras})
        counts.update(images=1, single_boxes=len(primary), added_boxes=len(extras),
                      images_with_single=bool(primary), images_with_additions=bool(extras))
        if row_id in requested_ids:
            training[row_id] = (
                export_row(row_id, scene, view, primary),
                export_row(row_id, scene, view, primary + extras),
            )
        if index % 100 == 0:
            print(f"Teacher {index}/1000, {counts['single_boxes']} original and {counts['added_boxes']} added in {time.monotonic() - started:.1f}s", flush=True)
    single, ladder = zip(*(training[row_id] for row_id, _, _ in supplemental), strict=True)
    summary = {
        **counts, "supplemental_frames": len(supplemental),
        "supplemental_single_boxes": sum(len(row["boxes_xyxy"]) for row in single),
        "supplemental_ladder_boxes": sum(len(row["boxes_xyxy"]) for row in ladder),
        "supplemental_single_empty": sum(row["allow_empty_partial"] for row in single),
        "supplemental_ladder_empty": sum(row["allow_empty_partial"] for row in ladder),
    }
    write_json(out / "raw-predictions.json", {"frames": raw})
    write_json(out / "admitted.json", {"frames": admitted, "counts": summary, "wall_seconds": time.monotonic() - started})
    write_json(out / "frames-single.json", {"frames": single})
    write_json(out / "frames-ladder.json", {"frames": ladder})
    write_json(out / "freeze.json", {
        name: sha256(out / name) for name in (
            "teacher-plan.json", "raw-predictions.json", "admitted.json", "frames-single.json", "frames-ladder.json",
        )
    })
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
