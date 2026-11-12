"""Generate and freeze single-pass YOLO detections from a label-blind image cohort."""
import argparse
import json
import resource
import time
from pathlib import Path

from common import initialize_runtime, local_path, pin_ultralytics, read_text, sha256, write_json

initialize_runtime()

import torch
from ultralytics import YOLO, __version__ as ultralytics_version


def predict_one(model, source, imgsz, conf, device, max_det, head="one2many"):
    result = model.predict(
        str(source) if isinstance(source, Path) else source,
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        max_det=max_det,
        device=device,
        half=False,
        rect=True,
        verbose=False,
        nms=head == "one2many",
    )[0]
    if bool(model.predictor.model.end2end) != (head == "one2one"):
        raise RuntimeError(f"Predictor backend did not select {head}")
    if device == "mps":
        torch.mps.synchronize()
    boxes = result.boxes.xyxy.cpu().numpy().round().astype(int).tolist()
    return [
        {"box_xyxy": box, "class": result.names[int(cls)], "confidence": float(score)}
        for box, cls, score in zip(boxes, result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--head", choices=("one2one", "one2many"), default="one2many")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable")
    pin_ultralytics()
    torch.set_num_threads(8)
    scenes = json.loads(read_text(args.manifest))["scenes"]
    if not scenes:
        raise ValueError("Empty cohort")
    args.out.mkdir(parents=True, exist_ok=False)
    weights_path = args.weights.resolve(strict=True)
    model = YOLO(str(weights_path))
    plan = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "weights": str(weights_path),
        "weights_sha256": sha256(weights_path),
        "code_sha256": sha256(__file__),
        "imgsz": args.imgsz,
        "head": args.head,
        "nms_iou": 0.7 if args.head == "one2many" else None,
        "confidence_floor": args.conf,
        "max_det": args.max_det,
        "device": args.device,
        "torch": str(torch.__version__),
        "ultralytics": ultralytics_version,
        "coordinate_convention": "Original RGB image pixels, integer xyxy from rounding",
    }
    write_json(args.out / "inference-plan.json", plan)
    outputs = []
    start = time.monotonic()
    for scene in scenes:
        predictions = {}
        for view in scene["views"]:
            source = local_path(view["path"])
            if sha256(source) != view["sha256"]:
                raise ValueError(f"Image changed after manifest freeze: {source}")
            started = time.monotonic()
            predictions[view["id"]] = {
                "sha256": view["sha256"],
                "detections": predict_one(
                    model, source, args.imgsz, args.conf, args.device, args.max_det, args.head
                ),
                "seconds": time.monotonic() - started,
            }
        if len(predictions) != len(scene["views"]):
            raise ValueError(f"Duplicate view ID in {scene['id']}")
        outputs.append({"id": scene["id"], "setup": scene["setup"], "views": predictions})
        if len(outputs) % 10 == 0 or len(outputs) == len(scenes):
            print(f"{len(outputs)}/{len(scenes)} scenes in {time.monotonic() - start:.1f}s", flush=True)
    write_json(args.out / "predictions.json", {
        "scenes": outputs,
        "wall_seconds": time.monotonic() - start,
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    write_json(args.out / "freeze.json", {
        name: sha256(args.out / name) for name in ("inference-plan.json", "predictions.json")
    })


if __name__ == "__main__":
    main()
