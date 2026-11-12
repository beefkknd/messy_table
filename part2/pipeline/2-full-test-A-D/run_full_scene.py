"""Run an inference-only vehicle matching baseline on the RoundaboutHD pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO

SCENES = Path(__file__).resolve().parents[1] / "data/roundabouthd/full-scene"
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO car, motorcycle, bus, truck
EMBEDDER = "facebook/dinov2-small"


def detect(path, detector, processor, encoder, device, classes=VEHICLE_CLASSES):
    result = detector.predict(str(path), imgsz=1536, conf=0.25, classes=classes,
                              device=device, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().round().astype(int)
    detections = [
        {"box_xyxy": box.tolist(), "class": result.names[int(cls)], "confidence": float(conf)}
        for box, cls, conf in zip(boxes, result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]
    return detections, encode_boxes(path, boxes, processor, encoder, device)


def encode_boxes(path, boxes, processor, encoder, device):
    if len(boxes) == 0:
        return np.empty((0, encoder.config.hidden_size))
    with Image.open(path) as image:
        crops = [image.crop(tuple(box)).convert("RGB") for box in boxes]
    return encode_crops(crops, processor, encoder, device)


def encode_crops(crops, processor, encoder, device):
    if not crops:
        return np.empty((0, encoder.config.hidden_size))
    inputs = processor(images=crops, do_center_crop=False, do_resize=True,
                       size={"height": 224, "width": 224}, return_tensors="pt")
    with torch.inference_mode():
        embeddings = encoder(**inputs.to(device)).last_hidden_state[:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, dim=1).cpu().numpy()
    return embeddings


def match(embeddings_a, embeddings_b, threshold):
    similarity = embeddings_a @ embeddings_b.T
    rows, cols = linear_sum_assignment(np.maximum(similarity - threshold, 0), maximize=True)
    return [
        {"A": int(a), "B": int(b), "cosine": float(similarity[a, b])}
        for a, b in zip(rows, cols) if similarity[a, b] > threshold
    ]


def map_detections(detections, annotations):
    if not detections or not annotations:
        return {}
    pred = np.array([d["box_xyxy"] for d in detections], dtype=float)
    truth = np.array(list(annotations.values()), dtype=float)
    truth[:, 2:] += truth[:, :2]
    span = np.maximum(
        np.minimum(pred[:, None, 2:], truth[None, :, 2:]) -
        np.maximum(pred[:, None, :2], truth[None, :, :2]), 0
    )
    intersection = span.prod(axis=2)
    union = ((pred[:, 2:] - pred[:, :2]).prod(axis=1)[:, None] +
             (truth[:, 2:] - truth[:, :2]).prod(axis=1)[None, :] - intersection)
    iou = intersection / union
    rows, cols = linear_sum_assignment(np.where(iou >= 0.5, iou, 0), maximize=True)
    ids = list(annotations)
    return {int(a): int(ids[b]) for a, b in zip(rows, cols) if iou[a, b] >= 0.5}


def score(predictions, key):
    outcomes = []
    for prediction, truth in zip(predictions, key["cases"], strict=True):
        assert prediction["case"] == truth["case"]
        maps = {
            side: map_detections(prediction["detections"][side], truth["scenes"][side]["vehicles"])
            for side in ("A", "B")
        }
        links = []
        for link in prediction["links"]:
            a, b = (maps[side].get(link[side]) for side in ("A", "B"))
            status = "unscorable" if a is None or b is None else "correct" if a == b else "wrong"
            links.append({**link, "vehicle_A": a, "vehicle_B": b, "status": status})
        recovered = {link["vehicle_A"] for link in links if link["status"] == "correct"}
        shared = set(truth["same_vehicle_ids"])
        outcomes.append({
            "case": prediction["case"],
            "expected": sorted(shared),
            "detected_in_both": sorted(shared & set(maps["A"].values()) & set(maps["B"].values())),
            "missed": sorted(shared - recovered),
            "links": links,
        })
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo11m.pt")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("full_scene_results.json"))
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running on {device}", flush=True)
    detector = YOLO(args.weights)
    processor = AutoImageProcessor.from_pretrained(EMBEDDER)
    encoder = AutoModel.from_pretrained(EMBEDDER).to(device).eval()
    predictions = []
    for case in sorted(SCENES.glob("case-*")):
        a, features_a = detect(case / "A.jpg", detector, processor, encoder, device)
        b, features_b = detect(case / "B.jpg", detector, processor, encoder, device)
        links = match(features_a, features_b, args.threshold)
        predictions.append({"case": int(case.name.split("-")[1]), "detections": {"A": a, "B": b}, "links": links})
        print(f"{case.name}: {len(a)} and {len(b)} detections, {len(links)} proposed matches", flush=True)
    key = json.loads((SCENES / "answer-key.json").read_text())
    result = {
        "configuration": {"detector": Path(args.weights).name, "embedder": EMBEDDER,
                          "match_threshold": args.threshold, "label_iou_threshold": 0.5},
        "predictions": predictions,
        "scores": score(predictions, key),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
