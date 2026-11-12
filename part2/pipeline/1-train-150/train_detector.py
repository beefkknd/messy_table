"""Fixed-budget, single-class YOLO26m fine-tuning for the isolated pilot."""
import argparse
import copy
import hashlib
import json
import random
import resource
import time
from pathlib import Path

from common import (
    WEIGHTS,
    WEIGHTS_SHA,
    initialize_runtime,
    local_path,
    pin_ultralytics,
    read_text,
    sha256,
    write_json,
)

initialize_runtime()

import cv2
import numpy as np
import torch
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import E2ELoss

from partial_loss import PartialDetectionLoss

COMPLETE = "complete_frame"
PARTIAL = {"partial_crop", "partial_frame"}


def load_rows(manifest, scopes):
    document = json.loads(read_text(manifest))
    rows = document["frames"] if isinstance(document, dict) else document
    train = [row for row in rows if row["partition"] == "train"]
    if not train or any(row["scope"] not in scopes for row in train):
        raise ValueError(f"Missing or inappropriate training scopes in {manifest}")
    for row in train:
        if "label" not in row and "boxes_xyxy" not in row:
            raise ValueError(f"Missing box labels for {row['image']}")
    return train


def image_and_boxes(row, imgsz):
    path = local_path(row["image"])
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    height, width = image.shape[:2]
    if "label" in row:
        yolo = np.asarray(
            [list(map(float, line.split())) for line in read_text(local_path(row["label"])).splitlines() if line.strip()],
            dtype=np.float32,
        ).reshape(-1, 5)
        if not (yolo[:, 0] == 0).all():
            raise ValueError(f"Expected tabletop_item class 0 in {row['label']}")
        center = yolo[:, 1:3] * [width, height]
        extent = yolo[:, 3:5] * [width, height]
        boxes = np.column_stack((center - extent / 2, center + extent / 2))
    else:
        boxes = np.asarray(row["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
    if row["scope"] in PARTIAL and not len(boxes) and not row.get("allow_empty_partial", False):
        raise ValueError(f"Partial training sample {row['image']} must have a positive")
    if "boxes" in row and len(boxes) != row["boxes"]:
        raise ValueError(f"Expected {row['boxes']} boxes in {row['image']}, read {len(boxes)}")
    if (
        not np.isfinite(boxes).all()
        or (boxes[:, :2] < -0.001).any()
        or (boxes[:, 2:] > np.asarray([width, height]) + 0.001).any()
        or (boxes[:, 2:] <= boxes[:, :2]).any()
    ):
        raise ValueError(f"Invalid original-frame box in {row['image']}")
    boxes = boxes.clip([0, 0, 0, 0], [width, height, width, height])
    scale = imgsz / max(height, width)
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (imgsz - rw) // 2, (imgsz - rh) // 2
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    square = cv2.copyMakeBorder(
        resized, top, imgsz - rh - top, left, imgsz - rw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    boxes = boxes * [rw / width, rh / height, rw / width, rh / height] + [left, top, left, top]
    xywh = np.column_stack(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2])) / imgsz
    rgb = square[:, :, ::-1].transpose(2, 0, 1).copy()
    return rgb, xywh.astype(np.float32)


def batch_from(rows, imgsz, device, treat_partial_as_background=False):
    imgs, boxes = zip(*(image_and_boxes(row, imgsz) for row in rows), strict=True)
    batch_indexes = np.repeat(np.arange(len(rows)), [len(box) for box in boxes])
    return {
        "img": torch.from_numpy(np.stack(imgs)).to(device=device, dtype=torch.float32) / 255,
        "bboxes": torch.from_numpy(np.concatenate(boxes)).to(device),
        "batch_idx": torch.from_numpy(batch_indexes).to(device),
        "cls": torch.zeros((len(batch_indexes), 1), device=device),
        "incomplete": torch.tensor(
            [row["scope"] in PARTIAL and not treat_partial_as_background for row in rows], device=device
        ),
    }


def sample_stream(rows, seed, count):
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    stream = []
    while len(stream) < count:
        rng.shuffle(indices)
        stream.extend(rows[index] for index in indices[:count - len(stream)])
    return stream


def make_schedule(trusted, supplement, seed, total_steps, batch, supplement_slots):
    trusted_count = batch - supplement_slots
    trusted_stream = sample_stream(trusted, seed, total_steps * trusted_count)
    extra_stream = sample_stream(supplement, seed + 1, total_steps * supplement_slots)
    return [
        trusted_stream[step * trusted_count:(step + 1) * trusted_count]
        + extra_stream[step * supplement_slots:(step + 1) * supplement_slots]
        for step in range(total_steps)
    ]


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_inputs(rows):
    files = {}
    for row in rows:
        for key in ("image", "label"):
            if key not in row:
                continue
            name = row[key]
            if name not in files:
                files[name] = sha256(local_path(name))
            expected = row.get(f"{key}_sha256")
            if expected is not None and files[name] != expected:
                raise ValueError(f"{key} changed after the manifest was created: {name}")
    return files


def inventory_digest(rows, files):
    images = [(row.get("row_id"), row["image"], files[row["image"]], row["scope"], row["scene"]) for row in rows]
    return hashlib.sha256(json.dumps(images).encode()).hexdigest()


def new_model(source, inference_head):
    model = DetectionModel(copy.deepcopy(source.yaml), nc=1, verbose=False)
    model.load(source, verbose=False)
    model.args = copy.deepcopy(DEFAULT_CFG)
    model.names = {0: "tabletop_item"}
    model.task = "detect"
    model.end2end = inference_head == "one2one"
    return model


def atomic_checkpoint(path, model, optimizer, step, arm_options, supplement_inventory):
    frozen_model = copy.deepcopy(model).cpu().float().eval()
    optimizer_state = optimizer.state_dict()
    optimizer_state["state"] = {
        parameter: {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
        for parameter, state in optimizer_state["state"].items()
    }
    pending = path.with_suffix(".tmp")
    torch.save({
        "model": frozen_model,
        "train_args": {**vars(model.args), "task": "detect"},
        "optimizer": optimizer_state,
        "step": step,
        "options": arm_options,
        "supplement_inventory_sha256": supplement_inventory,
    }, pending)
    pending.rename(path)


def synchronized(device):
    if device == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted", required=True, type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--continue-from", type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--supplement-slots", default=1, type=int)
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--freeze-layers", default=10, type=int)
    parser.add_argument("--seed", default=923, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--inference-head", choices=("one2one", "one2many"), default="one2one")
    parser.add_argument("--unsafe-partial-bce", action="store_true", help="Diagnostic control only")
    args = parser.parse_args()
    if not (0 < args.steps and 0 <= args.supplement_slots < args.batch and args.imgsz >= 32 and args.imgsz % 32 == 0):
        parser.error("Need positive steps, a complete frame per batch, and an image size divisible by 32")
    if args.supplement and not args.supplement_slots:
        parser.error("Supplement manifest supplied without any supplemental slots")
    if args.unsafe_partial_bce and not args.supplement:
        parser.error("Unmasked BCE requires a partial manifest")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable")
    pin_ultralytics()
    if sha256(WEIGHTS) != WEIGHTS_SHA:
        raise ValueError(f"Unexpected pretrained checkpoint hash for {WEIGHTS}")
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    trusted = load_rows(args.trusted, {COMPLETE})
    supplement = load_rows(args.supplement, {COMPLETE, *PARTIAL}) if args.supplement else trusted
    inputs = freeze_inputs(trusted + supplement)
    supplement_inventory = inventory_digest(supplement, inputs)
    options = {
        "seed": args.seed, "batch": args.batch, "supplement_slots": args.supplement_slots,
        "imgsz": args.imgsz, "freeze_layers": args.freeze_layers, "lr": args.lr,
        "device": args.device, "inference_head": args.inference_head,
        "trusted_sha256": sha256(args.trusted), "unsafe_partial_bce": args.unsafe_partial_bce,
        "trusted_inventory_sha256": inventory_digest(trusted, inputs),
    }
    optimizer_state = None
    if args.continue_from:
        previous = torch.load(args.continue_from, map_location="cpu", weights_only=False)
        if previous["options"] != options:
            raise ValueError("Continuation must preserve seed, trusted data, loss, and optimizer configuration")
        if previous["supplement_inventory_sha256"] != supplement_inventory:
            raise ValueError("Continuation must preserve ordered supplemental images and pixels")
        model = previous["model"]
        prior_steps = previous["step"]
        optimizer_state = previous["optimizer"]
    else:
        model = new_model(YOLO(str(WEIGHTS)).model, args.inference_head)
        prior_steps = 0
    input_state = state_digest(model)
    if not (0 <= args.freeze_layers < len(model.model)):
        raise ValueError(f"Must leave at least one of {len(model.model)} model blocks trainable")
    # Pin the stock 80:20 training branch mixture in every round. The stock
    # schedule is epoch-based, whereas this study controls optimizer steps.
    model.to(args.device).train()
    for block in list(model.model)[:args.freeze_layers]:
        block.requires_grad_(False).eval()
    for block in list(model.model)[args.freeze_layers:]:
        block.requires_grad_(True)
    criterion = E2ELoss(model, loss_fn=PartialDetectionLoss)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0005
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        del optimizer_state, previous
    # Advance the sample stream across rounds so trusted replay is comparable
    # while refreshed manifests can replace only the supplemental inventory.
    schedule = make_schedule(
        trusted, supplement, args.seed, prior_steps + args.steps, args.batch, args.supplement_slots
    )[prior_steps:]
    run = {
        "options": options,
        "source_weights_sha256": WEIGHTS_SHA,
        "incoming_checkpoint_sha256": sha256(args.continue_from) if args.continue_from else None,
        "incoming_model_state_sha256": input_state,
        "supplement_manifest_sha256": sha256(args.supplement) if args.supplement else None,
        "supplement_inventory_sha256": supplement_inventory,
        "trusted_manifest": str(args.trusted.resolve()),
        "supplement_manifest": str(args.supplement.resolve()) if args.supplement else None,
        "optimizer": "AdamW, weight_decay=0.0005, clip_grad_norm=10",
        "criterion": "Ultralytics E2ELoss at fixed one2many=0.8, one2one=0.2; no unknown negatives by default",
        "training_preprocessing": "Original RGB, square letterbox 114, no random image augmentation",
        "torch": str(torch.__version__),
        "ultralytics": ultralytics_version,
        "code_sha256": {name: sha256(Path(__file__).with_name(name)) for name in (
            "train_detector.py", "common.py", "partial_loss.py"
        )},
        "planned_optimizer_steps": args.steps,
        "previous_optimizer_steps": prior_steps,
        "input_sha256": inputs,
        "schedule": [
            [
                {"row_id": row.get("row_id", row["image"]), "image": row["image"], "scope": row["scope"],
                 "setup": row["setup"], "boxes": len(row["boxes_xyxy"]) if "boxes_xyxy" in row else row.get("boxes")}
                for row in batch
            ]
            for batch in schedule
        ],
    }
    write_json(args.out / "run-plan.json", run)
    metrics = []
    peak_driver_bytes = 0
    print(f"Setup completed in {time.monotonic() - start:.2f}s. Optimizing {args.steps} updates.", flush=True)
    for idx, rows in enumerate(schedule, 1):
        started = time.monotonic()
        batch = batch_from(rows, args.imgsz, args.device, args.unsafe_partial_bce)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = criterion(model(batch["img"]), batch)
        objective = loss.sum() / args.batch
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite loss on optimizer update {prior_steps + idx}")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        synchronized(args.device)
        if args.device == "mps":
            peak_driver_bytes = max(peak_driver_bytes, torch.mps.driver_allocated_memory())
        elapsed = time.monotonic() - started
        metrics.append({
            "step": prior_steps + idx,
            "seconds": elapsed,
            "loss": float(objective.detach()),
            "components_one2one": {key: float(value) for key, value in parts.items()},
        })
        if idx <= 3 or idx % 10 == 0 or idx == args.steps:
            print(f"Update {idx}/{args.steps} | loss {metrics[-1]['loss']:.4f} | {elapsed:.2f}s", flush=True)
    path = args.out / "checkpoint.pt"
    atomic_checkpoint(path, model, optimizer, prior_steps + args.steps, options, supplement_inventory)
    write_json(args.out / "training.json", {
        "completed_optimizer_steps": len(metrics),
        "total_optimizer_steps": prior_steps + len(metrics),
        "checkpoint_sha256": sha256(path),
        "wall_seconds": time.monotonic() - start,
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_sampled_mps_driver_bytes": peak_driver_bytes if args.device == "mps" else None,
        "per_step": metrics,
    })


if __name__ == "__main__":
    main()
