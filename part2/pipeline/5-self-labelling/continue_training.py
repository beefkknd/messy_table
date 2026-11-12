"""Fork the saved A model and AdamW state with a controlled supplemental stream."""
import argparse
import hashlib
from pathlib import Path
import resource
import sys
import time

from common import A_CHECKPOINT, HERE, PREVIOUS, PROGRAM, initialize_runtime, pin_ultralytics, sha256, write_json

initialize_runtime()
sys.path.append(str(PREVIOUS))

import cv2
import torch
from ultralytics.utils.loss import E2ELoss

from build_stronger_teacher import read, view_rows
from partial_loss import PartialDetectionLoss
from train_detector import (
    COMPLETE, PARTIAL, atomic_checkpoint, batch_from, freeze_inputs, inventory_digest, load_rows,
    sample_stream, state_digest, synchronized,
)

NEW_ARMS = {"single", "ladder", "oracle"}
REFRESH_ARMS = {"frozen", "refresh"}


def optimizer_digest(state_dict, group_names):
    digest = hashlib.sha256()
    layout = []
    group_count = len(state_dict["param_groups"])
    if group_count != len(group_names):
        raise ValueError("Named optimizer group count differs")
    for group, names in zip(state_dict["param_groups"], group_names, strict=True):
        values = {name: value for name, value in group.items() if name != "params"}
        digest.update(repr((sorted(values.items()), tuple(names))).encode())
        for param_id, name in zip(group["params"], names, strict=True):
            values = state_dict["state"][param_id]
            if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"Unexpected AdamW state for {name}")
            steps = int(values["step"].item())
            layout.append({"parameter": name, "shape": list(values["exp_avg"].shape), "step": steps})
            for key, value in sorted(values.items()):
                tensor = value.detach().cpu().contiguous()
                digest.update(repr((key, str(tensor.dtype), tuple(tensor.shape))).encode())
                digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), layout


def restore(checkpoint_path, options, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["options"] != options:
        raise ValueError("Checkpoint options differ from the approved original A recipe")
    model = checkpoint["model"]
    model_digest = state_digest(model)
    model.to(device).train()
    for block in list(model.model)[:options["freeze_layers"]]:
        block.requires_grad_(False).eval()
    for block in list(model.model)[options["freeze_layers"]:]:
        block.requires_grad_(True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable], lr=options["lr"], weight_decay=0.0005
    )
    names = {id(parameter): name for name, parameter in trainable}
    group_names = [[names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups]
    expected_state, layout = optimizer_digest(checkpoint["optimizer"], group_names)
    if not layout or any(
        entry["shape"] != list(parameter.shape) or entry["step"] != checkpoint["step"]
        for entry, (_, parameter) in zip(layout, trainable, strict=True)
    ):
        raise ValueError("Saved moments differ from current trainable parameter shapes or expected steps")
    optimizer.load_state_dict(checkpoint["optimizer"])
    if optimizer_digest(optimizer.state_dict(), group_names)[0] != expected_state:
        raise ValueError("AdamW restoration changed ordered optimizer state")
    return model, optimizer, {
        "incoming_model_state_sha256": model_digest,
        "incoming_optimizer_state_sha256": expected_state,
        "previous_optimizer_steps": checkpoint["step"],
        "previous_supplement_inventory_sha256": checkpoint["supplement_inventory_sha256"],
        "optimizer_groups": group_names,
        "optimizer_state_layout": layout,
    }


def verify_supplement(rows, metadata, expected_scope):
    expected = view_rows(metadata)
    if len(expected) != 240:
        raise ValueError("Expected 240 frozen supplemental image slots")
    for row, (row_id, scene, view) in zip(rows, expected, strict=True):
        fields = {
            "row_id": row_id, "image": view["path"], "image_sha256": view["sha256"],
            "scene": scene["id"], "setup": scene["setup"], "scope": expected_scope,
        }
        if any(row.get(key) != value for key, value in fields.items()):
            raise ValueError(f"Unexpected data or label scope in supplemental row {row_id}")


def frozen_block_digest(model, freeze_layers):
    digest = hashlib.sha256()
    for block in list(model.model)[:freeze_layers]:
        if block.training or any(parameter.requires_grad for parameter in block.parameters()):
            raise ValueError("A frozen block is trainable or updates normalization statistics")
        digest.update(state_digest(block).encode())
    return digest.hexdigest()


def batch_schedule(trusted, supplemental, options, prior_steps, steps, extra_offset):
    base_slots = options["batch"] - options["supplement_slots"]
    extra_slots = options["supplement_slots"]
    common = sample_stream(trusted, options["seed"], (prior_steps + steps) * base_slots)[prior_steps * base_slots:]
    extra = sample_stream(supplemental, options["seed"] + 1, (extra_offset + steps) * extra_slots)[extra_offset * extra_slots:]
    return [
        common[index * base_slots:(index + 1) * base_slots] + extra[index * extra_slots:(index + 1) * extra_slots]
        for index in range(steps)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted({"replay", *NEW_ARMS, *REFRESH_ARMS}))
    parser.add_argument("--trusted", type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=A_CHECKPOINT)
    parser.add_argument("--protocol", type=Path, default=PROGRAM / "PROTOCOL.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--audit-only", action="store_true", help="Restore checkpoint on CPU and verify schedule without updates")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.out = args.out.resolve()
    if not args.out.is_relative_to(HERE):
        parser.error("Outputs must stay within follow-up engineering")
    if not 0 < args.steps <= 300:
        parser.error("Require 1 to 300 additional updates")
    if args.arm == "replay":
        if args.supplement or args.metadata:
            parser.error("Replay uses trusted frames in all slots")
    elif not args.supplement or not args.metadata:
        parser.error("Require supplemental labels and a frozen metadata manifest")
    from_a = args.checkpoint == A_CHECKPOINT.resolve()
    if from_a == (args.arm in REFRESH_ARMS):
        parser.error("Initial arms fork A, and refresh arms start from their eligible student")
    pin_ultralytics()
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    device = "cpu" if args.audit_only else "mps"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS unavailable")
    base_plan = read(A_CHECKPOINT.parent / "run-plan.json")
    expected_options = base_plan["options"]
    protocol = read(args.protocol)
    training = protocol["training"]
    for option, field in (
        ("seed", "seed"), ("batch", "batch"), ("supplement_slots", "supplemental_slots"),
        ("imgsz", "imgsz"), ("freeze_layers", "frozen_backbone_blocks"), ("lr", "lr"),
    ):
        if expected_options[option] != training[field]:
            raise ValueError(f"Inherited option {option} differs from the approved protocol")
    approved_steps = training["additional_optimizer_steps"] if from_a else protocol["conditional_refresh"]["extra_steps_each"]
    if args.steps != approved_steps:
        raise ValueError("Additional optimizer budget differs from the approved protocol")
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != read(args.checkpoint.parent / "training.json")["checkpoint_sha256"]:
        raise ValueError("Unexpected incoming checkpoint hash")
    if from_a and checkpoint_sha != protocol["parent_checkpoint_sha256"]:
        raise ValueError("A checkpoint differs from the approved fork point")
    if args.metadata and sha256(args.metadata) != protocol["supplemental_sha256"]:
        raise ValueError("Supplemental metadata differs from the approved source inventory")
    if sha256(args.trusted) != expected_options["trusted_sha256"]:
        raise ValueError("Trusted manifest changed since A training")
    torch.manual_seed(expected_options["seed"])
    start = time.monotonic()
    trusted = load_rows(args.trusted, {COMPLETE})
    supplement = load_rows(args.supplement, {COMPLETE, *PARTIAL}) if args.supplement else trusted
    inputs = freeze_inputs(trusted + supplement)
    if inventory_digest(trusted, inputs) != expected_options["trusted_inventory_sha256"]:
        raise ValueError("Trusted frame inventory changed since A training")
    for row in trusted:
        for key in ("image", "label"):
            name = row[key]
            if base_plan["input_sha256"][name] != inputs[name]:
                raise ValueError(f"Trusted training bytes changed: {name}")
    if args.metadata:
        verify_supplement(supplement, args.metadata, COMPLETE if args.arm == "oracle" else "partial_frame")
        if {row["setup"] for row in supplement} & {row["setup"] for row in trusted}:
            raise ValueError("Supplemental and trusted setup inventories overlap")
    incoming, optimizer, provenance = restore(args.checkpoint, expected_options, device)
    frozen_before = frozen_block_digest(incoming, expected_options["freeze_layers"])
    prior_steps = provenance["previous_optimizer_steps"]
    supplement_digest = inventory_digest(supplement, inputs)
    if from_a and prior_steps != 300:
        raise ValueError("Initial fork requires the 300-step A model and optimizer")
    if (args.arm == "replay" or not from_a) and provenance["previous_supplement_inventory_sha256"] != supplement_digest:
        raise ValueError("Replay and refresh must preserve prior ordered supplemental images and pixels")
    extra_offset = prior_steps if args.arm == "replay" else prior_steps - 300
    schedule = batch_schedule(trusted, supplement, expected_options, prior_steps, args.steps, extra_offset)
    planned_rows = [
        [
            {"row_id": row.get("row_id", row["image"]), "image": row["image"], "scope": row["scope"],
             "setup": row["setup"], "boxes": len(row["boxes_xyxy"]) if "boxes_xyxy" in row else row.get("boxes")}
            for row in batch
        ]
        for batch in schedule
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "run-plan.json", {
        "arm": args.arm, "options": expected_options,
        "protocol_sha256": sha256(args.protocol),
        "incoming_checkpoint": str(args.checkpoint), "incoming_checkpoint_sha256": checkpoint_sha,
        **provenance,
        "restoration_device": device,
        "incoming_frozen_blocks_sha256": frozen_before,
        "supplement_manifest_sha256": sha256(args.supplement) if args.supplement else None,
        "supplement_inventory_sha256": supplement_digest, "supplemental_offset": extra_offset,
        "metadata_sha256": sha256(args.metadata) if args.metadata else None,
        "planned_optimizer_steps": args.steps, "audit_only": args.audit_only,
        "schedule": planned_rows, "input_sha256": inputs,
        "code_sha256": {
            str(path): sha256(path) for path in (
                Path(__file__).resolve(), HERE / "common.py", PREVIOUS / "train_detector.py", PREVIOUS / "partial_loss.py",
            )
        },
    })
    if args.audit_only:
        print(f"Verified {prior_steps}-step optimizer with {len(provenance['optimizer_state_layout'])} parameter states "
              f"and digest {provenance['incoming_optimizer_state_sha256']}", flush=True)
        return
    criterion = E2ELoss(incoming, loss_fn=PartialDetectionLoss)
    metrics, peak = [], 0
    for index, rows in enumerate(schedule, 1):
        step_start = time.monotonic()
        batch = batch_from(rows, expected_options["imgsz"], device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = criterion(incoming(batch["img"]), batch)
        objective = loss.sum() / expected_options["batch"]
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite objective at {prior_steps + index}")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(incoming.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        synchronized(device)
        peak = max(peak, torch.mps.driver_allocated_memory())
        metrics.append({
            "step": prior_steps + index, "loss": float(objective.detach()),
            "seconds": time.monotonic() - step_start,
            "components_one2one": {key: float(value) for key, value in parts.items()},
        })
        if index <= 3 or index % 30 == 0 or index == args.steps:
            print(f"{args.arm} update {prior_steps + index} | loss {metrics[-1]['loss']:.4f} | "
                  f"{metrics[-1]['seconds']:.2f}s", flush=True)
    path = args.out / "checkpoint.pt"
    frozen_after = frozen_block_digest(incoming, expected_options["freeze_layers"])
    if frozen_after != frozen_before:
        raise ValueError("Frozen model parameters or buffers changed during training")
    atomic_checkpoint(path, incoming, optimizer, prior_steps + args.steps, expected_options, supplement_digest)
    write_json(args.out / "training.json", {
        "completed_optimizer_steps": len(metrics), "total_optimizer_steps": prior_steps + len(metrics),
        "checkpoint_sha256": sha256(path), "wall_seconds": time.monotonic() - start,
        "outgoing_frozen_blocks_sha256": frozen_after,
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_sampled_mps_driver_bytes": peak, "per_step": metrics,
    })


if __name__ == "__main__":
    main()
