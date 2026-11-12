"""Verify locked optimizer budgets, sample schedules, and checkpoint hashes."""
import json
from pathlib import Path
import statistics

from common import HERE, local_path, read_text, sha256, write_json


def read(path):
    return json.loads(read_text(path))


def check_run(arm, root, settings):
    folder = root / "train" / arm
    plan = read(folder / "run-plan.json")
    results = read(folder / "training.json")
    options = plan["options"]
    for saved, frozen in (
        ("seed", "seed"), ("batch", "batch"), ("device", "device"), ("lr", "lr"),
        ("imgsz", "train_imgsz"), ("freeze_layers", "train_frozen_layers"), ("supplement_slots", "supplement_slots"),
    ):
        if options[saved] != settings[frozen]:
            raise ValueError(f"{arm} changed frozen setting {saved}")
    if options["unsafe_partial_bce"] or plan["previous_optimizer_steps"] != 0:
        raise ValueError(f"{arm} did not start the planned initial masked run")
    steps = settings["optimizer_steps"]
    if [row["step"] for row in results["per_step"]] != list(range(1, steps + 1)):
        raise ValueError(f"{arm} optimizer updates do not match frozen budget")
    if results["completed_optimizer_steps"] != steps or len(plan["schedule"]) != steps:
        raise ValueError(f"{arm} is incomplete")
    if results["checkpoint_sha256"] != sha256(folder / "checkpoint.pt"):
        raise ValueError(f"{arm} checkpoint hash mismatch")
    for path, expected in plan["input_sha256"].items():
        if sha256(local_path(path)) != expected:
            raise ValueError(f"{arm} training input changed: {path}")
    return plan, results


def main():
    root = HERE / "round2"
    settings = read(root / "freeze-plan.json")
    data = {arm: check_run(arm, root, settings) for arm in "ABCD"}
    common = {plan["incoming_model_state_sha256"] for plan, _ in data.values()}
    if len(common) != 1:
        raise ValueError("Arm initialization states differ")
    partial_label_stats = {}
    trusted_count = settings["batch"] - settings["supplement_slots"]
    for a, b, c, d in zip(*(data[arm][0]["schedule"] for arm in "ABCD"), strict=True):
        if not a[:trusted_count] == b[:trusted_count] == c[:trusted_count] == d[:trusted_count]:
            raise ValueError("Trusted replay slots differ across arms")
        if not all(len(batch) == settings["batch"] for batch in (a, b, c, d)):
            raise ValueError("Actual batch does not match the freeze")
        for slot_b, slot_c, slot_d in zip(b[trusted_count:], c[trusted_count:], d[trusted_count:], strict=True):
            if not slot_b["row_id"] == slot_c["row_id"] == slot_d["row_id"]:
                raise ValueError("Supplemental row IDs differ across B/C/D")
            if slot_b["image"] != slot_c["image"] or slot_b["boxes"] > slot_c["boxes"]:
                raise ValueError("B and C must use the same crop pixels and a subset of labels")
            if slot_c["boxes"] != slot_d["boxes"]:
                raise ValueError("C and D use different label counts")
    for arm, (plan, results) in data.items():
        extra_rows = [row for batch in plan["schedule"] for row in batch[trusted_count:]]
        partial_label_stats[arm] = {
            "actual_optimizer_steps": results["completed_optimizer_steps"],
            "wall_seconds": results["wall_seconds"],
            "median_step_seconds": statistics.median(row["seconds"] for row in results["per_step"]),
            "peak_sampled_mps_driver_bytes": results["peak_sampled_mps_driver_bytes"],
            "max_rss_bytes_macos": results["max_rss_bytes_macos"],
            "checkpoint_sha256": results["checkpoint_sha256"],
            "training_artifacts_sha256": {
                name: sha256(root / "train" / arm / name) for name in ("run-plan.json", "training.json")
            },
            "supplemental_row_exposures": len(extra_rows),
            "supplemental_positive_exposures": sum(row["boxes"] for row in extra_rows),
            "empty_supplemental_row_exposures": sum(not row["boxes"] for row in extra_rows),
        }
    write_json(root / "verified-training.json", {
        "matched_initial_model_state_sha256": common.pop(),
        "freeze_plan_sha256": sha256(root / "freeze-plan.json"),
        "trusted_replay_schedule_matches": True,
        "b_and_c_supplemental_crop_pixels_and_order_match": True,
        "c_and_d_supplemental_row_ids_and_label_counts_match": True,
        "arms": partial_label_stats,
    })


if __name__ == "__main__":
    main()
