"""Evaluate a soft epipolar penalty on the frozen MessyTable baseline."""
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
import torch
from transformers import AutoImageProcessor, AutoModel

from run_box_context import APPEARANCE_CUTOFF, HERE, SCENES, summarize
from run_full_scene import EMBEDDER, encode_boxes, match, score

MAX_PENALTY = 0.10
FREE_RESIDUAL = 0.25


def fundamental_matrix(camera_a, camera_b):
    def camera_to_reference(camera):
        pose = camera["camera_to_reference"]
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
        transform[:3, 3] = pose[:3]
        return transform

    a_to_b = np.linalg.solve(camera_to_reference(camera_b), camera_to_reference(camera_a))
    tx, ty, tz = a_to_b[:3, 3]
    cross_t = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    ka = np.array(camera_a["intrinsics"]).reshape(3, 3)
    kb = np.array(camera_b["intrinsics"]).reshape(3, 3)
    return np.linalg.inv(kb).T @ cross_t @ a_to_b[:3, :3] @ np.linalg.inv(ka)


def epipolar_residual(boxes_a, boxes_b, fundamental):
    """Average distance to each epipolar line in units of the box half-extent."""
    a = np.column_stack(((boxes_a[:, :2] + boxes_a[:, 2:]) / 2, np.ones(len(boxes_a))))
    b = np.column_stack(((boxes_b[:, :2] + boxes_b[:, 2:]) / 2, np.ones(len(boxes_b))))
    lines_b = a @ fundamental.T
    lines_a = b @ fundamental
    error = np.abs(lines_b @ b.T)
    support_a = (boxes_a[:, 2:] - boxes_a[:, :2]) / 2
    support_b = (boxes_b[:, 2:] - boxes_b[:, :2]) / 2
    denom_a = support_a @ np.abs(lines_a[:, :2]).T
    denom_b = np.abs(lines_b[:, :2]) @ support_b.T
    return (error / np.maximum(denom_a, 1e-12) + error / np.maximum(denom_b, 1e-12)) / 2


def geometry_links(similarity, residual):
    penalty = MAX_PENALTY * np.clip(residual - FREE_RESIDUAL, 0, 1)
    adjusted = similarity - penalty
    weights = np.maximum(adjusted - APPEARANCE_CUTOFF, 0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [
        {"A": int(a), "B": int(b), "cosine": float(similarity[a, b]),
         "epipolar_residual": float(residual[a, b]), "penalty": float(penalty[a, b]),
         "adjusted_score": float(adjusted[a, b])}
        for a, b in zip(rows, cols) if weights[a, b] > 0
    ]


def main():
    baseline_path = HERE / "messytable100_yolo26m_results.json"
    baseline = json.loads(baseline_path.read_text())
    calibration_path = SCENES / "camera-calibration.json"
    calibration = json.loads(calibration_path.read_text())
    assert baseline["configuration"]["match_threshold"] == APPEARANCE_CUTOFF
    assert len(calibration) == 80
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(EMBEDDER)
    encoder = AutoModel.from_pretrained(EMBEDDER).to(device).eval()
    predictions, diagnostics = [], []
    for pair in baseline["predictions"]:
        case_id = pair["case"]
        cameras = calibration.get(str(case_id))
        links = pair["links"]
        if cameras:
            folder = SCENES / f"case-{case_id:03}"
            boxes = {side: np.array([d["box_xyxy"] for d in pair["detections"][side]]).reshape(-1, 4)
                     for side in "AB"}
            a, b = (encode_boxes(folder / f"{side}.jpg", boxes[side], processor, encoder, device)
                    for side in "AB")
            assert edges(match(a, b, APPEARANCE_CUTOFF)) == edges(links), case_id
            matrix = fundamental_matrix(cameras["A"], cameras["B"])
            residual = epipolar_residual(boxes["A"], boxes["B"], matrix)
            links = geometry_links(a @ b.T, residual)
        predictions.append({**pair, "links": links})
        diagnostics.append({"case": case_id, "used_geometry": cameras is not None,
                            "changed": edges(links) != edges(pair["links"])})
        if case_id % 20 == 0:
            print(f"Processed {case_id} pairs", flush=True)

    key_path = SCENES / "answer-key.json"
    assert hashlib.sha256(key_path.read_bytes()).hexdigest() == baseline["configuration"]["answer_key_sha256"]
    key = json.loads(key_path.read_text())
    for case in key["cases"]:
        case["same_vehicle_ids"] = case["shared_object_ids"]
        for scene in case["scenes"].values():
            scene["vehicles"] = scene["objects"]
    outcomes = score(predictions, key)
    for outcome in outcomes:
        for link in outcome["links"]:
            link["object_A"] = link.pop("vehicle_A")
            link["object_B"] = link.pop("vehicle_B")
    result = {
        "configuration": {**baseline["configuration"], "detector_boxes": baseline_path.name,
                          "max_epipolar_penalty": MAX_PENALTY, "free_residual": FREE_RESIDUAL},
        "summary": summarize(outcomes, key), "diagnostics": diagnostics, "scores": outcomes,
    }
    (HERE / "messytable100_epipolar_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))
    print("Pairs with changed matches:", sum(d["changed"] for d in diagnostics))


def edges(links):
    return {(link["A"], link["B"]) for link in links}


if __name__ == "__main__":
    main()
