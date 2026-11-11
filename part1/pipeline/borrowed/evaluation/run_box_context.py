"""Test relative box positions as a tie-breaker on frozen MessyTable detections."""
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from transformers import AutoImageProcessor, AutoModel

from run_full_scene import EMBEDDER, encode_boxes, match, score

HERE = Path(__file__).resolve().parent
SCENES = HERE.parent / "data/messytable-100"
APPEARANCE_CUTOFF = 0.70
ANCHOR_MARGIN = 0.03
CONTEXT_WEIGHT = 0.05


def anchor_matches(similarity):
    if not similarity.size:
        return []
    best_b = similarity.argmax(axis=1)
    best_a = similarity.argmax(axis=0)
    anchors = []
    for a, b in enumerate(best_b):
        value = similarity[a, b]
        other_b = np.delete(similarity[a, :], b).max(initial=-1)
        other_a = np.delete(similarity[:, b], a).max(initial=-1)
        if best_a[b] == a and value > APPEARANCE_CUTOFF and value - max(other_a, other_b) >= ANCHOR_MARGIN:
            anchors.append((a, int(b)))
    return anchors


def box_distances(boxes):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    return np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)


def contextual_links(similarity, boxes_a, boxes_b):
    anchors = anchor_matches(similarity)
    context = np.zeros_like(similarity)
    votes = np.zeros(similarity.shape, dtype=int)
    if len(anchors) >= 2:
        distances_a, distances_b = box_distances(boxes_a), box_distances(boxes_b)
        for a, b in zip(*np.where(similarity > APPEARANCE_CUTOFF)):
            references = [(i, j) for i, j in anchors if i != a and j != b]
            comparisons = []
            for (i, j), (k, l) in combinations(references, 2):
                order_a = np.sign(distances_a[a, i] - distances_a[a, k])
                order_b = np.sign(distances_b[b, j] - distances_b[b, l])
                if order_a and order_b:
                    comparisons.append(order_a * order_b)
            if comparisons:
                context[a, b] = np.mean(comparisons)
                votes[a, b] = len(comparisons)

    adjusted = similarity + CONTEXT_WEIGHT * context
    eligible = similarity > APPEARANCE_CUTOFF
    weights = np.where(eligible, np.maximum(adjusted - APPEARANCE_CUTOFF, 0), 0)
    rows, cols = linear_sum_assignment(weights, maximize=True)
    links = [
        {"A": int(a), "B": int(b), "cosine": float(similarity[a, b]),
         "context_agreement": float(context[a, b]), "context_votes": int(votes[a, b]),
         "adjusted_score": float(adjusted[a, b])}
        for a, b in zip(rows, cols) if weights[a, b] > 0
    ]
    return links, anchors, int(np.count_nonzero(votes))


def summarize(outcomes, key):
    strata = {case["case"]: case["stratum"] for case in key["cases"]}
    summary = {}
    for group in ["overall", *sorted(set(strata.values()))]:
        cases = [case for case in outcomes if group in ("overall", strata[case["case"]])]
        counts = Counter(link["status"] for case in cases for link in case["links"])
        true = sum(len(case["expected"]) for case in cases)
        detected = sum(len(case["detected_in_both"]) for case in cases)
        scorable = counts["correct"] + counts["wrong"]
        summary[group] = {
            "pairs": len(cases), "expected_links": true, "detected_in_both": detected,
            "proposals": counts.total(), **{k: counts[k] for k in ("correct", "wrong", "unscorable")},
            "recall": counts["correct"] / true if true else None,
            "precision_among_scorable": counts["correct"] / scorable if scorable else None,
            "pairs_with_proposals": sum(bool(case["links"]) for case in cases),
        }
    return summary


def main():
    baseline_path = HERE / "messytable100_yolo26m_results.json"
    baseline = json.loads(baseline_path.read_text())
    assert baseline["configuration"]["match_threshold"] == APPEARANCE_CUTOFF
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(EMBEDDER)
    encoder = AutoModel.from_pretrained(EMBEDDER).to(device).eval()
    predictions = []
    diagnostics = []
    for pair in baseline["predictions"]:
        folder = SCENES / f"case-{pair['case']:03}"
        boxes = {side: np.array([d["box_xyxy"] for d in pair["detections"][side]]).reshape(-1, 4)
                 for side in "AB"}
        a, b = (encode_boxes(folder / f"{side}.jpg", boxes[side], processor, encoder, device)
                for side in "AB")
        original = match(a, b, APPEARANCE_CUTOFF)
        original_edges = {(link["A"], link["B"]) for link in original}
        assert original_edges == {(link["A"], link["B"]) for link in pair["links"]}, pair["case"]
        links, anchors, compared = contextual_links(a @ b.T, boxes["A"], boxes["B"])
        predictions.append({**pair, "links": links})
        diagnostics.append({
            "case": pair["case"], "anchors": anchors, "candidates_with_context": compared,
            "changed": {(link["A"], link["B"]) for link in links} != original_edges,
        })
        if pair["case"] % 20 == 0:
            print(f"Processed {pair['case']} pairs", flush=True)

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
    results = {
        "configuration": {
            **baseline["configuration"], "anchor_margin": ANCHOR_MARGIN, "context_weight": CONTEXT_WEIGHT,
            "detector_boxes": baseline_path.name,
        },
        "summary": summarize(outcomes, key),
        "diagnostics": diagnostics,
        "scores": outcomes,
    }
    (HERE / "messytable100_box_context_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["summary"], indent=2))
    print("Pairs with usable context:", sum(d["candidates_with_context"] > 0 for d in diagnostics))
    print("Pairs with changed matches:", sum(d["changed"] for d in diagnostics))


if __name__ == "__main__":
    main()
