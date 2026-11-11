"""Freeze and score dense correspondence evidence for every original pair."""
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

from roma_adapter import HERE, CHECKSUM, RESOLUTION, VENDOR, DINO, build_model, input_tensor, predict

ROOT = HERE.parents[1]
EVALUATION = ROOT / 'evaluation'
EXTENSION = EVALUATION / 'messytable-foundyou-native-retrieval-100'
sys.path[:0] = [str(EVALUATION), str(EXTENSION), str(ROOT / 'everyday-identity/scripts')]
from score_extension import load_extension
from score_foundyou_native_retrieval import read, save, sha, verified_member
from runtime_assets import model_file

CYCLE_PX = 4.0
MIN_PIXELS = 8
GATE = .05


def revision(repository):
    assert not subprocess.check_output(['git', '-C', str(repository), 'status', '--porcelain'])
    return subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'],
                                   text=True).strip()


def paths(case):
    return {side: ROOT / 'data/messytable-100' / f'case-{case:03}' / f'{side}.jpg' for side in 'AB'}


def regions(image_path, detections, segmenter):
    with Image.open(image_path) as image:
        width, height = image.size
    boxes = np.array([d['box_xyxy'] for d in detections], dtype=np.float32).reshape(-1, 4)
    normalized = boxes * np.array([RESOLUTION / width, RESOLUTION / height] * 2)
    centers = np.arange(RESOLUTION) + .5
    box_masks = np.array([
        (centers[None, :] >= x1) & (centers[None, :] < x2)
        & (centers[:, None] >= y1) & (centers[:, None] < y2)
        for x1, y1, x2, y2 in normalized
    ], dtype=bool).reshape(-1, RESOLUTION, RESOLUTION)
    masks = np.zeros_like(box_masks)
    quality = np.full(len(boxes), np.nan)
    if len(boxes):
        result = segmenter.predict(str(image_path), bboxes=boxes.tolist(), imgsz=1024, conf=0.,
                                   device='mps', verbose=False)[0]
        if result.masks is not None:
            assert tuple(result.masks.data.shape[1:]) == (height, width)
            indices = result.boxes.cls.int().tolist()
            assert len(set(indices)) == len(indices) and all(0 <= i < len(boxes) for i in indices)
            for index, mask, confidence in zip(indices, result.masks.data.cpu().numpy(),
                                               result.boxes.conf.tolist(), strict=True):
                masks[index] = cv2.resize(mask.astype(np.uint8), (RESOLUTION, RESOLUTION),
                                          interpolation=cv2.INTER_NEAREST_EXACT).astype(bool) & box_masks[index]
                quality[index] = confidence
    return {'box': box_masks, 'mask': masks, 'quality': quality}


def cycle_points(warp, reverse, overlap, reverse_overlap):
    height, width = overlap.shape
    x, y = ((warp + 1) * np.array([width / 2, height / 2], dtype=np.float32) - .5).transpose(2, 0, 1)
    opposite = cv2.remap(reverse, x.copy(), y.copy(), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    opposite_overlap = cv2.remap(reverse_overlap, x.copy(), y.copy(), cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    yy, xx = np.indices((height, width))
    returned = (opposite + 1) * np.array([width / 2, height / 2], dtype=np.float32) - .5
    error = np.hypot(returned[..., 0] - xx, returned[..., 1] - yy)
    good = ((x >= -.5) & (x < width - .5) & (y >= -.5) & (y < height - .5)
            & (error <= CYCLE_PX))
    target_x = np.clip(np.floor(x + .5).astype(int), 0, width - 1)
    target_y = np.clip(np.floor(y + .5).astype(int), 0, height - 1)
    return good, target_y, target_x, np.sqrt(np.maximum(overlap * opposite_overlap, 0))


def directed_evidence(source_masks, target_masks, edges, cycle):
    good, target_y, target_x, weights = cycle
    areas = source_masks.sum(axis=(1, 2))
    source_points = [np.where(mask & good) for mask in source_masks]
    coverage, counts = [], []
    for a, b in edges:
        ys, xs = source_points[a]
        inside = target_masks[b, target_y[ys, xs], target_x[ys, xs]]
        counts.append(int(inside.sum()))
        coverage.append(float(weights[ys[inside], xs[inside]].sum()) / areas[a] if areas[a] else 0.)
    return np.array(coverage), np.array(counts, dtype=int)


def edge_evidence(graph, outputs, region_masks):
    edges = [tuple(edge) for edge in graph['edges']]
    reverse_edges = [(b, a) for a, b in edges]
    cycles = {
        'AB': cycle_points(outputs['warp_AB'], outputs['warp_BA'], outputs['overlap_AB'], outputs['overlap_BA']),
        'BA': cycle_points(outputs['warp_BA'], outputs['warp_AB'], outputs['overlap_BA'], outputs['overlap_AB']),
    }
    result = {'edges': np.array(edges, dtype=int).reshape(-1, 2)}
    for lane in ('box', 'mask'):
        ab, count_ab = directed_evidence(region_masks['A'][lane], region_masks['B'][lane], edges, cycles['AB'])
        ba, count_ba = directed_evidence(region_masks['B'][lane], region_masks['A'][lane], reverse_edges, cycles['BA'])
        balanced = np.minimum(ab, ba)
        balanced[(count_ab < MIN_PIXELS) | (count_ba < MIN_PIXELS)] = 0
        result.update({lane: balanced, lane + '_AB': ab, lane + '_BA': ba,
                       lane + '_count_AB': count_ab, lane + '_count_BA': count_ba})
    for side in 'AB':
        result['mask_area_' + side] = region_masks[side]['mask'].sum(axis=(1, 2))
        result['mask_quality_' + side] = region_masks[side]['quality']
    return result


def main():
    assert not (HERE / 'plan.json').exists()
    smoke = read(HERE / 'smoke.json')
    assert smoke['passed'] and smoke['resolution'] == RESOLUTION
    original, graphs, _, _, historical = load_extension()
    assert len(original) == 97 and sorted(graphs) == list(range(1, 101))
    matched_budget = sum(len(links) for links in historical.values())
    sam_weights = model_file('sam2.1_s.pt')
    image_hashes = {str(path.relative_to(ROOT)): sha(path) for case in graphs for path in paths(case).values()}
    plan = {'cases': sorted(graphs), 'training': False, 'threshold_fitting': False,
            'input_image_sha256': image_hashes, 'source_inference_freeze_sha256': sha(EXTENSION / 'inference-freeze.json'),
            'original_inference_freeze_sha256': sha(EVALUATION / 'messytable-foundyou-native-retrieval/inference-freeze.json'),
            'romav2_revision': revision(VENDOR), 'dinov3_revision': revision(DINO), 'romav2_checkpoint_sha256': CHECKSUM,
            'sam_checkpoint_sha256': sha(sam_weights), 'smoke_sha256': sha(HERE / 'smoke.json'),
            'scripts': {name: sha(HERE / name) for name in ('roma_adapter.py', 'run_dense.py')},
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'opencv-python', 'ultralytics')},
            'resolution': RESOLUTION, 'device': 'mps', 'precision': 'float32',
            'dense_pass': 'One bidirectional RoMa v2 full-scene warp per original image pair, base resolution without the second high-resolution pass.',
            'lanes': ['box', 'mask'],
            'mask_rule': 'Frozen SAM2.1-small prompted with each detector box. Clip to that box. Missing masks give zero mask evidence.',
            'score_rule': 'Minimum of two directed confidence-weighted region coverage ratios. Weight is geometric mean of overlap and sampled reverse overlap. Require a valid round trip and a minimum sample count in both directions.',
            'max_round_trip_error_grid_px': CYCLE_PX, 'minimum_samples_per_direction': MIN_PIXELS,
            'comparison': 'Identical ASNet shortlist, row top1 and global greedy one-to-one with fixed proposal counts. Report forced zero-evidence winners.',
            'budgets': sorted(set([100, 200, 411, matched_budget])),
            'cutoff_control': 'Reuse original ASNet Hungarian assignments at distance 18.00142575056375.',
            'gate_control': 'Retain an ASNet cutoff link only if its SAM foreground dense score is at least 0.05. Fixed illustrative rule.',
            'gate_score': GATE}
    save(HERE / 'plan.json', plan)
    save(HERE / 'candidate-freeze.json', {str(case): sha(
        EXTENSION / f'case-{case:03}/graph.json' if case not in original else
        EVALUATION / f'messytable-foundyou-native-retrieval/graphs/case-{case:03}.json')
        for case in sorted(graphs)})
    (HERE / 'evidence').mkdir()
    from ultralytics import SAM
    segmenter = SAM(str(sam_weights))
    model, count = build_model('mps')
    assert count == smoke['checkpoint_entries_loaded']
    start = time.monotonic()
    for case in sorted(graphs):
        images = paths(case)
        outputs = predict(model, [input_tensor(images[side]) for side in 'AB'], 'mps')
        masks = {side: regions(images[side], graphs[case]['detections'][side], segmenter) for side in 'AB'}
        evidence = edge_evidence(graphs[case], outputs, masks)
        np.savez_compressed(HERE / 'evidence' / f'case-{case:03}.npz', **evidence)
        if case in (1, 100) or case % 5 == 0:
            print(f'{case}/100 pairs | {time.monotonic() - start:.1f}s | {len(evidence["edges"])} candidate edges',
                  flush=True)
    for name, digest in image_hashes.items():
        verified_member(ROOT, name, digest)
    files = [HERE / name for name in ('plan.json', 'candidate-freeze.json', 'smoke.json')]
    files.extend(sorted((HERE / 'evidence').glob('*.npz')))
    save(HERE / 'inference-freeze.json', {str(path.relative_to(HERE)): sha(path) for path in files})
    print('Dense scores frozen.', flush=True)


if __name__ == '__main__':
    main()
