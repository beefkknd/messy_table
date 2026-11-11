"""Evaluate a frozen sweep array with the original author metrics and solver."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/workspace')
HERE = ROOT / 'evaluation/messytable-official-reproduction'
SAMPLE = HERE / 'sample100'
OUT = SAMPLE / 'step2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    args = parser.parse_args()
    plan = json.loads((OUT / 'scores/plan.json').read_text())
    assert args.method in plan['methods']
    source = ROOT / 'evaluation/messytable-vit-p3de-100/assets/ViT-P3DE/src'
    sys.path.insert(0, str(source))
    import utils
    import KMSolver
    freeze = json.loads((OUT / 'scores/freeze.json').read_text())
    name = args.method + '.npy'
    assert sha(OUT / 'scores' / name) == freeze[name]
    assert sha(OUT / 'scores/plan.json') == freeze['plan.json']
    dist = np.load(OUT / 'scores' / name)
    truth = np.load(SAMPLE / 'vit/gt_inst.npy')
    pairs = json.loads((SAMPLE / 'vit/pair-index.json').read_text())
    assert dist.shape == truth.shape == (plan['candidates'], 1)
    start = time.monotonic()
    score = 1 - utils.scale_data(dist)
    ap = float(utils.cal_mAP(score, truth))
    fpr = float(utils.FPR_95(dist, truth))
    scale = float(np.percentile(dist, 95))
    assert scale > 0
    counts = {threshold: 0 for threshold in range(100, 40, -10)}
    accepted = {'correct': 0, 'wrong': 0, 'unmapped': 0}
    original_km = utils.apply_km
    active_truth = [None]

    def observed_km(matrix, threshold=.5):
        predicted = original_km(matrix, threshold)
        mapped = predicted.astype(bool)
        accepted['correct'] += int(np.sum(mapped & (active_truth[0] == 1)))
        accepted['wrong'] += int(np.sum(mapped & (active_truth[0] == 0)))
        return predicted

    utils.apply_km = observed_km
    scenes = {}
    for pair in pairs:
        lo, hi = pair['start'], pair['stop']
        active_truth[0] = truth[lo:hi].reshape(len(pair['main_bbox_id']), len(pair['sec_bbox_id']))
        sample = {'main_bbox_id': [str(i) for i in pair['main_bbox_id']],
                  'sec_bbox_id': [str(i) for i in pair['sec_bbox_id']],
                  'gt_inst': truth[lo:hi], 'gt_subcls': None}
        before = counts[100]
        utils.compute_IPAA(dist[lo:hi] / scale, sample, counts, .70)
        scene = scenes.setdefault(pair['scene'], {'setup': pair['setup'], 'difficulty': pair['difficulty'],
                                                  'image_pairs': 0, 'IPAA100_pairs': 0})
        scene['image_pairs'] += 1
        scene['IPAA100_pairs'] += counts[100] - before
    result = {
        'method': args.method, 'label': plan['method_labels'][args.method], 'training': False,
        'image_pairs': len(pairs), 'candidate_object_pairs': len(dist), 'positive_object_pairs': int(truth.sum()),
        'AP': ap, 'FPR95': fpr, 'IPAA100': counts[100] / len(pairs),
        'IPAA': {str(key): value / len(pairs) for key, value in counts.items()},
        'IPAA_counts': {str(key): value for key, value in counts.items()},
        'author_assignment': {**accepted, 'proposals': accepted['correct'] + accepted['wrong'],
                              'recall': accepted['correct'] / int(truth.sum())},
        'distance_p95': scale, 'km_threshold': .70,
        'metric_code_sha256': sha(source / 'utils.py'), 'native_solver_sha256': sha(Path(KMSolver.__file__)),
        'score_sha256': freeze[name], 'elapsed_seconds': time.monotonic() - start,
        'scene_outcomes': scenes,
    }
    destination = OUT / 'metrics' / (args.method + '.json')
    temporary = destination.with_suffix('.pending.json')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    print(args.method, round(100 * ap, 3), round(100 * fpr, 3), round(100 * result['IPAA100'], 3), flush=True)


if __name__ == '__main__':
    main()
