"""Run unchanged author metric functions and the shipped native Linux KMSolver."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/workspace')
HERE = ROOT / 'evaluation/messytable-official-reproduction'
REPOS = {
    'vit': ROOT / 'evaluation/messytable-vit-p3de-100/assets/ViT-P3DE',
    'asnet': ROOT / 'evaluation/messytable-asnet-ladder/assets/MessyTable',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('vit', 'asnet'), required=True)
    parser.add_argument('--selection', default='sample100')
    args = parser.parse_args()
    repo = REPOS[args.model]
    sys.path.insert(0, str(repo / 'src'))
    import utils
    import KMSolver
    folder = HERE / args.selection / args.model
    freeze = json.loads((folder / 'distance-freeze.json').read_text())
    for name, digest in freeze.items():
        assert sha(folder / name) == digest
    dist = np.load(folder / 'app_dist.npy')
    truth = np.load(folder / 'gt_inst.npy')
    pairs = json.loads((folder / 'pair-index.json').read_text())
    started = time.monotonic()
    score = 1 - utils.scale_data(dist)
    ap = float(utils.cal_mAP(score, truth))
    fpr = float(utils.FPR_95(dist, truth))
    # This global test-distribution normalization is part of the authors' IPAA evaluation.
    scale = float(np.percentile(dist, 95))
    counts = {threshold: 0 for threshold in range(100, 40, -10)}
    scene_counts = {}
    for index, pair in enumerate(pairs):
        lo, hi = pair['start'], pair['stop']
        sample = {
            'main_bbox_id': [str(value) for value in pair['main_bbox_id']],
            'sec_bbox_id': [str(value) for value in pair['sec_bbox_id']],
            'gt_inst': truth[lo:hi],
            'gt_subcls': None,
        }
        prior = counts.copy()
        utils.compute_IPAA(dist[lo:hi] / scale, sample, counts, .70)
        item = scene_counts.setdefault(pair['scene'], {'setup': pair['setup'], 'difficulty': pair['difficulty'],
                                                       'image_pairs': 0, 'IPAA100_pairs': 0})
        item['image_pairs'] += 1
        item['IPAA100_pairs'] += counts[100] - prior[100]
        if (index + 1) % 900 == 0:
            print(args.model, 'native metrics', index + 1, '/', len(pairs), flush=True)
    result = {
        'model': args.model, 'split': 'Official test.json, fixed100-scene sample',
        'training': False, 'image_pairs': len(pairs), 'candidate_object_pairs': len(dist),
        'positive_object_pairs': int(truth.sum()), 'AP': ap, 'FPR95': fpr,
        'IPAA': {str(key): value / len(pairs) for key, value in counts.items()},
        'IPAA_native_rounded': {str(key): value for key, value in utils.convert_IPAA(counts, len(pairs)).items()},
        'IPAA_counts': {str(key): value for key, value in counts.items()},
        'distance_p95': scale, 'km_threshold': .70, 'km_integer_scale': 1000,
        'metric_code_sha256': sha(repo / 'src/utils.py'), 'native_solver_sha256': sha(Path(KMSolver.__file__)),
        'numpy_version': np.__version__, 'runtime': sys.version,
        'distance_freeze_sha256': sha(folder / 'distance-freeze.json'),
        'elapsed_seconds': time.monotonic() - started,
        'scene_outcomes': scene_counts,
    }
    save(folder / 'native-metrics.json', result)
    print(args.model, {key: result[key] for key in ('AP', 'FPR95', 'IPAA_native_rounded', 'elapsed_seconds')}, flush=True)


if __name__ == '__main__':
    main()
