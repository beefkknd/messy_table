"""Independent count reconstruction without importing the evaluation pipeline."""
from pathlib import Path
from collections import Counter
from itertools import combinations
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def assign(detections, entries, minimum):
    ids = list(entries)
    if not detections or not ids:
        return set()
    a = np.array([x['box_xyxy'] for x in detections], dtype=float)
    b = np.array([entries[x]['pos'] for x in ids], dtype=float)
    intersection = np.prod(np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:])
                                   - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None), axis=2)
    aarea = np.prod(a[:, 2:]-a[:, :2], axis=1)
    barea = np.prod(b[:, 2:]-b[:, :2], axis=1)
    overlap = intersection/(aarea[:, None]+barea[None, :]-intersection)
    cost = np.where(overlap >= minimum, -overlap, 0.)
    rows, columns = linear_sum_assignment(cost)
    return {int(ids[j]) for i, j in zip(rows, columns) if overlap[i, j] >= minimum}


def main():
    frozen = read(HERE/'prediction-freeze.json')
    results = read(HERE/'results.json')
    assert digest(HERE/'prediction-freeze.json') == results['prediction_freeze_sha256']
    key = read(ROOT/'data/messytable-full/MessyTable/labels/test.json')['scenes']
    arms = list(results['arms'])
    count = {arm: Counter() for arm in arms}
    photo_ids = set()
    for filename, expected in frozen['scenes'].items():
        path = HERE/'predictions'/filename
        assert digest(path) == expected
        scene = read(path)
        sid = scene['id']
        visible = {c: set(map(int, v['instances'])) for c, v in key[sid]['cameras'].items()}
        found = {arm: {} for arm in arms}
        for camera, v in scene['views'].items():
            assert (sid, camera) not in photo_ids
            photo_ids.add((sid, camera))
            entries = key[sid]['cameras'][camera]['instances']
            base = v['arms']['fine_tuned']
            for arm in ('fine_tuned_tiles', 'fine_tuned_tiles_G'):
                assert v['arms'][arm][:len(base)] == base
            for arm in arms:
                detections = v['arms'][arm]
                assert all(d['confidence'] > .25 for d in detections)
                hits = assign(detections, entries, .5)
                found[arm][camera] = hits
                count[arm].update(images=1, ground_truth=len(entries),
                                  detections=len(detections), localized=len(hits),
                                  localized_at_iou75=len(assign(detections, entries, .75)),
                                  unmapped=len(detections)-len(hits))
        for a, b in combinations(visible, 2):
            common = visible[a] & visible[b]
            for arm in arms:
                count[arm].update(shared_objects=len(common), view_pairs=1,
                                  shared_localized_both=len(common & found[arm][a] & found[arm][b]))
    for arm, actual in count.items():
        expected = results['arms'][arm]['counts']
        for field, value in actual.items():
            assert value == expected[field], (arm, field, value, expected[field])
    assert len(photo_ids) == 16749
    report = {'passed': True, 'unique_photos': len(photo_ids), 'scenes': len(frozen['scenes']),
              'method': 'Separate NumPy overlap computation and minimum-cost assignment; no pipeline imports',
              'counts': {a: dict(c) for a, c in count.items()},
              'verified': ['Image and view uniqueness', 'Prediction hashes', 'Strict confidence cutoff',
                           'Baseline preservation in raw and gated inventories',
                           'IoU50 and IoU75 localization counts', 'All-view-pair endpoint coverage'],
              'results_sha256': digest(HERE/'results.json'), 'code_sha256': digest(Path(__file__))}
    (HERE/'independent-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': True, 'unique_photos': len(photo_ids), 'counts': report['counts']}), flush=True)


if __name__ == '__main__':
    main()
