"""Frozen full-test localization evaluation. No training or identity matching.

prepare hashes images and locks the protocol; infer consumes only the image
manifest; score reads native test labels after prediction integrity freezes.
"""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import argparse
import hashlib
import json
import logging
import os
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROGRAM = ROOT / 'research/self-learning-program'
ENG = PROGRAM / 'engineering'
TILE_ENG = PROGRAM / 'step0-tiles/engineering'
sys.path.insert(0, str(ENG))
sys.path.insert(0, str(TILE_ENG))
from common import initialize_runtime, pin_ultralytics, sha256
initialize_runtime()
from tile_detector import windows, project, merge
from score_detector import view_score
from run_full_scene import map_detections
import cv2
import numpy as np
import torch
from ultralytics import YOLO

DATA = ROOT / 'data/messytable-full/MessyTable'
WEIGHTS = {
    'vanilla': ROOT / 'everyday-identity/assets/models/yolo26m.pt',
    'fine_tuned': ENG / 'round2/train/A/checkpoint.pt',
}
EXPECTED = {
    'vanilla': '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7',
    'fine_tuned': '3f278134b873b122f6ccc75216f3156dc8be72e882bd1f3da06c2f9a129e2ac3',
}
ARMS = ['vanilla', 'fine_tuned', 'fine_tuned_tiles', 'fine_tuned_tiles_G']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def prepare():
    assert not (HERE/'protocol.json').exists(), 'Existing protocol must be preserved'
    native = read(DATA/'labels/test.json')['scenes']
    scenes = []
    for index, (sid, scene) in enumerate(sorted(native.items())):
        views = []
        for camera, view in sorted(scene['cameras'].items(), key=lambda x: int(x[0])):
            path = DATA/'images'/view['pathname']
            views.append({'id': camera, 'path': str(path.relative_to(ROOT)),
                          'sha256': sha256(path), 'width': 1920, 'height': 1080})
        assert len(views) == 9
        scenes.append({'id': sid, 'setup': sid.rsplit('-', 1)[0], 'views': views})
        if (index+1) % 200 == 0:
            emit({'phase': 'prepare', 'hashed_scenes': index+1, 'total_scenes': len(native)})
    manifest = {'scope': 'All scenes and cameras in the official native test split', 'scenes': scenes}
    save(HERE/'images.json', manifest)
    setup_ids = {s['setup'] for s in scenes}
    training = read(DATA/'labels/train.json')['scenes']
    assert not setup_ids & {s.rsplit('-', 1)[0] for s in training}
    assert len(scenes) == 1861 and sum(len(s['views']) for s in scenes) == 16749
    for arm, path in WEIGHTS.items():
        assert sha256(path) == EXPECTED[arm], f'Checkpoint changed: {arm}'
    pin_ultralytics()
    protocol = {
        'scope': 'Part Two full official test: fixed inference and localization only',
        'training': False, 'identity_matching': False,
        'created_unix': time.time(),
        'population': {'scenes': len(scenes), 'setups': len(setup_ids), 'photos': 16749,
                       'downloaded_photos': 50211, 'fraction_of_all_downloaded_photos': 16749/50211,
                       'fraction_of_official_test': 1.0, 'ground_truth_appearances': 408937},
        'image_manifest_sha256': sha256(HERE/'images.json'),
        'native_test_sha256': sha256(DATA/'labels/test.json'),
        'weights_sha256': EXPECTED,
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [
            Path(__file__), TILE_ENG/'tile_detector.py', ENG/'score_detector.py',
            ROOT/'evaluation/run_full_scene.py']},
        'software': {'torch': str(torch.__version__), 'ultralytics': '8.4.202', 'device': 'mps'},
        'prediction': {'imgsz': 768, 'confidence': 'strictly greater than .25',
                       'full_frame_raw_floor': .001, 'tile_raw_floor': .25,
                       'head': 'one2many', 'nms_iou': .7, 'max_det_per_observation': 300,
                       'half': False, 'rect': True, 'full_frame_batch': 6, 'tile_batch': 6},
        'tile': {'size': 768, 'stride': 576, 'count_per_image': 6,
                 'edge_margin': 4, 'deduplication_iou': .55,
                 'base_policy': 'Keep original full-frame boxes; append confidence-sorted, nonduplicate, nonedge tile observations'},
        'extra_G_gate': {'confidence_min': .8, 'min_short_side': 8, 'max_aspect_ratio': 12,
                         'max_frame_area_fraction': .25, 'min_artificial_edge_clearance': 12,
                         'true_frame_edges': 'exempt', 'baseline_boxes': 'always retained'},
        'arms': ARMS,
        'scoring': {'localization': 'class-agnostic one-to-one maximum-IoU at >=.50 and >=.75',
                    'both_view_coverage': 'all 36 unordered camera pairs in every scene, scene-local identity IDs',
                    'uncertainty': 'paired bootstrap over 188 setups; 20000 replicates; one training seed',
                    'threshold_selection': 'none; all settings reused from historical work',
                    'scope_warning': 'Whole native test now exposed; setup separation does not guarantee novel product identities'},
    }
    save(HERE/'protocol.json', protocol)
    save(HERE/'protocol-freeze.json', {'protocol.json': sha256(HERE/'protocol.json'),
                                       'images.json': sha256(HERE/'images.json')})
    emit({'phase': 'prepared', **protocol['population']})


def require_protocol():
    for name, expected in read(HERE/'protocol-freeze.json').items():
        assert sha256(HERE/name) == expected, name
    protocol = read(HERE/'protocol.json')
    for path, expected in protocol['code_sha256'].items():
        assert sha256(ROOT/path) == expected, path
    return protocol


def predict(model, images, floor):
    result = model.predict(images, imgsz=768, conf=floor, iou=.7, max_det=300,
                           device='mps', half=False, rect=True, verbose=False, nms=True)
    assert not model.predictor.model.end2end
    assert next(model.model.parameters()).dtype == torch.float32
    torch.mps.synchronize()
    return [[{'box_xyxy': box, 'class': r.names[int(cls)], 'confidence': float(score)}
             for box, cls, score in zip(r.boxes.xyxy.cpu().numpy().round().astype(int).tolist(),
                                       r.boxes.cls.tolist(), r.boxes.conf.tolist()) if score > .25]
            for r in result]


def batches(model, images, floor):
    rows = []
    started = time.monotonic()
    for first in range(0, len(images), 6):
        rows.extend(predict(model, images[first:first+6], floor))
    return rows, time.monotonic()-started


def gate_G(det, width, height):
    if det['confidence'] < .8:
        return False
    x1, y1, x2, y2 = det['box_xyxy']
    short, long = sorted((x2-x1, y2-y1))
    if short < 8 or long/short > 12 or short*long/(width*height) > .25:
        return False
    left, top, right, bottom = det['source_tile_xyxy']
    distances = []
    if left > 0:
        distances.append(x1-left)
    if top > 0:
        distances.append(y1-top)
    if right < width:
        distances.append(right-x2)
    if bottom < height:
        distances.append(bottom-y2)
    return not distances or min(distances) >= 12


def tiles_for_view(model, image, base):
    height, width = image.shape[:2]
    regions = windows(width, height)
    assert len(regions) == 6
    pixels = [image[t:b, l:r] for l, t, r, b in regions]
    outputs, seconds = batches(model, pixels, .25)
    tiles = []
    for tile_index, (region, detections) in enumerate(zip(regions, outputs, strict=True)):
        rows = []
        for det in detections:
            mapped, decision = project(det, region, width, height)
            mapped = {**mapped, 'source_tile_index': tile_index, 'source_tile_xyxy': list(region)}
            rows.append({'local': det, 'mapped': mapped, 'decision': decision})
        tiles.append({'xyxy': list(region), 'detections': rows})
    raw = merge(base, tiles)
    gated = [*base, *(d for d in raw[len(base):] if gate_G(d, width, height))]
    assert raw[:len(base)] == gated[:len(base)] == base
    return raw, gated, tiles, seconds


def replay(models):
    """Check the accelerated execution on archived historical inputs before expansion."""
    cohort = read(PROGRAM/'data/cohorts/original100.json')['scenes'][:3]
    paths = [ROOT/v['path'] for scene in cohort for v in scene['views']]
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for path in paths]
    source_runs = {'vanilla': ENG/'inference/pretrained768/original100/predictions.json',
                   'fine_tuned': ENG/'round2/inference/A/original100/predictions.json'}
    records = []
    for arm, model in models.items():
        output, _ = batches(model, images, .001)
        old = read(source_runs[arm])['scenes'][:3]
        previous = [[d for d in case['views'][view['id']]['detections'] if d['confidence'] > .25]
                    for case, scene in zip(old, cohort, strict=True) for view in scene['views']]
        geometry = [[(d['box_xyxy'], d['class']) for d in x] for x in output] == [
                    [(d['box_xyxy'], d['class']) for d in x] for x in previous]
        max_drift = max((abs(a['confidence']-b['confidence']) for x, y in zip(output, previous)
                         for a, b in zip(x, y)), default=0) if geometry else None
        record = {'arm': arm, 'frames': 6, 'geometry_and_order_identical': geometry,
                  'max_confidence_drift': max_drift, 'passed': geometry and max_drift <= 1e-5}
        records.append(record)
        assert record['passed'], f'Batch replay failed: {record}'
    old_tiles = read(PROGRAM/'step0-tiles/engineering/inference/original100/predictions.json')['scenes'][:3]
    for index, image in enumerate(images):
        base = [d for d in previous[index]]
        actual, _, _, _ = tiles_for_view(models['fine_tuned'], image, base)
        scene_index, view_index = divmod(index, 2)
        view_id = cohort[scene_index]['views'][view_index]['id']
        expected = old_tiles[scene_index]['views'][view_id]['detections']
        a = [(d['box_xyxy'], d['class']) for d in actual]
        b = [(d['box_xyxy'], d['class']) for d in expected]
        if a != b:
            # Batching can change rounding by a pixel; preserve this diagnostic.
            records.append({'arm': 'tile_replay', 'frame': index, 'exact_box_order': False,
                            'actual_count': len(a), 'historical_count': len(b),
                            'note': 'Batch floating-point sensitivity; no threshold was changed'})
        else:
            records.append({'arm': 'tile_replay', 'frame': index, 'exact_box_order': True,
                            'actual_count': len(a), 'historical_count': len(b)})
    save(HERE/'historical-replay.json', records)
    emit({'phase': 'historical_replay', 'records': records})


def infer():
    protocol = require_protocol()
    pin_ultralytics()
    assert torch.backends.mps.is_available()
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    torch.set_num_threads(8)
    models = {}
    for arm, path in WEIGHTS.items():
        assert sha256(path) == EXPECTED[arm]
        models[arm] = YOLO(str(path))
    replay(models)
    scene_rows = read(HERE/'images.json')['scenes']
    output = HERE/'predictions'
    output.mkdir(exist_ok=True)
    started = time.monotonic()
    existing = 0
    for index, scene in enumerate(scene_rows):
        destination = output/(scene['id']+'.json')
        if destination.exists():
            record = read(destination)
            assert record['protocol_sha256'] == sha256(HERE/'protocol.json')
            assert record['id'] == scene['id']
            existing += 1
            continue
        images = []
        for view in scene['views']:
            path = ROOT/view['path']
            assert sha256(path) == view['sha256'], path
            image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            assert image is not None and image.shape == (1080, 1920, 3)
            images.append(image)
        whole = {}
        runtime = {}
        for arm, model in models.items():
            whole[arm], runtime[arm] = batches(model, images, .001)
        views = {}
        for offset, (view, image) in enumerate(zip(scene['views'], images, strict=True)):
            base = whole['fine_tuned'][offset]
            raw, gated, provenance, tile_seconds = tiles_for_view(models['fine_tuned'], image, base)
            views[view['id']] = {
                'sha256': view['sha256'],
                'arms': {'vanilla': whole['vanilla'][offset], 'fine_tuned': base,
                         'fine_tuned_tiles': raw, 'fine_tuned_tiles_G': gated},
                'tile_observations': provenance, 'tile_seconds': tile_seconds,
            }
        save(destination, {'id': scene['id'], 'setup': scene['setup'],
                           'protocol_sha256': sha256(HERE/'protocol.json'), 'views': views,
                           'full_frame_seconds': runtime,
                           'tile_seconds': sum(v['tile_seconds'] for v in views.values())})
        if (index+1) % 25 == 0 or index+1 == len(scene_rows):
            done = index+1-existing
            elapsed = time.monotonic()-started
            emit({'phase': 'inference', 'scenes': index+1, 'total': len(scene_rows),
                  'photos': (index+1)*9, 'elapsed_seconds_this_session': round(elapsed, 1),
                  'estimated_seconds_remaining': round(elapsed/max(done, 1)*(len(scene_rows)-index-1), 1)})
    files = {path.name: sha256(path) for path in sorted(output.glob('*.json'))}
    assert len(files) == len(scene_rows)
    save(HERE/'prediction-freeze.json', {'protocol_sha256': sha256(HERE/'protocol.json'),
                                        'manifest_sha256': sha256(HERE/'images.json'),
                                        'created_unix': time.time(), 'scenes': files})
    emit({'phase': 'frozen', 'scenes': len(files), 'seconds_this_session': time.monotonic()-started})


def ratios(c):
    return {'detection_recall': c['localized']/c['ground_truth'],
            'detection_precision': c['localized']/c['detections'],
            'detection_recall_iou75': c['localized_at_iou75']/c['ground_truth'],
            'both_view_coverage': c['shared_localized_both']/c['shared_objects'],
            'unmapped_per_photo': c['unmapped']/c['images']}


def score():
    protocol = require_protocol()
    freeze = read(HERE/'prediction-freeze.json')
    assert freeze['protocol_sha256'] == sha256(HERE/'protocol.json')
    manifest = read(HERE/'images.json')['scenes']
    assert len(freeze['scenes']) == len(manifest)
    for name, expected in freeze['scenes'].items():
        assert sha256(HERE/'predictions'/name) == expected
    assert sha256(DATA/'labels/test.json') == protocol['native_test_sha256']
    labels = read(DATA/'labels/test.json')['scenes']
    difficulties = {sid: name for name in ('easy', 'medium', 'hard')
                    for sid in read(DATA/f'labels/test_{name}.json')['scenes']}
    totals = {arm: Counter() for arm in ARMS}
    setups = {arm: defaultdict(Counter) for arm in ARMS}
    strata = {arm: {'difficulty': defaultdict(Counter), 'camera': defaultdict(Counter),
                    'short_side': defaultdict(Counter)} for arm in ARMS}
    scene_results = []
    marginal = {arm: Counter() for arm in ('fine_tuned_tiles', 'fine_tuned_tiles_G')}
    for index, scene in enumerate(manifest):
        sid = scene['id']
        pred = read(HERE/'predictions'/(sid+'.json'))
        assert pred['id'] == sid and pred['setup'] == scene['setup']
        per_scene = {arm: Counter(images=9, scenes=1) for arm in ARMS}
        matched = {arm: {} for arm in ARMS}
        annotated = {}
        view_results = {}
        for view in scene['views']:
            camera = view['id']
            raw = pred['views'][camera]
            truth = {oid: obj['pos'] for oid, obj in labels[sid]['cameras'][camera]['instances'].items()}
            truth_view = {'objects_xyxy': truth}
            view_results[camera] = {}
            for arm in ARMS:
                detections = raw['arms'][arm]
                counts, ids, all_ids = view_score({'sha256': raw['sha256'], 'detections': detections},
                                                 truth_view, view, .25)
                per_scene[arm].update(counts)
                matched[arm][camera] = ids
                annotated[camera] = all_ids
                counts['images'] = 1
                strata[arm]['difficulty'][difficulties[sid]].update(counts)
                strata[arm]['camera'][camera].update(counts)
                for oid, b in truth.items():
                    short = min(b[2]-b[0], b[3]-b[1])
                    size = 'lt32' if short < 32 else '32_to_lt96' if short < 96 else 'ge96'
                    strata[arm]['short_side'][size]['ground_truth'] += 1
                    strata[arm]['short_side'][size]['localized'] += int(oid) in ids
                view_results[camera][arm] = {'counts': counts, 'localized_ids': sorted(ids)}
            truth_xywh = {oid: [b[0], b[1], b[2]-b[0], b[3]-b[1]] for oid, b in truth.items()}
            base_ids = matched['fine_tuned'][camera]
            for arm in marginal:
                count = len(raw['arms']['fine_tuned'])
                extras = raw['arms'][arm][count:]
                marginal[arm].update(extras=len(extras),
                    newly_covered=len(matched[arm][camera]-base_ids),
                    previously_covered_lost=len(base_ids-matched[arm][camera]))
                uncovered = {oid: box for oid, box in truth_xywh.items() if int(oid) not in base_ids}
                marginal[arm]['extras_mapping_to_previously_uncovered'] += len(map_detections(extras, uncovered))
        for a, b in combinations(annotated, 2):
            shared = annotated[a] & annotated[b]
            for arm in ARMS:
                both = shared & matched[arm][a] & matched[arm][b]
                per_scene[arm].update(view_pairs=1, shared_objects=len(shared),
                                      shared_localized_both=len(both),
                                      zero_overlap_pairs=int(not shared))
        for arm in ARMS:
            totals[arm].update(per_scene[arm])
            setups[arm][scene['setup']].update(per_scene[arm])
        scene_results.append({'scene': sid, 'setup': scene['setup'], 'difficulty': difficulties[sid],
                              'counts': {arm: dict(c) for arm, c in per_scene.items()}, 'views': view_results})
        if (index+1) % 250 == 0:
            emit({'phase': 'scoring', 'scenes': index+1, 'total': len(manifest)})
    for arm, c in totals.items():
        assert c['ground_truth'] == 408937 and c['images'] == 16749
        assert c['view_pairs'] == 66996 and c['localized']+c['unmapped'] == c['detections']
        assert len(setups[arm]) == 188
    setup_ids = sorted(setups['vanilla'])
    keys = ['localized', 'ground_truth', 'detections', 'shared_localized_both', 'shared_objects', 'unmapped']
    arrays = {arm: np.array([[setups[arm][s][key] for key in keys] for s in setup_ids], dtype=float)
              for arm in ARMS}
    rng = np.random.default_rng(20261111)
    draws = rng.integers(0, len(setup_ids), size=(20000, len(setup_ids)))
    pooled = {arm: a[draws].sum(axis=1) for arm, a in arrays.items()}
    contrasts = {}
    for arm, reference in [('fine_tuned', 'vanilla'), ('fine_tuned_tiles', 'fine_tuned'),
                           ('fine_tuned_tiles_G', 'fine_tuned')]:
        contrasts[arm+'_minus_'+reference] = {}
        for name, num, den in [('detection_recall', 0, 1), ('both_view_coverage', 3, 4),
                               ('detection_precision', 0, 2)]:
            sample = pooled[arm][:, num]/pooled[arm][:, den] - pooled[reference][:, num]/pooled[reference][:, den]
            contrasts[arm+'_minus_'+reference][name] = {
                'difference_percentage_points': 100*(ratios(totals[arm])[name]-ratios(totals[reference])[name]),
                'setup_bootstrap_ci95_percentage_points': (100*np.quantile(sample, [.025, .975])).tolist()}
    results = {'scope': protocol['population'], 'protocol_sha256': sha256(HERE/'protocol.json'),
               'prediction_freeze_sha256': sha256(HERE/'prediction-freeze.json'),
               'checkpoint_hashes': EXPECTED,
               'arms': {arm: {'counts': dict(c), 'metrics': ratios(c)} for arm, c in totals.items()},
               'marginal_tile_diagnostics': {a: dict(c) for a, c in marginal.items()},
               'paired_setup_bootstrap': contrasts, 'strata': strata,
               'runtime': {'note': 'Prediction timings include per-batch MPS synchronization; source decoding and artifact writes are separate.'},
               'limits': ['Detection only; no new training or identity-matching inference.',
                          'Same physical products may recur in train and test setups.',
                          'Fixed threshold comparison does not equalize prediction or error budgets.',
                          'Full official test is exposed by this evaluation.',
                          'Bootstrap reflects resampled setups, not different training seeds.']}
    save(HERE/'results.json', results)
    save(HERE/'per-scene-results.json', scene_results)
    save(HERE/'scoring-freeze.json', {n: sha256(HERE/n) for n in ['results.json', 'per-scene-results.json']})
    emit({'phase': 'complete', 'arms': results['arms'], 'marginal': results['marginal_tile_diagnostics']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'infer', 'score'])
    args = parser.parse_args()
    {'prepare': prepare, 'infer': infer, 'score': score}[args.phase]()
