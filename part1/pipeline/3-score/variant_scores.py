"""Freeze Step 2 score arrays for the exact2,484,124 sampled object pairs."""
import ast
from itertools import combinations
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from variant_common import OLD, OUT, SAMPLE, freeze, inputs, read, save, sha, verify
from run_box_context import CONTEXT_WEIGHT, anchor_matches, box_distances
from run_epipolar import MAX_PENALTY, FREE_RESIDUAL, fundamental_matrix, epipolar_residual

LABELS = {}


def old_geometry():
    path = OLD / 'compare.py'
    parsed = ast.parse(path.read_text())
    names = {'scene_rotation', 'neighbor_adjustment', 'epipolar_gate'}
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = {
        'np': np, 'linear_sum_assignment': linear_sum_assignment, 'combinations': combinations,
        'anchor_matches': anchor_matches, 'box_distances': box_distances, 'CONTEXT_WEIGHT': CONTEXT_WEIGHT,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def methods(a, b, dense, perspective, cameras, geometry, boxes):
    result = {}

    def add(name, label, values):
        LABELS.setdefault(name, label)
        result[name] = np.asarray(values, dtype=np.float64)

    def cosine(name, other=None):
        return a[name] @ b[other or name].T

    names = ('tight_1x', 'context_125', 'context_150', 'context_200')
    scales = [cosine(name) for name in names]
    for name, value in zip(names, scales, strict=True):
        add('dino_' + name, 'DINOv2 ' + name.replace('_', ' '), value)
    add('dino_context_mean', 'DINOv2 four-crop mean', np.mean(scales, axis=0))
    add('dino_context_max', 'DINOv2 four-crop maximum', np.max(scales, axis=0))
    add('dino_context_two_support', 'DINOv2 two-crop support', np.sort(scales, axis=0)[-2])
    for name in ('gray', 'canny', 'grabcut', 'sam'):
        add('dino_' + name, 'DINOv2 ' + {'gray': 'grayscale', 'canny': 'Canny edges',
                                      'grabcut': 'GrabCut foreground', 'sam': 'SAM foreground'}[name], cosine(name))
    for name in ('tight_1x', 'sam'):
        stack = np.stack([cosine(name, name if angle == 0 else f'{name}_r{angle}')
                          for angle in (0, 90, 180, 270)])
        for index, angle in enumerate((90, 180, 270), 1):
            add(f'dino_{name}_rotation_{angle}', f'DINOv2 {name} fixed{angle}°', stack[index])
        add(f'dino_{name}_rotation_max', f'DINOv2 {name} best quarter-turn per candidate', stack.max(axis=0))
        for mode in ('appearance', 'anchor'):
            add(f'dino_{name}_rotation_scene_{mode}', f'DINOv2 {name} shared-view rotation by {mode}',
                stack[geometry['scene_rotation'](stack, mode)])
    shift = geometry['neighbor_adjustment'](scales[0], boxes)
    add('dino_anchor_context', 'DINOv2 anchor context', scales[0] + shift)
    add('dino_anchor_reject_only', 'DINOv2 anchors, reject only', scales[0] + np.minimum(shift, 0))
    f = fundamental_matrix(cameras['A'], cameras['B'])
    residual = epipolar_residual(boxes['A'], boxes['B'], f)
    penalty = MAX_PENALTY * np.clip(residual - FREE_RESIDUAL, 0, 1)
    allowed = geometry['epipolar_gate'](boxes, f)
    add('dino_epipolar_soft', 'DINOv2 soft camera geometry', scales[0] - penalty)
    for name in ('dino_tight_1x', 'dino_sam', 'dino_context_mean'):
        add(name + '_geometry_gate', LABELS[name] + ' + camera geometry',
            np.where(allowed, result[name], -1.0))
    sam = result['dino_sam']
    for name in ('unwarped', 'warped'):
        supported = np.fmax(sam, perspective[name])
        add('dino_sam_geometry_' + name,
            'DINOv2 SAM + geometry + ' + ('rectified surface' if name == 'warped' else 'native surface control'),
            np.where(allowed, supported, -1.0))
    for name, label in (('box', 'RoMa box coverage'), ('mask', 'RoMa + SAM foreground'),
                         ('box_mean', 'RoMa box coverage, symmetric mean'),
                         ('mask_mean', 'RoMa + SAM, symmetric mean')):
        add('roma_' + name, label, dense[name])
    return result


def main():
    selection, scenes, intrinsics = inputs()
    for folder in ('views', 'dense', 'perspective'):
        while not (OUT / folder / 'freeze.json').exists():
            time.sleep(2)
        verify(OUT / folder)
    geometry = old_geometry()
    baseline = read(SAMPLE / 'vit/pair-index.json')
    assert len(baseline) == selection['image_pairs']
    out = OUT / 'scores'
    out.mkdir(exist_ok=True)
    accumulated = {}
    cached_scene = None
    cached_views = {}
    embedding_names = ('tight_1x', 'context_125', 'context_150', 'context_200',
                       'gray', 'canny', 'grabcut', 'sam', 'tight_1x_r90', 'tight_1x_r180',
                       'tight_1x_r270', 'sam_r90', 'sam_r180', 'sam_r270')
    for index, pair in enumerate(baseline):
        scene = pair['scene']
        left, right = pair['camera_A'], pair['camera_B']
        stem = f'{scene}-{left}-{right}'
        if cached_scene != scene:
            cached_views = {}
            for camera in range(1, 10):
                with np.load(OUT / 'views' / f'{scene}-0{camera}.npz', allow_pickle=False) as archive:
                    cached_views[camera] = {name: archive[name] for name in embedding_names}
            cached_scene = scene
        aa, bb = cached_views[left], cached_views[right]
        with np.load(OUT / 'dense' / (stem + '.npz'), allow_pickle=False) as dense, \
             np.load(OUT / 'perspective' / (stem + '.npz'), allow_pickle=False) as perspective:
            camera = scenes[scene]['cameras']
            boxes = {side: np.array([item['pos'] for item in camera[str(cam)]['instances'].values()])
                     for side, cam in zip('AB', (left, right))}
            calibration = {side: {'camera_to_reference': camera[str(cam)]['extrinsics'],
                                  'intrinsics': intrinsics[str(cam)]}
                           for side, cam in zip('AB', (left, right))}
            values = methods(aa, bb, dense, perspective, calibration, geometry, boxes)
        shape = (len(pair['main_bbox_id']), len(pair['sec_bbox_id']))
        for name, matrix in values.items():
            assert matrix.shape == shape and np.isfinite(matrix).all(), (stem, name)
            accumulated.setdefault(name, []).append(matrix.reshape(-1))
        if (index + 1) % 900 == 0:
            print('Assembled scores', index + 1, '/3600', flush=True)
    scores = {name: np.concatenate(parts) for name, parts in accumulated.items()}
    for name in scores:
        assert len(scores[name]) == selection['candidate_object_pairs']
    del accumulated
    for model, label in (('asnet', 'ASNet reference'), ('vit', 'ViT-P3DE Tiny reference')):
        folder = SAMPLE / model
        verify(folder, 'distance-freeze.json')
        assert read(folder / 'pair-index.json') == baseline
        scores[model + '_reference'] = -np.load(folder / 'app_dist.npy').reshape(-1)
        LABELS[model + '_reference'] = label
    normalize = lambda value: rankdata(value, method='average') / len(value)
    dense_rank = normalize(scores['roma_mask'])
    for name in ('dino_context_mean', 'dino_sam', 'asnet_reference', 'vit_reference'):
        appearance_rank = normalize(scores[name])
        for weight in (.5, .75):
            suffix = 'equal' if weight == .5 else 'dense_first'
            key = 'blend_roma_' + name + '_' + suffix
            scores[key] = weight * dense_rank + (1 - weight) * appearance_rank
            LABELS[key] = f'RoMa + SAM / {LABELS[name]} ({int(weight*100)}/{int((1-weight)*100)})'
    files = []
    for name, values in scores.items():
        if name.endswith('_reference'):
            distance = -values
        elif name == 'dino_anchor_context':
            distance = 1 + CONTEXT_WEIGHT - values
        else:
            distance = 1 - values
        assert np.isfinite(distance).all() and (distance >= -1e-6).all(), name
        # Tiny cosine rounding can put an exact self-alignment a few ulps above1.
        distance = np.maximum(distance, 0).reshape(-1, 1)
        file = out / (name + '.npy')
        np.save(file, distance)
        files.append(file)
    plan = {
        'training': False, 'selection_sha256': sha(SAMPLE / 'selection.json'),
        'pair_index_sha256': sha(SAMPLE / 'vit/pair-index.json'),
        'script_sha256': sha(__file__), 'geometry_source_sha256': sha(OLD / 'compare.py'),
        'method_labels': LABELS, 'methods': list(scores),
        'candidates': selection['candidate_object_pairs'], 'image_pairs': 3600,
        'source_freezes': {name: sha(OUT / name / 'freeze.json') for name in ('views', 'dense', 'perspective')},
        'rank_by': 'AP descending,then FPR95 ascending,then IPAA100 descending. References remain controls.',
        'blend_rule': 'Original global percentile ranks,equal or75%RoMa. All weights frozen before this sample scoring.',
        'distance_map': 'References original distance,anchor-context1.05-similarity,others1-similarity,clip tiny negatives to0.',
        'geometry_rejection': 'Keep rejected edges in population with DINO similarity-1,distance2.',
    }
    save(out / 'plan.json', plan)
    files.append(out / 'plan.json')
    freeze(out, files)
    print('All', len(scores), 'score arrays frozen before metric evaluation.', flush=True)


if __name__ == '__main__':
    main()
