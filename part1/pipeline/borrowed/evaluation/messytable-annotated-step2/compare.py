"""Rank each frozen Step 2 variant, then score the same four proposal budgets."""
from collections import defaultdict
from itertools import combinations
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from inputs import BUDGETS, EVAL, HERE, ROOT, VIT, graphs
from run_box_context import CONTEXT_WEIGHT, anchor_matches, box_distances
from run_epipolar import FREE_RESIDUAL, MAX_PENALTY, epipolar_residual, fundamental_matrix
from run_messytable_dino_crops import BRANCHES
from score_extension import score_methods
from score_foundyou_native_retrieval import global_greedy, prediction_rows, read, save, sha, verified_member

FUSION = EVAL / 'messytable-roma-asnet-fusion-100'
sys.path.insert(0, str(FUSION))
from compare_fusion import DENSE_WEIGHTS, anchor_and_fill, percentile_scores

CAMERAS = ROOT / 'data/messytable-100/camera-calibration.json'
ROTATIONS = (0, 90, 180, 270)
EXTENDED_BUDGETS = (800, 1200, 1600)


def verify_folder(folder):
    freeze = folder / 'inference-freeze.json'
    for path, digest in read(freeze).items():
        verified_member(folder, path, digest)
    return sha(freeze)


def scene_rotation(stack, mode):
    if mode == 'appearance':
        objectives = []
        for matrix in stack:
            excess = np.maximum(matrix - .70, 0)
            rows, cols = linear_sum_assignment(excess, maximize=True)
            objectives.append(excess[rows, cols].sum())
        return int(np.argmax(objectives))
    anchors = [(a, b) for a, b in anchor_matches(stack[0]) if stack[0, a, b] >= .75]
    if len(anchors) < 2:
        return 0
    differences = [stack[:, a, b] - stack[0, a, b] for a, b in anchors]
    return int(np.argmax(np.median(differences, axis=0)))


def neighbor_adjustment(matrix, boxes):
    anchors = anchor_matches(matrix)
    delta = np.zeros_like(matrix)
    if len(anchors) < 2:
        return delta
    da, db = (box_distances(boxes[side]) for side in 'AB')
    for a, b in zip(*np.where(matrix > .65)):
        references = [(i, j) for i, j in anchors if i != a and j != b]
        votes = [np.sign(da[a, i] - da[a, k]) * np.sign(db[b, j] - db[b, l])
                 for (i, j), (k, l) in combinations(references, 2)]
        nonzero = [vote for vote in votes if vote]
        if nonzero:
            delta[a, b] = CONTEXT_WEIGHT * np.mean(nonzero)
    return delta


def epipolar_gate(boxes, fundamental):
    # The previous viewpoint test expands each box, then checks the sign over all corner pairs.
    # A bilinear form takes its extrema over the two rectangles at their corners.
    points = {}
    for side, rect in boxes.items():
        pad = np.maximum(16., .10 * np.linalg.norm(rect[:, 2:] - rect[:, :2], axis=1))
        x1, y1 = (rect[:, :2] - pad[:, None]).T
        x2, y2 = (rect[:, 2:] + pad[:, None]).T
        ones = np.ones_like(x1)
        points[side] = np.stack([(x1, y1, ones), (x2, y1, ones),
                                 (x2, y2, ones), (x1, y2, ones)], axis=1).transpose(2, 1, 0)
    f = fundamental / np.linalg.norm(fundamental)
    values = np.einsum('aij,jk,blk->abil', points['A'], f.T, points['B'])
    return (values.min(axis=(2, 3)) <= 1e-10) & (values.max(axis=(2, 3)) >= -1e-10)


def appearance_matrices(features, boxes, cameras):
    def dino(name, other=None):
        return features[f'A_{name}'] @ features[f'B_{other or name}'].T

    crops = [dino(name) for name in BRANCHES]
    result = {f'dino_{name}': scores for name, scores in zip(BRANCHES, crops, strict=True)}
    tight = crops[0]
    result['dino_context_mean'] = np.mean(crops, axis=0)
    result['dino_context_max'] = np.max(crops, axis=0)
    result['dino_context_two_support'] = np.sort(crops, axis=0)[-2]
    for name in ('gray', 'canny', 'grabcut', 'sam'):
        result['dino_' + name] = dino(name)
    for name in ('tight_1x', 'sam'):
        stack = np.stack([dino(name, name if angle == 0 else f'{name}_r{angle}')
                          for angle in ROTATIONS])
        result[f'dino_{name}_rotate_max'] = stack.max(axis=0)
        for index, angle in enumerate(ROTATIONS[1:], 1):
            result[f'dino_{name}_rotate_{angle}'] = stack[index]
        for mode in ('appearance', 'anchor'):
            result[f'dino_{name}_rotate_scene_{mode}'] = stack[scene_rotation(stack, mode)]
    shift = neighbor_adjustment(tight, boxes)
    result['dino_anchor_context'] = tight + shift
    result['dino_anchor_reject_only'] = tight + np.minimum(shift, 0)
    result['asnet_2x'] = features['asnet']
    if cameras:
        f = fundamental_matrix(cameras['A'], cameras['B'])
        residual = epipolar_residual(boxes['A'], boxes['B'], f)
        penalty = MAX_PENALTY * np.clip(residual - FREE_RESIDUAL, 0, 1)
        allowed = epipolar_gate(boxes, f)
    else:
        penalty, allowed = 0, True
    result['dino_epipolar_soft'] = tight - penalty
    for name in ('dino_tight_1x', 'dino_sam', 'asnet_2x'):
        result[name + '_epipolar_gate'] = np.where(allowed, result[name], np.nan)
    return result


def crop_ladder(matrices, branches):
    # Keep the prior 0.70 floor and one-to-one Hungarian assignment for each rung.
    # Rank within a rung by score, and use the final crop as a fallback if needed.
    remaining = {case: np.ones(matrix.shape, dtype=bool)
                 for case, matrix in matrices['dino_tight_1x'].items()}
    ranked = []
    for branch in branches:
        lane = {case: np.where(remaining[case], matrix, -np.inf)
                for case, matrix in matrices[f'dino_{branch}'].items()}
        for link in hungarian_rank(lane, .70):
            ranked.append({**link, 'stage': branch})
            free = remaining[link['case']]
            free[link['A'], :] = False
            free[:, link['B']] = False
    last = {case: np.where(remaining[case], matrix, np.nan)
            for case, matrix in matrices[f'dino_{branches[-1]}'].items()}
    return ranked + global_greedy(last)


def hungarian_rank(matrices, floor):
    ranked = []
    for case, matrix in matrices.items():
        rows, cols = linear_sum_assignment(np.maximum(matrix - floor, 0), maximize=True)
        ranked.extend({'case': case, 'A': int(a), 'B': int(b), 'score': float(matrix[a, b])}
                      for a, b in zip(rows, cols) if matrix[a, b] > floor)
    return sorted(ranked, key=lambda edge: (-edge['score'], edge['case'], edge['A'], edge['B']))


def load_matrices(inputs):
    matrices = defaultdict(dict)
    cameras = read(CAMERAS)
    assert len(cameras) == 80
    with np.load(VIT / 'annotated-embeddings.npz', allow_pickle=False) as vit:
        for case, graph in sorted(inputs.items()):
            boxes = {side: np.asarray([d['box_xyxy'] for d in ds], dtype=float)
                     for side, ds in graph['detections'].items()}
            shape = (len(boxes['A']), len(boxes['B']))
            with np.load(HERE / 'appearance' / f'case-{case:03}.npz', allow_pickle=False) as f:
                lanes = appearance_matrices(f, boxes, cameras.get(str(case)))
            with np.load(HERE / 'roma' / f'case-{case:03}.npz', allow_pickle=False) as f:
                assert np.array_equal(f['edges'], np.indices(shape).reshape(2, -1).T)
                for name in ('box', 'mask'):
                    lanes[f'roma_{name}'] = f[name].reshape(shape)
                    lanes[f'roma_{name}_symmetric_mean'] = np.where(
                        (f[name + '_count_AB'] >= 8) & (f[name + '_count_BA'] >= 8),
                        .5 * (f[name + '_AB'] + f[name + '_BA']), 0).reshape(shape)
                    lanes[f'roma_{name}_forward'] = np.where(
                        f[name + '_count_AB'] >= 8, f[name + '_AB'], 0).reshape(shape)
            allowed = True
            if str(case) in cameras:
                cal = cameras[str(case)]
                f = fundamental_matrix(cal['A'], cal['B'])
                allowed = epipolar_gate(boxes, f)
                lanes['roma_mask_epipolar_gate'] = np.where(allowed, lanes['roma_mask'], np.nan)
            else:
                lanes['roma_mask_epipolar_gate'] = lanes['roma_mask']
            with np.load(HERE / 'perspective' / f'case-{case:03}.npz', allow_pickle=False) as f:
                sam = lanes['dino_sam']
                local = f['local']
                assert np.array_equal(np.isfinite(f['warped']), np.isfinite(f['unwarped']))
                lanes['sift_foreground_inliers'] = np.where(allowed, local, np.nan)
                for lane in ('unwarped', 'warped'):
                    enhanced = np.fmax(sam, f[lane])
                    lanes['dino_sam_epipolar_' + lane] = np.where(allowed, enhanced, np.nan)
            va, vb = (vit[f'{case:03}_{side}'] for side in 'AB')
            lanes['vit_control'] = -np.linalg.norm(va[:, None, :] - vb[None, :, :] + 1e-6, axis=-1)
            for name, matrix in lanes.items():
                assert matrix.shape == shape and np.isfinite(matrix).any()
                matrices[name][case] = matrix
    return matrices


def build_rankings(matrices):
    ranks = {name: global_greedy(lane) for name, lane in matrices.items()}
    for count in range(2, len(BRANCHES) + 1):
        end = BRANCHES[count - 1].rsplit('_', 1)[1]
        ranks[f'dino_context_ladder_{end}'] = crop_ladder(matrices, BRANCHES[:count])
    local_ranks = [link for link in ranks['sift_foreground_inliers'] if link['score'] > 0]
    ranks['sift_foreground_inliers'] = local_ranks
    free = {case: lane.copy() for case, lane in matrices['dino_sam_epipolar_gate'].items()}
    for link in local_ranks:
        free[link['case']][link['A'], :] = np.nan
        free[link['case']][:, link['B']] = np.nan
    ranks['sift_foreground_anchors_dino_sam_fill'] = local_ranks + global_greedy(free)
    ranks['dino_hungarian_070'] = hungarian_rank(matrices['dino_tight_1x'], .70)
    ranks['asnet_hungarian_original_cutoff'] = hungarian_rank(matrices['asnet_2x'], -18.00142575056375)
    dense = percentile_scores(matrices['roma_mask'])
    for appearance in ('dino_tight_1x', 'dino_context_mean', 'dino_sam', 'asnet_2x'):
        similarity = percentile_scores(matrices[appearance])
        for name, weight in DENSE_WEIGHTS.items():
            ranks[f'fusion_{appearance}_{name}'] = global_greedy({
                case: (1 - weight) * similarity[case] + weight * dense[case] for case in dense
            })
        progressive = anchor_and_fill(ranks['roma_mask'], matrices[appearance])
        ranks[f'fusion_roma_anchors_{appearance}_fill'] = [
            {**link, 'evidence': appearance + '_fill'} if link['evidence'] == 'asnet_fill' else link
            for link in progressive
        ]
    return ranks


def budget_predictions(inputs, ranks):
    for name, ranked in ranks.items():
        for budget in (*BUDGETS, *EXTENDED_BUDGETS):
            if len(ranked) < budget:
                continue
            links = defaultdict(list)
            for edge in ranked[:budget]:
                links[edge['case']].append({k: v for k, v in edge.items() if k != 'case'})
            for case_links in links.values():
                assert all(len({link[side] for link in case_links}) == len(case_links) for side in 'AB')
            yield f'{name}__B{budget}', prediction_rows(inputs, links)


def correct_ids(scored):
    return {(row['case'], link['vehicle_A']) for row in scored for link in row['links']
            if link['status'] == 'correct'}


def main():
    inputs = graphs()
    freeze = {name: verify_folder(HERE / name) for name in ('appearance', 'roma', 'perspective')}
    vit_freeze = read(VIT / 'prediction-freeze.json')
    for name, digest in vit_freeze.items():
        verified_member(VIT, name, digest)
    plan = {
        'training': False, 'cases': sorted(inputs), 'budgets': list(BUDGETS),
        'extended_budgets': list(EXTENDED_BUDGETS),
        'inputs': 'Same 4348 annotated boxes and 58351 candidate edges as the ViT all-pairs arm.',
        'decoder': 'Global score sort, one-to-one greedy per case, with explicit exceptions for the original cutoff assignment and crop ladder.',
        'dino': 'Frozen DINOv2-small CLS and cosine. Original crop/edge/mask/rotation preprocessing and fixed geometry and appearance-anchor rules.',
        'roma': 'Frozen bidirectional RoMa v2. Original 640-grid cycle/sample test; box and SAM coverage, min/mean and forward-only aggregation.',
        'asnet': 'Published ASNet with 2x crops and original distance.',
        'fusion': 'Previously used global percentile blends (50% and 75% RoMa foreground) and 200 positive RoMa anchors followed by appearance fills.',
        'geometry': 'Original 80 calibrated pairs only. Missing calibration leaves scores unchanged. Padded box gate is a feasibility test.',
        'perspective': 'Foreground SIFT candidate evidence and validated homography patch reranking. Match the warped arm to its unwarped control.',
        'label_access': 'Freeze all ranked predictions before reading the identity answer key.',
        'interpretation': 'Exploratory reuse of the same 100 pairs. Largest score across multiple variants is descriptive.',
        'freeze_sha256': freeze, 'vit_prediction_freeze_sha256': sha(VIT / 'prediction-freeze.json'),
        'camera_calibration_sha256': sha(CAMERAS),
        'script_sha256': sha(__file__),
    }
    save(HERE / 'plan.json', plan)
    ranks = build_rankings(load_matrices(inputs))
    selected = {name: ranked[:max(EXTENDED_BUDGETS)] for name, ranked in ranks.items()}
    save(HERE / 'rankings.json', selected)
    save(HERE / 'prediction-freeze.json', {name: sha(HERE / name) for name in ('plan.json', 'rankings.json')})
    print(f'Frozen {len(selected)} rankings. Scoring.', flush=True)

    truth = {}
    for row in read(ROOT / 'data/messytable-100/answer-key.json')['cases']:
        row['same_vehicle_ids'] = row['shared_object_ids']
        for scene in row['scenes'].values():
            scene['vehicles'] = scene['objects']
        truth[row['case']] = row
    expected = read(VIT / 'scores-summary.json')['methods']
    early = HERE / 'roma'
    for path, digest in read(early / 'primary-ranking-freeze.json').items():
        verified_member(early, path, digest)
    early_rank = read(early / 'primary-rankings.json')
    results, identity_sets = {}, {}
    for name, rows in budget_predictions(inputs, selected):
        scored, outcomes = score_methods({name: rows}, truth, sorted(inputs))
        results[name] = scored[name]
        metrics = results[name]['overall']
        assert metrics['unscorable'] == 0 and metrics['detected_in_both'] == metrics['expected_links'] == 1696
        if name.startswith(('roma_box__', 'roma_mask__')):
            assert metrics['proposals'] == int(name.split('__B')[1])
            if metrics['proposals'] <= max(BUDGETS):
                assert selected[name.split('__')[0]][:metrics['proposals']] == early_rank[
                    name.split('__')[0]][:metrics['proposals']]
        if name.startswith('vit_control__'):
            prior_key = 'annotated_all_pairs__' + name.split('__')[1]
            if prior_key in expected:
                assert results[name] == expected[prior_key], name
        identity_sets[name] = correct_ids(outcomes[name])
    for name, groups in results.items():
        budget = name.split('__')[1]
        prior = identity_sets[f'vit_control__{budget}']
        now = identity_sets[name]
        groups['versus_vit'] = {'new_correct': len(now - prior), 'lost_correct': len(prior - now),
                                'shared_correct': len(prior & now)}
    diagnostics = {}
    for budget in (*BUDGETS, *EXTENDED_BUDGETS):
        diagnostics[f'B{budget}'] = [name for name, ranked in ranks.items() if len(ranked) < budget]
    patch_counts = defaultdict(int)
    for case in inputs:
        for status, count in read(HERE / 'perspective' / f'case-{case:03}.json')['status'].items():
            patch_counts[status] += count
    save(HERE / 'scores-summary.json', {'plan': plan, 'methods': results,
                                      'below_budget': diagnostics, 'perspective_status': dict(patch_counts)})
    print('431-proposal leaders', flush=True)
    ordered = sorted(((result['overall']['correct'], name, result['overall']['wrong'])
                      for name, result in results.items() if name.endswith('__B431')), reverse=True)
    for correct, name, wrong in ordered:
        print(f'{name:<65} {correct:3} correct {wrong:3} wrong', flush=True)


if __name__ == '__main__':
    main()
