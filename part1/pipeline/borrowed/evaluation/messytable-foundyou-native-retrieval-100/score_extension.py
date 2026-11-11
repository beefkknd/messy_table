"""Compare frozen option 1 scores on the original 97 and the full 100."""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from score_foundyou_native_retrieval import (
    ASNET_CUTOFF, BUDGETS, METHODS, asnet_assign, global_greedy, load_verified,
    prediction_rows, presence_metrics, read, row_top1, save, sha, verified_member,
)
from run_box_context import summarize
from run_full_scene import map_detections, score

PRIOR = ROOT / 'evaluation/messytable-foundyou-native-retrieval'
ADDED = (41, 60, 67)
HUNGARIAN = 'asnet__historical_hungarian_distance_lt18'


def load_extension():
    plan = read(OUT / 'plan.json')
    assert tuple(plan['new_inference_cases']) == ADDED
    assert plan['asnet_cutoff'] == ASNET_CUTOFF
    assert tuple(plan['historical_fixed_proposal_budgets']) == BUDGETS
    assert sha(PRIOR / 'inference-freeze.json') == plan['prior_inference_freeze_sha256']
    assert sha(PRIOR / 'input-images.json') == plan['prior_input_manifest_sha256']
    freeze = read(OUT / 'inference-freeze.json')
    for path, digest in freeze.items():
        verified_member(OUT, path, digest)
    for path, digest in plan['added_image_sha256'].items():
        verified_member(ROOT, path, digest)
    for path, digest in read(OUT / 'shortlist-freeze.json').items():
        assert freeze[path] == digest
    _, graphs, matrices, scenes, historical = load_verified()
    original_cases = sorted(graphs)
    assert set(original_cases) | set(ADDED) == set(range(1, 101))
    assert not set(original_cases) & set(ADDED)
    for case in ADDED:
        directory = OUT / f'case-{case:03}'
        graph = read(directory / 'graph.json')
        assert graph['case'] == case
        graphs[case] = graph
        with np.load(directory / 'asnet-distances.npz', allow_pickle=False) as blob:
            distance = blob['distance']
        shape = tuple(len(graph['detections'][side]) for side in 'AB')
        assert distance.shape == shape and np.isfinite(distance).all()
        historical[case] = asnet_assign(distance)
        edges = list(map(tuple, graph['edges']))
        assert edges == sorted(set(edges))
        mask = np.zeros(shape, dtype=bool)
        for edge in edges:
            mask[edge] = True
        assert np.allclose(distance[mask], graph['asnet_distances'], atol=0, rtol=0)
        assert all(mask[link['A'], link['B']] for link in historical[case])
        matrices['asnet'][case] = np.where(mask, -distance, np.nan)
        with np.load(directory / 'native-scores.npz', allow_pickle=False) as blob:
            for method in ('forward', 'reverse', 'symmetric'):
                assert blob[method].shape == shape
                assert np.array_equal(np.isfinite(blob[method]), mask)
                assert np.isnan(blob[method][~mask]).all()
            assert np.array_equal(blob['symmetric'][mask],
                                  np.minimum(blob['forward'], blob['reverse'])[mask])
            for method in ('forward', 'symmetric'):
                matrices[method][case] = blob[method]
            scenes[case] = {'AtoB': blob['scene_AtoB'], 'BtoA': blob['scene_BtoA']}
        for direction, count in zip(('AtoB', 'BtoA'), shape, strict=True):
            assert scenes[case][direction].shape == (count,)
            assert np.isfinite(scenes[case][direction]).all()
    return original_cases, graphs, matrices, scenes, historical


def decode(graphs, matrices, historical, cases, budgets):
    selected = {case: graphs[case] for case in cases}
    methods = {HUNGARIAN: prediction_rows(selected, historical)}
    for method in METHODS:
        values = {case: matrices[method][case] for case in cases}
        methods[f'{method}__row_top1'] = prediction_rows(
            selected, {case: row_top1(matrix)[0] for case, matrix in values.items()})
        ranked = global_greedy(values)
        for budget in budgets:
            links = {}
            assert len(ranked) >= budget
            for link in ranked[:budget]:
                links.setdefault(link['case'], []).append({k: v for k, v in link.items() if k != 'case'})
            methods[f'{method}__global_B{budget}'] = prediction_rows(selected, links)
        if method != 'asnet':
            links = {}
            for link in ranked:
                if link['score'] > 0:
                    links.setdefault(link['case'], []).append({k: v for k, v in link.items() if k != 'case'})
            methods[f'{method}__greedy_logit_gt0'] = prediction_rows(selected, links)
    return methods


def score_methods(predictions, truths, cases):
    key = {'cases': [truths[case] for case in cases]}
    results, outcomes = {}, {}
    for method, rows in predictions.items():
        outcomes[method] = score(rows, key)
        results[method] = summarize(outcomes[method], key)
        for group, summary in results[method].items():
            included = [row for row in outcomes[method]
                        if group == 'overall' or truths[row['case']]['stratum'] == group]
            zeros = [row for row in included if not row['expected']]
            summary['zero_overlap_pairs'] = len(zeros)
            summary['zero_overlap_pairs_with_proposals'] = sum(bool(row['links']) for row in zeros)
            summary['unique_correct'] = sum(len({link['vehicle_A'] for link in row['links']
                                                 if link['status'] == 'correct'}) for row in included)
            summary['unique_recall'] = (summary['unique_correct'] / summary['expected_links']
                                        if summary['expected_links'] else None)
    return results, outcomes


def scene_presence(graphs, scenes, truths, cases):
    records = []
    for case in cases:
        truth = truths[case]
        for source, target in ('AB', 'BA'):
            direction = source + 'to' + target
            source_map = map_detections(graphs[case]['detections'][source],
                                        truth['scenes'][source]['vehicles'])
            target_ids = set(map(int, truth['scenes'][target]['vehicles']))
            for index, logit in enumerate(scenes[case][direction]):
                identity = source_map.get(index)
                present = identity in target_ids if identity is not None else None
                classification = ('unmapped_source' if present is None else
                                  'mapped_present_gt' if present else 'mapped_absent_annotation')
                records.append({'direction': direction, 'classification': classification,
                                'present_in_target_annotation': present, 'logit': float(logit),
                                'zero_overlap_case': not truth['same_vehicle_ids']})
    result = {}
    for direction in ('pooled', 'AtoB', 'BtoA'):
        selected = [r for r in records if direction in ('pooled', r['direction'])]
        result[direction] = {'all': presence_metrics(selected),
                             'zero_overlap_cases': presence_metrics([r for r in selected if r['zero_overlap_case']]),
                             'nonzero_overlap_cases': presence_metrics([r for r in selected if not r['zero_overlap_case']])}
    return result


def main():
    assert not (OUT / 'scores-summary.json').exists(), 'Preserve completed scoring'
    original, graphs, matrices, scenes, historical = load_extension()
    all_cases = sorted(graphs)
    matched_budget = sum(len(links) for links in historical.values())
    full_budgets = sorted(set(BUDGETS) | {matched_budget})
    predictions = {
        'original97': decode(graphs, matrices, historical, original, BUDGETS),
        'full100': decode(graphs, matrices, historical, all_cases, full_budgets),
        'added3': decode(graphs, matrices, historical, list(ADDED), ()),
    }
    save(OUT / 'predictions.json', predictions)
    save(OUT / 'prediction-freeze.json', {
        'predictions.json': sha(OUT / 'predictions.json'),
        'inference-freeze.json': sha(OUT / 'inference-freeze.json'),
        'score_extension.py': sha(__file__),
    })

    # Scoring starts after all cohort decisions and rankings have been frozen.
    key_path = ROOT / 'data/messytable-100/answer-key.json'
    key = read(key_path)
    truths = {}
    for case in key['cases']:
        case['same_vehicle_ids'] = case['shared_object_ids']
        for scene in case['scenes'].values():
            scene['vehicles'] = scene['objects']
        truths[case['case']] = case
    assert sorted(truths) == all_cases
    results, outcomes, presence = {}, {}, {}
    for name, cases in [('original97', original), ('full100', all_cases), ('added3', list(ADDED))]:
        results[name], outcomes[name] = score_methods(predictions[name], truths, cases)
        presence[name] = scene_presence(graphs, scenes, truths, cases)

    previous = read(PRIOR / 'scored-predictions.json')['methods']
    assert set(previous) == set(outcomes['original97'])
    for method, rows in outcomes['original97'].items():
        signature = lambda rows: [(r['case'], r['expected'], r['detected_in_both'],
                                  [(l['A'], l['B'], l['status']) for l in r['links']]) for r in rows]
        assert signature(rows) == signature(previous[method]), method

    save(OUT / 'scores-summary.json', {
        'scored_pairs': 100, 'reused_pairs': 97, 'new_inference_cases': ADDED,
        'matched_full100_proposal_budget': matched_budget, 'historical_budgets': BUDGETS,
        'historical_97_per_case_scores_reproduced': True, 'answer_key_sha256': sha(key_path),
        'methods_by_cohort': results, 'scene_presence_by_cohort': presence,
    })
    save(OUT / 'scored-predictions.json', outcomes)
    lines = [
        'Native FoundYou retrieval on all 100 original MessyTable pairs',
        'The full cohort includes cases 041, 060 and 067. The original 97 scores are reused and reproduce exactly.',
        'All model parameters, ranking rules and existing budgets are frozen. The extra budget equals the full-100 ASNet cutoff proposal count.',
        '',
    ]
    for cohort, methods in results.items():
        control = methods[HUNGARIAN]['overall']
        lines.extend([f'{cohort} | {control["expected_links"]} true links | {control["detected_in_both"]} mapped in both views',
                      'Method | Correct | Wrong | Unmapped | Proposals | Unique recall | Zero-overlap pairs with proposals'])
        for method, strata in methods.items():
            row = strata['overall']
            recall = 'n/a' if row['unique_recall'] is None else f'{row["unique_recall"]:.2%}'
            lines.append(f'{method} | {row["correct"]} | {row["wrong"]} | {row["unscorable"]} | '
                         f'{row["proposals"]} | {recall} | '
                         f'{row["zero_overlap_pairs_with_proposals"]}/{row["zero_overlap_pairs"]}')
        lines.append('')
    (OUT / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
