"""Score frozen native FoundYou outputs; never perform model inference or tune rules.

Run only after the lead confirms inference is complete:
  python evaluation/score_foundyou_native_retrieval.py --confirm-inference-frozen
All input/hash/schema checks and label-free decoding precede answer-key access.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/messytable-foundyou-native-retrieval'
SOURCE = ROOT / 'evaluation/messytable-asnet-ladder'
BLIND = {41, 60, 67}
CASES = set(range(1, 101)) - BLIND
METHODS = ('asnet', 'forward', 'symmetric')
BUDGETS = (100, 200, 411)
ASNET_CUTOFF = 18.00142575056375


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verified_member(base, name, digest):
    path = (base / name).resolve()
    assert path.is_relative_to(base.resolve()), name
    assert path.is_file() and sha(path) == digest, f'Hash mismatch: {path}'


def make_link(a, b, value):
    return {'A': int(a), 'B': int(b), 'score': float(value)}


def row_top1(matrix):
    links = []
    empty = 0
    for a, row in enumerate(matrix):
        indices = np.flatnonzero(np.isfinite(row))
        if not len(indices):
            empty += 1
            continue
        b = min(indices, key=lambda j: (-float(row[j]), int(j)))
        links.append(make_link(a, b, row[b]))
    targets = Counter(link['B'] for link in links)
    return links, {'queries': len(matrix), 'empty': empty,
                   'collision_target_count': sum(n > 1 for n in targets.values()),
                   'collision_excess_proposals': sum(max(0, n - 1) for n in targets.values()),
                   'proposals_in_collisions': sum(n for n in targets.values() if n > 1)}


def global_greedy(matrices):
    # Endpoint uniqueness is per case; the budget is global over all 97 cases.
    ranked = sorted(((-float(matrix[a, b]), case, int(a), int(b))
                     for case, matrix in matrices.items()
                     for a, b in zip(*np.where(np.isfinite(matrix)))),
                    key=lambda item: item)
    used_a, used_b, accepted = set(), set(), []
    for negative_score, case, a, b in ranked:
        if (case, a) in used_a or (case, b) in used_b:
            continue
        used_a.add((case, a))
        used_b.add((case, b))
        accepted.append({'case': case, **make_link(a, b, -negative_score)})
    return accepted


def asnet_assign(distance):
    rows, cols = linear_sum_assignment(np.maximum(ASNET_CUTOFF - distance, 0), maximize=True)
    return [{'A': int(a), 'B': int(b), 'distance': float(distance[a, b])}
            for a, b in zip(rows, cols) if distance[a, b] < ASNET_CUTOFF]


def prediction_rows(graphs, links):
    return [{'case': case, 'detections': graphs[case]['detections'],
             'links': links.get(case, [])} for case in sorted(graphs)]


def load_verified():
    freeze_path = OUT / 'inference-freeze.json'
    assert freeze_path.is_file(), 'Inference freeze missing; scoring forbidden.'
    freeze = read(freeze_path)
    required = {'plan.json', 'shortlist-freeze.json', 'checkpoint-audit.json',
                'smoke-check.json', 'baseline-integrity.json', 'inference.json'}
    required |= {f'{folder}/case-{case:03}.{ext}' for case in CASES
                 for folder, ext in [('graphs', 'json'), ('scores', 'npz')]}
    assert required <= set(freeze), 'Incomplete inference freeze membership'
    assert {int(Path(p).stem.split('-')[1]) for p in freeze
            if p.startswith('graphs/')} == CASES
    assert {int(Path(p).stem.split('-')[1]) for p in freeze
            if p.startswith('scores/')} == CASES
    for name, digest in freeze.items():
        verified_member(OUT, name, digest)
    inference = read(OUT / 'inference.json')
    assert inference['pairs'] == 97 and set(inference['processed_cases']) == CASES
    shortlist = read(OUT / 'shortlist-freeze.json')
    assert set(shortlist) == {f'graphs/case-{case:03}.json' for case in CASES}
    assert all(freeze[name] == digest for name, digest in shortlist.items())
    plan = read(OUT / 'plan.json')
    assert set(plan['blind_excluded_from_inference_and_scoring']) == BLIND
    provenance = read(OUT / 'input-provenance-freeze.json')
    assert set(provenance) == {'input-images.json'}
    verified_member(OUT, 'input-images.json', provenance['input-images.json'])
    image_manifest = read(OUT / 'input-images.json')
    assert image_manifest['matches_prior_foundyou_preinference_image_manifest'] is True
    images = image_manifest['images']
    assert len(images) == 194
    assert {(r['case'], r['side']) for r in images} == {(case, side) for case in CASES for side in 'AB'}
    prior_plan_path = ROOT / 'evaluation/messytable-foundyou-ladder/plan.json'
    prior_plan_hash = plan['historical_hashes_before'][str(prior_plan_path.relative_to(ROOT))]
    assert sha(prior_plan_path) == prior_plan_hash
    prior_image_hashes = read(prior_plan_path)['identity']['image_hashes']
    for record in images:
        expected_path = f"data/messytable-100/case-{record['case']:03}/{record['side']}.jpg"
        assert record['path'] == expected_path
        assert record['sha256'] == prior_image_hashes[f"{record['case']:03}-{record['side']}"]
        verified_member(ROOT, record['path'], record['sha256'])
    assert sha(SOURCE / 'inference-freeze.json') == plan['source_freeze_sha256']
    source_freeze = read(SOURCE / 'inference-freeze.json')
    baseline_path = SOURCE / 'predictions/final_ladder__conservative_frozen.json'
    for rel in ['predictions/final_ladder__conservative_frozen.json',
                *[f'distances/case-{case:03}.npz' for case in sorted(CASES)]]:
        assert rel in source_freeze, rel
        verified_member(SOURCE, rel, source_freeze[rel])
    baseline = {r['case']: r for r in read(baseline_path)['predictions']}
    assert set(baseline) == CASES
    graphs, matrices, scene_scores, historical = {}, {m: {} for m in METHODS}, {}, {}
    for case in sorted(CASES):
        graph = read(OUT / f'graphs/case-{case:03}.json')
        assert graph['case'] == case and graph['detections'] == baseline[case]['detections']
        shape = tuple(len(graph['detections'][side]) for side in 'AB')
        edges = [tuple(edge) for edge in graph['edges']]
        assert edges == sorted(set(edges))
        with np.load(SOURCE / f'distances/case-{case:03}.npz', allow_pickle=False) as blob:
            distance = blob['distance'].copy()
        assert distance.shape == shape and np.isfinite(distance).all()
        expected = set()
        for a in range(shape[0]):
            expected.update((a, b) for b in sorted(range(shape[1]), key=lambda b: (float(distance[a, b]), b))[:3])
        for b in range(shape[1]):
            expected.update((a, b) for a in sorted(range(shape[0]), key=lambda a: (float(distance[a, b]), a))[:3])
        expected.update((l['A'], l['B']) for l in baseline[case]['links'])
        assert set(edges) == expected, f'Graph membership differs in case {case}'
        assert np.array_equal(np.array(graph['asnet_distances']), np.array([float(distance[a, b]) for a, b in edges]))
        mask = np.zeros(shape, dtype=bool)
        for a, b in edges:
            assert 0 <= a < shape[0] and 0 <= b < shape[1]
            mask[a, b] = True
        with np.load(OUT / f'scores/case-{case:03}.npz', allow_pickle=False) as blob:
            assert set(blob.files) == {'forward', 'reverse', 'symmetric', 'scene_AtoB', 'scene_BtoA'}
            arrays = {name: blob[name].copy() for name in blob.files}
        for name in ['forward', 'reverse', 'symmetric']:
            assert arrays[name].shape == shape
            assert np.array_equal(np.isfinite(arrays[name]), mask), (case, name)
            assert np.isnan(arrays[name][~mask]).all()
        assert np.array_equal(arrays['symmetric'][mask], np.minimum(arrays['forward'][mask], arrays['reverse'][mask]))
        assert arrays['scene_AtoB'].shape == (shape[0],) and arrays['scene_BtoA'].shape == (shape[1],)
        assert np.isfinite(arrays['scene_AtoB']).all() and np.isfinite(arrays['scene_BtoA']).all()
        matrices['asnet'][case] = np.where(mask, -distance, np.nan)
        for name in ['forward', 'symmetric']:
            matrices[name][case] = arrays[name]
        scene_scores[case] = {'AtoB': arrays['scene_AtoB'], 'BtoA': arrays['scene_BtoA']}
        historical[case] = asnet_assign(distance)
        assert [(l['A'], l['B']) for l in historical[case]] == [(l['A'], l['B']) for l in baseline[case]['links']]
        graphs[case] = graph
    return freeze, graphs, matrices, scene_scores, historical


def presence_metrics(records):
    counts = Counter(r['classification'] for r in records)
    valid = [r for r in records if r['classification'] != 'unmapped_source']
    positives = [r for r in valid if r['present_in_target_annotation']]
    negatives = [r for r in valid if not r['present_in_target_annotation']]
    auc = None
    if positives and negatives:
        labels = np.array([r['present_in_target_annotation'] for r in valid], dtype=bool)
        ranks = rankdata([r['logit'] for r in valid], method='average')
        auc = float((ranks[labels].sum() - len(positives) * (len(positives) + 1) / 2) /
                    (len(positives) * len(negatives)))
    tp = sum(r['logit'] > 0 for r in positives)
    fp = sum(r['logit'] > 0 for r in negatives)
    return {'all_source_queries': len(records), 'classification_counts': dict(counts),
            'excluded_unmapped_source': counts['unmapped_source'],
            'valid_mapped_queries': len(valid), 'positive_queries': len(positives),
            'negative_queries': len(negatives), 'roc_auc': auc,
            'logit_zero': {'true_positive': tp, 'false_positive': fp,
                           'true_negative': len(negatives) - fp, 'false_negative': len(positives) - tp,
                           'true_positive_rate': tp / len(positives) if positives else None,
                           'false_positive_rate': fp / len(negatives) if negatives else None},
            'unmapped_logit_positive': sum(r['logit'] > 0 for r in records if r['classification'] == 'unmapped_source')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-inference-frozen', action='store_true',
                        help='Lead confirms inference is complete and annotation scoring may begin.')
    args = parser.parse_args()
    if not args.confirm_inference_frozen:
        parser.error('--confirm-inference-frozen is required; no annotation access occurred.')
    freeze, graphs, matrices, scene_scores, historical = load_verified()
    # All decoding and decision settings are frozen before annotation access.
    predictions, ranking_details, top_diagnostics = {}, {}, {}
    for method in METHODS:
        top_links, diagnostics = {}, {}
        for case, matrix in matrices[method].items():
            top_links[case], diagnostics[case] = row_top1(matrix)
        predictions[f'{method}__row_top1'] = prediction_rows(graphs, top_links)
        top_diagnostics[method] = diagnostics
        ranked = global_greedy(matrices[method])
        ranking_details[method] = ranked
        for budget in BUDGETS:
            links = {case: [] for case in graphs}
            for link in ranked[:budget]:
                links[link['case']].append({k: v for k, v in link.items() if k != 'case'})
            predictions[f'{method}__global_B{budget}'] = prediction_rows(graphs, links)
        if method != 'asnet':
            links = {case: [] for case in graphs}
            for link in ranked:
                if link['score'] > 0:
                    links[link['case']].append({k: v for k, v in link.items() if k != 'case'})
            predictions[f'{method}__greedy_logit_gt0'] = prediction_rows(graphs, links)
    predictions['asnet__historical_hungarian_distance_lt18'] = prediction_rows(graphs, historical)
    decision = {'script_sha256': sha(__file__), 'inference_freeze_sha256': sha(OUT / 'inference-freeze.json'),
                'input_provenance_freeze_sha256': sha(OUT / 'input-provenance-freeze.json'),
                'input_images_sha256': sha(OUT / 'input-images.json'),
                'cases': sorted(CASES), 'budgets': list(BUDGETS), 'asnet_cutoff': ASNET_CUTOFF,
                'methods': list(predictions), 'decoder': 'Descending raw score, ties case/A/B ascending; global prefix, per-case endpoint uniqueness.',
                'row_top1': 'One per A source with finite candidate; ties B ascending; collisions retained.',
                'presence': 'Separate AtoB/BtoA scene logits; annotation-scope presence; unmapped source excluded from AUC only and reported.',
                'rules': 'No threshold fitting or method selection. All output predictions decoded before answer-key access.'}
    old_decision = OUT / 'scoring-decision-freeze.json'
    if old_decision.exists():
        assert read(old_decision) == decision, 'Scoring decisions changed; preserve original before a reviewed rerun.'
    else:
        save(old_decision, decision)
    save(OUT / 'predictions.json', {'methods': predictions, 'top1_diagnostics': top_diagnostics})
    save(OUT / 'rankings.json', {'methods': ranking_details, 'note': 'Complete label-free greedy accepted rankings; prefixes determine budgets.'})

    # First annotation access: only after complete hash/graph checks and decision freeze.
    from run_full_scene import map_detections, score
    from run_box_context import summarize
    key = read(ROOT / 'data/messytable-100/answer-key.json')
    key['cases'] = sorted((r for r in key['cases'] if r['case'] not in BLIND), key=lambda r: r['case'])
    assert {r['case'] for r in key['cases']} == CASES
    for truth in key['cases']:
        truth['same_vehicle_ids'] = truth['shared_object_ids']
        for scene in truth['scenes'].values():
            scene['vehicles'] = scene['objects']
    truths = {r['case']: r for r in key['cases']}
    maps = {case: {side: map_detections(graph['detections'][side], truths[case]['scenes'][side]['vehicles'])
                   for side in 'AB'} for case, graph in graphs.items()}
    queries, presence, ceiling_rows = [], [], []
    for case in sorted(graphs):
        truth, graph = truths[case], graphs[case]
        all_ids = {side: {int(i) for i in truth['scenes'][side]['vehicles']} for side in 'AB'}
        mapped_ids = {side: set(maps[case][side].values()) for side in 'AB'}
        positive_edges = [(a, b) for a, b in graph['edges']
                          if a in maps[case]['A'] and b in maps[case]['B'] and maps[case]['A'][a] == maps[case]['B'][b]]
        candidate_ids = sorted({maps[case]['A'][a] for a, b in positive_edges})
        shared = set(truth['shared_object_ids'])
        ceiling_rows.append({'case': case, 'expected': len(shared),
                             'detected_in_both': len(shared & mapped_ids['A'] & mapped_ids['B']),
                             'candidate_correct_edges': len(positive_edges), 'candidate_unique_true_ids': candidate_ids})
        for direction, source_side, target_side in [('AtoB', 'A', 'B'), ('BtoA', 'B', 'A')]:
            for index in range(len(graph['detections'][source_side])):
                identity = maps[case][source_side].get(index)
                present = identity in all_ids[target_side] if identity is not None else None
                classification = 'unmapped_source' if identity is None else 'mapped_present_gt' if present else 'mapped_absent_annotation'
                rec = {'case': case, 'direction': direction, 'source_index': index,
                       'source_identity': identity, 'classification': classification,
                       'present_in_target_annotation': present,
                       'target_detection_available': identity in mapped_ids[target_side] if identity is not None else None,
                       'zero_overlap_case': not bool(shared), 'logit': float(scene_scores[case][direction][index])}
                presence.append(rec)
                if direction != 'AtoB':
                    continue
                candidates = [b for a, b in graph['edges'] if a == index]
                rec = {**rec, 'candidate_count': len(candidates),
                       'positive_candidate_available': any(maps[case]['B'].get(b) == identity for b in candidates) if identity is not None else None}
                queries.append(rec)
    expected_total = sum(r['expected'] for r in ceiling_rows)
    detected_total = sum(r['detected_in_both'] for r in ceiling_rows)
    assert expected_total == 1647 and detected_total == 593, (expected_total, detected_total)
    results, all_outcomes = {}, {}
    for method, rows in predictions.items():
        outcomes = score(rows, key)
        summary = summarize(outcomes, key)
        # Keep all status buckets. Explicit unique recall avoids accidental repeated-ID counting.
        for group, metrics in summary.items():
            selected = [r for r in outcomes if group == 'overall' or truths[r['case']]['stratum'] == group]
            unique = sum(len({l['vehicle_A'] for l in r['links'] if l['status'] == 'correct'}) for r in selected)
            metrics['correct_unique_links'] = unique
            metrics['unique_true_link_recall'] = unique / metrics['expected_links'] if metrics['expected_links'] else None
            metrics['unmapped'] = metrics['unscorable']
            metrics['confirmed_correct_share_all_proposals'] = metrics['correct'] / metrics['proposals'] if metrics['proposals'] else None
            zeros = [r for r in selected if not r['expected']]
            metrics['zero_overlap_pairs'] = len(zeros)
            metrics['zero_overlap_pairs_with_proposals'] = sum(bool(r['links']) for r in zeros)
            metrics['zero_overlap_proposals'] = sum(len(r['links']) for r in zeros)
            metrics['zero_overlap_wrong_links'] = sum(l['status'] == 'wrong' for r in zeros for l in r['links'])
            metrics['zero_overlap_unmapped_links'] = sum(l['status'] == 'unscorable' for r in zeros for l in r['links'])
        if method.endswith('__row_top1'):
            base = method.split('__')[0]
            summary['overall']['query_diagnostics'] = {name: sum(r[name] for r in top_diagnostics[base].values())
                for name in ['queries', 'empty', 'collision_target_count', 'collision_excess_proposals', 'proposals_in_collisions']}
        # Attach explicit source classification to every A->B decision.
        query_lookup = {(q['case'], q['source_index']): q for q in queries}
        for row in outcomes:
            for link in row['links']:
                link['source_classification'] = query_lookup[row['case'], link['A']]['classification']
        class_counts = {}
        for classification in ['mapped_present_gt', 'mapped_absent_annotation', 'unmapped_source']:
            selected_links = [l for r in outcomes for l in r['links'] if l['source_classification'] == classification]
            counts = Counter(l['status'] for l in selected_links)
            class_counts[classification] = {'source_queries': sum(q['classification'] == classification for q in queries),
                'proposals': len(selected_links), **{s: counts[s] for s in ['correct', 'wrong', 'unscorable']}}
        results[method] = {'summary': summary, 'by_source_classification': class_counts}
        all_outcomes[method] = outcomes
    hist = results['asnet__historical_hungarian_distance_lt18']['summary']['overall']
    assert (hist['correct'], hist['wrong'], hist['unscorable'], hist['proposals']) == (221, 54, 136, 411), hist
    # Exact historical status regression using previously frozen scorer outputs.
    previous_path = SOURCE / 'scores.json'
    expected_hash = read(OUT / 'plan.json')['historical_hashes_before'].get(str(previous_path.relative_to(ROOT)))
    assert expected_hash and sha(previous_path) == expected_hash
    previous = read(previous_path)['methods']['final_ladder__conservative_frozen']['scores']
    previous_by_case = {r['case']: r for r in previous}
    for row in all_outcomes['asnet__historical_hungarian_distance_lt18']:
        signature = lambda links: [(l['A'], l['B'], l['status']) for l in links]
        assert signature(row['links']) == signature(previous_by_case[row['case']]['links'])
    candidate_total = sum(len(r['candidate_unique_true_ids']) for r in ceiling_rows)
    query_class = Counter(q['classification'] for q in queries)
    ceiling = {'expected_links': expected_total, 'detected_both': detected_total,
               'candidate_unique_true_links': candidate_total,
               'candidate_recall_all_true_links': candidate_total / expected_total,
               'candidate_recall_detected_true_links': candidate_total / detected_total,
               'positive_mapped_edges': sum(r['candidate_correct_edges'] for r in ceiling_rows),
               'query_classifications': dict(query_class), 'all_A_source_queries': len(queries),
               'present_gt_no_target_detection': sum(q['classification'] == 'mapped_present_gt' and not q['target_detection_available'] for q in queries),
               'present_detected_but_missing_shortlist': sum(q['classification'] == 'mapped_present_gt' and q['target_detection_available'] and not q['positive_candidate_available'] for q in queries),
               'empty_candidate_queries': sum(q['candidate_count'] == 0 for q in queries), 'cases': ceiling_rows}
    presence_summary = {}
    for direction in ['pooled', 'AtoB', 'BtoA']:
        records = [r for r in presence if direction == 'pooled' or r['direction'] == direction]
        presence_summary[direction] = {'all': presence_metrics(records),
                                      'zero_overlap_cases': presence_metrics([r for r in records if r['zero_overlap_case']]),
                                      'nonzero_overlap_cases': presence_metrics([r for r in records if not r['zero_overlap_case']])}
    notes = ['Reused 97 development pairs; no fresh-holdout or truck-transfer claim.',
             'All wrong/unmapped proposals retained. Unmapped is unknown localization, not verified wrong identity.',
             'Global fixed budgets are descriptive: raw cross-query score comparability is unvalidated.',
             'Independent row top1 always chooses when candidates exist; collisions are retained and no absence claim follows.',
             'Scene scores concern annotation-scope physical-ID presence and cannot select a target endpoint; unannotated objects remain unknown.',
             'Native logit>0 is an uncalibrated fixed rule, not a fitted absence detector.',
             'ASNet trained in-domain by original authors; FoundYou crop target distribution differs from native whole-image retrieval.',
             'Thresholds, budgets, shortlist and decoder frozen before annotation access; no method selection.']
    summary = {'scored_pairs': 97, 'excluded': sorted(BLIND), 'decision': decision,
               'shortlist_ceiling': ceiling, 'methods': results, 'full_scene_presence': presence_summary,
               'regression': {'historical_asnet_assignments_exact': True, 'historical_asnet_link_status_exact': True,
                              'historical_asnet_counts_exact': True, 'detection_ceiling_593': True,
                              'inference_hashes_verified': len(freeze), 'shortlist_membership_exact': True}, 'notes': notes}
    save(OUT / 'scores-summary.json', summary)
    save(OUT / 'scored-predictions.json', {'methods': all_outcomes, 'source_queries': queries, 'full_scene_queries': presence})
    lines = ['Native FoundYou retrieval: frozen development comparison\n',
             '97 pairs; exclusions041/060/067; no training or threshold fitting.\n',
             f'1647 true links; fixed detection ceiling593; shortlist true-link ceiling{candidate_total}.\n\n',
             'Method | correct | wrong | unmapped | proposals | unique recall | confirmed correct/all | zero-overlap pairs with proposals\n']
    for method, result in results.items():
        m = result['summary']['overall']
        lines.append(f"{method} | {m['correct']} | {m['wrong']} | {m['unmapped']} | {m['proposals']} | {m['unique_true_link_recall']:.4%} | {m['confirmed_correct_share_all_proposals'] if m['proposals'] else 'n/a'} | {m['zero_overlap_pairs_with_proposals']}/{m['zero_overlap_pairs']}\n")
    lines += ['\nSource-query diagnostics:\n', json.dumps({k: v for k, v in ceiling.items() if k != 'cases'}, indent=2),
              '\nFull-scene presence (separate task; AUC excludes reported unmapped sources):\n', json.dumps(presence_summary, indent=2),
              '\n\nLimitations:\n', '\n'.join(notes), '\n']
    (OUT / 'RESULTS.txt').write_text(''.join(lines))
    # Ensure scoring did not alter inference artifacts.
    for name, digest in freeze.items():
        verified_member(OUT, name, digest)
    print('Scoring complete; all frozen inputs unchanged.', flush=True)
    print(''.join(lines[:5]), flush=True)


if __name__ == '__main__':
    main()
