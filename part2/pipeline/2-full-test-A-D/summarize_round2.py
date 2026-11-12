#!/usr/bin/env python3
"""Independently aggregate frozen per-case detector scores and paired cluster contrasts."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare import OUT, ROOT, digest, load, save

ARMS = ('pretrained768', 'A', 'B', 'C', 'D')
COHORTS = {'development100': 100, 'original100': 83}
RATIOS = {'both_view_coverage': ('shared_localized_both', 'shared_objects'),
          'endpoint_recall_iou50': ('localized', 'ground_truth'),
          'endpoint_recall_iou75': ('localized_at_iou75', 'ground_truth'),
          'unmapped_fraction_of_predictions': ('unmapped', 'detections'),
          'predictions_per_pair': ('detections', 'pairs'),
          'unmapped_per_pair': ('unmapped', 'pairs')}
CONTRASTS = tuple((arm, 'pretrained768') for arm in 'ABCD') + tuple((arm, 'A') for arm in 'BCD') + (('C', 'B'), ('C', 'D'))
REPLICATES = 20000
SEED = 20261110


def compute_ratio(counts, numerator, denominator):
    return counts[numerator] / counts[denominator] if counts[denominator] else None


def verify_scoring_artifact(root, cohort, arm, score_path, document, truth_path):
    base = root.parent / 'inference' if arm == 'pretrained768' else root / 'inference'
    run = base / arm / cohort
    freeze = load(run / 'freeze.json')
    if set(freeze) != {'predictions.json', 'inference-plan.json'}:
        raise ValueError(f'Incomplete prediction freeze for {cohort}/{arm}')
    if any(digest(run / name) != expected for name, expected in freeze.items()):
        raise ValueError(f'Prediction artifacts changed for {cohort}/{arm}')
    if document['prediction_sha256'] != freeze['predictions.json'] or document['truth_sha256'] != digest(truth_path):
        raise ValueError(f'Score inputs changed for {score_path}')
    if document['mapping_sha256'] != digest(ROOT / 'evaluation/run_full_scene.py'):
        raise ValueError(f'Historical mapper changed since scoring {score_path}')
    plan = load(run / 'inference-plan.json')
    if plan['manifest_sha256'] != digest(OUT / 'cohorts' / f'{cohort}.json'):
        raise ValueError(f'Prediction used a different cohort for {cohort}/{arm}')
    return {'score_sha256': digest(score_path), 'prediction_freeze_sha256': digest(run / 'freeze.json'),
            'source_weights_sha256': plan['weights_sha256']}


def case_counts(at_threshold, members, expected_groups):
    by_id = {row['id']: row for row in at_threshold['cases']}
    assert len(at_threshold['cases']) == len(members) == len(by_id) and by_id.keys() == members.keys()
    clusters = defaultdict(Counter)
    total = Counter()
    for case_id, row in by_id.items():
        cluster = members[case_id]
        counts = row['counts']
        assert cluster == row['setup'] and counts['pairs'] == 1 and counts['images'] == 2
        assert counts['localized'] + counts['unmapped'] == counts['detections']
        assert sum(counts[f'shared_localized_{count}'] for count in ('both', 'one', 'neither')) == counts['shared_objects']
        assert len(set(row['shared_ids_localized_both'])) == counts['shared_localized_both']
        clusters[cluster].update(counts)
        total.update(counts)
    assert len(clusters) == expected_groups and dict(total) == at_threshold['counts']
    assert {cluster: dict(counts) for cluster, counts in clusters.items()} == at_threshold['setup_counts']
    return by_id, clusters, total


def paired_effect(candidate, reference, resamples):
    candidate_rows, candidate_clusters, candidate_total = candidate
    reference_rows, reference_clusters, reference_total = reference
    cluster_ids = sorted(candidate_clusters)
    assert candidate_clusters.keys() == reference_clusters.keys()
    gained = lost = pairs_up = pairs_down = 0
    for case_id, left in candidate_rows.items():
        right = reference_rows[case_id]
        for denominator in ('pairs', 'images', 'ground_truth', 'shared_objects'):
            assert left['counts'][denominator] == right['counts'][denominator]
        c_ids = set(left['shared_ids_localized_both'])
        r_ids = set(right['shared_ids_localized_both'])
        gained += len(c_ids - r_ids)
        lost += len(r_ids - c_ids)
        pairs_up += len(c_ids) > len(r_ids)
        pairs_down += len(c_ids) < len(r_ids)
    effects = {}
    for label, (num, den) in RATIOS.items():
        observed = [compute_ratio(counts, num, den) for counts in (candidate_total, reference_total)]
        arrays = [np.array([[clusters[group][num], clusters[group][den]] for group in cluster_ids], dtype=np.int64)
                  for clusters in (candidate_clusters, reference_clusters)]
        pooled = [array[resamples].sum(axis=1) for array in arrays]
        valid = (pooled[0][:, 1] > 0) & (pooled[1][:, 1] > 0)
        draws = pooled[0][valid, 0] / pooled[0][valid, 1] - pooled[1][valid, 0] / pooled[1][valid, 1]
        effects[label] = {'observed_delta': None if None in observed else observed[0] - observed[1],
                          'paired_cluster_ci95': np.quantile(draws, [0.025, 0.975]).tolist() if draws.size else None,
                          'valid_bootstrap_draws': int(valid.sum())}
    assert gained - lost == candidate_total['shared_localized_both'] - reference_total['shared_localized_both']
    return {'effects': effects, 'both_view_instances_gained': gained, 'both_view_instances_lost': lost,
            'pairs_with_more_both_view': pairs_up, 'pairs_with_fewer_both_view': pairs_down,
            'raw_predictions_delta': candidate_total['detections'] - reference_total['detections'],
            'raw_unmapped_delta': candidate_total['unmapped'] - reference_total['unmapped']}


def operating_point(documents, thresholds, members, setup_groups):
    values = {arm: case_counts(document['thresholds'][str(thresholds[arm])], members, setup_groups)
              for arm, document in documents.items()}
    resamples = np.random.default_rng(SEED).integers(setup_groups, size=(REPLICATES, setup_groups))
    return {'arms': {arm: {'threshold': thresholds[arm], 'counts': dict(value[2]),
                           'ratios': {metric: compute_ratio(value[2], *fields) for metric, fields in RATIOS.items()}}
                     for arm, value in values.items()},
            'paired': {f'{candidate}_minus_{reference}': paired_effect(values[candidate], values[reference], resamples)
                       for candidate, reference in CONTRASTS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round2', type=Path, default=OUT.parent / 'engineering/round2')
    args = parser.parse_args()
    root = args.round2.resolve()
    output = OUT / 'round2-detector-comparison.json'
    assert not output.exists(), 'Preserve first scored comparison'
    selection_path = root / 'calibrated-thresholds.json'
    selection = load(selection_path)
    plan_path = root / 'freeze-plan.json'
    if digest(plan_path) != selection['frozen_plan_sha256']:
        raise ValueError('Calibration came from a different run plan')
    plan = load(plan_path)
    fixed = plan['fixed_confidence']
    assert fixed == .25 and set(selection['thresholds']) == set(ARMS)
    calibration = {}
    for arm in ARMS:
        score_path = root / 'scores/calibration30' / f'{arm}.json'
        assert digest(score_path) == selection['calibration_scores_sha256'][arm]
        calibration[arm] = load(score_path)['thresholds']
    ceilings = {field: calibration['A'][str(fixed)]['counts'][field] for field in ('detections', 'unmapped')}
    assert ceilings == selection['ceilings']
    for arm in 'BCD':
        eligible = [threshold for threshold in sorted(plan['confidence_grid'])
                    if all(calibration[arm][str(threshold)]['counts'][field] <= ceiling for field, ceiling in ceilings.items())]
        assert eligible and selection['thresholds'][arm] == eligible[0], arm
    assert selection['thresholds']['A'] == selection['thresholds']['pretrained768'] == fixed
    cohorts = {}
    for cohort, groups in COHORTS.items():
        manifest = load(OUT / 'cohorts' / f'{cohort}.json')
        members = {row['id']: row['setup'] for row in manifest['scenes']}
        assert len(members) == 100 and len(set(members.values())) == groups
        documents, provenance = {}, {}
        for arm in ARMS:
            score_path = root / 'scores' / cohort / f'{arm}.json'
            documents[arm] = load(score_path)
            provenance[arm] = verify_scoring_artifact(root, cohort, arm, score_path, documents[arm],
                                                     OUT / 'scoring' / f'{cohort}.json')
        cohorts[cohort] = {'paired_groups': groups, 'cohort_sha256': digest(OUT / 'cohorts' / f'{cohort}.json'),
                           'arm_sources': provenance,
                           'fixed_0_25': operating_point(documents, dict.fromkeys(ARMS, fixed), members, groups),
                           'calibrated': operating_point(documents, selection['thresholds'], members, groups)}
        if cohort == 'original100':
            assert all(arm['counts']['shared_objects'] == 1696 for arm in cohorts[cohort]['fixed_0_25']['arms'].values())
    save(output, {'round2_plan_sha256': digest(plan_path), 'calibrated_thresholds_sha256': digest(selection_path),
                  'verified_calibration_thresholds': selection['thresholds'], 'verified_calibration_ceilings': ceilings,
                  'cluster_bootstrap_replicates': REPLICATES, 'bootstrap_seed': SEED,
                  'bootstrap': 'Paired percentile intervals resample setup groups with replacement and pool numerator/denominator counts before taking differences. Original100 groups are connected source-setup components.',
                  'cohorts': cohorts})
    save(OUT / 'round2-detector-comparison-freeze.json', {str(path.relative_to(ROOT)): digest(path)
                                                          for path in (output, Path(__file__).resolve())})


if __name__ == '__main__':
    main()
