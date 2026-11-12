#!/usr/bin/env python3
"""Recompute stronger-teacher comparisons from frozen inference outputs."""

import argparse
from pathlib import Path
import sys

import numpy as np

from freeze_cohorts import COHORTS, OLD_DATA, OUT, ROOT, digest, load, save

LEGACY = OLD_DATA.parent / 'engineering'
sys.path[:0] = [str(LEGACY), str(OLD_DATA)]
from score_detector import score_at_threshold
from summarize_round2 import RATIOS, REPLICATES, SEED, case_counts, compute_ratio, paired_effect

ARMS = ('T', 'S', 'L', 'O', 'A300')
SCORED = {'development100': 100, 'original100': 83}
CONTRASTS = (('S', 'T'), ('L', 'T'), ('O', 'T'), ('L', 'S'), ('O', 'S'), ('O', 'L'), ('T', 'A300'))
FIXED = .25
ARM_RATIOS = {**RATIOS, 'endpoint_precision_iou50': ('localized', 'detections')}
COMMON_INFERENCE = {'imgsz': 768, 'head': 'one2many', 'nms_iou': .7, 'confidence_floor': .001, 'max_det': 300}


def verified_run(root, cohort, arm, manifest_path, members, parent_weights):
    source_root = LEGACY / 'round2' if arm == 'A300' else root
    actual_arm = 'A' if arm == 'A300' else arm
    checkpoint = source_root / f'train/{actual_arm}/checkpoint.pt'
    prediction_path = source_root / f'inference/{actual_arm}/{cohort}'
    freeze_path = prediction_path / 'freeze.json'
    freeze = load(freeze_path)
    assert set(freeze) == {'inference-plan.json', 'predictions.json'}
    assert all(digest(prediction_path / name) == value for name, value in freeze.items())
    plan = load(prediction_path / 'inference-plan.json')
    assert all(plan[name] == value for name, value in COMMON_INFERENCE.items()), (cohort, arm)
    assert plan['manifest_sha256'] == digest(manifest_path) and plan['weights_sha256'] == digest(checkpoint)
    assert Path(plan['weights']).resolve() == checkpoint.resolve()
    if arm == 'A300':
        assert plan['weights_sha256'] == parent_weights
    rows = load(prediction_path / 'predictions.json')['scenes']
    assert len(rows) == len(members) and {row['id'] for row in rows} == members.keys()
    for row in rows:
        scene = members[row['id']]
        assert row['setup'] == scene['setup']
        assert {key: value['sha256'] for key, value in row['views'].items()} == {view['id']: view['sha256'] for view in scene['views']}
    return rows, {'prediction_freeze_sha256': digest(freeze_path), **freeze, 'checkpoint_sha256': plan['weights_sha256']}


def cohort_inputs(root, cohort, parent_weights):
    manifest = COHORTS / f'{cohort}.json'
    ground_truth = OLD_DATA / f'scoring/{cohort}.json'
    cohort_rows = load(manifest)['scenes']
    members = {row['id']: row for row in cohort_rows}
    truth_document = load(ground_truth)
    assert truth_document['cohort_sha256'] == digest(manifest)
    truths = {row['id']: row for row in truth_document['scenes']}
    assert len(truths) == len(cohort_rows) and truths.keys() == members.keys()
    predictions, provenance = {}, {}
    for arm in ARMS:
        predictions[arm], provenance[arm] = verified_run(root, cohort, arm, manifest, members, parent_weights)
    provenance['cohort_manifest_sha256'] = digest(manifest)
    provenance['scoring_key_sha256'] = digest(ground_truth)
    return predictions, truths, members, provenance



def training_schedule(root, protocol, protocol_hash):
    details, plans = {}, {}
    metadata_path = OUT / 'cohorts/supplemental240.json'
    metadata = {f"{scene['id']}-{view['id']}": (scene['setup'], view['path'], view['sha256'])
                for scene in load(metadata_path)['scenes'] for view in scene['views']}
    for arm in ('T', 'S', 'L', 'O'):
        directory = root / 'train' / arm
        plan_path, result_path = directory / 'run-plan.json', directory / 'training.json'
        plan, result = load(plan_path), load(result_path)
        plans[arm] = plan
        assert plan['protocol_sha256'] == protocol_hash and not plan['audit_only']
        assert plan['incoming_checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
        assert plan['previous_optimizer_steps'] == plan['planned_optimizer_steps'] == result['completed_optimizer_steps'] == 300
        assert result['total_optimizer_steps'] == 600 and digest(directory / 'checkpoint.pt') == result['checkpoint_sha256']
        assert [item['step'] for item in result['per_step']] == list(range(301, 601))
        assert plan['incoming_frozen_blocks_sha256'] == result['outgoing_frozen_blocks_sha256']
        schedule = plan['schedule']
        assert len(schedule) == 300 and all(len(batch) == 4 for batch in schedule)
        extra = [batch[3] for batch in schedule]
        unique_rows = {row['row_id'] for row in extra}
        expected_scope = 'complete_frame' if arm in ('T', 'O') else 'partial_frame'
        assert all(row['scope'] == expected_scope for row in extra)
        if arm == 'T':
            assert len(unique_rows) == 150
        else:
            assert plan['metadata_sha256'] == digest(metadata_path) and unique_rows == metadata.keys()
            for row in extra:
                assert (row['setup'], row['image'], plan['input_sha256'][row['image']]) == metadata[row['row_id']]
        details[arm] = {'optimizer_updates': result['completed_optimizer_steps'], 'final_optimizer_step': result['total_optimizer_steps'],
                        'incoming_model_state_sha256': plan['incoming_model_state_sha256'],
                        'incoming_optimizer_state_sha256': plan['incoming_optimizer_state_sha256'],
                        'checkpoint_sha256': result['checkpoint_sha256'], 'run_plan_sha256': digest(plan_path), 'training_sha256': digest(result_path),
                        'trusted_slot_exposures': 900, 'supplemental_slot_exposures': 300, 'supplemental_unique_frames': len(unique_rows),
                        'supplemental_unique_setups': len({row['setup'] for row in extra}),
                        'supplemental_repeated_exposures': 300 - len(unique_rows),
                        'supplemental_positive_box_exposures': sum(row['boxes'] for row in extra),
                        'supplemental_empty_exposures': sum(row['boxes'] == 0 for row in extra),
                        'supplemental_scope': expected_scope}
    assert len({item['incoming_model_state_sha256'] for item in details.values()}) == 1
    assert len({item['incoming_optimizer_state_sha256'] for item in details.values()}) == 1
    trusted_stream = [[batch[:3] for batch in plan['schedule']] for plan in plans.values()]
    assert all(stream == trusted_stream[0] for stream in trusted_stream[1:])
    pseudo_stream = [[(batch[3]['row_id'], batch[3]['image']) for batch in plans[arm]['schedule']] for arm in ('S', 'L', 'O')]
    assert all(stream == pseudo_stream[0] for stream in pseudo_stream[1:])
    return details


def calibration(root, protocol):
    predictions, truths, members, provenance = cohort_inputs(root, 'calibration30', protocol['parent_checkpoint_sha256'])
    grid = protocol['evaluation']['calibration_grid']
    assert grid == sorted(set(grid)) and FIXED in grid
    reference = score_at_threshold(predictions['T'], truths, members, FIXED)['counts']
    limits = {name: reference[name] for name in ('detections', 'unmapped')}
    thresholds = {'T': FIXED}
    budget_counts = {'T': limits}
    for arm in ARMS:
        if arm == 'T':
            continue
        for threshold in grid:
            scores = score_at_threshold(predictions[arm], truths, members, threshold)['counts']
            counts = {name: scores[name] for name in limits}
            if all(value <= limits[name] for name, value in counts.items()):
                thresholds[arm], budget_counts[arm] = threshold, counts
                break
    assert thresholds.keys() == set(ARMS)
    return {'thresholds': thresholds, 'reference': 'T at .25', 'count_ceilings': limits, 'selected_counts': budget_counts,
            'provenance': provenance}


def aggregate(root, protocol, selected):
    results = {}
    for cohort, group_count in SCORED.items():
        predictions, truths, members, provenance = cohort_inputs(root, cohort, protocol['parent_checkpoint_sha256'])
        setups = {name: row['setup'] for name, row in members.items()}
        assert len(set(setups.values())) == group_count
        draws = np.random.default_rng(SEED).integers(group_count, size=(REPLICATES, group_count))
        operating_points = {}
        for regime, thresholds in [('fixed_0_25', dict.fromkeys(ARMS, FIXED)), ('calibrated', selected)]:
            parts = {arm: case_counts(score_at_threshold(predictions[arm], truths, members, threshold), setups, group_count)
                     for arm, threshold in thresholds.items()}
            operating_points[regime] = {'arms': {arm: {'threshold': thresholds[arm], 'counts': dict(value[2]),
                                                       'ratios': {metric: compute_ratio(value[2], *fields) for metric, fields in ARM_RATIOS.items()}}
                                                 for arm, value in parts.items()},
                                         'contrasts': {f'{arm}_minus_{reference}': paired_effect(parts[arm], parts[reference], draws)
                                                       for arm, reference in CONTRASTS}}
        if cohort == 'original100':
            assert all(arm['counts']['shared_objects'] == 1696 for arm in operating_points['fixed_0_25']['arms'].values())
        results[cohort] = {'resampling_groups': group_count, 'sources': provenance, **operating_points}
    return results


def promotion_eligibility(results, protocol, quality):
    rule = protocol['refresh_qualification']
    assert rule['development_gain_pp_min'] == 3 and rule['original100_gain_pp_min'] == -1
    assert rule['development_prediction_count_ratio_max'] == rule['development_unmapped_count_ratio_max'] == 1.05
    assert rule['development_gain_lower95_min_exclusive'] == 0 and rule['teacher_quality_gate_required']
    development = results['development100']['calibrated']
    original = results['original100']['calibrated']
    options = {}
    for arm in ('S', 'L'):
        d_effect = development['contrasts'][f'{arm}_minus_T']['effects']['both_view_coverage']
        d_c, d_t = (development['arms'][lane]['counts'] for lane in (arm, 'T'))
        o_c, o_t = (original['arms'][lane]['counts'] for lane in (arm, 'T'))
        checks = {'teacher_quality': quality['qualification'][arm],
                  'development_gain_3pp': 100 * (d_c['shared_localized_both'] - d_t['shared_localized_both']) >= 3 * d_t['shared_objects'],
                  'development_ci_lower_positive': d_effect['paired_cluster_ci95'][0] > 0,
                  'development_prediction_cap': 20 * d_c['detections'] <= 21 * d_t['detections'],
                  'development_unmapped_cap': 20 * d_c['unmapped'] <= 21 * d_t['unmapped'],
                  'original100_regression_within_1pp': 100 * (o_c['shared_localized_both'] - o_t['shared_localized_both']) >= -o_t['shared_objects']}
        options[arm] = {'eligible': all(checks.values()), 'checks': checks,
                        'failed_checks': [name for name, passed in checks.items() if not passed]}
    eligible = [arm for arm in options if options[arm]['eligible']]
    winner = min(eligible, key=lambda arm: (-development['arms'][arm]['counts']['shared_localized_both'],
                                           development['arms'][arm]['counts']['unmapped'], arm != 'S')) if eligible else None
    return {'arms': options, 'selected_for_possible_refresh': winner}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--runs', type=Path, default=OUT.parent / 'engineering/runs')
    args = parser.parse_args()
    assert digest(args.protocol) == args.protocol_sha256
    protocol = load(args.protocol)
    assert protocol['evaluation']['cohorts'] == ['calibration30', 'development100', 'original100']
    assert protocol['evaluation']['fixed_detection_cutoff'] == FIXED
    output = OUT / 'independent-evaluation.json'
    calibrations = OUT / 'independent-calibration.json'
    freeze = OUT / 'independent-evaluation-freeze.json'
    assert not (output.exists() or freeze.exists()), 'Preserve a completed independent evaluation'
    quality_path = OUT / 'teacher-quality.json'
    assert digest(quality_path) == load(OUT / 'audit-freeze.json')[quality_path.relative_to(ROOT).as_posix()]
    quality = load(quality_path)
    assert quality['protocol_sha256'] == args.protocol_sha256
    root = args.runs.resolve()
    schedules = training_schedule(root, protocol, args.protocol_sha256)
    selected = calibration(root, protocol)
    calibration_document = {**selected, 'protocol_sha256': args.protocol_sha256}
    if calibrations.exists():
        assert load(calibrations) == calibration_document, 'Calibration differs from a prior incomplete analysis'
    else:
        save(calibrations, calibration_document)
    results = aggregate(root, protocol, selected['thresholds'])
    eligibility = promotion_eligibility(results, protocol, quality)
    save(output, {'protocol_sha256': args.protocol_sha256, 'teacher_quality_sha256': digest(quality_path),
                  'calibration_sha256': digest(calibrations), 'paired_bootstrap_draws': REPLICATES,
                  'paired_bootstrap_seed': SEED, 'primary_mapper_sha256': digest(ROOT / 'evaluation/run_full_scene.py'),
                  'threshold_scorer_sha256': digest(LEGACY / 'score_detector.py'),
                  'calibration': selected, 'training_exposures': schedules, 'cohorts': results, 'refresh_qualification': eligibility,
                  'scope': 'Previously exposed native development and historical regression cohorts. No fresh test accessed.'})
    save(freeze, {path.relative_to(ROOT).as_posix(): digest(path) for path in (output, calibrations, Path(__file__).resolve())})
    print('Calibration:', selected['thresholds'], selected['count_ceilings'])
    print('Refresh qualification:', eligibility)


if __name__ == '__main__':
    main()
