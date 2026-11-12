#!/usr/bin/env python3
"""Score the Step0 tiled detector against its frozen full-frame A300 inputs."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
PROGRAM = OUT.parents[1]
ROOT = PROGRAM.parents[1]
PRIOR = PROGRAM / 'data'
ENGINEERING = PROGRAM / 'engineering'
sys.path[:0] = [str(ENGINEERING), str(PRIOR)]
from prepare import digest, load, save
from score_detector import map_detections, score_at_threshold, xyxy_to_xywh
from summarize_round2 import RATIOS, REPLICATES, SEED, case_counts, compute_ratio, paired_effect

COHORTS = {'calibration30': (30, 30, 672), 'development100': (100, 100, 2173), 'original100': (100, 83, 1696)}
CONFIDENCE = .25
A300 = ENGINEERING / 'round2/train/A/checkpoint.pt'
FULL_ROOT = ENGINEERING / 'round2/inference/A'
TILES_ROOT = OUT.parent / 'engineering/inference'
SIZE_BUCKETS = ('under_32', '32_to_under_96', 'at_least_96')
ARM_RATIOS = {**RATIOS, 'localization_precision_iou50': ('localized', 'detections')}


def cohort_inputs(cohort):
    cohort_path = PRIOR / 'cohorts' / f'{cohort}.json'
    truth_path = PRIOR / 'scoring' / f'{cohort}.json'
    manifest, key = load(cohort_path), load(truth_path)
    assert key['cohort_sha256'] == digest(cohort_path)
    scenes = {row['id']: row for row in manifest['scenes']}
    truths = {row['id']: row for row in key['scenes']}
    pairs, groups, expected = COHORTS[cohort]
    assert len(manifest['scenes']) == len(key['scenes']) == len(scenes) == pairs and scenes.keys() == truths.keys()
    assert len({row['setup'] for row in scenes.values()}) == groups
    gt_shared = 0
    for case, scene in scenes.items():
        gold = truths[case]
        assert gold['setup'] == scene['setup'] and set(gold['views']) == {'A', 'B'}
        gt_shared += len(gold['views']['A']['objects_xyxy'].keys() & gold['views']['B']['objects_xyxy'].keys())
    assert gt_shared == expected
    return cohort_path, truth_path, scenes, truths


def verified_predictions(path, manifest_path, scenes, weights_sha):
    freeze_path = path / 'freeze.json'
    freeze = load(freeze_path)
    assert set(freeze) == {'inference-plan.json', 'predictions.json'}
    assert all(digest(path / filename) == expected for filename, expected in freeze.items()), path
    plan = load(path / 'inference-plan.json')
    assert plan['manifest_sha256'] == digest(manifest_path)
    assert plan['weights_sha256'] == weights_sha and plan['confidence_floor'] <= CONFIDENCE
    document = load(path / 'predictions.json')
    rows = document['scenes']
    by_id = {row['id']: row for row in rows}
    assert len(rows) == len(by_id) and by_id.keys() == scenes.keys()
    for scene_id, prediction in by_id.items():
        expected = scenes[scene_id]
        assert prediction['setup'] == expected['setup']
        actual = {key: value['sha256'] for key, value in prediction['views'].items()}
        assert actual == {view['id']: view['sha256'] for view in expected['views']}
    timing_keys = ('wall_seconds', 'frames', 'tile_passes', 'total_detector_passes_conceptual',
                   'median_cached_full_frame_seconds', 'median_tile_pipeline_seconds', 'median_combined_seconds')
    return rows, plan, {'freeze_sha256': digest(freeze_path), **freeze,
                        'recorded_runtime': {key: document[key] for key in timing_keys if key in document}}


def at_cutoff(row):
    return [detection for detection in row['detections'] if detection['confidence'] > CONFIDENCE]


def historical_boxes(view):
    return view.get('objects_xywh_historical') or {key: xyxy_to_xywh(box) for key, box in view['objects_xyxy'].items()}


def size_bucket(box):
    short_side = min(box[2:])
    return 'under_32' if short_side < 32 else '32_to_under_96' if short_side < 96 else 'at_least_96'


def endpoint_effects(full, tiles, truths, full_cases, tile_cases, draws, groups):
    full_by_id = {row['id']: row for row in full}
    tile_by_id = {row['id']: row for row in tiles}
    stats = defaultdict(Counter)
    case_rows = []
    timings = Counter()
    for case_id, full_row in full_by_id.items():
        current = Counter()
        tile_row = tile_by_id[case_id]
        case_views = {}
        for side in 'AB':
            base_view, combined_view = full_row['views'][side], tile_row['views'][side]
            base, combined = at_cutoff(base_view), at_cutoff(combined_view)
            assert combined[:len(base)] == base, (case_id, side, 'Tiled merge did not preserve the full-frame prefix')
            boxes = historical_boxes(truths[case_id]['views'][side])
            base_map, combined_map = map_detections(base, boxes), map_detections(combined, boxes)
            full_ids, tiled_ids = set(base_map.values()), set(combined_map.values())
            assert full_ids == set(full_cases[case_id]['views'][side]['localized_ids'])
            assert tiled_ids == set(tile_cases[case_id]['views'][side]['localized_ids'])
            gained, lost = tiled_ids - full_ids, full_ids - tiled_ids
            marginal = {index: object_id for index, object_id in combined_map.items() if index >= len(base)}
            count = Counter(tile_additions=len(combined) - len(base), additions_mapped_in_combined=len(marginal),
                            additions_to_previously_unmapped=len(set(marginal.values()) - full_ids),
                            endpoint_gains=len(gained), endpoint_losses=len(lost))
            for oid, xywh in boxes.items():
                bucket = size_bucket(xywh)
                numeric = int(oid)
                count[f'{bucket}_gt'] += 1
                count[f'{bucket}_full_mapped'] += numeric in full_ids
                count[f'{bucket}_tiles_mapped'] += numeric in tiled_ids
                count[f'{bucket}_gains'] += numeric in gained
                count[f'{bucket}_losses'] += numeric in lost
            current.update(count)
            case_views[side] = {'counts': dict(count), 'gained_ids': sorted(gained), 'lost_ids': sorted(lost)}
            for key in ('tile_passes', 'tile_seconds', 'tile_pipeline_seconds', 'cached_full_frame_seconds'):
                if isinstance(combined_view.get(key), (int, float)):
                    timings[key] += combined_view[key]
            if isinstance(combined_view.get('added_detections'), int):
                assert combined_view['added_detections'] == count['tile_additions']
        stats[full_row['setup']].update(current)
        case_rows.append({'id': case_id, 'setup': full_row['setup'], 'counts': dict(current), 'views': case_views,
                          'both_view_ids_gained': sorted(set(tile_cases[case_id]['shared_ids_localized_both']) - set(full_cases[case_id]['shared_ids_localized_both'])),
                          'both_view_ids_lost': sorted(set(full_cases[case_id]['shared_ids_localized_both']) - set(tile_cases[case_id]['shared_ids_localized_both']))})
    assert len(stats) == groups
    overall = Counter()
    for count in stats.values():
        overall.update(count)
    sizes = {}
    setup_ids = sorted(stats)
    for bucket in SIZE_BUCKETS:
        fields = (f'{bucket}_gt', f'{bucket}_full_mapped', f'{bucket}_tiles_mapped', f'{bucket}_gains', f'{bucket}_losses')
        gt, baseline, candidate, gain, loss = (overall[field] for field in fields)
        assert candidate - baseline == gain - loss
        matrix = np.array([[stats[setup][field] for field in fields[:3]] for setup in setup_ids])
        sample = matrix[draws].sum(axis=1)
        valid = sample[:, 0] > 0
        differences = (sample[valid, 2] - sample[valid, 1]) / sample[valid, 0]
        sizes[bucket] = {'gt_endpoints': gt, 'full_frame_mapped': baseline, 'tiled_merged_mapped': candidate,
                         'gained': gain, 'lost': loss, 'recall_delta': (candidate - baseline) / gt if gt else None,
                         'paired_setup_ci95': np.quantile(differences, [.025, .975]).tolist() if differences.size else None}
    return {'counts': {key: value for key, value in overall.items() if not any(key.startswith(bucket) for bucket in SIZE_BUCKETS)},
            'native_short_side': sizes, 'recorded_view_runtime_totals': dict(timings)}, case_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--tile-runs', type=Path, default=TILES_ROOT)
    args = parser.parse_args()
    assert digest(args.protocol) == args.protocol_sha256
    protocol = load(args.protocol)
    assert protocol['evaluation']['detector_confidence'] == CONFIDENCE and protocol['evaluation']['mapping_iou'] == .5
    assert protocol['evaluation']['short_side_bins_native_px'] == [32, 96]
    assert protocol['evaluation']['bootstrap_draws'] == REPLICATES and protocol['evaluation']['seed'] == SEED
    assert set(protocol['cohorts']) == set(COHORTS) and not protocol['training']
    global_freeze_path = args.tile_runs / 'freeze.json'
    global_freeze = load(global_freeze_path)
    for relative, expected_hash in global_freeze.items():
        assert digest(args.tile_runs.parent / relative) == expected_hash, relative
    lock_path = args.tile_runs / 'pre-inference-lock.json'
    lock = load(lock_path)
    assert lock['protocol_sha256'] == args.protocol_sha256 and args.protocol.stat().st_mtime_ns <= lock_path.stat().st_mtime_ns
    replay = load(args.tile_runs / 'replay.json')
    assert replay['passed'] and len(replay['frames']) == 18
    output, case_output, freeze = (OUT / name for name in ('detector-comparison.json', 'detector-cases.json', 'detector-freeze.json'))
    assert not any(path.exists() for path in (output, case_output, freeze))
    weights_sha = digest(A300)
    assert weights_sha == protocol['weights_sha256'] == lock['weights_sha256']
    comparisons, per_case = {}, {}
    for cohort, (pairs, groups, expected) in COHORTS.items():
        manifest, key, scenes, truths = cohort_inputs(cohort)
        full, base_plan, base_prov = verified_predictions(FULL_ROOT / cohort, manifest, scenes, weights_sha)
        tiles, tile_plan, tile_prov = verified_predictions(args.tile_runs / cohort, manifest, scenes, weights_sha)
        assert digest(manifest) == protocol['cohorts'][cohort]
        assert (base_plan['imgsz'], base_plan['head'], base_plan['nms_iou']) == (768, 'one2many', .7)
        assert (tile_plan['imgsz'], tile_plan['head'], tile_plan['nms_iou']) == (768, 'one2many', .7)
        assert tile_plan['protocol_sha256'] == args.protocol_sha256
        assert tile_plan['cached_full_frame_freeze_sha256'] == base_prov['freeze_sha256']
        assert tile_plan['tile_stride_px'] == protocol['tiling']['stride']
        assert tile_plan['artificial_edge_rejection_inclusive_px'] == protocol['tiling']['interior_edge_margin_px']
        assert tile_plan['greedy_dedup_iou_inclusive'] == protocol['tiling']['merge_iou']
        members = {case: row['setup'] for case, row in scenes.items()}
        scored = {name: case_counts(score_at_threshold(rows, truths, scenes, CONFIDENCE), members, groups)
                  for name, rows in [('full_frame', full), ('tiled_merged', tiles)]}
        resamples = np.random.default_rng(SEED).integers(groups, size=(REPLICATES, groups))
        base, candidate = scored['full_frame'], scored['tiled_merged']
        assert base[2]['shared_objects'] == candidate[2]['shared_objects'] == expected
        effects, case_rows = endpoint_effects(full, tiles, truths, base[0], candidate[0], resamples, groups)
        assert effects['counts']['endpoint_gains'] - effects['counts']['endpoint_losses'] == candidate[2]['localized'] - base[2]['localized']
        assert effects['counts']['tile_additions'] == candidate[2]['detections'] - base[2]['detections']
        comparisons[cohort] = {'pairs': pairs, 'resampling_groups': groups,
                               'manifest_sha256': digest(manifest), 'scoring_key_sha256': digest(key),
                               'full_frame_source': base_prov, 'tiles_source': tile_prov,
                               'arms': {name: {'counts': dict(parts[2]), 'ratios': {metric: compute_ratio(parts[2], *fields)
                                                                                 for metric, fields in ARM_RATIOS.items()}}
                                        for name, parts in scored.items()},
                               'tiles_minus_full_frame': paired_effect(candidate, base, resamples), 'endpoints': effects}
        per_case[cohort] = case_rows
    save(output, {'protocol_sha256': args.protocol_sha256, 'checkpoint_sha256': weights_sha,
                  'detector_confidence_strictly_greater_than': CONFIDENCE,
                  'historical_mapper_sha256': digest(ROOT / 'evaluation/run_full_scene.py'),
                  'tile_inference_global_freeze_sha256': digest(global_freeze_path),
                  'historical_replay_frames': len(replay['frames']),
                  'bootstrap_seed': SEED, 'paired_setup_bootstrap_draws': REPLICATES, 'cohorts': comparisons})
    save(case_output, {'protocol_sha256': args.protocol_sha256, 'cohorts': per_case})
    save(freeze, {path.relative_to(ROOT).as_posix(): digest(path) for path in (output, case_output, Path(__file__).resolve())})
    print({name: {arm: record['counts']['shared_localized_both'] for arm, record in item['arms'].items()}
           for name, item in comparisons.items()})


if __name__ == '__main__':
    main()
