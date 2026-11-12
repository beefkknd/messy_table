#!/usr/bin/env python3
"""Verify a frozen stronger teacher, score its matched 240 frames, and export the complete oracle control."""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys

import numpy as np

from freeze_cohorts import COHORTS, OLD_DATA, OUT, ROOT, digest, load, rank, save, verify_native_pool

sys.path.insert(0, str(OLD_DATA))
from prepare import LABELS, truth_rows
from audit_pseudo180_diagnostics import match_at_iou, montage, target_size, unmapped_proxy

SUPPLEMENT = OUT / 'cohorts/supplemental240.json'
POOL = COHORTS / 'pseudo1000.json'
REQUIRED = {'teacher-plan.json', 'raw-predictions.json', 'admitted.json', 'frames-single.json', 'frames-ladder.json'}
BRANCHES = {'rgb_768': 768, 'rgb_1536': 1536, 'gray_768': 768}
DRAWS = 20000
SEED = 20261110


def frame_rows(manifest):
    return [(f"{row['id']}-{view['id']}", row, view) for row in load(manifest)['scenes'] for view in row['views']]


def box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / (box_area(a) + box_area(b) - area)


def valid_seed(seed, view, plan):
    x1, y1, x2, y2 = seed['box_xyxy']
    return (all(map(math.isfinite, (x1, y1, x2, y2))) and seed['class'] == plan['class']
            and 0 <= x1 < x2 <= view['width'] and 0 <= y1 < y2 <= view['height']
            and min(x2 - x1, y2 - y1) >= plan['min_side_px']
            and box_area(seed['box_xyxy']) <= plan['max_frame_area_fraction'] * view['width'] * view['height'])


def validate_seed_source(seed, branches):
    branch, index = seed['source_branch'], seed['source_index']
    assert 0 <= index < len(branches[branch])
    actual = branches[branch][index]
    assert all(seed[field] == actual[field] for field in ('class', 'confidence', 'box_xyxy'))


def validate_run(run, expected_freeze, protocol_path, expected_protocol, amendment_path, expected_amendment):
    assert digest(protocol_path) == expected_protocol
    assert digest(amendment_path) == expected_amendment
    protocol = load(protocol_path)
    amendment = load(amendment_path)
    assert amendment['parent_protocol_sha256'] == expected_protocol and amendment['bootstrap_seed'] == SEED
    assert amendment['support_confidence_operator'] == 'strict >0.45; seed>=0.80;raw floor>0.25'
    assert protocol['teacher_pool_sha256'] == digest(POOL) and protocol['supplemental_sha256'] == digest(SUPPLEMENT)
    freeze_path = run / 'freeze.json'
    assert digest(freeze_path) == expected_freeze
    frozen = load(freeze_path)
    assert set(frozen) == REQUIRED and all(digest(run / name) == expected for name, expected in frozen.items())
    for relative, expected in load(OUT / 'selection-freeze.json').items():
        assert digest(ROOT / relative) == expected, relative
    source_hashes = load(OUT / 'inventory.json')['source_sha256']
    for relative, expected in source_hashes.items():
        assert digest(ROOT / relative) == expected, relative
    plan_path = run / 'teacher-plan.json'
    assert max(protocol_path.stat().st_mtime_ns, amendment_path.stat().st_mtime_ns) <= plan_path.stat().st_mtime_ns
    assert SUPPLEMENT.stat().st_mtime_ns <= plan_path.stat().st_mtime_ns
    plan = load(plan_path)
    assert Path(plan['pool']).resolve() == POOL and plan['pool_sha256'] == digest(POOL)
    assert Path(plan['supplemental']).resolve() == SUPPLEMENT and plan['supplemental_sha256'] == digest(SUPPLEMENT)
    checkpoint = (OLD_DATA.parent / 'engineering/round2/train/A/checkpoint.pt').resolve()
    assert Path(plan['checkpoint']).resolve() == checkpoint and digest(checkpoint) == plan['source_weights_sha256'] == protocol['parent_checkpoint_sha256']
    for file, expected in plan['code_sha256'].items():
        path = Path(file)
        assert path.is_relative_to(ROOT) and digest(path) == expected, path
    assert plan['branches'] == BRANCHES and plan['inference_head'] == 'one2many'
    assert plan['nms_iou'] == .7 and plan['max_det'] == 300
    assert plan['confidence_floor'] == protocol['teacher']['raw_confidence_floor'] == .25
    assert plan['minimum_alternate_support_confidence_exclusive'] == protocol['teacher']['support_confidence'] == .45
    assert plan['seed_confidence'] == .8 and plan['dedup_iou'] == plan['support_iou'] == .55
    assert plan['min_side_px'] == 12 and plan['max_frame_area_fraction'] == .30 and plan['max_accepted_per_frame'] == 64
    assert plan['class'] == 'tabletop_item'
    pool_rows = frame_rows(POOL)
    subset_rows = frame_rows(SUPPLEMENT)
    verify_native_pool(load(POOL)['scenes'])
    raw = load(run / 'raw-predictions.json')['frames']
    admitted = load(run / 'admitted.json')['frames']
    single = load(run / 'frames-single.json')['frames']
    ladder = load(run / 'frames-ladder.json')['frames']
    assert len(pool_rows) == len(raw) == len(admitted) == 1000
    assert len(subset_rows) == len(single) == len(ladder) == 240
    assert len({scene['setup'] for _, scene, _ in subset_rows}) == 91
    admitted_index = {}
    for (row_id, scene, view), raw_row, row in zip(pool_rows, raw, admitted, strict=True):
        for item in (raw_row, row):
            assert item['row_id'] == row_id and item['scene'] == scene['id'] and item['setup'] == scene['setup']
            assert item['view'] == view
        assert row_id not in admitted_index and raw_row['branches'].keys() == BRANCHES.keys()
        admitted_index[row_id] = row
        branches = raw_row['branches']
        joined = row['original'] + row['supplements']
        assert len(joined) <= plan['max_accepted_per_frame']
        for index, seed in enumerate(joined):
            validate_seed_source(seed, branches)
            assert valid_seed(seed, view, plan) and seed['confidence'] >= plan['seed_confidence']
            assert all(box_iou(seed['box_xyxy'], other['box_xyxy']) < plan['dedup_iou'] for other in joined[:index])
            if index < len(row['original']):
                assert seed['admission'] == 'one_pass_seed' and seed['source_branch'] == 'rgb_768'
            else:
                assert seed['admission'] == 'supported_supplement' and seed['source_branch'] != 'rgb_768'
                support = seed['support']
                validate_seed_source(support, branches)
                assert valid_seed(support, view, plan) and support['source_branch'] != seed['source_branch']
                assert support['confidence'] > plan['minimum_alternate_support_confidence_exclusive']
                overlap = box_iou(support['box_xyxy'], seed['box_xyxy'])
                assert overlap >= plan['support_iou'] and math.isclose(seed['support_iou'], overlap)
    subset_admissions = []
    for (row_id, scene, view), s, c in zip(subset_rows, single, ladder, strict=True):
        admitted = admitted_index[row_id]
        subset_admissions.append(admitted)
        expected = {'row_id': row_id, 'scene': scene['id'], 'setup': scene['setup'], 'view': view['id'],
                    'partition': 'train', 'scope': 'partial_frame', 'image': view['path'], 'image_sha256': view['sha256']}
        for row, seeds in ((s, admitted['original']), (c, admitted['original'] + admitted['supplements'])):
            assert all(row[field] == value for field, value in expected.items())
            boxes = [seed['box_xyxy'] for seed in seeds]
            assert row['boxes_xyxy'] == boxes and row['allow_empty_partial'] == (not boxes)
    return frozen, plan, subset_admissions, single, list(admitted_index.values()), protocol['teacher_quality_gate']


def mapped(predictions, truth, threshold):
    return match_at_iou(predictions, truth, threshold)


def metrics(count):
    result = {}
    for threshold in (50, 75):
        for arm in ('single', 'ladder', 'supplement'):
            n = count[f'{arm}_mapped_{threshold}']
            denom = count[f'{arm}_seeds']
            result[f'{arm}_precision_iou{threshold}'] = n / denom if denom else None
            if arm != 'supplement':
                result[f'{arm}_recall_iou{threshold}'] = n / count['gt_endpoints']
        result[f'ladder_minus_single_recall_iou{threshold}'] = (count[f'ladder_mapped_{threshold}'] - count[f'single_mapped_{threshold}']) / count['gt_endpoints']
    for arm in ('single', 'ladder'):
        result[f'{arm}_omitted_endpoint_rate_iou50'] = 1 - result[f'{arm}_recall_iou50']
        result[f'{arm}_AB_both_coverage_iou50'] = count[f'{arm}_AB_both'] / count['AB_shared'] if count['AB_shared'] else None
    return result


def bootstrap(setups):
    setup_ids = sorted(setups)
    weights = np.random.default_rng(SEED).multinomial(len(setup_ids), np.full(len(setup_ids), 1 / len(setup_ids)), size=DRAWS)
    fields = sorted({field for counts in setups.values() for field in counts})
    matrix = np.array([[setups[setup][field] for field in fields] for setup in setup_ids], dtype=np.int64)
    measures = defaultdict(list)
    for draw in (weights @ matrix).tolist():
        pooled = Counter(dict(zip(fields, draw, strict=True)))
        for name, value in metrics(pooled).items():
            if value is not None:
                measures[name].append(value)
    return {name: np.quantile(values, [.025, .975]).tolist() for name, values in measures.items()}


def summarize(admissions, truth, previously_audited):
    setup_counts = defaultdict(Counter)
    strata_counts = defaultdict(Counter)
    geometry = defaultdict(Counter)
    mapping50 = {}
    for row in admissions:
        side = row['view']['id']
        boxes = truth[row['scene']]['views'][side]['objects_xyxy']
        initial, additions = row['original'], row['supplements']
        final = initial + additions
        count = setup_counts[row['setup']]
        count.update(frames=1, gt_endpoints=len(boxes), single_seeds=len(initial), ladder_seeds=len(final),
                     supplement_seeds=len(additions), single_empty=not initial, ladder_empty=not final,
                     images_with_supplements=bool(additions))
        for threshold in (50, 75):
            b, c = mapped(initial, boxes, threshold / 100), mapped(final, boxes, threshold / 100)
            add_matches = {index: object_id for index, object_id in c.items() if index >= len(initial)}
            b_ids, c_ids = set(b.values()), set(c.values())
            count.update({f'single_mapped_{threshold}': len(b), f'ladder_mapped_{threshold}': len(c),
                          f'supplement_mapped_{threshold}': len(add_matches),
                          f'newly_mapped_{threshold}': len(c_ids - b_ids), f'previously_mapped_lost_{threshold}': len(b_ids - c_ids),
                          f'supplement_direct_new_{threshold}': len(set(add_matches.values()) - b_ids)})
            if threshold == 50:
                mapping50[row['scene'], side] = b_ids, c_ids, set(map(int, boxes))
                for index, seed in enumerate(final):
                    band = 'single' if index < len(initial) else 'supplement'
                    lane = f"{band}|{seed['source_branch']}"
                    geometry[lane].update(candidates=1, mapped=index in c)
                    bucket = '0.80-<0.85' if seed['confidence'] < .85 else '0.85-<0.90' if seed['confidence'] < .90 else '0.90-1.00'
                    geometry[f'{band}|seed_confidence|{bucket}'].update(candidates=1, mapped=index in c)
                    if index not in c:
                        geometry[lane][unmapped_proxy(seed['box_xyxy'], boxes)] += 1
                for oid, box in boxes.items():
                    bucket = target_size(box)
                    strata_counts[f'short_side|{bucket}'].update(gt=1, single_mapped=int(oid) in b_ids, ladder_mapped=int(oid) in c_ids)
                    if min(box[:2]) == 0 or box[2] == row['view']['width'] or box[3] == row['view']['height']:
                        strata_counts['frame_edge'].update(gt=1, single_mapped=int(oid) in b_ids, ladder_mapped=int(oid) in c_ids)
    for scene, info in truth.items():
        a, b = (mapping50[scene, side] for side in 'AB')
        shared = a[2] & b[2]
        setup_counts[info['setup']].update(AB_shared=len(shared), single_AB_both=len(shared & a[0] & b[0]),
                                           ladder_AB_both=len(shared & a[1] & b[1]))
    total = Counter()
    historical_setup_strata = defaultdict(Counter)
    for setup, counts in setup_counts.items():
        total.update(counts)
        historical_setup_strata['previously_seen_setup' if setup in previously_audited else 'new_setup'].update(counts)
    for threshold in (50, 75):
        total[f'newly_mapped_setups_{threshold}'] = sum(count[f'newly_mapped_{threshold}'] > 0 for count in setup_counts.values())
        total[f'supplement_direct_new_setups_{threshold}'] = sum(count[f'supplement_direct_new_{threshold}'] > 0 for count in setup_counts.values())
        for arm in ('single', 'ladder'):
            total[f'{arm}_omitted_{threshold}'] = total['gt_endpoints'] - total[f'{arm}_mapped_{threshold}']
    return {'counts': total, 'metrics': metrics(total), 'paired_setup_bootstrap_ci95': bootstrap(setup_counts),
            'by_setup': setup_counts, 'by_target_size_and_edge': strata_counts, 'by_admission_source_iou50': geometry,
            'by_prior_setup_exposure': {name: {'counts': counts, 'metrics': metrics(counts)} for name, counts in historical_setup_strata.items()}}


def review_sheet(admissions, truth, destination):
    examples = {'RECOVERED': {}, 'UNMAPPED': {}}
    for row in admissions:
        side = row['view']['id']
        boxes = truth[row['scene']]['views'][side]['objects_xyxy']
        initial, additions = row['original'], row['supplements']
        old = set(mapped(initial, boxes, .50).values())
        all_matches = mapped(initial + additions, boxes, .50)
        for index, seed in enumerate(additions, len(initial)):
            result = all_matches.get(index)
            outcome = 'UNMAPPED' if result is None else 'RECOVERED' if result not in old else None
            if outcome is None:
                continue
            identity = f"{row['row_id']}|{seed['source_branch']}|{seed['source_index']}"
            caption = f"{outcome} | {seed['source_branch']} | {seed['confidence']:.2f}"
            examples[outcome][identity] = row['setup'], (row['view']['path'], seed['box_xyxy'], list(boxes.values()), caption)
    tiles = []
    counts = {}
    for outcome, candidates in examples.items():
        setups = set()
        for identity in rank(candidates, f'post-freeze-sheet|{outcome}'):
            setup, example = candidates[identity]
            if setup not in setups:
                setups.add(setup)
                tiles.append(example)
            if len(setups) == 6:
                break
        counts[outcome] = len(setups)
    assert tiles
    montage(tiles, destination)
    return {'sample_counts': counts, 'selection': 'SHA-ranked additions, one per setup in each lane. Native boxes green, admitted addition pink. Diagnosis after prediction and admission freeze; illustrative sample.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--amendment', type=Path, required=True)
    parser.add_argument('--amendment-sha256', required=True)
    parser.add_argument('--teacher-run', type=Path, required=True)
    parser.add_argument('--teacher-freeze-sha256', required=True)
    parser.add_argument('--reveal-selected-native-train', action='store_true')
    parser.add_argument('--audit-full-pool-descriptively', action='store_true')
    args = parser.parse_args()
    if args.audit_full_pool_descriptively and not args.reveal_selected_native_train:
        parser.error('Descriptive scoring requires explicitly opening the selected native-train audit lane.')
    output = OUT / 'teacher-quality.json'
    oracle_path = OUT / 'oracle/supplemental240-complete.json'
    freeze_path = OUT / 'audit-freeze.json'
    sheet_path = OUT / 'teacher-additions-review.jpg'
    frozen, plan, subset, pseudo_rows, pool, gate = validate_run(args.teacher_run.resolve(), args.teacher_freeze_sha256,
        args.protocol.resolve(), args.protocol_sha256, args.amendment.resolve(), args.amendment_sha256)
    print('Frozen 1,000-frame run and exact ordered 240-frame admission rows verified.')
    if not args.reveal_selected_native_train:
        return
    assert not any(path.exists() for path in (output, oracle_path, freeze_path, sheet_path))
    source = LABELS['train']
    source_hash = digest(source)
    assert source_hash == load(OLD_DATA / 'audit.json')['source_sha256'][source.relative_to(ROOT).as_posix()]
    # Only the selected native-training scene records are decoded after verifying all frozen inputs.
    truth = {row['id']: row for row in truth_rows(load(SUPPLEMENT)['scenes'])}
    old_scenes = load(COHORTS / 'pseudo180.json')['scenes']
    audited_setups = {row['setup'] for row in old_scenes}
    report = summarize(subset, truth, audited_setups)
    overall = {arm: report['metrics'][f'{arm}_precision_iou50'] >= gate['overall_point_precision_min']
               and report['paired_setup_bootstrap_ci95'][f'{arm}_precision_iou50'][0] >= gate['overall_setup_bootstrap_lower95_min']
               for arm in ('single', 'ladder')}
    marginal = report['metrics']['supplement_precision_iou50']
    ladder_checks = {'overall': overall['ladder'], 'marginal_precision': marginal is not None and marginal >= gate['marginal_ladder_precision_min'],
                     'new_correct_endpoints': report['counts']['supplement_direct_new_50'] >= gate['new_correct_endpoint_min'],
                     'new_endpoint_setups': report['counts']['supplement_direct_new_setups_50'] >= gate['new_endpoint_setups_min']}
    report['qualification'] = {'scope': 'Supplemental240 only', 'S': overall['single'], 'L': all(ladder_checks.values()),
                               'L_checks': ladder_checks, 'frozen_requirements': gate}
    descriptive = {}
    if args.audit_full_pool_descriptively:
        full_truth = {row['id']: row for row in truth_rows(load(POOL)['scenes'])}
        old_ids = {row['id'] for row in old_scenes}
        groups = {'all1000': pool, 'previously_audited180': [row for row in pool if row['scene'] in old_ids],
                  'other820': [row for row in pool if row['scene'] not in old_ids]}
        assert [len(values) for values in groups.values()] == [1000, 180, 820]
        for name, values in groups.items():
            group_truth = {row['scene']: full_truth[row['scene']] for row in values}
            descriptive[name] = summarize(values, group_truth, audited_setups)
    oracle = []
    for row in pseudo_rows:
        boxes = truth[row['scene']]['views'][row['view']]['objects_xyxy']
        assert boxes
        keys = ('row_id', 'scene', 'setup', 'view', 'partition', 'image', 'image_sha256')
        oracle.append({**{key: row[key] for key in keys}, 'scope': 'complete_frame',
                       'boxes_xyxy': [boxes[oid] for oid in sorted(boxes, key=int)], 'boxes': len(boxes)})
    assert sum(row['boxes'] for row in oracle) == report['counts']['gt_endpoints']
    header = {'version': 1, 'selection_sha256': digest(SUPPLEMENT), 'teacher_freeze_sha256': args.teacher_freeze_sha256,
              'protocol_sha256': args.protocol_sha256, 'amendment_sha256': args.amendment_sha256,
              'source_native_train_sha256': source_hash}
    sheet = review_sheet(subset, truth, sheet_path)
    save(output, {**header, 'teacher_checkpoint_sha256': plan['source_weights_sha256'], 'teacher_artifact_hashes': frozen,
                  'setup_bootstrap_draws': DRAWS, 'setup_bootstrap_seed': SEED,
                  'mapping': 'One-to-one Hungarian native-pixel assignment, class-agnostic, separately at IoU 0.50 and 0.75.',
                  'scope': 'Primary scoring and eligibility use only the 240 metadata-selected frames. Pool/subgroup results are descriptive.',
                  'descriptive_pool_audits': descriptive, 'post_freeze_contact_sheet': sheet, **report})
    save(oracle_path, {**header, 'scope': 'Privileged complete-label control on the exact same ordered RGB frames; no instance IDs or category labels.',
                       'frames': oracle})
    save(freeze_path, {path.relative_to(ROOT).as_posix(): digest(path) for path in (output, oracle_path, sheet_path, Path(__file__).resolve())})
    print(report['counts'])
    print(report['metrics'])


if __name__ == '__main__':
    main()
