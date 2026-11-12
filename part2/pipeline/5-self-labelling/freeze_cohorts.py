#!/usr/bin/env python3
"""Verify the prior pseudo pool and select a label-blind matched supplemental cohort."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parents[1]
ROOT = PRIOR.parents[1]
OLD_DATA = PRIOR / 'data'
COHORTS = OLD_DATA / 'cohorts'
FULL = ROOT / 'data/messytable-full'
NATIVE = FULL / 'MessyTable/images'
SEED = '2026-11-10-stronger-teacher-supplemental240-v1'
SOURCE_NAMES = ('pseudo1000', 'pseudo180', 'trusted300', 'trusted150', 'calibration30', 'development100', 'original100', 'component40')


def digest(path):
    with path.open('rb') as source:
        value = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def rank(values, lane):
    return sorted(values, key=lambda value: hashlib.sha256(f'{SEED}|{lane}|{value}'.encode()).digest())


def views(rows):
    return [view for row in rows for view in row['views']]


def hashes(rows):
    return {view['sha256'] for view in views(rows)}


def setups(rows):
    return {row['setup'] for row in rows}


def summary(rows):
    return {'scenes': len(rows), 'setups': len(setups(rows)), 'frames': len(views(rows)),
            'camera_counts': dict(sorted(Counter(view['camera_id'] for view in views(rows)).items())),
            'date_count': len({row['id'][:8] for row in rows}),
            'scenes_by_view_count': dict(sorted(Counter(len(row['views']) for row in rows).items())),
            'scenes_per_setup_histogram': dict(sorted(Counter(Counter(row['setup'] for row in rows).values()).items()))}


def select(pool, seen):
    seen_ids = {row['id'] for row in seen}
    seen_setups = setups(seen)
    eligible = defaultdict(dict)
    for row in pool:
        if row['id'] not in seen_ids and len(row['views']) == 3:
            eligible[row['setup']][row['id']] = row
    assert eligible.keys() == setups(pool)
    unused, prior = set(eligible) - seen_setups, set(eligible) & seen_setups
    assert len(unused) == 31 and len(prior) == 60
    trio_setups = unused | set(rank(prior, 'prior-trio-setups')[:27])
    rows = []
    for setup in rank(eligible, 'output-order'):
        scene = rank(eligible[setup], 'trio-scene')[0]
        source = eligible[setup][scene]
        assert [view['id'] for view in source['views']] == list('ABC')
        kept = source['views'] if setup in trio_setups else source['views'][:2]
        rows.append({**source, 'views': kept})
    return rows


def verify_native_pool(rows):
    seen_paths, seen_hashes = set(), set()
    for row in rows:
        assert row['native_split'] == 'train' and row['id'].rsplit('-', 1)[0] == row['setup']
        assert [view['id'] for view in row['views']] in [list('AB'), list('ABC')]
        for view in row['views']:
            path = ROOT / view['path']
            assert path.parent == NATIVE and path.is_file() and not path.is_symlink()
            assert path.name == f"{row['id']}-{int(view['camera_id']):02d}.jpg"
            assert (view['width'], view['height']) == (1920, 1080)
            assert view['path'] not in seen_paths and view['sha256'] not in seen_hashes
            assert digest(path) == view['sha256'], path
            seen_paths.add(view['path'])
            seen_hashes.add(view['sha256'])


def source_records():
    main_freeze = OLD_DATA / 'freeze.json'
    frozen = load(main_freeze)
    original_freeze = OLD_DATA / 'original100-freeze.json'
    component_freeze = OLD_DATA / 'component40-selection-freeze.json'
    separately_frozen = {**load(original_freeze), **load(component_freeze)}
    sources = {}
    for name in SOURCE_NAMES:
        path = COHORTS / f'{name}.json'
        key = path.relative_to(OLD_DATA).as_posix()
        expected = frozen.get(key) or separately_frozen[path.relative_to(ROOT).as_posix()]
        assert digest(path) == expected, path
        sources[name] = load(path)['scenes']
    reserve_path = COHORTS / 'test-reservations.json'
    assert digest(reserve_path) == frozen[reserve_path.relative_to(OLD_DATA).as_posix()]
    history_audit = FULL / 'sources/split-audit.json'
    old_audit = OLD_DATA / 'audit.json'
    prior_val = ROOT / 'evaluation/messytable-multiview-graph/manifest.json'
    prior_third = ROOT / 'evaluation/messytable-multiview-graph-97/manifest.json'
    prior_pilot_images = sorted((ROOT / 'data/messytable').glob('case-*/*.jpg'))
    assert len(prior_pilot_images) == 6
    for path in (history_audit, prior_val, prior_third):
        assert digest(path) == load(old_audit)['source_sha256'][path.relative_to(ROOT).as_posix()]
    source_paths = [*(COHORTS / f'{name}.json' for name in SOURCE_NAMES), main_freeze, original_freeze, component_freeze,
                    reserve_path, history_audit, old_audit, prior_val, prior_third, *prior_pilot_images]
    legacy_exposure_setups = set(load(history_audit)['existing_experiment_camera_setups']) | setups(load(prior_val)['scenes'])
    legacy_exposure_hashes = {digest(path) for path in prior_pilot_images} | hashes(load(prior_val)['scenes']) | hashes(load(prior_third)['scenes'])
    return sources, source_paths, legacy_exposure_setups, legacy_exposure_hashes


def prepare():
    destinations = [OUT / 'cohorts/supplemental240.json', OUT / 'inventory.json', OUT / 'selection-freeze.json']
    assert not any(path.exists() for path in destinations), 'Preserve existing selection freeze'
    src, paths, legacy_setups, legacy_hashes = source_records()
    pool, audited = src['pseudo1000'], src['pseudo180']
    assert pool[:60] == audited and len(pool) == 334 and len(setups(pool)) == 91 and len(views(pool)) == 1000
    assert Counter(len(row['views']) for row in pool) == {3: 332, 2: 2}
    verify_native_pool(pool)
    selected = select(pool, audited)
    assert len(selected) == len(setups(selected)) == 91 and len(views(selected)) == len(hashes(selected)) == 240
    assert Counter(len(row['views']) for row in selected) == {3: 58, 2: 33}
    assert not hashes(selected) & hashes(audited)
    assert not {row['id'] for row in selected} & {row['id'] for row in audited}
    assert not setups(pool) & legacy_setups and not hashes(pool) & legacy_hashes
    comparator_exclusion = {}
    for name in ('trusted300', 'trusted150', 'calibration30', 'development100', 'original100', 'component40'):
        rows = src[name]
        actual_setups = {view['source_setup'] for view in views(rows)} if name == 'original100' else setups(rows)
        setup_overlap = len(setups(pool) & actual_setups)
        image_overlap = len(hashes(pool) & hashes(rows))
        assert setup_overlap == image_overlap == 0, name
        comparator_exclusion[name] = {'source_setups': len(actual_setups), 'source_frames': len(views(rows)),
                                      'shared_setups': setup_overlap, 'shared_image_sha256': image_overlap}
    reserves = load(COHORTS / 'test-reservations.json')['setup_groups']
    assert not setups(pool) & {row['setup'] for group in reserves.values() for row in group}
    seen_ids = {row['id'] for row in audited}
    novel = [row for row in pool if row['id'] not in seen_ids]
    source_digests = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    selection_path, inventory_path, freeze_path = destinations
    save(selection_path, {'version': 1, 'cohort': 'supplemental240', 'seed': SEED,
         'source_cohort': (COHORTS / 'pseudo1000.json').relative_to(ROOT).as_posix(),
         'source_cohort_sha256': digest(COHORTS / 'pseudo1000.json'),
         'selection': 'Exclude all exact Pseudo180 scenes. SHA-256 rank remaining complete-trio scenes within each setup; select one from all 91 setups. Keep all three cameras in the 31 never-audited setups and 27 hash-ranked previously audited setups; keep A/B elsewhere. Remaining full-trio source observations are still accessible to the full-pool teacher.',
         'summary': summary(selected), 'scenes': selected})
    save(inventory_path, {'purpose': 'Metadata-only discovery and matched training/audit population freeze', 'seed': SEED,
         'discovery_pool': summary(pool), 'prior_pseudo180': summary(audited), 'pool_after_excluding_exact_prior_scenes': summary(novel),
         'matched_subset': {**summary(selected), 'exact_prior_audit_frames': 0, 'exact_prior_audit_scenes': 0,
                            'prior_audit_setups_reused_with_different_scenes': len(setups(selected) & setups(audited)),
                            'previously_unaudited_setups': len(setups(selected) - setups(audited)),
                            'extra_source_C_views_for_matched_pairs': 33},
         'discovery_overlap': {'prior_pseudo180_frames': len(hashes(pool) & hashes(audited)), 'prior_pseudo180_setups': len(setups(pool) & setups(audited)),
                               'comparator_exclusion': comparator_exclusion, 'all_recorded_legacy_setup_overlap': 0,
                               'all_recorded_legacy_image_sha256_overlap': 0, 'test_reservation_setup_overlap': 0},
         'source_sha256': source_digests, 'scope': 'Source cohort manifests, prior inventory and JPEG hashes only. No native annotation file read. Integration30 was verified by setup metadata only.'})
    save(freeze_path, {path.relative_to(ROOT).as_posix(): digest(path) for path in (selection_path, inventory_path, Path(__file__).resolve())})
    print(summary(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        freeze = load(OUT / 'selection-freeze.json')
        source = load(OUT / 'inventory.json')['source_sha256']
        for relative, expected in {**freeze, **source}.items():
            assert digest(ROOT / relative) == expected, relative
        verify_native_pool(load(COHORTS / 'pseudo1000.json')['scenes'])
        print('Verified source manifests, metadata selection and 1,000 native JPEG hashes.')


if __name__ == '__main__':
    main()
