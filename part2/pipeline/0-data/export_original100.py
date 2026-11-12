#!/usr/bin/env python3
"""Export exact historical A/B pixels and answer-key coordinates for regression."""

from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from prepare import NATIVE, ORIGINAL, OUT, ROOT, check_box, digest, load, save, setup


def cluster_setup_graph(cases):
    neighbors = defaultdict(set)
    for case in cases:
        a, b = (setup(case['scenes'][side]['source_scene']) for side in 'AB')
        neighbors[a].add(b)
        neighbors[b].add(a)
    clusters = {}
    for group in sorted(neighbors):
        if group in clusters:
            continue
        pending = [group]
        while pending:
            item = pending.pop()
            if item in clusters:
                continue
            clusters[item] = group
            pending.extend(neighbors[item])
    return clusters


def main():
    cohort_path = OUT / 'cohorts/original100.json'
    truth_path = OUT / 'scoring/original100.json'
    freeze_path = OUT / 'original100-freeze.json'
    assert not any(path.exists() for path in (cohort_path, truth_path, freeze_path))
    cases = sorted(load(ORIGINAL)['cases'], key=lambda row: row['case'])
    assert [row['case'] for row in cases] == list(range(1, 101))
    groups = cluster_setup_graph(cases)
    scenes, truth = [], []
    counts = Counter()
    for case in cases:
        case_id = f'case-{case["case"]:03d}'
        case_views = case['scenes']
        views = []
        truth_views = {}
        source_setups = set()
        for side in 'AB':
            view = case_views[side]
            source_setup = setup(view['source_scene'])
            source_setups.add(source_setup)
            path = ORIGINAL.parent / view['image']
            assert path == ORIGINAL.parent / case_id / f'{side}.jpg'
            fingerprint = digest(path)
            assert fingerprint == digest(NATIVE / 'images' / view['source_image'])
            with Image.open(path) as image:
                width, height = image.size
            views.append({'id': side, 'camera_id': str(view['camera']), 'path': path.relative_to(ROOT).as_posix(),
                          'sha256': fingerprint, 'width': width, 'height': height,
                          'source_scene': view['source_scene'], 'source_setup': source_setup})
            boxes = {}
            for instance, (x, y, w, h) in view['objects'].items():
                box = [x, y, x + w, y + h]
                check_box(box, width, height)
                boxes[instance] = box
            truth_views[side] = {'camera_id': str(view['camera']), 'objects_xyxy': boxes,
                                 'objects_xywh_historical': view['objects']}
            counts['boxes'] += len(boxes)
        group_ids = {groups[key] for key in source_setups}
        assert len(group_ids) == 1
        group = group_ids.pop()
        shared = set(map(int, truth_views['A']['objects_xyxy'].keys() & truth_views['B']['objects_xyxy'].keys()))
        assert shared == set(case['shared_object_ids'])
        counts['pairs'] += 1
        counts['frames'] += 2
        counts['cross_setup_pairs'] += len(source_setups) > 1
        counts['same_source_scene_pairs'] += case_views['A']['source_scene'] == case_views['B']['source_scene']
        counts['shared_instances'] += len(shared)
        scenes.append({'id': case_id, 'setup': group, 'views': views})
        truth.append({'id': case_id, 'setup': group, 'views': truth_views, 'shared_object_ids': sorted(shared)})
    assert counts['shared_instances'] == 1696
    counts['source_setups'] = len(groups)
    counts['source_setup_components'] = len(set(groups.values()))
    summary = dict(counts)
    save(cohort_path, {'version': 1, 'cohort': 'original100', 'purpose': 'Exposed regression only',
                       'setup_rule': 'Connected components of source setups co-occurring in A/B pairs. IDs are minimum member setup IDs.',
                       'summary': summary, 'scenes': scenes})
    save(truth_path, {'version': 1, 'cohort': 'original100', 'cohort_sha256': digest(cohort_path),
                      'source_annotation_sha256': digest(ORIGINAL), 'statistics': summary, 'scenes': truth})
    save(freeze_path, {path.relative_to(ROOT).as_posix(): digest(path)
                       for path in (cohort_path, truth_path, ORIGINAL, Path(__file__))})
    print(summary)


if __name__ == '__main__':
    main()
