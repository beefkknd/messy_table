#!/usr/bin/env python3
"""Select setup-disjoint MessyTable inputs and export complete one-class boxes."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FULL = ROOT / 'data/messytable-full'
NATIVE = FULL / 'MessyTable'
SOURCES = FULL / 'sources'
SEED = '2026-11-10-self-learning-r1-v1'
ORIGINAL = ROOT / 'data/messytable-100/answer-key.json'
PILOT = ROOT / 'data/messytable/answer-key.json'
THIRD = ROOT / 'evaluation/messytable-multiview-graph-97/manifest.json'
OLD_VAL = ROOT / 'evaluation/messytable-multiview-graph/manifest.json'
AUDIT = SOURCES / 'split-audit.json'
FRESH = SOURCES / 'fresh-scene-candidates.json'
INVENTORY = SOURCES / 'inventory.json'
LABELS = {split: NATIVE / f'labels/{split}.json' for split in ('train', 'val')}
CAMERA_NAME = re.compile(r'(\d{8}-\d{5}-\d{2})-(0[1-9])\.jpg')
CAMERA_META = re.compile(rb'"pathname": "([^"\r\n]+)",\s+"extrinsics": (\[[^\]]+\])')
SCENE_OPEN = re.compile(rb'^        "(\d{8}-\d{5}-\d{2})": (\{)', re.MULTILINE)
SCENE_CLOSE = re.compile(rb'^        \}', re.MULTILINE)


def digest(path):
    with path.open('rb') as source:
        value = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def setup(scene):
    return scene.rsplit('-', 1)[0]


def order(values, lane):
    return sorted(values, key=lambda value: hashlib.sha256(f'{SEED}|{lane}|{value}'.encode()).digest())


def historical():
    history = {}
    for path in (ORIGINAL, PILOT):
        cases = load(path)['cases']
        views = [(case.get('source_scene') or view['source_scene'], view['source_image'])
                 for case in cases for view in case['scenes'].values()]
        history[path.relative_to(ROOT).as_posix()] = views
    for path in (THIRD, OLD_VAL):
        history[path.relative_to(ROOT).as_posix()] = [
            (view.get('source_scene', row['id']), view.get('source_image') or view['source_pathname'])
            for row in load(path)['scenes'] for view in row['views']
        ]
    for views in history.values():
        for scene, image in views:
            assert CAMERA_NAME.fullmatch(image)[1] == scene, (scene, image)
    base_setups = {setup(scene) for path, views in history.items() if 'answer-key' in path
                   for scene, _ in views}
    assert base_setups == set(load(AUDIT)['existing_experiment_camera_setups'])
    return history


def camera_metadata(split):
    """Scan camera pathname and extrinsics tokens without decoding label payloads."""
    scenes = defaultdict(dict)
    with LABELS[split].open('rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        for match in CAMERA_META.finditer(blob):
            filename = match[1].decode()
            scene, camera = CAMERA_NAME.fullmatch(filename).groups()
            camera = str(int(camera))
            assert camera not in scenes[scene]
            extrinsics = json.loads(match[2])
            assert len(extrinsics) == 6 and all(math.isfinite(x) for x in extrinsics)
            scenes[scene][camera] = (filename, extrinsics)
    assert all(set(cameras) == set('123456789') for cameras in scenes.values())
    return scenes


def scene_labels(split, selected=None):
    """Decode publisher scene JSON only for requested scene IDs."""
    with LABELS[split].open('rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as blob:
        found = set()
        for start in SCENE_OPEN.finditer(blob):
            scene = start[1].decode()
            if selected is not None and scene not in selected:
                continue
            assert scene not in found
            found.add(scene)
            end = SCENE_CLOSE.search(blob, start.end())
            yield scene, json.loads(blob[start.start(2):end.end()])
        if selected is not None:
            assert found == selected, selected - found


def eligible_groups(scenes, used):
    groups = defaultdict(list)
    for scene in scenes:
        if setup(scene) not in used:
            groups[setup(scene)].append(scene)
    return groups


def viewing_vector(extrinsics):
    # Publisher ASNet compute_angle convention: T @ [0, 0, -1, 1].
    x, y, z, roll, pitch, yaw = extrinsics
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    vector = (x - (cy * sp * cr + sy * sr), y - (sy * sp * cr - cy * sr), z - cp * cr)
    norm = math.hypot(*vector)
    assert norm > 0
    return tuple(value / norm for value in vector)


def angle(left, right):
    cosine = sum(a * b for a, b in zip(left, right))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def select_views(scene, cameras, index, count):
    vectors = {cam: viewing_vector(values[1]) for cam, values in cameras.items()}
    pairs = sorted((angle(vectors[a], vectors[b]), a, b) for a, b in itertools.combinations(sorted(cameras), 2))
    band = index % 3
    band_pairs = [(a, b) for _, a, b in pairs[12 * band:12 * (band + 1)]]
    primary = order(band_pairs, f'pair|{scene}')[0]
    selected = order(primary, f'direction|{scene}')
    if count == 3:
        third = max((cam for cam in sorted(cameras) if cam not in primary),
                    key=lambda cam: min(angle(vectors[cam], vectors[other]) for other in primary))
        selected.append(third)
    views = []
    for letter, camera in zip('ABC', selected):
        path = NATIVE / 'images' / cameras[camera][0]
        with Image.open(path) as image:
            width, height = image.size
            assert image.getexif().get(274) in (None, 1), path
        views.append({'id': letter, 'camera_id': camera, 'path': path.relative_to(ROOT).as_posix(),
                      'width': width, 'height': height, 'sha256': digest(path)})
    return views, round(angle(*(vectors[camera] for camera in primary)), 6), ('lower', 'middle', 'upper')[band]


def records(split, scene_ids, meta, n_views):
    rows = []
    for index, (scene, count) in enumerate(zip(scene_ids, n_views, strict=True)):
        views, angle_ab, band = select_views(scene, meta[scene], index, count)
        rows.append({'id': scene, 'setup': setup(scene), 'native_split': split, 'views': views,
                     'primary_view_angle_deg': angle_ab, 'primary_view_angle_tercile': band})
    return rows


def first_scene(setups, groups, lane):
    return [order(groups[group], lane)[0] for group in setups]


def summary(rows):
    angles = [row['primary_view_angle_deg'] for row in rows if len(row['views']) >= 2]
    cameras = Counter(view['camera_id'] for row in rows for view in row['views'])
    angle_edges = (30, 60, 90, 120)
    angle_labels = ('<30', '30-<60', '60-<90', '90-<120', '>=120')
    return {'setups': len({row['setup'] for row in rows}), 'scenes': len(rows),
            'frames': sum(cameras.values()), 'dates': len({row['id'][:8] for row in rows}),
            'primary_pairs': len(angles), 'camera_counts': dict(sorted(cameras.items())),
            'angle_range_deg': [min(angles), max(angles)] if angles else None,
            'angle_bin_counts': dict(sorted(Counter(angle_labels[bisect_right(angle_edges, a)] for a in angles).items()))}


def manifest(name, rows):
    return {'version': 1, 'cohort': name, 'seed': SEED, 'summary': summary(rows), 'scenes': rows}


def check_box(box, width, height):
    assert len(box) == 4 and all(math.isfinite(value) for value in box), box
    x1, y1, x2, y2 = box
    assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, (box, width, height)


def yolo_box(box, width, height):
    check_box(box, width, height)
    x1, y1, x2, y2 = box
    cx, cy, w, h = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height
    line = f'0 {cx:.10f} {cy:.10f} {w:.10f} {h:.10f}'
    _, cx, cy, w, h = map(float, line.split())
    restored = ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)
    assert max(abs(a - b) for a, b in zip(box, restored)) < 0.001
    return line


def truth_rows(rows):
    assert len({row['native_split'] for row in rows}) == 1
    by_id = {row['id']: row for row in rows}
    selected = dict(scene_labels(rows[0]['native_split'], set(by_id)))
    truth = []
    for row in rows:
        scene = selected[row['id']]
        views = {}
        for view in row['views']:
            camera = scene['cameras'][view['camera_id']]
            assert Path(view['path']).name == camera['pathname']
            objects = camera['instances']
            assert objects
            for instance in objects.values():
                assert 1 <= instance['cls'] <= 42 and 1 <= instance['subcls'] <= 120
                check_box(instance['pos'], view['width'], view['height'])
            views[view['id']] = {
                'camera_id': view['camera_id'],
                'objects_xyxy': {key: value['pos'] for key, value in objects.items()},
                'object_classes': {key: value['cls'] for key, value in objects.items()},
                'object_subclasses': {key: value['subcls'] for key, value in objects.items()},
            }
        truth.append({'id': row['id'], 'setup': row['setup'], 'views': views})
    return truth


def annotation_stats(rows, truth):
    sizes = Counter()
    counts = Counter()
    classes, subclasses = set(), set()
    shared = 0
    for row, gold in zip(rows, truth, strict=True):
        if len(row['views']) >= 2:
            shared += len(gold['views']['A']['objects_xyxy'].keys() & gold['views']['B']['objects_xyxy'].keys())
        for view in row['views']:
            details = gold['views'][view['id']]
            counts['frames'] += 1
            counts['boxes'] += len(details['objects_xyxy'])
            classes.update(details['object_classes'].values())
            subclasses.update(details['object_subclasses'].values())
            for x1, y1, x2, y2 in details['objects_xyxy'].values():
                sizes['edge_boxes'] += min(x1, y1) == 0 or x2 == view['width'] or y2 == view['height']
                sizes['short_side_under_16px'] += min(x2 - x1, y2 - y1) < 16
                sizes['short_side_under_32px'] += min(x2 - x1, y2 - y1) < 32
    return {**counts, **sizes, 'classes': len(classes), 'subclasses': len(subclasses), 'AB_shared_instances': shared}


def export_yolo(stage, destination, rows, truth, partition):
    frames = []
    for row, gold in zip(rows, truth, strict=True):
        for view in row['views']:
            source = ROOT / view['path']
            image_path = Path('yolo/images') / partition / source.name
            label_path = Path('yolo/labels') / partition / source.with_suffix('.txt').name
            for path in (image_path, label_path):
                (stage / path).parent.mkdir(parents=True, exist_ok=True)
            (stage / image_path).symlink_to(os.path.relpath(source, (destination / image_path).parent))
            boxes = gold['views'][view['id']]['objects_xyxy']
            lines = [yolo_box(boxes[key], view['width'], view['height']) for key in sorted(boxes, key=int)]
            (stage / label_path).write_text('\n'.join(lines) + '\n')
            frames.append({'image': (destination / image_path).relative_to(ROOT).as_posix(),
                           'label': (destination / label_path).relative_to(ROOT).as_posix(),
                           'setup': row['setup'], 'scene': row['id'], 'scope': 'complete_frame',
                           'partition': partition, 'image_sha256': view['sha256'], 'boxes': len(lines)})
    return frames


def preview(stage, rows, truth):
    cases = []
    for row, gold in zip(rows, truth, strict=True):
        for view in row['views']:
            objects = gold['views'][view['id']]['objects_xyxy']
            cases.append((len(objects), view['path'], view, objects))
    cases.sort()
    examples = [cases[i * (len(cases) - 1) // 5] for i in range(6)]
    w, h, caption = 640, 360, 30
    panel = Image.new('RGB', (3 * w, 2 * (h + caption) + 30), '#1b2330')
    draw = ImageDraw.Draw(panel)
    draw.text((12, 9), 'MessyTable training annotations | One class: visible publisher-scope tabletop item', fill='white')
    for index, (count, name, view, objects) in enumerate(examples):
        left, top = index % 3 * w, index // 3 * (h + caption) + 30
        with Image.open(ROOT / name) as source:
            image = source.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
        panel.paste(image, (left, top))
        for instance, (x1, y1, x2, y2) in objects.items():
            x1, x2 = (left + x * w / view['width'] for x in (x1, x2))
            y1, y2 = (top + y * h / view['height'] for y in (y1, y2))
            draw.rectangle((x1, y1, min(x2, left + w - 1), min(y2, top + h - 1)), outline='#f5e269', width=1)
            if x2 - x1 > 20 and y2 - y1 > 14:
                draw.text((x1 + 2, y1 + 1), instance, fill='#f5e269', stroke_width=1, stroke_fill='#1b2330')
        draw.text((left + 10, top + h + 9), f'{Path(name).stem}   |   {count} item boxes', fill='white')
    panel.save(stage / 'trusted-annotation-review.jpg', quality=93)


def prepare(destination):
    history = historical()
    used = {setup(scene) for views in history.values() for scene, _ in views}
    old_images = {image for views in history.values() for _, image in views}
    index = load(FRESH)
    prior_audit = load(AUDIT)['native_splits']
    train, val = camera_metadata('train'), camera_metadata('val')
    assert set(train) == set(index['train']) and set(val) == set(index['val'])
    groups = {split: eligible_groups(scenes, used) for split, scenes in index.items()}
    assert len(groups['train']) == 191 and len(groups['val']) == 176 and len(groups['test']) == 82
    for a, b in itertools.combinations(groups.values(), 2):
        assert not (a.keys() & b.keys())
    for scene in index['test']:
        assert all((NATIVE / 'images' / f'{scene}-{camera:02d}.jpg').is_file() for camera in range(1, 10))
    train_order = order(groups['train'], 'train-setups')
    trusted150_ids = first_scene(train_order[:50], groups['train'], 'trusted-scene')
    extension_ids = first_scene(train_order[50:100], groups['train'], 'trusted-scene')
    trusted = records('train', trusted150_ids, train, itertools.repeat(3, 50))
    extension = records('train', extension_ids, train, itertools.repeat(3, 50))
    nested = [{**row, 'views': row['views'][:1], 'primary_view_angle_deg': None,
               'primary_view_angle_tercile': None} for row in trusted]
    pseudo_by_setup = [order(groups['train'][group], 'pseudo-scene') for group in train_order[100:]]
    pseudo_ids = [sc for layer in itertools.zip_longest(*pseudo_by_setup) for sc in layer if sc is not None][:334]
    pseudo = records('train', pseudo_ids, train, [3] * 332 + [2, 2])
    val_order = order(groups['val'], 'val-setups')
    calibration_ids = first_scene(val_order[:30], groups['val'], 'val-scene')
    development_ids = first_scene(val_order[30:130], groups['val'], 'val-scene')
    calibration = records('val', calibration_ids, val, itertools.repeat(2, 30))
    development = records('val', development_ids, val, itertools.repeat(2, 100))
    test_order = order(groups['test'], 'test-setups')
    reserves = {'component40': test_order[:40], 'integration30': test_order[40:70], 'unreserved12': test_order[70:]}
    partitions = [trusted, extension, pseudo, calibration, development]
    setup_groups = [{row['setup'] for row in rows} for rows in partitions]
    for a, b in itertools.combinations([*setup_groups, *map(set, reserves.values())], 2):
        assert not a & b
    for row in itertools.chain.from_iterable(partitions):
        assert row['setup'] not in used and not any(Path(view['path']).name in old_images for view in row['views'])
    images = [view['sha256'] for rows in partitions for row in rows for view in row['views']]
    assert len(set(images)) == len(images)
    for path, views in history.items():
        assert not set(images) & {digest(NATIVE / 'images' / image) for _, image in set(views)}, path
    cohorts = {'trusted50': nested, 'trusted150': trusted, 'trusted300': trusted + extension,
               'pseudo180': pseudo[:60], 'pseudo1000': pseudo, 'calibration30': calibration, 'development100': development}
    expected = {'trusted50': (50, 50), 'trusted150': (50, 150), 'trusted300': (100, 300), 'pseudo180': (60, 180),
                'pseudo1000': (91, 1000), 'calibration30': (30, 60), 'development100': (100, 200)}
    for name, rows in cohorts.items():
        counts = summary(rows)
        assert (counts['setups'], counts['frames']) == expected[name], name
    sources = [ORIGINAL, PILOT, THIRD, OLD_VAL, AUDIT, FRESH, INVENTORY, *LABELS.values(), Path(__file__).resolve()]
    source_hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in sources}
    with TemporaryDirectory(prefix='.cohort-stage-', dir=destination) as temporary:
        stage = Path(temporary)
        # Selection is complete and committed to stage before any labels are decoded.
        for name, rows in cohorts.items():
            save(stage / 'cohorts' / f'{name}.json', manifest(name, rows))
        save(stage / 'cohorts/test-reservations.json', {'seed': SEED, 'native_split': 'test',
             'purpose': 'Metadata-only setup reservations; native test labels and images are sealed.',
             'setup_groups': {name: [{'setup': group, 'available_scenes': len(groups['test'][group])}
                                     for group in setups] for name, setups in reserves.items()}})
        trusted_truth, calibration_truth = truth_rows(trusted), truth_rows(calibration)
        save(stage / 'scoring/calibration30.json', {'version': 1, 'cohort': 'calibration30',
             'cohort_sha256': digest(stage / 'cohorts/calibration30.json'), 'scenes': calibration_truth})
        frames = export_yolo(stage, destination, trusted, trusted_truth, 'train')
        frames.extend(export_yolo(stage, destination, calibration, calibration_truth, 'val'))
        save(stage / 'yolo/supervised-frames.json', {'version': 1, 'frames': frames})
        (stage / 'yolo/trusted150.yaml').write_text('train: images/train\nval: images/val\nnames:\n  0: tabletop_item\n')
        preview(stage, trusted, trusted_truth)
        save(stage / 'audit.json', {'seed': SEED, 'native_inventory_snapshot': prior_audit,
             'eligible_after_history': {name: {'setups': len(group), 'scenes': sum(map(len, group.values())),
                 'images': 9 * sum(map(len, group.values()))} for name, group in groups.items()},
             'historical_union': {'setups': len(used), 'unique_source_images': len(old_images)},
             'historical_sources': {name: {'setups': len({setup(scene) for scene, _ in views}),
                 'scenes': len({scene for scene, _ in views}), 'view_observations': len(views),
                 'unique_source_images': len({image for _, image in views})} for name, views in history.items()},
             'cohorts': {name: summary(rows) for name, rows in cohorts.items()},
             'test_reservations': {name: {'setups': len(setups), 'scenes': sum(len(groups['test'][g]) for g in setups)}
                                   for name, setups in reserves.items()},
             'complete_label_export': {'trusted150': annotation_stats(trusted, trusted_truth),
                                       'calibration30': annotation_stats(calibration, calibration_truth)},
             'validation_scope': 'Calibration labels decoded for selected views. Development labels await prediction freeze.',
             'pseudo_scope': 'Metadata-only membership. No pseudo truth export during preparation.',
             'test_scope': 'Uses a prior metadata-only scene list. This script has no native test annotation path.',
             'source_sha256': source_hashes, 'recorded_archive_sha256': load(INVENTORY)['archive_sha256']})
        freeze = {path.relative_to(stage).as_posix(): digest(path) for path in sorted(stage.rglob('*'))
                  if not path.is_symlink() and path.is_file()}
        save(stage / 'freeze.json', freeze)
        # Preserve an existing freeze. Only publish staged outputs once all checks succeed.
        for source in sorted(stage.iterdir(), key=lambda path: path.name == 'freeze.json'):
            source.rename(destination / source.name)


def verify(destination, images=False):
    for relative, expected in load(destination / 'freeze.json').items():
        assert digest(destination / relative) == expected, relative
    if images:
        seen = set()
        for path in (destination / 'cohorts').glob('*.json'):
            for scene in load(path).get('scenes', []):
                for view in scene['views']:
                    if view['path'] not in seen:
                        assert digest(ROOT / view['path']) == view['sha256'], view['path']
                        seen.add(view['path'])
        for frame in load(destination / 'yolo/supervised-frames.json')['frames']:
            image = ROOT / frame['image']
            label = ROOT / frame['label']
            assert image.is_symlink() and digest(image) == frame['image_sha256'], image
            assert len(label.read_text().splitlines()) == frame['boxes'], label


def reveal(destination, cohort, predictions, expected, result):
    verify(destination)
    assert digest(predictions) == expected, 'Prediction artifact does not match the supplied frozen digest'
    assert not result.exists(), result
    manifest_path = destination / 'cohorts' / f'{cohort}.json'
    rows = load(manifest_path)['scenes']
    split = rows[0]['native_split']
    source_digest = load(destination / 'audit.json')['source_sha256'][LABELS[split].relative_to(ROOT).as_posix()]
    assert digest(LABELS[split]) == source_digest
    truth = truth_rows(rows)
    save(result, {'version': 1, 'cohort': cohort, 'cohort_sha256': digest(manifest_path),
                  'predictions_sha256': expected, 'source_annotation_sha256': source_digest,
                  'statistics': annotation_stats(rows, truth), 'scenes': truth})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=OUT)
    subcommands = parser.add_subparsers(dest='action', required=True)
    subcommands.add_parser('prepare')
    verifier = subcommands.add_parser('verify')
    verifier.add_argument('--images', action='store_true')
    scorer = subcommands.add_parser('reveal-for-scoring')
    scorer.add_argument('cohort', choices=('development100', 'pseudo180', 'pseudo1000'))
    scorer.add_argument('--predictions', type=Path, required=True)
    scorer.add_argument('--predictions-sha256', required=True)
    scorer.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if args.action == 'prepare':
        if any((destination / name).exists() for name in ('cohorts', 'scoring', 'yolo', 'audit.json', 'freeze.json', 'trusted-annotation-review.jpg')):
            parser.error('Preserve existing preparation outputs. Select an empty destination under the workspace.')
        destination.mkdir(parents=True, exist_ok=True)
        prepare(destination)
    elif args.action == 'verify':
        verify(destination, args.images)
    else:
        reveal(destination, args.cohort, args.predictions, args.predictions_sha256, args.out)


if __name__ == '__main__':
    main()
