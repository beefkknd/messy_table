#!/usr/bin/env python3
"""Inventory native splits and mark existing evaluation scenes; no model training."""
import collections, itertools, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
NATIVE = ROOT / 'MessyTable'

def setup(scene):
    return scene.rsplit('-', 1)[0]

def main():
    native_scenes, native_setups, native_images = {}, {}, {}
    summaries = {}
    for split in ('train', 'val', 'test'):
        path = NATIVE / 'labels' / (split + '.json')
        data = json.loads(path.read_text())
        scenes = data['scenes']
        native_scenes[split] = set(scenes)
        native_setups[split] = {setup(s) for s in scenes}
        images, classes, subclasses = set(), collections.Counter(), collections.Counter()
        boxes = invalid = missing = 0
        camera_pair_count = 0
        for scene in scenes.values():
            cameras = scene['cameras']
            camera_pair_count += len(cameras) * (len(cameras)-1) // 2
            for camera in cameras.values():
                image = camera['pathname']
                images.add(image)
                missing += not (NATIVE / 'images' / image).is_file()
                for obj in camera['instances'].values():
                    boxes += 1
                    classes[str(obj['cls'])] += 1
                    subclasses[str(obj['subcls'])] += 1
                    x1, y1, x2, y2 = obj['pos']
                    invalid += not (x2 > x1 and y2 > y1)
        native_images[split] = images
        summaries[split] = {'scenes': len(scenes), 'camera_setups': len(native_setups[split]),
                            'images': len(images), 'boxes': boxes, 'classes': len(classes),
                            'subclasses': len(subclasses), 'same_scene_camera_pair_combinations': camera_pair_count,
                            'missing_image_files': missing, 'nonpositive_boxes': invalid}
        print(split, summaries[split], flush=True)
        del data, scenes
    overlaps = {}
    for a, b in itertools.combinations(native_scenes, 2):
        overlaps[a+'__'+b] = {'scenes': len(native_scenes[a] & native_scenes[b]),
                              'camera_setups': len(native_setups[a] & native_setups[b]),
                              'images': len(native_images[a] & native_images[b])}
    used_scenes, used_images = set(), set()
    for folder in ('messytable', 'messytable-100'):
        path = WORKSPACE / 'data' / folder / 'answer-key.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for case in data.get('cases', []):
            if case.get('source_scene'):
                used_scenes.add(case['source_scene'])
            for view in case.get('scenes', {}).values():
                if view.get('source_scene'):
                    used_scenes.add(view['source_scene'])
                if view.get('source_image'):
                    used_images.add(view['source_image'])
    used_setups = {setup(s) for s in used_scenes}
    fresh = {split: sorted(s for s in scenes if setup(s) not in used_setups)
             for split, scenes in native_scenes.items()}
    result = {'native_splits': summaries, 'cross_split_overlap': overlaps,
              'existing_experiment_source_scenes': sorted(used_scenes),
              'existing_experiment_camera_setups': sorted(used_setups),
              'fresh_scene_counts_excluding_used_camera_setups': {k: len(v) for k,v in fresh.items()},
              'notes': ['Native scene IDs scope physical instance IDs.',
                        'Difficulty label files are subsets, not additional independent images.',
                        'Camera pair combinations are correlated views, not independent scenes.',
                        'Fresh means excluded by available local source metadata; no model has been run by this acquisition task.',
                        'Original 100-pair files and 3 blind pairs were not changed or scored.']}
    (ROOT / 'sources' / 'split-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT / 'sources' / 'fresh-scene-candidates.json').write_text(json.dumps(fresh, indent=2)+'\n')
    if any(s['missing_image_files'] or s['nonpositive_boxes'] for s in summaries.values()):
        raise SystemExit('Dataset integrity issue: see split-audit.json')
    print('Split overlap:', overlaps, flush=True)
    print('Fresh scene counts:', result['fresh_scene_counts_excluding_used_camera_setups'], flush=True)

if __name__ == '__main__':
    main()
