"""Select complete test scenes by fixed hash, stratified by official difficulty."""
from collections import Counter
import hashlib

from common import DATA, HERE, LABELS, read, save, sha

SEED = 'messytable-official-protocol-sample100-2026-11-11-v1'
COUNTS = {'easy': 30, 'medium': 50, 'hard': 20}


def main():
    target = HERE / 'sample100'
    target.mkdir(exist_ok=True)
    original = read(LABELS)['scenes']
    selected, used_setups = [], set()
    sources = {'test.json': sha(LABELS)}
    for difficulty, count in COUNTS.items():
        path = DATA / 'labels' / f'test_{difficulty}.json'
        sources[path.name] = sha(path)
        choices = read(path)['scenes']
        assert set(choices).issubset(original)
        ordered = sorted(choices, key=lambda scene: hashlib.sha256(
            (SEED + '|' + difficulty + '|' + scene).encode()).hexdigest())
        for scene in ordered:
            setup = scene.rsplit('-', 1)[0]
            if setup in used_setups:
                continue
            selected.append({'scene': scene, 'setup': setup, 'difficulty': difficulty})
            used_setups.add(setup)
            if sum(row['difficulty'] == difficulty for row in selected) == count:
                break
        assert sum(row['difficulty'] == difficulty for row in selected) == count
    chosen = {row['scene'] for row in selected}
    rows = [next(row for row in selected if row['scene'] == scene) for scene in original if scene in chosen]
    assert len(rows) == len(used_setups) == 100
    appearances = sum(len(camera['instances']) for scene in chosen
                      for camera in original[scene]['cameras'].values())
    candidates = positives = 0
    for scene in chosen:
        cameras = original[scene]['cameras']
        for a in range(1, 10):
            for b in range(a + 1, 10):
                left = cameras[str(a)]['instances']
                right = cameras[str(b)]['instances']
                candidates += len(left) * len(right)
                positives += len(set(left) & set(right))
    manifest = {
        'seed': SEED, 'selection': 'SHA256 order within difficulty, admit a scene only from an unused setup.',
        'selection_did_not_use_predictions': True, 'training': False,
        'full_split_scenes': 1861, 'sample_scenes': 100, 'sample_setups': 100,
        'images': 900, 'image_pairs': 3600, 'camera_pairs_per_scene': 36,
        'annotated_appearances': appearances, 'candidate_object_pairs': candidates,
        'positive_object_pairs': positives, 'negative_object_pairs': candidates - positives,
        'difficulty_counts': dict(Counter(row['difficulty'] for row in rows)),
        'source_sha256': sources, 'scenes': rows,
        'interpretation': 'An exploratory difficulty-stratified sample across100 setups from the official test split. This checks the official evaluation on a manageable sample. Full-split published numbers remain unverified.',
    }
    path = target / 'selection.json'
    if path.exists():
        assert read(path) == manifest
    else:
        save(path, manifest)
    print({key: manifest[key] for key in ('sample_scenes', 'images', 'image_pairs', 'annotated_appearances',
                                         'candidate_object_pairs', 'positive_object_pairs', 'difficulty_counts')})


if __name__ == '__main__':
    main()
