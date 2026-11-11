"""Replay foreground local evidence and validated perspective on all sampled pairs."""
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import hashlib
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

from variant_common import LOCAL, OUT, VIEWPOINT, freeze, inputs, read, save, sha, view_path
from run_full_scene import encode_crops
from runtime_assets import dinov2_source

sys.path.insert(0, str(LOCAL))
import local_correspondence as local


def viewpoint_functions():
    names = {'hom', 'project', 'corners', 'dedup', 'support', 'spread_ok', 'detail_info',
             'fit_surface', 'patch_views'}
    path = VIEWPOINT / 'run_study.py'
    parsed = ast.parse(path.read_text())
    nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {'np': np, 'cv2': cv2, 'hashlib': hashlib}
    namespace['native_crop'] = lambda row, masks, side, index: row['crops'][side][index][:2]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def shortlist(lanes):
    edges = set()
    for scores in lanes:
        for a, row in enumerate(np.argsort(-scores, axis=1, kind='stable')[:, :2]):
            edges.update((a, int(b)) for b in row)
        for b, row in enumerate(np.argsort(-scores.T, axis=1, kind='stable')[:, :2]):
            edges.update((int(a), b) for a in row)
    return sorted(edges)


def prepare(scene, camera, extractor):
    filename = OUT / 'views' / f'{scene}-0{camera}.npz'
    while not filename.exists():
        time.sleep(.5)
    with np.load(filename, allow_pickle=False) as archive:
        values = {name: archive[name] for name in ('tight_1x', 'sam', 'gray', 'canny', 'grabcut')}
        boxes = archive['rounded_boxes']
        meta = read(filename.with_suffix('.json'))
        rgb = np.array(Image.open(view_path(scene, camera)).convert('RGB'))
        rows = []
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            pixels = rgb[y1:y2, x1:x2].copy()
            status = meta['masks']['sam'][index]
            mask = archive[f'native_{index}'] if status['status'] == 'usable' else None
            record = {**status, 'mask_path': None, 'mask_sha256': None}
            with patch.object(local, 'sam_foreground', return_value=mask):
                features = local.crop_features(Image.fromarray(pixels), record, None, extractor)
            rows.append((pixels, mask, features))
    return rows, values


def pair_evidence(scene, a, b, all_crops, all_values, functions):
    crops = {'A': all_crops[a], 'B': all_crops[b]}
    shape = (len(crops['A']), len(crops['B']))
    lanes = [all_values[a][name] @ all_values[b][name].T for name in all_values[a]]
    edges = shortlist(lanes)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    support = np.zeros(shape, dtype=float)
    patches, keys, counts = [], [], Counter()
    for left, right in edges:
        pa, _, fa = crops['A'][left]
        pb, _, fb = crops['B'][right]
        sizes = {'A': pa.shape[1::-1], 'B': pb.shape[1::-1]}
        measured = local.measure(fa, fb, matcher)
        detail = functions['detail_info'](measured['matches'], sizes)
        if detail['eligible']:
            support[left, right] = detail['unique_affine_inliers']
        fit = functions['fit_surface'](measured['matches'], sizes, f'{scene}-{a}-{b}-{left}-{right}')
        counts[fit['status']] += 1
        if fit['status'] == 'geometry_valid':
            images, status = functions['patch_views']({'crops': crops}, None, left, right, fit)
            counts[status['status']] += 1
            if images is not None:
                keys.append((left, right))
                patches.extend(images)
    return support, patches, keys, dict(counts), len(edges)


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    selection, scenes, _ = inputs()
    out = OUT / 'perspective'
    out.mkdir(exist_ok=True)
    plan = {
        'training': False, 'source_plan_sha256': sha(OUT / 'plan.json'),
        'script_sha256': sha(__file__), 'local_source_sha256': sha(LOCAL / 'local_correspondence.py'),
        'perspective_source_sha256': sha(VIEWPOINT / 'run_study.py'),
        'shortlist': 'Same prior union of top2 per endpoint,direction and RGB/gray/Canny/GrabCut/SAM lane.',
        'fit': 'Reuse unchanged reciprocal SIFT,held-out homography validation and foreground common-patch extraction.',
        'final_population': 'All Cartesian candidates retained,unmeasured edges fall back to original SAM DINO similarity.',
        'workers': 12,
    }
    if (out / 'plan.json').exists():
        assert read(out / 'plan.json') == plan
    else:
        save(out / 'plan.json', plan)
    functions = viewpoint_functions()
    extractor, _ = local.runtime()
    processor = AutoImageProcessor.from_pretrained(dinov2_source(), local_files_only=True)
    model = AutoModel.from_pretrained(dinov2_source(), local_files_only=True).to('mps').eval().requires_grad_(False)
    pool = ThreadPoolExecutor(max_workers=12)
    files = [out / 'plan.json']
    start = time.monotonic()
    done = 0
    for scene in scenes:
        crops, values = {}, {}
        pending = []
        for a, b in combinations(range(1, 10), 2):
            path = out / f'{scene}-{a}-{b}.npz'
            metadata = path.with_suffix('.json')
            files.extend([path, metadata])
            if path.exists() and metadata.exists():
                done += 1
                continue
            for camera in (a, b):
                if camera not in crops:
                    crops[camera], values[camera] = prepare(scene, camera, extractor)
            pending.append((path, pool.submit(pair_evidence, scene, a, b, crops, values, functions)))
        for path, future in pending:
            support, patches, keys, counts, slate = future.result()
            unwarped = np.full(support.shape, np.nan)
            warped = np.full(support.shape, np.nan)
            if patches:
                features = np.concatenate([encode_crops(patches[start:start+64], processor, model, 'mps')
                                           for start in range(0, len(patches), 64)])
                for index, (a, b) in enumerate(keys):
                    left, right, aligned = features[3*index:3*index+3]
                    unwarped[a, b] = left @ right
                    warped[a, b] = left @ aligned
            temporary = path.with_suffix('.pending.npz')
            np.savez_compressed(temporary, local=support, unwarped=unwarped, warped=warped)
            temporary.replace(path)
            save(path.with_suffix('.json'), {'shortlist_edges': slate, 'status': counts, 'patches': len(keys)})
            done += 1
        if done % 360 == 0 or done == 36:
            print(f'Perspective {done}/3600 pairs, elapsed {time.monotonic()-start:.1f}s', flush=True)
    pool.shutdown()
    assert done == selection['image_pairs']
    freeze(out, files)
    print('All3600 perspective pairs frozen.', flush=True)


if __name__ == '__main__':
    main()
