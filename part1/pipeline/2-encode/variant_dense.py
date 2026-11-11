"""Dense scene correspondence on all36 view pairs in every preserved scene."""
from itertools import combinations, product
from contextlib import nullcontext
from unittest.mock import patch
import time

import cv2
import numpy as np
import torch

from variant_common import OUT, annotations, freeze, inputs, read, save, sha, view_path
from roma_adapter import CHECKSUM, build_model, input_tensor, predict
from run_dense import MIN_PIXELS, cycle_points, edge_evidence


def all_directed(source, target, cycle):
    good, yy, xx, weight = cycle
    areas = source.sum(axis=(1, 2))
    values = np.zeros((len(source), len(target)), dtype=float)
    counts = np.zeros_like(values, dtype=int)
    for index, mask in enumerate(source):
        y, x = np.where(mask & good)
        if not len(y):
            continue
        membership = target[:, yy[y, x], xx[y, x]]
        counts[index] = membership.sum(axis=1)
        weighted = np.ascontiguousarray(np.where(membership, weight[y, x], np.float32(0)))
        values[index] = weighted.sum(axis=1) / areas[index] if areas[index] else 0
    return values, counts


def evidence(outputs, masks):
    cycles = {
        'AB': cycle_points(outputs['warp_AB'], outputs['warp_BA'], outputs['overlap_AB'], outputs['overlap_BA']),
        'BA': cycle_points(outputs['warp_BA'], outputs['warp_AB'], outputs['overlap_BA'], outputs['overlap_AB']),
    }
    result = {}
    for lane in ('box', 'mask'):
        ab, count_ab = all_directed(masks['A'][lane], masks['B'][lane], cycles['AB'])
        reverse, count_reverse = all_directed(masks['B'][lane], masks['A'][lane], cycles['BA'])
        ba, count_ba = reverse.T, count_reverse.T
        supported = (count_ab >= MIN_PIXELS) & (count_ba >= MIN_PIXELS)
        result[lane] = np.where(supported, np.minimum(ab, ba), 0)
        result[lane + '_mean'] = np.where(supported, (ab + ba) / 2, 0)
        result[lane + '_AB'] = ab
        result[lane + '_BA'] = ba
        result[lane + '_count_AB'] = count_ab
        result[lane + '_count_BA'] = count_ba
    return result



def cloned(value):
    if isinstance(value, dict):
        return {key: cloned(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cloned(item) for item in value]
    return value.clone()


@torch.inference_mode()
def cached_features(model, tensor):
    with patch.object(torch, 'autocast', lambda *args, **kwargs: nullcontext()):
        image = tensor.to('mps')
        return model.f(image), model.refiner_features(image)


def cached_predict(model, tensors, features):
    descriptors = iter([cloned(value[0]) for value in features])
    fine = iter([cloned(value[1]) for value in features])
    with patch.object(model.f, 'forward', side_effect=lambda image: next(descriptors)), \
         patch.object(model.refiner_features, 'forward', side_effect=lambda image: next(fine)):
        output = predict(model, tensors, 'mps')
    assert next(descriptors, None) is None and next(fine, None) is None
    return output


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    selection, scenes, _ = inputs()
    out = OUT / 'dense'
    out.mkdir(exist_ok=True)
    plan = {
        'training': False, 'source_plan_sha256': sha(OUT / 'plan.json'),
        'script_sha256': sha(__file__), 'checkpoint_sha256': CHECKSUM,
        'source_rule': 'Exact previous RoMa adapter and cycle/support rule. Vectorized target mask evaluation,validate against old edge_evidence.',
        'all_pairs': 3600, 'resolution': 640, 'precision': 'float32', 'device': 'mps',
        'cache': 'Cache each source image descriptor and refiner features within scene. Native pair forward receives cloned cached features. Save only object-level scores.',
        'previous_source': 'uncached-plan.json preserves the original execution plan for earlier score files.',
        'cache_equivalence': 'Compare cached and ordinary predictions for first new pair,require exact arrays.',
    }
    if (out / 'plan.json').exists():
        assert read(out / 'plan.json') == plan
    else:
        save(out / 'plan.json', plan)
    model, _ = build_model('mps')
    files = [out / 'plan.json', out / 'uncached-plan.json']
    comparisons = []
    cache_check = out / 'cache-equivalence.json'
    checked = cache_check.exists()
    start = time.monotonic()
    done = 0
    for scene, record in scenes.items():
        paths = {cam: view_path(scene, cam) for cam in record['cameras']}
        masks = {}
        tensors = {}
        cache = {}
        for a, b in combinations(range(1, 10), 2):
            left, right = str(a), str(b)
            destination = out / f'{scene}-{a}-{b}.npz'
            files.append(destination)
            if destination.exists():
                done += 1
                continue
            for side in (left, right):
                if side in masks:
                    continue
                filename = OUT / 'views' / f'{scene}-0{side}.npz'
                while not filename.exists():
                    time.sleep(.5)
                with np.load(filename, allow_pickle=False) as data:
                    masks[side] = {name: data[name] for name in ('box', 'mask', 'quality')}
                tensors[side] = input_tensor(paths[side])
                cache[side] = cached_features(model, tensors[side])
            outputs = cached_predict(model, [tensors[left], tensors[right]], [cache[left], cache[right]])
            if not checked:
                original = predict(model, [tensors[left], tensors[right]], 'mps')
                deltas = {name: float(np.max(np.abs(original[name] - outputs[name]))) for name in original}
                assert all(np.array_equal(original[name], outputs[name]) for name in original), deltas
                save(cache_check, {'scene': scene, 'cameras': [a,b], 'max_absolute_deltas': deltas, 'bit_identical': True})
                checked = True
            pair_masks = {'A': masks[left], 'B': masks[right]}
            values = evidence(outputs, pair_masks)
            if done == 0 or (done % 360 == 0):
                shape = values['mask'].shape
                edges = list(product(range(shape[0]), range(shape[1])))
                old = edge_evidence({'edges': edges}, outputs, pair_masks)
                delta = max(float(np.max(np.abs(values[name] - old[name].reshape(shape))))
                            for name in ('box', 'mask'))
                assert delta < 1e-6, delta
                for lane in ('box', 'mask'):
                    for direction in ('AB', 'BA'):
                        key = lane + '_count_' + direction
                        assert np.array_equal(values[key], old[key].reshape(shape))
                comparisons.append({'scene': scene, 'cameras': [a, b], 'maximum_score_delta': delta})
            tmp = destination.with_suffix('.pending.npz')
            np.savez_compressed(tmp, **values)
            tmp.replace(destination)
            done += 1
            if done == 1 or done % 100 == 0:
                elapsed = time.monotonic() - start
                save(out / 'progress.json', {'pairs_complete': done, 'total_pairs': 3600, 'elapsed_seconds': elapsed,
                                             'pair_rate_including_waits': done / elapsed})
                print(f'RoMa {done}/3600 pairs, elapsed {elapsed:.1f}s', flush=True)
    assert done == selection['image_pairs']
    save(out / 'equivalence.json', comparisons)
    files.extend([out / 'equivalence.json', cache_check])
    freeze(out, files)
    print('All3600 dense pairs frozen.', flush=True)


if __name__ == '__main__':
    main()
