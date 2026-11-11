"""Use each upstream compute_distance method on the official ordered pairs."""
import argparse
import ast
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import ASNET_REPO, HERE, VIT_REPO, read, save, sha


def native_distance(model_name):
    path = (VIT_REPO if model_name == 'vit' else ASNET_REPO) / 'src/model.py'
    class_name = 'TripletNet_cam_pos' if model_name == 'vit' else 'ASNet'
    parsed = ast.parse(path.read_text())
    selected = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in selected.body if isinstance(node, ast.FunctionDef) and node.name == 'compute_distance')
    namespace = {'F': F, 'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['compute_distance']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', default='sample100')
    parser.add_argument('--model', choices=('vit', 'asnet'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sample = HERE / args.selection
    selection = read(sample / 'selection.json')
    folder = sample / args.model
    freeze = read(folder / 'inference-freeze.json')
    for name, digest in freeze.items():
        assert sha(folder / name) == digest, name
    distance = native_distance(args.model)
    distances, labels, rows = [], [], []
    offset = 0
    smoke = None
    start = time.monotonic()
    with torch.inference_mode():
        for index, scene in enumerate(selection['scenes']):
            file = folder / 'features' / (scene['scene'] + '.npz')
            with np.load(file, allow_pickle=False) as f:
                attributes = f['attributes']
                keys = ['feat'] if args.model == 'vit' else ['center', 'surrnd']
                for a in range(1, 10):
                    left_mask = attributes[:, 0] == a
                    ids_a = attributes[left_mask, 1].tolist()
                    for b in range(a + 1, 10):
                        right_mask = attributes[:, 0] == b
                        ids_b = attributes[right_mask, 1].tolist()
                        left = {key: torch.as_tensor(f[key][left_mask]).repeat_interleave(len(ids_b), dim=0)
                                for key in keys}
                        right = {key: torch.as_tensor(f[key][right_mask]).repeat(len(ids_a), 1)
                                 for key in keys}
                        result = distance(None, {k: v.to('mps') for k, v in left.items()},
                                          {k: v.to('mps') for k, v in right.items()}).cpu().numpy()
                        assert result.shape == (len(ids_a) * len(ids_b),) and np.isfinite(result).all()
                        if smoke is None:
                            cpu = distance(None, left, right).numpy()
                            delta = float(np.max(np.abs(cpu - result)))
                            assert np.allclose(cpu, result, atol=3e-4, rtol=2e-4), delta
                            smoke = {'scene': scene['scene'], 'cameras': [a, b], 'cpu_mps_max_abs_distance_delta': delta}
                        truth = (np.array(ids_a)[:, None] == np.array(ids_b)[None, :]).astype(float).reshape(-1)
                        distances.append(result.astype(np.float64))
                        labels.append(truth)
                        rows.append({'scene': scene['scene'], 'setup': scene['setup'], 'difficulty': scene['difficulty'],
                                     'camera_A': a, 'camera_B': b, 'main_bbox_id': ids_a, 'sec_bbox_id': ids_b,
                                     'start': offset, 'stop': offset + len(result)})
                        offset += len(result)
            if (index + 1) % 25 == 0:
                print(args.model, 'distances', index + 1, '/', len(selection['scenes']),
                      'scenes', f'{time.monotonic()-start:.1f}s', flush=True)
    dist = np.concatenate(distances)
    truth = np.concatenate(labels)
    assert len(rows) == selection['image_pairs'] == 3600
    assert len(dist) == len(truth) == selection['candidate_object_pairs']
    assert truth.sum() == selection['positive_object_pairs']
    np.save(folder / 'app_dist.npy', dist.reshape(-1, 1))
    np.save(folder / 'gt_inst.npy', truth.reshape(-1, 1))
    save(folder / 'pair-index.json', rows)
    plan = {
        'training': False, 'distance_method': 'Unmodified upstream compute_distance, FP32 MPS, stored as float64 to mirror native list->np.array conversion.',
        'inference_freeze_sha256': sha(folder / 'inference-freeze.json'),
        'selection_sha256': sha(sample / 'selection.json'), 'script_sha256': sha(__file__),
        'pairs': len(rows), 'object_pairs': len(dist), 'positive_pairs': int(truth.sum()),
        'smoke': smoke, 'elapsed_seconds': time.monotonic() - start,
    }
    save(folder / 'distance-plan.json', plan)
    save(folder / 'distance-freeze.json', {name: sha(folder / name) for name in (
        'distance-plan.json', 'app_dist.npy', 'gt_inst.npy', 'pair-index.json')})
    print(args.model, 'distances frozen.', plan['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
