"""Extract every official test annotation using the released model and crops."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torchvision import models
import yaml

from common import (
    ASNET, ASNET_REPO, DATA, HERE, LABELS, ROOT, VIT, VIT_REPO,
    load_module, load_native_dataset, read, save, sha,
)


def configuration(model_name):
    if model_name == 'vit':
        repo = VIT_REPO
        config_path = repo / 'models/tripletnet_1gpu_all_pairs_vit_tiny_p3de/train.yaml'
        weights = VIT / 'assets/TripletNet_cam_pos.pth'
    else:
        repo = ASNET_REPO
        config_path = repo / 'models/asnet/train.yaml'
        weights = ASNET / 'assets/ASNet.pth'
    config = yaml.safe_load(config_path.read_text())
    config.update(device='mps', original_img_size=(1080, 1920),
                  cropped_img_size=(224, 224), img_dir=str(DATA / 'images'),
                  zoomout_crop_num='single_crop')
    return repo, config_path, config, weights


def load_model(model_name, config):
    if model_name == 'vit':
        adapter = load_module('published_vit_adapter', VIT / 'run.py')
        return adapter.load_model(torch.device('mps'))
    source = load_module('published_asnet_model', ASNET_REPO / 'src/model.py')
    resnet18 = models.resnet18
    with patch.object(models, 'resnet18', side_effect=lambda **_: resnet18(weights=None)):
        model = source.ASNet(config)
    state = torch.load(ASNET / 'assets/ASNet.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def position_values(sample):
    position = sample['instance_pos']
    assert position.dtype == torch.float64
    expected = ((position[:, :2] + position[:, 2:]) / 120).long()
    single = position.float()
    actual = ((single[:, :2] + single[:, 2:]) / 120).long()
    if not torch.equal(actual, expected):
        # Only the two center bins enter P3DE. Use their exact integer
        # representatives if float32 rounding crosses a boundary.
        bad = (actual != expected).any(dim=1)
        center = expected[bad].float() * 60 + 30
        single[bad] = torch.cat([center, center], dim=1)
        assert torch.equal(((single[:, :2] + single[:, 2:]) / 120).long(), expected)
        return single, int(bad.sum())
    return single, 0


@torch.inference_mode()
def encode_sample(model_name, model, sample, device, batch):
    pixels = sample['single_crop']
    attributes = sample['instance_attributes']
    position, adjusted = position_values(sample)
    output = {}
    for start in range(0, len(pixels), batch):
        end = start + batch
        images = pixels[start:end].clone().to(device)
        if model_name == 'vit':
            pos = position[start:end] if str(device) == 'mps' else sample['instance_pos'][start:end]
            feature = model.forward_once(
                images, torch.as_tensor(attributes[start:end, 0], device=device), pos.to(device))
        else:
            feature = model.forward_once(images)
        for name, values in feature.items():
            output.setdefault(name, []).append(values.cpu().numpy())
    encoded = {name: np.concatenate(values) for name, values in output.items()}
    assert all(np.isfinite(values).all() and len(values) == len(attributes) for values in encoded.values())
    return encoded, adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('vit', 'asnet'), required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--selection', type=str)
    parser.add_argument('--batch', type=int, default=96)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    repo, config_path, config, weights = configuration(args.model)
    dataset = load_native_dataset(repo, config)
    assert len(dataset) == 1861
    selection = None
    if args.selection:
        selection_path = HERE / args.selection / 'selection.json'
        selection = read(selection_path)
        assert selection['source_sha256']['test.json'] == sha(LABELS)
        chosen = {row['scene'] for row in selection['scenes']}
        dataset.scene_list = [scene for scene in dataset.scene_list if scene in chosen]
        dataset.scene_labels = {scene: dataset.scene_labels[scene] for scene in dataset.scene_list}
        assert len(dataset) == selection['sample_scenes']
    box_count = sum(len(camera['instances']) for scene in dataset.scene_labels.values()
                    for camera in scene['cameras'].values())
    assert box_count == (selection['annotated_appearances'] if selection else 408937)
    out = HERE / args.selection / args.model if args.selection else HERE / args.model
    out.mkdir(exist_ok=True)
    (out / 'features').mkdir(exist_ok=True)
    plan = {
        'training': False, 'split': str(LABELS.relative_to(ROOT)), 'split_sha256': sha(LABELS),
        'model': args.model, 'weights_sha256': sha(weights), 'config_sha256': sha(config_path),
        'source_sha256': {name: sha(repo / 'src' / name) for name in ('model.py', 'data.py', 'utils.py', 'config.py', 'test.py')},
        'runner_sha256': sha(__file__), 'common_sha256': sha(HERE / 'common.py'),
        'scenes': len(dataset), 'images': len(dataset) * 9, 'annotated_appearances': box_count,
        'selection_sha256': sha(selection_path) if selection else None,
        'image_pairs': len(dataset) * 36, 'camera_pairs': [[a, b] for a in range(1, 10) for b in range(a + 1, 10)],
        'batch': args.batch, 'preprocess_workers': 4, 'device': 'mps', 'precision': 'float32',
        'crop': 'Unmodified upstream MessyTableDatasetFeatures and crop_feat functions. OpenCV BGR,2x crop,224x224 torchvision Resize and ToTensor.',
        'batching': 'Chunk the scene crop batch for memory. Verify CPU/device and batch-size invariance before full extraction.',
        'position_port': 'Verify native float64 center bins against MPS float32 for every item. Preserve exact bins via representative coordinates if rounding crosses a boundary.',
        'metrics': 'Evaluate full official36 camera pairs per scene. Original Linux KMSolver and upstream AP,FPR95,IPAA functions in pinned Python3.7 runtime.',
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python')},
        'scope': selection['interpretation'] if selection else 'Full official test split, including previously reserved project test scenes. Authorized by explicit full-test reproduction request.',
    }
    if (out / 'plan.json').exists():
        assert read(out / 'plan.json') == plan
    else:
        save(out / 'plan.json', plan)
    model = load_model(args.model, config)
    example = dataset[0]
    cpu, _ = encode_sample(args.model, model, example, torch.device('cpu'), args.batch)
    model.to('mps')
    trial, _ = encode_sample(args.model, model, example, 'mps', args.batch)
    smaller, _ = encode_sample(args.model, model, example, 'mps', 17)
    checks = {}
    for name in trial:
        checks[name] = {
            'cpu_mps_max_abs_delta': float(np.max(np.abs(cpu[name] - trial[name]))),
            'batch_size_max_abs_delta': float(np.max(np.abs(smaller[name] - trial[name]))),
        }
        assert np.allclose(cpu[name], trial[name], atol=3e-4, rtol=2e-4), checks
        assert np.allclose(smaller[name], trial[name], atol=3e-4, rtol=2e-4), checks
    save(out / 'smoke.json', {'passed': True, 'scene': example['scene_name'], 'features': checks})
    print(args.model, 'CPU/MPS and batch checks passed.', checks, flush=True)
    model.to('mps')
    wanted = min(args.limit or len(dataset), len(dataset))
    indices = [index for index in range(wanted)
               if not (out / 'features' / (dataset.scene_list[index] + '.npz')).exists()]
    pool = ThreadPoolExecutor(max_workers=4)
    pending = deque()
    iterator = iter(indices)
    for _ in range(6):
        index = next(iterator, None)
        if index is not None:
            pending.append((index, pool.submit(dataset.__getitem__, index)))
    started = time.monotonic()
    processed_boxes, processed_scenes, adjusted_total = 0, 0, 0
    while pending:
        index, future = pending.popleft()
        sample = future.result()
        upcoming = next(iterator, None)
        if upcoming is not None:
            pending.append((upcoming, pool.submit(dataset.__getitem__, upcoming)))
        features, adjusted = encode_sample(args.model, model, sample, 'mps', args.batch)
        path = out / 'features' / (sample['scene_name'] + '.npz')
        temp = path.with_suffix('.pending.npz')
        np.savez_compressed(temp, **features, attributes=sample['instance_attributes'],
                            positions=sample['instance_pos'].numpy())
        temp.replace(path)
        processed_boxes += len(sample['instance_attributes'])
        processed_scenes += 1
        adjusted_total += adjusted
        elapsed = time.monotonic() - started
        progress = {
            'scenes_complete': len(list((out / 'features').glob('*.npz'))), 'total_scenes': len(dataset),
            'this_session_scenes': processed_scenes, 'this_session_boxes': processed_boxes,
            'elapsed_seconds': elapsed, 'crops_per_second': processed_boxes / elapsed,
            'adjusted_position_bins_this_session': adjusted_total,
            'remaining_estimate_seconds': max(0, (len(dataset) - (index + 1)) / processed_scenes * elapsed),
        }
        save(out / 'progress.json', progress)
        if processed_scenes == 1 or processed_scenes % 25 == 0 or index == wanted - 1:
            print(args.model, f'{index+1}/{len(dataset)} scenes, {processed_boxes} crops, {elapsed:.1f}s, {processed_boxes/elapsed:.1f} crops/s', flush=True)
    pool.shutdown()
    if wanted == len(dataset):
        files = [out / 'features' / (scene + '.npz') for scene in dataset.scene_list]
        assert all(path.is_file() for path in files)
        save(out / 'inference-freeze.json', {str(path.relative_to(out)): sha(path)
                                            for path in [out / 'plan.json', out / 'smoke.json', *files]})
    print(args.model, 'extraction pass finished.', flush=True)


if __name__ == '__main__':
    main()
