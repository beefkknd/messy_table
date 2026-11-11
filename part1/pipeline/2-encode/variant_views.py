"""Cache nine-view SAM masks and frozen DINO crop variations for every scene."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from ultralytics import SAM

from variant_common import EDGES, OUT, annotations, freeze, inputs, read, save, sha, view_path
from run_full_scene import encode_crops
from run_messytable_dino_crops import BRANCHES, SCALES, expanded
from runtime_assets import dinov2_source, dinov2_weights, model_file

sys.path.insert(0, str(EDGES))
from transform_crops import MaskResult, SETTINGS, grabcut, grayscale_and_edges, masked_rgb, native_crop

RESOLUTION = 640


def mask_view(rgb, boxes_float, boxes, segmenter, path):
    height, width = rgb.shape[:2]
    centers = np.arange(RESOLUTION) + .5
    scaled = boxes_float * np.array([RESOLUTION / width, RESOLUTION / height] * 2)
    grids = np.array([(centers[None, :] >= x1) & (centers[None, :] < x2)
                      & (centers[:, None] >= y1) & (centers[:, None] < y2)
                      for x1, y1, x2, y2 in scaled], dtype=bool)
    foreground = np.zeros_like(grids)
    quality = np.full(len(boxes), np.nan)
    cropped = [MaskResult(None, 'missing_prompt_mask') for _ in boxes]
    result = segmenter.predict(str(path), bboxes=boxes_float.tolist(), imgsz=1024,
                               conf=0., device='mps', verbose=False)[0]
    if result.masks is not None:
        assert tuple(result.masks.data.shape[1:]) == (height, width)
        prompt = result.boxes.cls.int().tolist()
        assert len(set(prompt)) == len(prompt)
        for index, mask, confidence in zip(prompt, result.masks.data.cpu().numpy(),
                                           result.boxes.conf.tolist(), strict=True):
            assert 0 <= index < len(boxes)
            foreground[index] = cv2.resize(mask.astype(np.uint8), (RESOLUTION, RESOLUTION),
                                            interpolation=cv2.INTER_NEAREST_EXACT).astype(bool) & grids[index]
            quality[index] = confidence
            x1, y1, x2, y2 = boxes[index]
            cropped[index] = MaskResult(mask[y1:y2, x1:x2].astype(bool, copy=True), 'candidate', confidence)
    return grids, foreground, quality, cropped


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    selection, scenes, _ = inputs()
    out = OUT / 'views'
    out.mkdir(exist_ok=True)
    plan = {
        'training': False, 'source_plan_sha256': sha(OUT / 'plan.json'),
        'scripts_sha256': sha(__file__), 'dino_weights_sha256': sha(dinov2_weights()),
        'sam_weights_sha256': sha(model_file('sam2.1_s.pt')),
        'old_transform_sha256': sha(EDGES / 'transform_crops.py'), 'settings': SETTINGS,
        'SAM': 'Native float annotation box prompts,1024; preserve native mask,round boxes for DINO crop,640grid clipped to float box for RoMa.',
        'rotations': [0, 90, 180, 270], 'crop_scales': SCALES,
        'grabcut_execution': 'Prefetch8 views with4 view tasks and16 mask workers. Identical per-crop function and RNG reset.',
        'prefix_provenance': 'Existing view files retained from serial-plan.json; math and model preprocessing unchanged.',
    }
    if (out / 'plan.json').exists():
        assert read(out / 'plan.json') == plan
    else:
        save(out / 'plan.json', plan)
    processor = AutoImageProcessor.from_pretrained(dinov2_source(), local_files_only=True)
    model = AutoModel.from_pretrained(dinov2_source(), local_files_only=True).to('mps').eval().requires_grad_(False)
    segmenter = SAM(str(model_file('sam2.1_s.pt')))
    workers = ThreadPoolExecutor(max_workers=16)
    prefetch = ThreadPoolExecutor(max_workers=4)

    def grabcut_job(scene, camera, ds):
        image = np.array(Image.open(view_path(scene, camera)).convert('RGB'))
        boxes = np.array([d['box_xyxy'] for d in ds]).round().astype(int)
        boxes = np.clip(boxes, 0, [1920, 1080] * 2)
        return list(workers.map(grabcut, repeat(image), boxes))

    remaining = []
    for scene, record in scenes.items():
        for camera, ds in annotations(record).items():
            destination = out / (scene + '-0' + camera + '.npz')
            if not (destination.exists() and destination.with_suffix('.json').exists()):
                remaining.append((scene, camera, ds))
    queue = iter(remaining)
    pending = {}
    for _ in range(8):
        task = next(queue, None)
        if task is not None:
            pending[task[:2]] = prefetch.submit(grabcut_job, *task)
    start = time.monotonic()
    done = 0
    files = [out / 'plan.json', out / 'serial-plan.json']
    for scene, record in scenes.items():
        for camera, ds in annotations(record).items():
            stem = scene + '-0' + camera
            destination = out / (stem + '.npz')
            metadata = out / (stem + '.json')
            files.extend([destination, metadata])
            if destination.exists() and metadata.exists():
                done += 1
                continue
            path = view_path(scene, camera)
            with Image.open(path) as image:
                rgb = np.array(image.convert('RGB'))
            assert rgb.shape == (1080, 1920, 3)
            boxes_float = np.array([d['box_xyxy'] for d in ds], dtype=float)
            boxes = boxes_float.round().astype(int)
            boxes = np.clip(boxes, 0, [1920, 1080] * 2)
            assert (boxes[:, 2:] > boxes[:, :2]).all()
            grid, masks, quality, sam = mask_view(rgb, boxes_float, boxes, segmenter, path)
            crops = {name: [native_crop(rgb, box) for box in expanded(boxes, scale, 1920, 1080)]
                     for name, scale in zip(BRANCHES, SCALES, strict=True)}
            raw = crops['tight_1x']
            crops['gray'], crops['canny'] = map(list, zip(*(grayscale_and_edges(crop)[:2] for crop in raw)))
            mask_meta = {}
            grabcut_masks = pending.pop((scene, camera)).result()
            task = next(queue, None)
            if task is not None:
                pending[task[:2]] = prefetch.submit(grabcut_job, *task)
            for name, results in (('sam', sam), ('grabcut', grabcut_masks)):
                transformed, diagnostic = zip(*(masked_rgb(crop, mask)
                                                for crop, mask in zip(raw, results, strict=True)))
                crops[name] = list(transformed)
                mask_meta[name] = list(diagnostic)
            for lane in ('tight_1x', 'sam'):
                for angle in (90, 180, 270):
                    crops[f'{lane}_r{angle}'] = [np.ascontiguousarray(np.rot90(crop, angle // 90))
                                                for crop in crops[lane]]
            labels = list(crops)
            pixels = [crop for label in labels for crop in crops[label]]
            vectors = np.concatenate([encode_crops(pixels[start:start+64], processor, model, 'mps')
                                      for start in range(0, len(pixels), 64)])
            arrays = dict(zip(labels, np.split(vectors, len(labels)), strict=True))
            arrays.update(box=grid, mask=masks, quality=quality, rounded_boxes=boxes)
            for index, mask in enumerate(sam):
                arrays[f'native_{index}'] = mask.mask if mask.mask is not None else np.zeros(raw[index].shape[:2], bool)
            temporary = destination.with_suffix('.pending.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)
            save(metadata, {'scene': scene, 'camera': camera, 'image_sha256': sha(path),
                            'boxes_xyxy': boxes_float.tolist(), 'masks': mask_meta})
            done += 1
            if done == 1 or done % 50 == 0:
                elapsed = time.monotonic() - start
                print(f'Views {done}/900, elapsed {elapsed:.1f}s', flush=True)
    prefetch.shutdown()
    workers.shutdown()
    assert done == selection['images'] == 900
    freeze(out, files)
    print('All900 views frozen.', flush=True)


if __name__ == '__main__':
    main()
