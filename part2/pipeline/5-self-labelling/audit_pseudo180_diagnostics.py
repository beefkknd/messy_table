#!/usr/bin/env python3
"""Diagnose frozen pseudo-label localization, scope and misses without exporting truth."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from audit_pseudo180 import area_overlap, checked_inputs, key, map_detections
from prepare import LABELS, OUT, ROOT, digest, load, order, save, truth_rows, verify


def match_at_iou(predictions, annotations, threshold):
    if not predictions or not annotations:
        return {}
    left = np.asarray([row['box_xyxy'] for row in predictions], dtype=float)
    right = np.asarray(list(annotations.values()), dtype=float)
    spans = np.maximum(np.minimum(left[:, None, 2:], right[None, :, 2:]) -
                       np.maximum(left[:, None, :2], right[None, :, :2]), 0)
    intersection = spans.prod(axis=2)
    union = np.prod(left[:, 2:] - left[:, :2], axis=1)[:, None] + np.prod(right[:, 2:] - right[:, :2], axis=1)[None, :] - intersection
    overlap = intersection / union
    rows, cols = linear_sum_assignment(np.where(overlap >= threshold, overlap, 0), maximize=True)
    keys = list(annotations)
    return {int(a): int(keys[b]) for a, b in zip(rows, cols) if overlap[a, b] >= threshold}


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def unmapped_proxy(box, truth):
    overlaps = [(area_overlap(box, target), area(target)) for target in truth.values()]
    target_iou = max((intersection / (area(box) + target_area - intersection)
                      for intersection, target_area in overlaps), default=0)
    if target_iou >= .50:
        return 'duplicate_candidate'
    strongly_covered = [intersection for intersection, target_area in overlaps if intersection / target_area >= .50]
    if len(strongly_covered) >= 2 and sum(strongly_covered) >= .35 * area(box):
        return 'possible_group'
    if target_iou == 0:
        return 'no_target_overlap'
    if target_iou < .10:
        return 'low_target_overlap'
    return 'localization_or_scope_mismatch'


def target_size(box):
    length = min(box[2] - box[0], box[3] - box[1])
    return '<32px' if length < 32 else '32-<96px' if length < 96 else '>=96px'


def montage(examples, out):
    cols, width, height = 6, 270, 205
    rows = (len(examples) + cols - 1) // cols
    if not rows:
        return
    panel = Image.new('RGB', (cols * width, rows * height), '#202735')
    draw = ImageDraw.Draw(panel)
    for i, (path, anchor, gold, caption) in enumerate(examples):
        x1, y1, x2, y2 = anchor
        dx, dy = max(18, (x2 - x1) * .55), max(18, (y2 - y1) * .55)
        with Image.open(ROOT / path) as source:
            left, top = max(0, int(x1 - dx)), max(0, int(y1 - dy))
            right, bottom = min(source.width, int(x2 + dx + 1)), min(source.height, int(y2 + dy + 1))
            window = source.crop((left, top, right, bottom))
        window.thumbnail((width - 6, height - 25))
        ox, oy = i % cols * width + 3, i // cols * height + 3
        panel.paste(window, (ox, oy))
        sx, sy = window.width / (right - left), window.height / (bottom - top)
        for box, color in [(box, '#58e8b7') for box in gold] + [(anchor, '#ff718d')]:
            if not area_overlap(box, [left, top, right, bottom]):
                continue
            draw.rectangle((ox + (max(box[0], left) - left) * sx, oy + (max(box[1], top) - top) * sy,
                            ox + (min(box[2], right) - left) * sx, oy + (min(box[3], bottom) - top) * sy), outline=color, width=2)
        draw.text((ox, oy + height - 20), f'{i + 1}. {caption}', fill='white')
    panel.save(out, quality=91)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-run', required=True, type=Path)
    parser.add_argument('--temporary-review-directory', type=Path, help='Optional private temporary montage directory. Remove after inspection.')
    args = parser.parse_args()
    output = OUT / 'pseudo180-diagnostics.json'
    assert not output.exists()
    verify(OUT)
    cohort_path = OUT / 'cohorts/pseudo180.json'
    cohort = load(cohort_path)['scenes']
    frozen, raw, admissions, _ = checked_inputs(args.teacher_run.resolve(), cohort_path, cohort)
    native = LABELS['train']
    native_sha = digest(native)
    assert native_sha == load(OUT / 'audit.json')['source_sha256'][native.relative_to(ROOT).as_posix()]
    truth = {scene['id']: scene for scene in truth_rows(cohort)}
    sample_scenes = set(order(truth, 'pseudo-missedbox-audit-scene')[:18])
    full = defaultdict(Counter)
    per_view = []
    mistakes = defaultdict(Counter)
    confidence = defaultdict(Counter)
    miss_size = defaultdict(Counter)
    missed_samples, unmatched_samples = [], []
    for (scene, side), row in admissions.items():
        boxes = truth[scene]['views'][side]['objects_xyxy']
        xywh = {oid: [x1, y1, x2 - x1, y2 - y1] for oid, (x1, y1, x2, y2) in boxes.items()}
        original, supplements = row['original'], row['supplements']
        combined = original + supplements
        maps = {('B', .50): map_detections(original, xywh), ('C', .50): map_detections(combined, xywh),
                ('B', .75): match_at_iou(original, boxes, .75), ('C', .75): match_at_iou(combined, boxes, .75)}
        for threshold in (.50, .75):
            for arm, count in [('B', len(original)), ('C', len(combined)), ('supplement_in_C', len(supplements))]:
                mapped = len(maps[arm, threshold]) if arm in ('B', 'C') else sum(i >= len(original) for i in maps['C', threshold])
                full[f'{arm}@{threshold:.2f}'].update(candidates=count, mapped=mapped, gt_endpoints=len(boxes))
        for index, detection in enumerate(combined):
            score = detection['confidence']
            band = '0.80–<0.85' if score < .85 else '0.85–<0.90' if score < .90 else '0.90–<0.95' if score < .95 else '0.95–1.00'
            seed_lane = 'original' if index < len(original) else 'supplement'
            confidence[f'{seed_lane}|{band}'].update(candidates=1, mapped_iou50=index in maps['C', .50], mapped_iou75=index in maps['C', .75])
            if index not in maps['C', .50]:
                proxy = unmapped_proxy(detection['box_xyxy'], boxes)
                mistakes[seed_lane][f'{proxy}|{detection["class"]}'] += 1
                unmatched_samples.append((row['view']['path'], detection['box_xyxy'], list(boxes.values()),
                                          f'{seed_lane[0].upper()} {detection["class"]} {score:.2f}'))
        mapped_gt = set(maps['C', .50].values())
        missed = {oid: box for oid, box in boxes.items() if int(oid) not in mapped_gt}
        by_size = Counter()
        for oid, box in boxes.items():
            size = target_size(box)
            is_edge = min(box[:2]) == 0 or box[2] == row['view']['width'] or box[3] == row['view']['height']
            miss_size[size].update(gt=1, missed=oid in missed, gt_edge=is_edge, missed_edge=is_edge and oid in missed)
            if oid in missed:
                by_size[size] += 1
        if scene in sample_scenes and side == order('ABC', f'pseudo-missedbox-audit-view|{scene}')[0]:
            per_view.append({'setup': row['setup'], 'scene': scene, 'view': side, 'gt': len(boxes),
                             'mapped_C_iou50': len(mapped_gt), 'missed_by_short_side': dict(by_size)})
            if missed:
                box_id = order(missed, f'pseudo-missedbox-audit-object|{scene}|{side}')[0]
                crop = missed[box_id]
                matched_predictions = [combined[index]['box_xyxy'] for index in maps['C', .50]]
                missed_samples.append((row['view']['path'], crop, matched_predictions, target_size(crop)))
    stats = {lane: {**counts, 'precision': counts['mapped'] / counts['candidates'],
                    'recall_over_all_gt_endpoints': counts['mapped'] / counts['gt_endpoints']}
             for lane, counts in full.items()}
    save(output, {'cohort_sha256': digest(cohort_path), 'source_annotation_sha256': native_sha, 'teacher_artifact_hashes': frozen,
                  'inventory_metrics': stats, 'confidence_counts': confidence, 'unmapped_geometry_proxy_by_teacher_class': mistakes,
                  'C_misses_by_target_short_side': miss_size, 'imagewise_miss_sample': per_view,
                  'definition': 'Unmapped geometry proxies use native GT boxes and are not independent background/furniture annotations. '
                  'Imagewise sample uses 18 hash-ranked setups and one hash-ranked view per setup. No boxes or instance IDs are exported.'})
    save(OUT / 'pseudo180-diagnostics-freeze.json', {str(path.relative_to(ROOT)): digest(path) for path in (output, Path(__file__).resolve())})
    if args.temporary_review_directory:
        review = args.temporary_review_directory
        assert not review.is_relative_to(ROOT), 'Use a private temporary directory outside the workspace'
        review.mkdir(parents=True, exist_ok=True)
        montage(unmatched_samples, review / 'unmapped.jpg')
        montage(missed_samples, review / 'missed.jpg')
    print(stats)
    print('Unmapped diagnosis proxies', mistakes)
    print('Miss size', miss_size)


if __name__ == '__main__':
    main()
