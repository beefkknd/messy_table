"""Section 2 flow photos: one real test photo through Step 0, Step 1 and the combine step.

Draws four panels from saved results only: the best detector's (H: 2,400 labelled photos, 1024 px
full frame + 768 px tiles) full-frame and tile predictions, the frozen tile windows and merge rule,
and the human labels (used only to count found objects). The rebuilt merge must equal the saved
combined boxes before anything is drawn. No inference.

Photo choice is a fixed rule, stated on the page: among the 900 Part One sample photos where tiles
add at least two correctly placed objects, the one whose labelled-object count is closest to the
sample median (ties by name). That is scene 20191004-00180-03, camera 7.
"""
from __future__ import annotations

import hashlib
import html
import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

SCENE, CAM = '20191004-00180-03', '7'
WIDTH, HEIGHT = 1920, 1080
PANEL_W = 1200
TEAL, AMBER, RED = '#107d71', '#d18a00', '#c0392b'
TILE_COLORS = ['#42679a', '#107d71', '#b85300', '#8744a4', '#007989', '#a0522d']
FULL = (0, 0, WIDTH, HEIGHT)
PANELS = ('pipeline-1-photo.jpg', 'pipeline-2-views.jpg', 'pipeline-3-candidates.jpg', 'pipeline-4-combined.jpg')


def _merge(base: list, tiles: list, overlaps: Callable, clearance: Callable) -> tuple[list, list]:
    """The study's raw-tile merge (research/detector-scale-and-integration/detect.py)."""
    kept = [d for d in base if d['confidence'] > .25]
    candidates = []
    for ti, tile in enumerate(tiles):
        for di, row in enumerate(tile['detections']):
            d = row['mapped']
            b = d['box_xyxy']
            if d['confidence'] <= .25 or b[0] >= b[2] or b[1] >= b[3] or clearance(b, tile['xyxy'], WIDTH, HEIGHT) <= 4:
                continue
            candidates.append((ti, di, d))
    candidates.sort(key=lambda x: (-x[2]['confidence'], x[0], x[1]))
    added = []
    for _, _, d in candidates:
        if kept and (overlaps([d['box_xyxy']], [p['box_xyxy'] for p in kept])[0] >= .55).any():
            continue
        kept.append(d)
        added.append(d)
    return kept, added


def _found(dets: list, instances: dict, overlaps: Callable) -> dict:
    """One-to-one box-to-object assignment at IoU >= 0.50, maximum total IoU (the report's rule)."""
    if not dets or not instances:
        return {}
    ids = list(instances)
    iou = overlaps([d['box_xyxy'] for d in dets], [instances[i]['pos'] for i in ids])[0]
    rows, cols = linear_sum_assignment(np.where(iou >= .5, iou, 0), maximize=True)
    return {int(r): ids[c] for r, c in zip(rows, cols) if iou[r, c] >= .5}


def evidence(root: Path, read: Callable, labels: dict) -> dict:
    sys.path.insert(0, str(root / 'research/detector-small-object-round'))
    from tile_policy import clearance, overlaps, windows
    study = root / 'research/detector-scale-and-integration'
    view = read(study / f'predictions/2400/{SCENE}.json')['views'][CAM]
    saved = {s['id']: s for s in read(study / 'step2/boxes.json')['scenes']}
    instances = labels[SCENE]['cameras'][CAM]['instances']
    full = [d for d in view['full_1024'] if d['confidence'] > .25]
    kept, added = _merge(view['full_1024'], view['tiles'], overlaps, clearance)
    assert [d['box_xyxy'] for d in kept] == [d['box_xyxy'] for d in saved[SCENE]['views'][CAM]['detections']['2400_1024']]
    assert [tuple(t['xyxy']) for t in view['tiles']] == windows(WIDTH, HEIGHT)
    tile_boxes = [(ti, r['mapped']) for ti, t in enumerate(view['tiles']) for r in t['detections'] if r['mapped']['confidence'] > .25]
    hit = _found(kept, instances, overlaps)
    facts = {'labelled': len(instances), 'full_boxes': len(full), 'tile_boxes': len(tile_boxes),
             'combined_boxes': len(kept), 'tile_added': len(added),
             'found_full': len(_found(full, instances, overlaps)), 'found_combined': len(hit)}
    assert facts['combined_boxes'] == facts['full_boxes'] + facts['tile_added']
    missed = [instances[i]['pos'] for i in instances if i not in hit.values()]
    return {'facts': facts, 'full': full, 'kept': kept, 'added': added, 'tiles': view['tiles'],
            'tile_boxes': tile_boxes, 'missed': missed}


def _zoom(boxes: list, margin: int = 70) -> tuple:
    a = np.asarray(boxes, float)
    x0, y0, x1, y1 = a[:, 0].min() - margin, a[:, 1].min() - margin, a[:, 2].max() + margin, a[:, 3].max() + margin
    w = max(x1 - x0, (y1 - y0) * 16 / 9)
    h = w * 9 / 16
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = min(max(cx - w / 2, 0), WIDTH - w), min(max(cy - h / 2, 0), HEIGHT - h)
    return (x0, y0, x0 + w, y0 + h)


class _Panel:
    def __init__(self, photo: Image.Image, crop: tuple = FULL):
        self.crop, self.scale = crop, PANEL_W / (crop[2] - crop[0])
        size = (PANEL_W, round(PANEL_W * 9 / 16))
        self.image = photo.crop(tuple(round(v) for v in crop)).resize(size, Image.LANCZOS)
        self.draw = ImageDraw.Draw(self.image, 'RGBA')
        if crop != FULL:
            bold = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold')), 24)
            self.draw.text((PANEL_W - 20, 14), 'zoomed in', font=bold, fill='white', anchor='ra', stroke_width=3, stroke_fill='#193039')

    def xy(self, b: list) -> list:
        x0, y0 = self.crop[:2]
        return [(b[0] - x0) * self.scale, (b[1] - y0) * self.scale, (b[2] - x0) * self.scale, (b[3] - y0) * self.scale]

    def box(self, b: list, color: str, width: int, dashed: bool = False) -> None:
        x0, y0, x1, y1 = self.xy(b)
        if not dashed:
            self.draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
            return
        for ax, ay, bx, by in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
            length = max(abs(bx - ax), abs(by - ay), 1e-9)
            for s in np.arange(0, length, 14):
                f0, f1 = s / length, min(s + 8, length) / length
                self.draw.line([ax + (bx - ax) * f0, ay + (by - ay) * f0, ax + (bx - ax) * f1, ay + (by - ay) * f1], fill=color, width=width)

    def save(self, path: Path) -> None:
        self.image.save(path, quality=86, optimize=True)


def draw(ev: dict, photo_path: Path, assets: Path) -> None:
    photo = Image.open(photo_path).convert('RGB')
    _Panel(photo).save(assets / PANELS[0])
    views = _Panel(photo)
    for color, tile in zip(TILE_COLORS, ev['tiles']):
        x0, y0, x1, y1 = views.xy(tile['xyxy'])
        views.draw.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=color, width=5, fill=color + '14')
    views.save(assets / PANELS[1])
    window = _zoom([d['box_xyxy'] for d in ev['kept']] + [d['box_xyxy'] for _, d in ev['tile_boxes']] + ev['missed'])
    candidates = _Panel(photo, window)
    for ti, d in ev['tile_boxes']:
        candidates.box(d['box_xyxy'], TILE_COLORS[ti], 3)
    for d in ev['full']:
        candidates.box(d['box_xyxy'], '#ffffff', 3)
    candidates.save(assets / PANELS[2])
    combined = _Panel(photo, window)
    added = {id(d) for d in ev['added']}
    for d in ev['kept']:
        combined.box(d['box_xyxy'], AMBER if id(d) in added else TEAL, 5 if id(d) in added else 3)
    for b in ev['missed']:
        combined.box(b, RED, 3, dashed=True)
    combined.save(assets / PANELS[3])


def _key(color: str, label: str) -> str:
    return f'<span class="key" style="--key:{color}"></span>{label}'


def flow_parts(facts: dict) -> dict:
    """HTML pieces for the Section 2 flow: one photo and one example note per block, plus a caption."""
    n = {k: f'{v:,}' for k, v in facts.items()}
    candidates = facts['full_boxes'] + facts['tile_boxes']
    notes = [
        f"Example: one camera’s view; people labelled {n['labelled']} objects (used only to score).",
        'Example: the whole photo plus these six overlapping 768-pixel tiles.',
        f"Example: {candidates} candidate boxes, {_key('#ffffff', n['full_boxes'] + ' whole photo')}, "
        f"{_key(TILE_COLORS[0], n['tile_boxes'] + ' tiles')} (one colour per tile). Many are duplicates.",
        f"Example: {n['combined_boxes']} boxes, {_key(TEAL, 'whole photo')} {_key(AMBER, n['tile_added'] + ' from tiles')}; "
        f"objects found {n['found_full']} → {n['found_combined']} of {n['labelled']}. "
        f"{_key(RED, 'Missed')}: two bottles in line got one box.",
    ]
    titles = ('Original photo', 'Step 0 · Prepare views', 'Step 1 · Find objects', 'Combine boxes')
    parts = {}
    for i, (name, title, note) in enumerate(zip(PANELS, titles, notes)):
        parts[f'FLOW_IMG_{i}'] = (f'<a class="flow-photo" href="assets/{name}" target="_blank" rel="noopener">'
                                  f'<img src="assets/{name}" alt="{html.escape(title)}: {html.escape(_plain(note))}" loading="lazy"></a>')
        parts[f'FLOW_NOTE_{i}'] = f'<span class="flow-note">{note}</span>'
    parts['PIPELINE_CAPTION'] = (
        f'<p class="flow-caption">Example photo: scene {SCENE}, camera {CAM}, run through detector H (2,400 labelled photos, '
        '1024-pixel full frame plus tiles). Boxes come from its saved predictions, and the rebuilt combine step reproduces the '
        'saved final boxes exactly. The Step 1 and Combine photos are zoomed in. The photo was chosen to show where tiles help: among the 900 Part One '
        'sample photos where tiles add at least two found objects, it has the most typical amount of clutter. In many easier '
        'photos the whole frame already finds everything.</p>')
    return parts


def _plain(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)


def build(root: Path, read: Callable, labels: dict, assets: Path, sources: dict) -> str:
    ev = evidence(root, read, labels)
    photo = root / f'data/messytable-full/MessyTable/images/{SCENE}-0{CAM}.jpg'
    sources[str(photo.relative_to(root))] = hashlib.sha256(photo.read_bytes()).hexdigest()
    draw(ev, photo, assets)
    return flow_parts(ev['facts'])
