"""Plain-language walkthrough of the winning matcher, drawn from one real benchmark view pair.

Every number and picture comes from frozen artifacts: the human boxes, the saved SAM masks, the
saved RoMa match counts, the saved ViT feature vectors, the frozen score arrays and a replay of the
author's native solver (flow-native-links.json, produced by flow_native_links.py in the pinned
Docker image). Nothing is re-inferred. Assertions tie each value back to its source.
"""
import html
import json
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion
from scipy.stats import rankdata

SCENE, CAM_A, CAM_B = '20190921-00003-02', 2, 8
MUGS = (('1', 'Mug 1', '#b85300'), ('2', 'Mug 2', '#007989'))
CROPS = {CAM_A: (170, 40, 1110, 820), CAM_B: (330, 60, 1270, 700)}
METHOD, PIXEL, LOOK, PIXEL_WEIGHT = 'blend_roma_vit_reference_dense_first', 'roma_mask', 'vit_reference', .75
CORRECT, WRONG, BOX = '#1b8a5a', '#c0392b', (255, 255, 255)
FONT = font_manager.findfont('DejaVu Sans')
FONT_BOLD = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
STRIPES = LinearSegmentedColormap.from_list('stripes', ['#16324f', '#42679a', '#eef2f5', '#5fae9f', '#0b5f55'])


def _font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))


def _read(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- evidence (numbers only)

def _pair_block(src):
    pairs = _read(src.pair_index)
    hits = [p for p in pairs if p['scene'] == SCENE and (p['camera_A'], p['camera_B']) == (CAM_A, CAM_B)]
    assert len(hits) == 1
    return hits[0]


def _percentile(values):
    return rankdata(values, method='average') / len(values)


def _scores(src, pair, mug_rows):
    load = lambda name: np.load(src.scores/f'{name}.npy').reshape(-1)
    pixel, look, blend = 1 - load(PIXEL), -load(LOOK), 1 - load(METHOD)
    pixel_rank, look_rank = _percentile(pixel), _percentile(look)
    assert np.allclose(PIXEL_WEIGHT*pixel_rank + (1-PIXEL_WEIGHT)*look_rank, blend, rtol=0, atol=1e-12)
    lo, width = pair['start'], len(pair['sec_bbox_id'])
    at = lambda array, i, j: float(array[lo + mug_rows[i]*width + mug_rows[j]])
    cells = {}
    for i, j in ((0, 0), (0, 1)):
        cells[f'{i}{j}'] = {'pixel': at(pixel_rank, i, j), 'look': at(look_rank, i, j),
                            'blend': at(blend, i, j), 'look_distance': -at(look, i, j),
                            'pixel_share': at(pixel, i, j)}
    zero_share = float(np.mean(pixel == 0))
    return cells, zero_share, len(blend)


def _spots(src, mug_rows):
    with np.load(src.dense/f'{SCENE}-{CAM_A}-{CAM_B}.npz', allow_pickle=False) as dense:
        forward, backward = dense['mask_count_AB'], dense['mask_count_BA']
        pick = lambda counts, i, j: int(counts[mug_rows[i], mug_rows[j]])
        return {f'{i}{j}': {'forward': pick(forward, i, j), 'backward': pick(backward, i, j)}
                for i in (0, 1) for j in (0, 1)}


def _fingerprints(src):
    with np.load(src.features/f'{SCENE}.npz', allow_pickle=False) as data:
        attributes, features = data['attributes'], data['feat']
    def vector(camera, identity):
        rows = np.where((attributes[:, 0] == camera) & (attributes[:, 1] == int(identity)))[0]
        assert len(rows) == 1
        return features[rows[0]].copy()
    return {(camera, ident): vector(camera, ident) for camera in (CAM_A, CAM_B) for ident, _, _ in MUGS}, features


def _links(src, result, pair, truth):
    replay = _read(src.native_links)
    official = result['methods'][METHOD]['author_assignment']
    assert replay['method'] == METHOD
    assert (replay['correct'], replay['wrong']) == (official['correct'], official['wrong'])
    links = replay['links'][f'{SCENE}/{CAM_A}-{CAM_B}']
    width = len(pair['sec_bbox_id'])
    def is_true(a, b):
        return bool(truth[pair['start'] + pair['main_bbox_id'].index(a)*width + pair['sec_bbox_id'].index(b)])
    return tuple((str(a), str(b), is_true(a, b)) for a, b in links)


def evidence(src, labels, result):
    pair = _pair_block(src)
    cameras = labels[SCENE]['cameras']
    for camera, key in ((CAM_A, 'main_bbox_id'), (CAM_B, 'sec_bbox_id')):
        assert [str(i) for i in pair[key]] == list(cameras[str(camera)]['instances'])
    mug_rows = [pair['main_bbox_id'].index(int(ident)) for ident, _, _ in MUGS]
    assert mug_rows == [pair['sec_bbox_id'].index(int(ident)) for ident, _, _ in MUGS]
    truth = np.load(src.gt_inst).reshape(-1)
    cells, zero_share, candidates = _scores(src, pair, mug_rows)
    prints, all_features = _fingerprints(src)
    for key, (a, b) in (('00', ('1', '1')), ('01', ('1', '2'))):
        distance = float(np.linalg.norm(prints[(CAM_A, a)] - prints[(CAM_B, b)]))
        assert abs(distance - cells[key]['look_distance']) < 1e-3
    best = result['methods'][METHOD]
    cutoff = 1 - best['km_threshold']*best['distance_p95']  # combined score needed to be allowed a link
    assert cells['00']['blend'] > cells['01']['blend'] > cutoff  # wrong pair clears the bar; one-partner rule decides
    blend = 1 - np.load(src.scores/f'{METHOD}.npy').reshape(-1)[pair['start']:pair['stop']]
    outcomes = best['scene_outcomes']
    by_level = {}
    for row in outcomes.values():
        by_level.setdefault(row['difficulty'], []).append(row['IPAA100_pairs'] / row['image_pairs'])
    return {'cells': cells, 'spots': _spots(src, mug_rows), 'zero_share': zero_share,
            'cutoff': cutoff, 'passing_in_pair': int(np.sum(blend > cutoff)),
            'scene_outcome': outcomes[SCENE], 'fully_correct': best['IPAA100'],
            'fully_correct_by_level': {k: float(np.mean(v)) for k, v in by_level.items()},
            'candidates': candidates, 'prints': prints, 'feature_scale': float(all_features.std()),
            'links': _links(src, result, pair, truth), 'objects': len(pair['main_bbox_id']),
            'boxes': {camera: {k: v['pos'] for k, v in cameras[str(camera)]['instances'].items()}
                      for camera in (CAM_A, CAM_B)}}


# ---------------------------------------------------------------- pictures

def _photo(src, camera):
    return Image.open(src.images/f'{SCENE}-0{camera}.jpg').convert('RGB')


def _crop_box(box, camera, scale):
    x0, y0 = CROPS[camera][:2]
    return [(box[0]-x0)*scale, (box[1]-y0)*scale, (box[2]-x0)*scale, (box[3]-y0)*scale]


def _side_by_side(src, height, gap):
    views, scales = [], []
    for camera in (CAM_A, CAM_B):
        crop = _photo(src, camera).crop(CROPS[camera])
        scale = height / crop.height
        views.append(crop.resize((round(crop.width*scale), height), Image.LANCZOS))
        scales.append(scale)
    canvas = Image.new('RGB', (views[0].width + gap + views[1].width, height + 46), 'white')
    canvas.paste(views[0], (0, 46))
    canvas.paste(views[1], (views[0].width + gap, 46))
    draw = ImageDraw.Draw(canvas)
    for camera, x in ((CAM_A, 0), (CAM_B, views[0].width + gap)):
        draw.text((x + 2, 8), f'Camera {camera}', font=_font(26, True), fill='#193039')
    return canvas, draw, scales, [0, views[0].width + gap]


def _outline(draw, rect, color, width, label=None):
    draw.rectangle(rect, outline=color, width=width)
    if label:
        font = _font(24, True)
        right, bottom = font.getbbox(label)[2:]
        draw.rectangle([rect[0], rect[1]-bottom-10, rect[0]+right+14, rect[1]], fill=color)
        draw.text((rect[0]+7, rect[1]-bottom-6), label, font=font, fill='white')


def draw_photos(ev, src, destination):
    canvas, draw, scales, offsets = _side_by_side(src, 620, 36)
    highlighted = {ident: (name, color) for ident, name, color in MUGS}
    for index, camera in enumerate((CAM_A, CAM_B)):
        ordered = sorted(ev['boxes'][camera].items(), key=lambda item: item[0] in highlighted)
        for ident, box in ordered:
            rect = _crop_box(box, camera, scales[index])
            rect = [rect[0]+offsets[index], rect[1]+46, rect[2]+offsets[index], rect[3]+46]
            if ident in highlighted:
                name, color = highlighted[ident]
                _outline(draw, rect, _rgb(color), 6, name)
            else:
                _outline(draw, rect, BOX, 3)
    canvas.save(destination, quality=86, optimize=True)


def _cutout(src, labels, camera, identity, size):
    entries = labels[SCENE]['cameras'][str(camera)]['instances']
    index = list(entries).index(identity)
    with np.load(src.views/f'{SCENE}-0{camera}.npz', allow_pickle=False) as views:
        x1, y1, x2, y2 = views['rounded_boxes'][index]
        mask = views[f'native_{index}'].astype(bool)
    pixels = np.asarray(_photo(src, camera))[y1:y2, x1:x2].astype(float)
    faded = np.where(mask[..., None], pixels, 255*.82 + pixels*.18)
    edge = mask & ~binary_erosion(mask, iterations=3)
    faded[edge] = _rgb({i: c for i, _, c in MUGS}[identity])
    tile = Image.fromarray(faded.astype(np.uint8))
    tile.thumbnail((size, size), Image.LANCZOS)
    square = Image.new('RGB', (size, size), (246, 249, 250))
    square.paste(tile, ((size-tile.width)//2, (size-tile.height)//2))
    return square


def draw_cutouts(src, labels, destination, size=300):
    margin, top, gap = 130, 50, 18
    canvas = Image.new('RGB', (margin + 2*size + gap, top + 2*size + gap), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, camera in enumerate((CAM_A, CAM_B)):
        draw.text((margin + col*(size+gap) + 8, 10), f'Camera {camera}', font=_font(24, True), fill='#193039')
    for row, (ident, name, color) in enumerate(MUGS):
        y = top + row*(size+gap)
        draw.text((4, y + size//2 - 16), name, font=_font(26, True), fill=_rgb(color))
        for col, camera in enumerate((CAM_A, CAM_B)):
            canvas.paste(_cutout(src, labels, camera, ident, size), (margin + col*(size+gap), y))
    canvas.save(destination, optimize=True)


def _context(src, labels, camera, identity, size):
    x1, y1, x2, y2 = labels[SCENE]['cameras'][str(camera)]['instances'][identity]['pos']
    dx, dy = (x2-x1)/2, (y2-y1)/2
    crop = _photo(src, camera).crop((int(max(0, x1-dx)), int(max(0, y1-dy)),
                                      int(min(1920, x2+1+dx)), int(min(1080, y2+1+dy))))
    crop.thumbnail((size, size), Image.LANCZOS)
    return crop


def _stripes(vector, scale, width, height):
    values = np.clip(vector/(2.4*scale), -1, 1)*.5 + .5
    colors = (STRIPES(values)[:, :3]*255).astype(np.uint8)
    strip = np.repeat(colors[None, :, :], height, axis=0)
    return Image.fromarray(strip).resize((width, height), Image.NEAREST)


def _difference(vector, reference, scale, width, height):
    gap = np.clip(np.abs(vector - reference)/(1.8*scale), 0, 1)
    ink = (255 - gap*215).astype(np.uint8)
    strip = np.repeat(np.stack([ink, ink, ink], axis=1)[None, :, :], height, axis=0)
    return Image.fromarray(strip).resize((width, height), Image.NEAREST)


def draw_fingerprints(ev, src, labels, destination):
    rows = ((CAM_A, '1'), (CAM_B, '1'), (CAM_B, '2'))
    thumb, bar_w, bar_h, diff_h, row_h, label_w = 190, 640, 90, 46, 262, 270
    canvas = Image.new('RGB', (thumb + 24 + label_w + bar_w, row_h*len(rows) - 30), 'white')
    draw = ImageDraw.Draw(canvas)
    names = {ident: (name, color) for ident, name, color in MUGS}
    reference = ev['prints'][rows[0]]
    order = np.argsort(reference)  # one shared stripe order for every row
    left = thumb + 24 + label_w
    for index, (camera, ident) in enumerate(rows):
        y = index*row_h + 12
        context = _context(src, labels, camera, ident, thumb)
        canvas.paste(context, ((thumb-context.width)//2, y + (thumb-context.height)//2))
        name, color = names[ident]
        draw.text((thumb + 24, y + 42), name, font=_font(44, True), fill=_rgb(color))
        draw.text((thumb + 24, y + 100), f'camera {camera}', font=_font(36), fill='#526970')
        vector = ev['prints'][(camera, ident)]
        canvas.paste(_stripes(vector[order], ev['feature_scale'], bar_w, bar_h), (left, y + 12))
        if index == 0:
            draw.text((left, y + bar_h + 22), 'reference', font=_font(28), fill='#526970')
            continue
        canvas.paste(_difference(vector[order], reference[order], ev['feature_scale'], bar_w, diff_h),
                     (left, y + bar_h + 18))
        draw.rectangle([left, y + bar_h + 18, left + bar_w - 1, y + bar_h + 18 + diff_h], outline='#bacace', width=2)
        draw.text((left, y + bar_h + diff_h + 22), 'difference from reference', font=_font(26), fill='#526970')
    canvas.save(destination, optimize=True)


def _link_line(draw, start, end, color, width):
    draw.line([start, end], fill=(255, 255, 255, 230), width=width + 4)
    draw.line([start, end], fill=color, width=width)
    for x, y in (start, end):
        r = width + 5
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline='white', width=3)


def draw_pairs(ev, src, destination):
    canvas, _, scales, offsets = _side_by_side(src, 560, 150)
    def centre(box, i):
        x0, y0 = CROPS[(CAM_A, CAM_B)[i]][:2]
        return (((box[0]+box[2])/2 - x0)*scales[i] + offsets[i], ((box[1]+box[3])/2 - y0)*scales[i] + 46)
    mug_colors = {ident: _rgb(color) for ident, _, color in MUGS}
    others = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    faint = ImageDraw.Draw(others)
    for a, b, correct in ev['links']:
        if a not in mug_colors:
            color = (*_rgb(CORRECT if correct else WRONG), 150)
            _link_line(faint, centre(ev['boxes'][CAM_A][a], 0), centre(ev['boxes'][CAM_B][b], 1), color, 2)
    canvas = Image.alpha_composite(canvas.convert('RGBA'), others)
    bold = ImageDraw.Draw(canvas)
    for a, b, _ in ev['links']:
        if a in mug_colors:
            _link_line(bold, centre(ev['boxes'][CAM_A][a], 0), centre(ev['boxes'][CAM_B][b], 1), mug_colors[a], 7)
    canvas.convert('RGB').save(destination, quality=86, optimize=True)


def draw_all(ev, src, labels, assets):
    draw_photos(ev, src, assets/'flow-1-photos.jpg')
    draw_cutouts(src, labels, assets/'flow-2a-cutouts.png')
    draw_fingerprints(ev, src, labels, assets/'flow-2b-fingerprints.png')
    draw_pairs(ev, src, assets/'flow-4-pairs.jpg')


# ---------------------------------------------------------------- page text

def _out_of_100(value):
    return f'{100*value:.1f}'


def _vote_row(label, value, kind, note=''):
    width = f'{100*value:.1f}%'
    extra = f'<small>{html.escape(note)}</small>' if note else ''
    return (f'<div class="vote {kind}"><span class="vote-label">{html.escape(label)}{extra}</span>'
            f'<span class="vote-track"><span class="vote-fill" style="width:{width}"></span></span>'
            f'<b>{_out_of_100(value)}</b></div>')


def _vote_card(title, color, cell, zero_note):
    return (f'<div class="vote-card" style="--pair:{color}"><h4>{html.escape(title)}</h4>'
            + _vote_row('Pixel witness · counts ×3', cell['pixel'], 'pixel', zero_note)
            + _vote_row('Look-alike judge · counts ×1', cell['look'], 'look')
            + _vote_row('Combined verdict', cell['blend'], 'total') + '</div>')


def _step(number, title, body, figure='', kind=''):
    return (f'<li class="story-step {kind}"><span class="step-no" aria-hidden="true">{number}</span>'
            f'<div class="step-body"><h3>{title}</h3>{body}{figure}</div></li>')


def _img(src, alt, wide=False):
    return (f'<a class="story-figure{" wide" if wide else ""}" href="assets/{src}" target="_blank" rel="noopener">'
            f'<img src="assets/{src}" alt="{html.escape(alt)}" loading="lazy"></a>')


def flow_html(ev, result):
    cells, spots, links = ev['cells'], ev['spots'], ev['links']
    right = sum(correct for _, _, correct in links)
    best, vit = result['methods'][METHOD], result['methods'][LOOK]
    counts = best['author_assignment']
    fmt = lambda n: f'{n:,}'
    pct = lambda v: f'{100*v:.2f}%'
    whole = lambda v: f'{round(100*v)}%'
    mug1, mug2 = MUGS[0][2], MUGS[1][2]
    glance = ''.join(f'<li><span>{i}</span>{t}</li>' for i, t in enumerate(
        ('Two photos', 'Two witnesses', 'Weigh the votes', 'Pair up', 'Check the answers'), 1))
    steps = [
        _step(1, 'Start with two photos of the same table',
              '<p>Nine cameras photograph the same table. We take two of those photos. A person has drawn a box '
              'around every object. <strong>The boxes say <em>where</em> things are, not <em>which</em> is which.</strong> '
              'The computer’s job is to work out which box on the left shows the same physical object as which box on the right.</p>'
              '<p class="step-aside">We follow two nearly identical blue mugs. They are hard to tell apart, which makes them a good test. '
              f'<strong>This is an easy scene where things went well</strong>: {ev["scene_outcome"]["IPAA100_pairs"]} of its '
              f'{ev["scene_outcome"]["image_pairs"]} photo pairs came out fully correct. Across the whole benchmark the figure is '
              f'{whole(ev["fully_correct"])}, and in the hardest scenes only {whole(ev["fully_correct_by_level"]["hard"])}.</p>',
              _img('flow-1-photos.jpg', 'Two camera views of the same table. Every object has a white box. Mug 1 is outlined in orange and Mug 2 in teal in both photos.', True)),
        _step(2, 'Ask two independent witnesses',
              '<p>Each witness looks at every possible pairing and says how convinced it is that the two boxes show the same object.</p>'
              '<div class="witnesses">'
              '<article class="witness pixel"><h4>The pixel witness</h4>'
              '<p><strong>First it lines up the two whole photos point by point</strong>, working out where each spot in one photo '
              'appears in the other. It keeps only points that survive a there-and-back check. '
              '<strong>Then it cuts each object out of its background</strong> and counts how many matched points land on '
              'that object in both photos.</p>'
              + _img('flow-2a-cutouts.png', 'Cut-outs of Mug 1 and Mug 2 from cameras 2 and 8. The object stays in colour; the background is faded.')
              + f'<ul class="facts"><li><b>{fmt(spots["00"]["forward"])}</b> matched points from Mug 1 land on Mug 1 '
              f'({fmt(spots["00"]["backward"])} going the other way)</li>'
              f'<li><b>{fmt(spots["01"]["forward"])}</b> land on Mug 2</li></ul>'
              f'<p class="step-aside">Those points cover only about {whole(cells["00"]["pixel_share"])} of the mug, '
              'yet that is already strong evidence, because most pairs have none.</p>'
              '<p class="tech">Technical names: SAM cuts out the object. RoMa finds matching pixels.</p></article>'
              '<article class="witness look"><h4>The look-alike judge</h4>'
              '<p><strong>It looks at the object plus a generous border around it</strong>, and notes which camera took the photo '
              'and where in the frame the object sits. '
              'It sums up what it sees as a <strong>fingerprint of 192 numbers</strong>, drawn below as coloured stripes. '
              'The top row is the reference, sorted to fade from blue to green, and the other rows use the same order. '
              '<strong>The grey strip under each row shows where it differs from the reference: '
              'the darker the strip, the bigger the difference.</strong></p>'
              + _img('flow-2b-fingerprints.png', 'Context photos and 192-number fingerprints for Mug 1 in camera 2, Mug 1 in camera 8 and Mug 2 in camera 8.')
              + f'<ul class="facts"><li>Mug 1 vs Mug 1: total difference <b>{cells["00"]["look_distance"]:.1f}</b>, '
              f'closer than {whole(cells["00"]["look"])} of all pairs</li>'
              f'<li>Mug 1 vs Mug 2: total difference <b>{cells["01"]["look_distance"]:.1f}</b>, still closer than '
              f'{whole(cells["01"]["look"])}. The twins fool it a little.</li></ul>'
              '<p class="tech">Technical name: ViT-P3DE, a pretrained vision transformer.</p></article></div>', kind='split'),
        _step(3, 'Weigh the votes',
              '<p>The witnesses speak in different units, so each opinion becomes a ranking: '
              f'<em>how does this pair compare with all {fmt(ev["candidates"])} possible pairs in the benchmark?</em> '
              'A score of 97 means “more convincing than about 97 out of 100 pairs”. Pairs that tie share the middle of their tied group. '
              '<strong>The pixel witness then counts three times, and the look-alike judge once.</strong></p>'
              '<div class="vote-cards">'
              + _vote_card('Mug 1 ↔ Mug 1 · the right pair', mug1, cells['00'], '')
              + _vote_card('Mug 1 ↔ Mug 2 · the wrong pair', mug2, cells['01'], 'no matched points')
              + '</div>'
              f'<p class="step-aside">About {whole(ev["zero_share"])} of all possible pairs have no matched points, or too few to count. '
              'They all tie, so each sits in the middle of that group, which is why the wrong pair still gets a pixel score near 50. '
              'The combined verdict is worked out from the unrounded scores.</p>'),
        _step(4, 'Pair up, one partner each',
              '<p>Now the computer chooses the set of pairings with the strongest combined verdicts. Two rules apply. '
              f'<strong>A pairing scoring below about {ev["cutoff"]*100:.0f} is refused outright</strong>, so an object hidden in the '
              '<em>other</em> photo can stay unmatched. <strong>Each object can have at most one partner.</strong></p>'
              f'<p>The wrong mug pair scored {_out_of_100(cells["01"]["blend"])}, just above the bar, so it <em>could</em> have been chosen. '
              f'It lost because Mug 1 already had a much stronger partner ({_out_of_100(cells["00"]["blend"])}). '
              f'In this photo pair {ev["passing_in_pair"]} pairings cleared the bar, and the one-partner rule picked {len(links)} of them.</p>',
              _img('flow-4-pairs.jpg', f'The two photos with lines joining each object to its chosen partner. All {len(links)} pairings are correct; the mug lines are orange and teal.', True)
              + '<p class="legend-line"><span class="key ok"></span>Green line: correct pairing. '
              '<span class="key bad"></span>Red line: a mistake (none here). Orange and teal: the two mugs.</p>'
              + f'<p class="step-aside">Here, all <strong>{right} of {ev["objects"]}</strong> objects were paired with the right partner. '
              'The pairing was done by the original authors’ own pairing program, fed with our combined scores.</p>'),
        _step(5, 'Only now, check the answer key',
              '<p>Within each run, the human identity labels are used only at the very end. Repeating these steps on every pair of photos in the benchmark gives:</p>'
              '<div class="tiles">'
              f'<div class="tile"><b>{fmt(counts["correct"])}</b><span>of {fmt(best["positive_object_pairs"])} true links found ({pct(counts["recall"])})</span></div>'
              f'<div class="tile"><b>{fmt(counts["wrong"])}</b><span>wrong links, out of {fmt(counts["proposals"])} proposed</span></div>'
              f'<div class="tile"><b>{100*best["AP"]:.1f}</b><span>ranking quality out of 100: how well true pairs rise above false ones, '
              f'before pairing up. The look-alike judge alone scores {100*vit["AP"]:.1f}. (Technical name: average precision.)</span></div>'
              '</div>'
              '<p class="step-aside">A caution: we chose this setup after comparing 45 options on these same photos, '
              'so these numbers may be a little optimistic for new photos.</p>', kind='final'),
    ]
    return (f'<div class="story"><ol class="glance" aria-label="The five steps">{glance}</ol>'
            f'<ol class="story-steps">{"".join(steps)}</ol>'
            f'<p class="muted">Example: scene {SCENE}, cameras {CAM_A} and {CAM_B}, chosen after scoring to explain the method. '
            'Pictures and numbers come from the saved results; no model was rerun.</p></div>')
