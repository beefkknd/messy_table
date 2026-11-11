"""Build Part One from preserved scene photographs and frozen matching results."""
from pathlib import Path
import hashlib
import html
import json
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paths import ASSETS, PART1, PART_ONE_RESULTS, SAMPLE, SITE, STEP2, sources
from scatter_charts import save_chart, tradeoff_chart
import flow_story

SRC = sources()  # verifies every large external input by sha256 before anything is drawn
PAGE = SITE / 'index.html'

EXAMPLES = [
    {
        'slug': 'mugs', 'scene': '20190921-00003-02', 'cameras': [2, 8],
        'title': 'Two similar mugs, two identities',
        'question': 'Which blue mug in the left photo is the same physical mug in the right photo?',
        'explanation': 'The orange rectangle marks mug 1 in both views. The cyan rectangle marks mug 2. Their openings, handles and visible surfaces change with the camera angle.',
        'items': [
            {'id': '1', 'name': 'Mug 1', 'color': '#b85300'},
            {'id': '2', 'name': 'Mug 2', 'color': '#007989'},
        ],
    },
    {
        'slug': 'viewpoint', 'scene': '20190923-00015-06', 'cameras': [1, 7],
        'title': 'Viewpoint and occlusion change the evidence',
        'question': 'Can we preserve identity when only part of an object is visible?',
        'explanation': 'Banana 24 lies beside the green pepper in the overhead view. From the second camera, other items cover much of it. Banana 5 remains a separate physical object across the two views.',
        'items': [
            {'id': '5', 'name': 'Banana 5', 'color': '#b85300'},
            {'id': '24', 'name': 'Banana 24', 'color': '#007989'},
        ],
    },
    {
        'slug': 'unmatched', 'scene': '20190924-00032-02', 'cameras': [1, 7],
        'title': 'Some objects should remain unmatched',
        'question': 'Which items have a counterpart, and which have no annotated visible counterpart?',
        'explanation': 'Bags 13 and 6 appear in both views. Bottle 11 has an annotation in camera 1 and no corresponding annotation in camera 7. The output needs to allow an unmatched item when the evidence does not provide a visible counterpart.',
        'items': [
            {'id': '13', 'name': 'Bag 13', 'color': '#b85300'},
            {'id': '6', 'name': 'Bag 6', 'color': '#007989'},
            {'id': '11', 'name': 'Bottle 11 · only annotated in camera 1', 'color': '#8744a4'},
        ],
    },
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def percent(value):
    return f'{100*value:.2f}%'


def plot_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
        'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#bacace', 'text.color': '#193039', 'axes.labelcolor': '#193039',
        'xtick.color': '#526970', 'ytick.color': '#193039',
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def baseline_chart(result):
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    published = [52.4, 82.8]
    sampled = [100*result['methods'][name]['AP'] for name in ('asnet_reference', 'vit_reference')]
    ax.plot([0, 100], [0, 100], '--', color='#a3b8bf', linewidth=1, label='Equal AP on the two populations')
    for name, x, y, color in zip(['ASNet', 'ViT-P3DE Tiny'], published, sampled, ['#879a9d', '#42679a'], strict=True):
        ax.scatter(x, y, s=115, color=color, edgecolors='white', zorder=3)
        ax.annotate(f'{name}\nPublished {x:.2f}% · sample {y:.2f}%', (x, y),
                    xytext=(-130, 23), textcoords='offset points', fontsize=10, color=color,
                    arrowprops={'arrowstyle': '-', 'color': color})
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel='Published AP · full official test',
           ylabel='Our reproduced AP · fixed 100-scene sample')
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_title('Reference scores · proximity to the diagonal shows agreement', loc='left', pad=18)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    fig.text(.5, .02, 'The populations differ. Agreement on a sample does not verify the complete published test claim.',
             ha='center', fontsize=9, color='#526970')
    fig.tight_layout(rect=(0, .04, 1, 1), pad=1.4)
    save_chart(fig, ASSETS/'reference-ap.png')


def result_chart(result):
    names = ['asnet_reference', 'vit_reference', *result['top_three_own_variants']]
    labels = ['ASNet', 'ViT-P3DE Tiny', 'RoMa+SAM / ViT · 75/25',
              'RoMa+SAM / ViT · 50/50', 'RoMa+SAM / ASNet · 75/25']
    colors = ['#879a9d', '#42679a', '#107d71', '#45a79d', '#bb8736']
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    valid = {name: r for name, r in result['methods'].items() if r['FPR95'] is not None}
    other = [r for name, r in valid.items() if name not in names]
    ax.scatter([100*r['FPR95'] for r in other], [100*r['AP'] for r in other],
               s=38, color='#bdcbd0', alpha=.75, label=f'Other {len(other)} configurations')
    for name, label, color in zip(names, labels, colors, strict=True):
        r = result['methods'][name]
        ax.scatter(100*r['FPR95'], 100*r['AP'], s=105, color=color,
                   edgecolors='white', linewidths=.9, zorder=4, label=label)
    maximum_fpr = max(100*r['FPR95'] for r in valid.values())
    ax.set(xlim=(0, min(100, maximum_fpr*1.08)), ylim=(0, 100), xlabel='FPR95 · wrong candidate pairs accepted at 95% recall · fewer ←',
           ylabel='Average precision · higher ↑')
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_title('Matching quality · better toward the upper left ↖', loc='left', pad=16)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    zoom = ax.inset_axes([.61, .55, .35, .35])
    for name, color in zip(names[2:4], colors[2:4], strict=True):
        r = result['methods'][name]
        zoom.scatter(100*r['FPR95'], 100*r['AP'], s=65, color=color, edgecolors='white', zorder=3)
    zoom.set(xlim=(2.65, 3.25), ylim=(90.48, 90.73))
    zoom.set_title('Leading blends · enlarged', fontsize=10)
    zoom.set_xlabel('FPR95 (%)', fontsize=9)
    zoom.set_ylabel('AP (%)', fontsize=9)
    zoom.tick_params(labelsize=9)
    zoom.grid(alpha=.18)
    ax.legend(loc='lower left', ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .025, f'Same 100 scenes / 3,600 view pairs · {len(valid)} plotted · '
             f'{len(result["methods"])-len(valid)} omitted for ambiguous cutoff ties · human boxes',
             ha='center', fontsize=9, color='#526970')
    fig.tight_layout(rect=(0, .04, 1, 1), pad=1.4)
    save_chart(fig, ASSETS/'matching-comparison.png')
    save(ASSETS/'matching-comparison.json', {
        'axes': {'x': 'FPR95 percent', 'y': 'AP percent'},
        'plotted': [{'method': name, 'x': 100*r['FPR95'], 'y': 100*r['AP'],
                     'fpr95_status': r['FPR95_status']} for name, r in valid.items()],
        'omitted': [name for name, r in result['methods'].items() if r['FPR95'] is None],
        'audit': '../../results/part-one/fpr95-tie-audit.json',
    })


def link_chart(result):
    names = ['vit_reference', result['top_three_own_variants'][0]]
    labels = ['ViT-P3DE Tiny', 'RoMa+SAM / ViT · 75/25']
    def point(name, label, color):
        a = result['methods'][name]['author_assignment']
        return {'label': label, 'good': a['correct'], 'bad': a['wrong']+a['unmapped'],
                'target': result['methods'][name]['positive_object_pairs'], 'color': color}
    points = [point(name, label, color) for name, label, color in
              zip(names, labels, ['#42679a', '#107d71'], strict=True)]
    background = [point(name, r['label'], '#bdcbd0') for name, r in result['methods'].items() if name not in names]
    tradeoff_chart(points, ASSETS/'correct-and-wrong.png',
                   'Actual identity assignments · 78,335 true links / 3,600 view pairs',
                   'Correct identity links', 'Wrong identity links', background=background,
                   pr_offsets=[(-145, -40), (-175, 20)], count_offsets=[(30, -35), (60, 10)])


def examples_html(labels, selection):
    included = {row['scene'] for row in selection['scenes']}
    output = []
    evidence = []
    for example in EXAMPLES:
        assert example['scene'] in included
        scene = labels[example['scene']]
        left_ids = scene['cameras'][str(example['cameras'][0])]['instances']
        right_ids = scene['cameras'][str(example['cameras'][1])]['instances']
        if example['slug'] == 'unmatched':
            assert '11' in left_ids and '11' not in right_ids
        else:
            assert all(item['id'] in left_ids and item['id'] in right_ids for item in example['items'])
        output.append(f'<article class="example"><h3>{html.escape(example["title"])}</h3>')
        output.append('<p>'+html.escape(example['question'])+'</p><div class="scene-pair">')
        record = {'scene': example['scene'], 'cameras': example['cameras'], 'items': example['items'], 'views': []}
        for camera in example['cameras']:
            source = SRC.images/f'{example["scene"]}-0{camera}.jpg'
            destination = ASSETS/source.name
            shutil.copy2(source, destination)
            boxes = []
            for item in example['items']:
                entry = scene['cameras'][str(camera)]['instances'].get(item['id'])
                if entry is None:
                    continue
                x1, y1, x2, y2 = entry['pos']
                style = (f'left:{100*x1/1920:.6f}%;top:{100*y1/1080:.6f}%;'
                         f'width:{100*(x2-x1)/1920:.6f}%;height:{100*(y2-y1)/1080:.6f}%;'
                         f'--object-color:{item["color"]};')
                boxes.append(f'<span class="identity-box" style="{style}"><b>{html.escape(item["id"])}</b></span>')
            output.append(
                f'<figure><a class="scene" href="assets/{source.name}" target="_blank" rel="noopener">'
                f'<img src="assets/{source.name}" alt="{html.escape(example["title"])} in camera {camera}">'
                + ''.join(boxes) + f'</a><figcaption>Camera {camera} · full scene</figcaption></figure>')
            record['views'].append({'camera': camera, 'source': str(source.relative_to(SRC.workspace)), 'sha256': sha(source)})
        output.append('</div><div class="identity-legend">')
        for item in example['items']:
            output.append(f'<span><i style="background:{item["color"]}"></i>{html.escape(item["name"])}</span>')
        output.append('</div><p class="example-note">'+html.escape(example['explanation'])+'</p></article>')
        evidence.append(record)
    return '\n'.join(output), evidence


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    result = read(STEP2/'results.json')
    tie_audit = read(PART_ONE_RESULTS/'fpr95-tie-audit.json')
    assert set(tie_audit['invalid_fpr95_methods']) == {'roma_box', 'roma_mask', 'roma_box_mean', 'roma_mask_mean'}
    assert tie_audit['source_results_sha256'] == sha(STEP2/'results.json')
    assert tie_audit['source_score_freeze_sha256'] == sha(SRC.score_freeze)
    for name, row in result['methods'].items():
        audited = tie_audit['methods'][name]
        assert row['FPR95'] == audited['author_reproduced_fpr95']
        row['FPR95_author_reproduction'] = row['FPR95']
        row['FPR95'] = audited['report_fpr95']
        row['FPR95_status'] = audited['status']
    result['FPR95_display_policy'] = tie_audit['display_policy']
    result['FPR95_na_methods'] = tie_audit['invalid_fpr95_methods']
    selection = read(SAMPLE/'selection.json')
    labels = read(SRC.labels)['scenes']
    assert result['selection_sha256'] == sha(SAMPLE/'selection.json')
    for name in result['methods']:
        assert result['methods'][name]['candidate_object_pairs'] == 2484124
    plot_style()
    baseline_chart(result)
    result_chart(result)
    link_chart(result)
    story = flow_story.evidence(SRC, labels, result)
    flow_story.draw_all(story, SRC, labels, ASSETS)
    examples, evidence = examples_html(labels, selection)
    template = (HERE/'template.html').read_text()
    assert template.count('%%EXAMPLES%%') == template.count('%%DATA%%') == template.count('%%FLOW%%') == 1
    payload = json.dumps(result, separators=(',', ':')).replace('<', '\\u003c')
    page = template.replace('%%FLOW%%', flow_story.flow_html(story, result)).replace('%%EXAMPLES%%', examples).replace('%%DATA%%', payload)
    PAGE.write_text(page)
    save(SITE/'provenance.json', {
        'source_results_sha256': sha(STEP2/'results.json'),
        'selection_sha256': sha(SAMPLE/'selection.json'),
        'scenes': 100, 'view_pairs': 3600, 'human_boxes': True, 'new_inference': False,
        'example_annotations': 'Human identity labels, chosen after scoring to explain the task.',
        'examples': evidence, 'build_script_sha256': sha(Path(__file__)),
        'scatter_helper_sha256': sha(HERE/'scatter_charts.py'),
        'template_sha256': sha(HERE/'template.html'),
        'paths_module_sha256': sha(HERE/'paths.py'),
        'external_inputs_sha256': sha(HERE/'external-inputs.json'),
        'flow_story': {'scene': flow_story.SCENE, 'cameras': [flow_story.CAM_A, flow_story.CAM_B],
                       'module_sha256': sha(HERE/'flow_story.py'),
                       'native_links_sha256': sha(SRC.native_links),
                       'native_links_script_sha256': sha(PART1/'tools/flow_native_links.py'),
                       'chosen_after_scoring': True},
        'fpr95_audit_sha256': sha(PART_ONE_RESULTS/'fpr95-tie-audit.json'),
        'fpr95_omitted_from_chart': tie_audit['invalid_fpr95_methods'],
        'assets_sha256': {path.name: sha(path) for path in ASSETS.iterdir() if path.is_file()},
        'html_sha256': sha(PAGE),
    })
    print('Saved', PAGE)


if __name__ == '__main__':
    main()
