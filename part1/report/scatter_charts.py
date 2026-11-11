"""Shared x/y comparison plots for the two local research reports."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter


def number_ticks(value, _):
    return f'{value/1000:g}k' if abs(value) >= 1000 else f'{value:g}'


def save_chart(fig, path):
    path = Path(path)
    fig.savefig(path, dpi=175)
    fig.savefig(path.with_suffix('.svg'))
    plt.close(fig)


def frontier(points):
    """Return non-dominated precision/recall points, without fitting anything."""
    values = [(100*p['good']/p['target'], 100*p['good']/(p['good']+p['bad'])) for p in points]
    return sorted(set(p for p in values if not any(
        q[0] >= p[0] and q[1] >= p[1] and q != p for q in values)))


def tradeoff_chart(points, path, title, good_name, bad_name, background=(),
                   pr_offsets=None, count_offsets=None):
    """One common-population scatter in ratios and another in absolute counts."""
    all_points = [*background, *points]
    assert all(p['target'] == all_points[0]['target'] for p in all_points)
    target = all_points[0]['target']
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5))
    precision, counts = axes
    defaults_pr = [(0, -38), (-100, 27), (-125, -45), (-335, 50), (-140, 50)]
    defaults_count = [(12, -7), (-100, -35), (-140, 23), (15, -16), (18, -48)]
    pr_offsets = pr_offsets or defaults_pr
    count_offsets = count_offsets or defaults_count
    for p in background:
        precision.scatter(100*p['good']/target, 100*p['good']/(p['good']+p['bad']),
                          s=24, color='#bac9ce', alpha=.65, zorder=1)
        counts.scatter(p['bad'], p['good'], s=24, color='#bac9ce', alpha=.65, zorder=1)
    edge = frontier(all_points)
    if len(edge) > 1:
        precision.plot(*zip(*edge), color='#91a8ae', linestyle='--', linewidth=1,
                       label='Best observed tradeoffs', zorder=1)
    for i, p in enumerate(points):
        x = 100*p['good']/target
        y = 100*p['good']/(p['good']+p['bad'])
        color = p['color']
        text_color = {'#879a9d': '#526970', '#b78937': '#93631b'}.get(color, color)
        for ax, xy, offset, detail in [
            (precision, (x, y), pr_offsets[i], f'{x:.2f}% recall · {y:.2f}% precision'),
            (counts, (p['bad'], p['good']), count_offsets[i], f'{p["good"]:,} / {p["bad"]:,}'),
        ]:
            ax.scatter(*xy, s=88, color=color, edgecolors='white', linewidths=1, zorder=4)
            ax.annotate(p['label']+'\n'+detail, xy=xy, xytext=offset, textcoords='offset points',
                        fontsize=9, color=text_color, ha='left', va='center', zorder=5,
                        bbox={'facecolor': 'white', 'alpha': .94, 'edgecolor': 'none', 'pad': 2},
                        arrowprops={'arrowstyle': '-', 'color': color, 'linewidth': .8},
                        annotation_clip=False)
    precision.set(xlim=(0, 103), ylim=(0, 106),
                  xlabel='Recall · fraction of true targets found →',
                  ylabel='Precision · fraction of output that is correct ↑')
    precision.set_xticks(range(0, 101, 20))
    precision.set_yticks(range(0, 101, 20))
    precision.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    precision.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    precision.set_title('Ratios · better toward the upper right ↗', loc='left', pad=16, fontsize=11)
    counts.set_xlim(-max(p['bad'] for p in all_points)*.04,
                    max(p['bad'] for p in all_points)*1.24)
    counts.set_ylim(0, target*1.18)
    counts.axhline(target, linestyle=':', linewidth=1.2, color='#91a8ae')
    counts.text(.02, target*1.035, f'All {target:,} true targets',
                transform=counts.get_yaxis_transform(), color='#526970', fontsize=9)
    counts.set(xlabel=bad_name+' · fewer ←', ylabel=good_name+' · more ↑')
    counts.set_title('Counts · better toward the upper left ↖', loc='left', pad=16, fontsize=11)
    counts.xaxis.set_major_formatter(FuncFormatter(number_ticks))
    counts.yaxis.set_major_formatter(FuncFormatter(number_ticks))
    for ax in axes:
        ax.grid(alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=12, y=.99)
    note = ('One point = one frozen configuration. Dashed line = observed tradeoffs; no single combined score. '
            'Gray points = other tested configurations.' if background else
            'One point = one frozen configuration. Dashed line = observed tradeoffs; no single combined score.')
    fig.text(.5, .025, note, ha='center', fontsize=9, color='#526970')
    fig.subplots_adjust(left=.07, right=.96, bottom=.16, top=.85, wspace=.25)
    Path(path).with_suffix('.json').write_text(json.dumps({
        'title': title, 'common_target': target,
        'ratio_axes': {'x': '100 * good / target', 'y': '100 * good / (good + bad)'},
        'count_axes': {'x': bad_name, 'y': good_name},
        'highlighted': [{**p, 'recall_percent': 100*p['good']/target,
                         'precision_percent': 100*p['good']/(p['good']+p['bad'])} for p in points],
        'background': [{**p, 'recall_percent': 100*p['good']/target,
                        'precision_percent': 100*p['good']/(p['good']+p['bad'])} for p in background],
        'observed_frontier': edge,
    }, indent=2)+'\n')
    save_chart(fig, path)
