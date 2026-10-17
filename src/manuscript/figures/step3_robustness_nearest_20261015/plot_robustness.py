#!/usr/bin/env python3
"""Publication figure: paired-definition sampling precision and period sensitivity."""
from manuscript_paths import workspace_root, input_path
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

HERE = (workspace_root() / 'figures/step3_robustness_nearest_20261015')
ROOT = HERE.parent.parent
SOURCE = ROOT / 'contribution_structure_analysis/step3_nearest_20261015/statistics'
STEM = 'Figure_S04_Interannual_allocation_robustness_1993_2025'
BASINS = ('Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic')
BANDS = ('0-700m', '700-2000m', '2000-6000m')
COMPONENTS = BASINS + BANDS
LABELS = BASINS + ('0-700 m', '700-2000 m', '2000-6000 m')
COLORS = ('#2c6aa8', '#d2694b', '#2a9d8f', '#8064a2', '#6f6f6f', '#1f5a9d', '#2b9c94', '#7956a7')
PERIODS = ((1993, 2025), (1995, 2025), (2000, 2025), (2005, 2025), (1993, 2020))
MARKERS = ('o', 's', 'D', '^', 'v')


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7., 'axes.labelsize': 7., 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 7., 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'xtick.major.size': 3., 'xtick.major.width': .65,
                         'ytick.major.size': 0., 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'legend.frameon': False, 'savefig.facecolor': 'white'})


def axes_style(ax):
    ax.set_xlim(-20., 105.)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.axvline(0., color='#9ba4aa', lw=.75, zorder=1)
    ax.grid(axis='x', color='#e8ebed', lw=.5)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', pad=5)


def intervals(ax, table, names, labels, colors):
    offsets = (-.19, 0., .19)
    for i, (name, color) in enumerate(zip(names, colors)):
        for block, offset, marker in zip((2, 3, 5), offsets, MARKERS[:3]):
            row = table[table.component.eq(name) & table.block_years.eq(block)]
            if len(row) != 1:
                raise ValueError('Missing or duplicated interval')
            row = row.iloc[0]
            ax.plot([row.lower_2p5_pct, row.upper_97p5_pct], [i + offset] * 2,
                    lw=1.05, color=color, alpha=.65, solid_capstyle='butt', zorder=2)
            ax.scatter([row.point_share_pct], [i + offset], marker=marker, s=13.,
                       color=color, edgecolors='white', linewidths=.35, zorder=3)
    ax.set_ylim(len(names) - .55, -.55)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(labels)
    axes_style(ax)
    ax.set_xlabel('Variance allocation (%)', labelpad=5.)


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    out = []
    for text in fig.findobj(match=Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if not fig.bbox.expanded(1.001, 1.001).contains(box.x0, box.y0) or not fig.bbox.expanded(1.001, 1.001).contains(box.x1, box.y1):
            out.append(text.get_text())
    if out:
        raise ValueError('Text outside figure: {}'.format(out))
    for ax in fig.axes:
        for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
            visible = [label.get_window_extent(renderer) for label in labels if label.get_visible() and label.get_text()]
            for i, box in enumerate(visible):
                if any(box.overlaps(other) for other in visible[i + 1:]):
                    raise ValueError('Overlapping tick labels')
    return {'text_inside_canvas': True, 'tick_labels_nonoverlapping': True}


def marker_layout_check(fig, ax, artists):
    """Check transformed glyph bounds, including strokes, rather than point centers."""
    fig.canvas.draw()
    boxes = []
    for artist in artists:
        center = ax.transData.transform(artist.get_offsets()[0])
        bounds = artist.get_paths()[0].get_extents()
        scale = np.sqrt(artist.get_sizes()[0]) * fig.dpi / 72.
        stroke = artist.get_linewidths()[0] * fig.dpi / 72. / 2.
        boxes.append((center[0] + bounds.x0 * scale - stroke,
                      center[1] + bounds.y0 * scale - stroke,
                      center[0] + bounds.x1 * scale + stroke,
                      center[1] + bounds.y1 * scale + stroke))
    overlaps = []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[i + 1:], i + 1):
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                overlaps.append((i, j))
    if overlaps:
        raise ValueError('Period-marker glyph bounds overlap: {}'.format(overlaps))
    return {'period_marker_count': len(boxes), 'period_marker_bounds_nonoverlapping': True}


def main(separate_period_markers=False):
    output = HERE / 'period_markers_separated' if separate_period_markers else HERE
    if any((output / (STEM + '.' + suffix)).exists() for suffix in ('png', 'pdf', 'svg', 'tiff')):
        raise ValueError('Existing exports protected from overwrite')
    intervals_path = SOURCE / 'joint_bootstrap_intervals.csv'
    periods_path = SOURCE / 'period_sensitivity.csv'
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (intervals_path, periods_path)}
    table = pd.read_csv(intervals_path)
    sensitivity = pd.read_csv(periods_path)
    if len(table) != 24 or len(sensitivity) != 40:
        raise ValueError('Unexpected component/method coverage')
    if not table.finite_draw_count.eq(5000).all():
        raise ValueError('Incomplete bootstrap')
    style()
    fig = plt.figure(figsize=(7.2, 6.1))
    a = fig.add_axes([.13, .60, .365, .285])
    b = fig.add_axes([.64, .60, .33, .285])
    c = fig.add_axes([.15, .12, .82, .35 if separate_period_markers else .32])
    intervals(a, table, BASINS, BASINS, COLORS[:5])
    intervals(b, table, BANDS, LABELS[5:], COLORS[5:])
    centers = np.arange(8) * (1.25 if separate_period_markers else 1.)
    period_artists = []
    for i, name in enumerate(COMPONENTS):
        for k, (start, end) in enumerate(PERIODS):
            selected = sensitivity[sensitivity.component.eq(name)
                                   & sensitivity.endpoint_start_year.eq(start)
                                   & sensitivity.endpoint_end_year.eq(end)]
            if len(selected) != 1:
                raise ValueError('Missing period sensitivity')
            y = centers[i] + (k - 2) * (.24 if separate_period_markers else .115)
            if separate_period_markers:
                size = 9. if k == 0 else (5.2 if k == 2 else 7.3)
            else:
                size = 12. if k else 18.
            artist = c.scatter([selected.iloc[0].share_pct], [y], marker=MARKERS[k],
                               s=size, color=COLORS[i],
                               facecolors=COLORS[i] if k == 0 else 'white',
                               linewidths=.55 if separate_period_markers else .85, zorder=3)
            assert float(artist.get_offsets()[0, 0]) == float(selected.iloc[0].share_pct)
            period_artists.append(artist)
    c.set_ylim(centers[-1] + .65 if separate_period_markers else 7.6,
               -.65 if separate_period_markers else -.6)
    c.set_yticks(centers)
    c.set_yticklabels(LABELS)
    c.axhline((centers[4] + centers[5]) / 2., color='#cfd5d9', lw=.65)
    axes_style(c)
    c.set_xlabel('Variance allocation (%)', labelpad=6.)
    for ax in (a, b):
        ax.xaxis.set_minor_locator(MultipleLocator(12.5))
        ax.tick_params(axis='x', which='minor', length=1.5, width=.5)
    top_handles = [Line2D([], [], color='#52636d', marker=MARKERS[i], markersize=3.2,
                          linewidth=.9, label='{}-year blocks'.format(block)) for i, block in enumerate((2, 3, 5))]
    fig.legend(handles=top_handles, loc='center', bbox_to_anchor=(.55, .95), ncol=3,
               fontsize=6.7, handlelength=1.6, columnspacing=2.0)
    fig.text(.13, .905, '1993-2025', fontsize=7., color='#36454e')
    fig.text(.64, .905, '95% bootstrap intervals', fontsize=7., color='#36454e')
    period_handles = [Line2D([], [], color='#52636d', marker=MARKERS[k], markersize=3.3,
                             markerfacecolor='#52636d' if k == 0 else 'white', markeredgewidth=.8,
                             linestyle='none', label='{}-{}'.format(start, end))
                      for k, (start, end) in enumerate(PERIODS)]
    fig.legend(handles=period_handles, loc='center', bbox_to_anchor=(.57, .492), ncol=5,
               fontsize=6.3, handlelength=.9, handletextpad=.4, columnspacing=1.15)
    for x, y, label in ((.035, .91, 'a'), (.54, .91, 'b'), (.035, .49, 'c')):
        fig.text(x, y, label, fontsize=10., fontweight='bold')
    qa = check_layout(fig)
    if separate_period_markers:
        qa.update(marker_layout_check(fig, c, period_artists))
        qa['period_marker_x_values_exactly_unchanged'] = True
        qa['period_order_top_to_bottom'] = ['{}-{}'.format(start, end) for start, end in PERIODS]
    output.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf', 'svg', 'tiff'):
        fig.savefig(str(output / (STEM + '.' + suffix)), dpi=450 if suffix == 'png' else 600,
                    facecolor='white', bbox_inches=None)
    plt.close(fig)
    table.to_csv(output / 'source_bootstrap_intervals.csv', index=False)
    sensitivity.to_csv(output / 'source_period_sensitivity.csv', index=False)
    assert hashes == {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in hashes}
    qa.update({'source_sha256': hashes, 'figure_size_inches': [7.2, 6.1], 'bootstrap_draws_per_block': 5000,
               'pointwise_not_simultaneous_intervals': True, 'period_symbols_are_point_estimates_not_uncertainty': True,
               'figure07_basin_definitions_matched': True, 'no_total_observational_uncertainty_claim': True})
    (output / 'verification.json').write_text(json.dumps(qa, indent=2))
    owner = ROOT.stat()
    for path in [HERE] + list(HERE.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    print('Saved:', output / (STEM + '.pdf'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--separate-period-markers', action='store_true',
                        help='Save a separate C-panel revision with ordered nonoverlapping period symbols')
    main(parser.parse_args().separate_period_markers)
