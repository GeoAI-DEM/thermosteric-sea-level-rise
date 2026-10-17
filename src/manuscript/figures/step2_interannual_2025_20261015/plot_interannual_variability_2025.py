#!/usr/bin/env python3
"""Plot archived interannual variability through 2025 without changing prior figures."""
from manuscript_paths import workspace_root, input_path
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = (workspace_root() / 'figures/step2_interannual_2025_20261015')
ROOT = HERE.parent.parent
INPUT = ROOT / 'contribution_structure_analysis/step2_interannual_20261014'
PERIOD = 'recent_extended'
BASINS = ('Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic', 'Unassigned')
BANDS = ('0-700m', '700-2000m', '2000-6000m')
BASIN_COLORS = ('#2c6aa8', '#d2694b', '#2a9d8f', '#8064a2', '#6f6f6f', '#acb2b7')
DEPTH_COLORS = ('#1f5a9d', '#2b9c94', '#7956a7')
OFFSET, REINFORCE, RESIDUAL = '#507b91', '#c97662', '#929ba2'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_data():
    files = {name: INPUT / (name + '.csv') for name in
             ('variability_contributions', 'component_covariance', 'decomposition_summary', 'detrended_annual_series')}
    frames = {name: pd.read_csv(path) for name, path in files.items()}
    selected = {name: frame[frame.period == PERIOD].copy() for name, frame in frames.items()}
    for frame in selected.values():
        if frame.empty or not (frame.endpoint_start_year.eq(1993).all()
                               and frame.endpoint_end_year.eq(2025).all()):
            raise ValueError('Expected the archived 1993-2025 endpoint period')
    contributions = selected['variability_contributions']
    basin = contributions[contributions.decomposition == 'global_by_basin'].set_index('region').loc[list(BASINS)].reset_index()
    depth = contributions[contributions.decomposition == 'global_by_depth'].set_index('depth_band').loc[list(BANDS)].reset_index()
    pairs = selected['component_covariance']
    pairs = pairs[(pairs.decomposition == 'global_by_basin') & (~pairs.is_diagonal)].copy()
    pairs['basin_a'] = pairs.component_a.str.split(r'\|').str[1]
    pairs['basin_b'] = pairs.component_b.str.split(r'\|').str[1]
    summary = selected['decomposition_summary']
    summary = summary[summary.decomposition == 'global_by_basin'].iloc[0]
    residuals = selected['detrended_annual_series']

    def residual(kind, region, band):
        rows = residuals[(residuals.component_type == kind) & (residuals.region == region)
                         & (residuals.depth_band == band)].sort_values('target_year')
        if not np.array_equal(rows.target_year.to_numpy(), np.arange(1994, 2026)):
            raise ValueError('Missing/duplicate annual residuals, including 2024 or 2025')
        result = rows.detrended_annual_change_mm.to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite residuals')
        return result

    global_residual = residual('global', 'Global', '0-6000m')
    variance = np.var(global_residual, ddof=1)
    np.testing.assert_allclose(variance, summary.reference_detrended_variance_mm2, rtol=1e-12)
    for frame in (basin, depth):
        np.testing.assert_allclose(frame.variance_contribution_pct.sum(), 100., atol=1e-10)
        for _, row in frame.iterrows():
            values = residual(row.component_type, row.region, row.depth_band)
            covariance = np.cov(values, global_residual, ddof=1)[0, 1]
            np.testing.assert_allclose(covariance / variance * 100., row.variance_contribution_pct, atol=1e-10)
    for _, row in pairs.iterrows():
        covariance = np.cov(residual('basin', row.basin_a, '0-6000m'),
                            residual('basin', row.basin_b, '0-6000m'), ddof=1)[0, 1]
        np.testing.assert_allclose(covariance, row.covariance_mm2, atol=1e-12)
        np.testing.assert_allclose(2. * covariance / variance * 100.,
                                   row.term_contribution_to_reference_variance_pct, atol=1e-10)
    np.testing.assert_allclose(2. * pairs.covariance_mm2.sum(), summary.twice_sum_cross_covariances_mm2,
                               atol=1e-11)
    return files, frames, basin, depth, pairs, summary, residuals


def pair_display(pairs):
    unassigned = pairs.basin_a.eq('Unassigned') | pairs.basin_b.eq('Unassigned')
    main = pairs[~unassigned].copy()
    if len(main) != 10 or unassigned.sum() != 5:
        raise ValueError('Need all ten five-basin pairs and five residual pairs')
    main = main.assign(magnitude=main.term_contribution_to_reference_variance_pct.abs())
    main = main.sort_values('magnitude', ascending=False)
    rows = [{'label': '{} $\\times$ {}'.format(row.basin_a, row.basin_b),
             'covariance_term_pct': float(row.term_contribution_to_reference_variance_pct),
             'pair_count': 1, 'includes_unassigned': False} for _, row in main.iterrows()]
    rows.append({'label': 'Unassigned interactions',
                 'covariance_term_pct': float(pairs[unassigned].term_contribution_to_reference_variance_pct.sum()),
                 'pair_count': 5, 'includes_unassigned': True})
    return pd.DataFrame(rows)


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 7., 'axes.labelsize': 6.7, 'xtick.labelsize': 6.3,
                         'ytick.labelsize': 6.6, 'axes.linewidth': .65,
                         'xtick.major.width': .65, 'ytick.major.width': .65,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def contribution_axis(axis, labels, values, colors):
    positions = np.arange(len(values))
    axis.barh(positions, values, height=.52, color=colors, linewidth=0, zorder=3)
    axis.scatter(values, positions, s=9, c=colors, edgecolors='white', linewidths=.3, zorder=4)
    for y, value in enumerate(values):
        axis.text(max(value, 0.) + 1.7, y, '{:.2f}%'.format(value),
                  va='center', ha='left', fontsize=6.5, color='#30383d')
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_ylim(len(values) - .55, -.55)
    axis.set_xlim(-7.5, 104.)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.tick_params(axis='y', length=0)
    axis.axvline(0, color='#9ca6ad', linewidth=.7, zorder=2)
    axis.grid(axis='x', color='#e8eaec', linewidth=.5)
    axis.set_axisbelow(True)
    axis.set_xlabel('Global variance contribution (%)', labelpad=4)


def draw(basin, depth, display):
    fig = plt.figure(figsize=(7.2, 4.8))
    a = fig.add_axes([.13, .52, .295, .35])
    b = fig.add_axes([.13, .13, .295, .23])
    c = fig.add_axes([.75, .13, .215, .74])
    contribution_axis(a, list(BASINS), basin.variance_contribution_pct.to_numpy(), BASIN_COLORS)
    contribution_axis(b, ['0-700 m', '700-2000 m', '2000-6000 m'],
                      depth.variance_contribution_pct.to_numpy(), DEPTH_COLORS)
    a.set_title('1993-2025', loc='left', fontsize=7.6, fontweight='bold', pad=9)
    positions = np.r_[np.arange(10), 11.]
    values = display.covariance_term_pct.to_numpy()
    for y, value, unassigned in zip(positions, values, display.includes_unassigned):
        color = RESIDUAL if unassigned else (OFFSET if value < 0 else REINFORCE)
        c.plot([0, value], [y, y], color=color, linewidth=2., solid_capstyle='round', zorder=3)
        c.scatter([value], [y], s=14, color=color, edgecolors='white', linewidths=.4, zorder=4)
        c.text(value + .9, y, '{:+.2f}'.format(value), va='center', ha='left', fontsize=6.1,
               color='#30383d', zorder=5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .45})
    c.set_yticks(positions)
    c.set_yticklabels(display.label)
    c.set_ylim(11.6, -.6)
    c.set_xlim(-36., 8.)
    c.set_xticks([-30, -20, -10, 0])
    c.tick_params(axis='y', length=0, pad=5)
    c.axvline(0, color='#9ca6ad', linewidth=.7, zorder=1)
    c.axhline(10.25, color='#e8eaec', linewidth=.5, zorder=0)
    c.grid(axis='x', color='#e8eaec', linewidth=.5, zorder=0)
    c.set_xlabel('Covariance contribution (%)\n$2\\,\\mathrm{Cov}(i,j)/\\mathrm{Var}(G)$', labelpad=4, fontsize=6.4)
    c.legend(handles=[Line2D([], [], color=OFFSET, marker='o', ms=3., lw=1., label='Offsetting'),
                      Line2D([], [], color=REINFORCE, marker='o', ms=3., lw=1., label='Reinforcing')],
              loc='lower left', bbox_to_anchor=(-.02, 1.045), ncol=2, frameon=False,
              fontsize=6., handlelength=1.1, handletextpad=.35, columnspacing=.8, borderaxespad=0.)
    for x, y, label in ((.045, .905, 'a'), (.045, .395, 'b'), (.525, .905, 'c')):
        fig.text(x, y, label, fontweight='bold', fontsize=10.)
    return fig


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    for axis in fig.axes:
        texts = list(axis.texts) + [axis.xaxis.label, axis.yaxis.label, axis.title]
        texts += axis.get_xticklabels() + axis.get_yticklabels()
        if axis.get_legend():
            texts += axis.get_legend().get_texts()
        for text in texts:
            if not text.get_visible() or not text.get_text():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1:
                issues.append('Outside canvas: ' + text.get_text())
        for labels in (axis.get_xticklabels(), axis.get_yticklabels()):
            visible = [item for item in labels if item.get_visible() and item.get_text()]
            for i, first in enumerate(visible):
                for second in visible[i + 1:]:
                    if first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)):
                        issues.append('Tick overlap: ' + first.get_text() + ' / ' + second.get_text())
    if issues:
        raise ValueError(issues)
    return {'text_outside_canvas': [], 'tick_label_overlaps': [], 'canvas_inches': list(fig.get_size_inches())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = 'Figure_interannual_variability_1993_2025'
    if any((out / (stem + '.' + suffix)).exists() for suffix in ('pdf', 'svg', 'png', 'tiff')):
        raise ValueError('Figure exists; select a separate output directory')
    files, frames, basin, depth, pairs, summary, residuals = load_data()
    protected = list(files.values()) + sorted((ROOT / 'figures/paper_figures_final_20261014/main_text').glob('*.pdf'))
    original_hashes = {path: sha256(path) for path in protected}
    display = pair_display(pairs)
    style()
    fig = draw(basin, depth, display)
    qa = check_layout(fig)
    for suffix in ('png', 'pdf', 'svg', 'tiff'):
        fig.savefig(str(out / (stem + '.' + suffix)), dpi=450 if suffix == 'png' else 600,
                    facecolor='white', bbox_inches=None)
    plt.close(fig)
    for name, frame in [('source_basin_variance_contributions.csv', basin),
                        ('source_depth_variance_contributions.csv', depth),
                        ('source_all_basin_covariance_pairs.csv', pairs),
                        ('source_displayed_covariance_terms.csv', display),
                        ('source_detrended_annual_values_1994_2025.csv', residuals)]:
        frame.to_csv(out / name, index=False)
    frames['variability_contributions'][frames['variability_contributions'].period.eq('historical_full')].to_csv(
        out / 'source_full_history_variance_reference_1941_2025.csv', index=False)
    pd.DataFrame([summary]).to_csv(out / 'source_variance_decomposition_summary.csv', index=False)
    if any(sha256(path) != expected for path, expected in original_hashes.items()):
        raise RuntimeError('Protected input or original manuscript figure changed')
    metadata = {'endpoint_years': [1993, 2025], 'annual_target_years': [1994, 2025],
                'annual_increment_count': 32, '2024_and_2025': 'computed_main_results_not_forecasts',
                'estimator': 'Cov(component_detrended_annual_increment,global_detrended_annual_increment)/Var(global)',
                'covariance_display': '2*Cov(pair)/Var(global); five residual pairs summed in final row',
                'basin_shares_sum_pct': float(basin.variance_contribution_pct.sum()),
                'depth_shares_sum_pct': float(depth.variance_contribution_pct.sum()),
                'net_covariance_reduction_pct_of_diagonal_sum': float(summary.net_cancellation_pct_of_diagonal_sum),
                'global_residual_sd_mm': float(summary.reference_detrended_std_mm),
                'descriptive_only': True, 'confidence_intervals_shown': False,
                'temperature_ensemble_uncertainty_used': False, 'primary_results_recomputed': False,
                'original_sources_unchanged': True, 'protected_source_count': len(protected),
                'source_sha256': {str(i): digest for i, digest in enumerate(original_hashes.values())},
                'script_sha256': sha256(Path(__file__)), 'qa': qa}
    (out / 'verification.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    owner = ROOT.stat()
    for path in [out] + list(out.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    print('Saved:', out)
    print('1993-2025 endpoints; all 32 annual increments 1994-2025 verified')
    print('Basin and depth shares each sum to 100%; all residual pair terms retained')


if __name__ == '__main__':
    main()
