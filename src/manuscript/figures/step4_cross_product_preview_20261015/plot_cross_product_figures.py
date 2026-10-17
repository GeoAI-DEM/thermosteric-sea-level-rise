#!/usr/bin/env python3
"""Read-only cross-product figure package with separate comparison contracts."""
from manuscript_paths import workspace_root, input_path
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = (workspace_root() / 'figures/step4_cross_product_preview_20261015')
ROOT = HERE.parent.parent
ANALYSIS = ROOT / 'contribution_structure_analysis'
pass  # Source imports are configured by the launcher.
from calculate_interannual_variability import detrend

FOLDERS = {
    'noaa': ANALYSIS / 'step4_noaa_comparison_20261014',
    'ishii': ANALYSIS / 'step4_ishii_comparison_20261014',
    'copernicus': ANALYSIS / 'step4_copernicus_comparison_20261015',
    'csiro': ANALYSIS / 'step4_csiro_comparison_20261014',
    'en4': ANALYSIS / 'en4_uniform_recalculation/results_1940_2023_g10_20261015',
}
BANDS = ('0-700m', '700-2000m', '0-2000m')
BAND_LABELS = ('0-700 m', '700-2000 m', '0-2000 m')
BLUE, TEAL, CORAL = '#285e8e', '#3a9991', '#c97662'
GRAY = '#65747b'
STEMS = ('Figure_08_Cross_product_thermosteric_comparison',
         'Figure_S05_EN4_historical_sensitivity',
         'Figure_S06_Regional_external_product_comparisons')


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class Inputs:
    def __init__(self):
        self.signatures = {}

    def read(self, product, filename):
        path = FOLDERS[product] / filename
        signature = sha256(path)
        if str(path) in self.signatures and self.signatures[str(path)] != signature:
            raise ValueError('Input changed during plotting')
        self.signatures[str(path)] = signature
        return pd.read_csv(path)

    def verify(self):
        assert self.signatures == {name: sha256(Path(name)) for name in self.signatures}


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7., 'axes.labelsize': 7.,
                         'xtick.labelsize': 6.5, 'ytick.labelsize': 7., 'axes.linewidth': .65,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'xtick.major.width': .65, 'ytick.major.width': .65,
                         'xtick.major.size': 3., 'ytick.major.size': 3.,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False,
                         'savefig.facecolor': 'white'})


def one(frame, **conditions):
    rows = select(frame, **conditions)
    if len(rows) != 1:
        raise ValueError('Expected one metric for {}: {}'.format(conditions, len(rows)))
    return rows.iloc[0]


def select(frame, **conditions):
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= frame[column].eq(value).to_numpy()
    return frame[mask].copy()


def years_checked(frame, start, end, column='year'):
    frame = frame.sort_values(column)
    np.testing.assert_array_equal(frame[column], np.arange(start, end + 1))
    return frame


def trend(years, values):
    x = np.asarray(years, dtype=float)
    x -= x.mean()
    y = np.asarray(values, dtype=float)
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


def check_trend(years, values, expected):
    np.testing.assert_allclose(trend(years, values), expected, rtol=0., atol=1e-10)


def end_labels(ax, x, series, min_gap, label_x, size=7.):
    endpoints = sorted([(float(values[-1]), label, color) for label, values, color in series])
    previous = -np.inf
    for y, label, color in endpoints:
        target = max(y, previous + min_gap)
        ax.plot([x, label_x - .12], [y, target], color=color, lw=.65, clip_on=False)
        ax.text(label_x, target, label, color=color, fontsize=size, va='center', clip_on=False)
        previous = target
    return previous


def clean_axis(ax, grid='y'):
    ax.grid(axis=grid, color='#e8ebed', linewidth=.5)
    ax.set_axisbelow(True)


def panel(fig, label, x, y):
    fig.text(x, y, label, fontsize=10., fontweight='bold')


def product_handles(labels, colors):
    return [Line2D([], [], color=color, marker='o', lw=0., markersize=3.2, label=label)
            for label, color in zip(labels, colors)]


def dot_trends(ax, rows, labels, colors, markers=None):
    rows = np.asarray(rows)
    if rows.shape != (len(labels), len(colors)) or not np.isfinite(rows).all():
        raise ValueError('Trend layout does not match data')
    markers = markers or ['o'] * len(colors)
    offsets = np.linspace(-.20, .20, len(colors))
    for i, values in enumerate(rows):
        for j, value in enumerate(values):
            ax.scatter(value, i + offsets[j], s=18., marker=markers[j], color=colors[j],
                       edgecolors='white', linewidths=.4, zorder=3)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.set_xlim(0., 1.65)
    ax.set_xticks([0., .5, 1., 1.5])
    ax.set_xlabel('Trend (mm yr$^{-1}$)', labelpad=6.)
    clean_axis(ax, 'x')


def verify_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if not fig.bbox.expanded(1.001, 1.001).contains(box.x0, box.y0) or not fig.bbox.expanded(1.001, 1.001).contains(box.x1, box.y1):
            outside.append(item.get_text())
    if outside:
        raise ValueError('Text outside figure: {}'.format(outside))
    for ax in fig.axes:
        xmin, xmax = sorted(ax.get_xlim())
        ymin, ymax = sorted(ax.get_ylim())
        for line in ax.lines:
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if np.any((x < xmin - 1e-9) | (x > xmax + 1e-9) | (y < ymin - 1e-9) | (y > ymax + 1e-9)):
                raise ValueError('Curve or leader extends outside data limits')
        for artist in ax.collections:
            points = np.asarray(artist.get_offsets(), dtype=float)
            if len(points) and np.any((points[:, 0] < xmin - 1e-9) | (points[:, 0] > xmax + 1e-9)
                                     | (points[:, 1] < ymin - 1e-9) | (points[:, 1] > ymax + 1e-9)):
                raise ValueError('Scatter point extends outside data limits')
        for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
            boxes = [item.get_window_extent(renderer) for item in ticks if item.get_visible() and item.get_text()]
            for i, box in enumerate(boxes):
                if any(box.overlaps(other) for other in boxes[i + 1:]):
                    raise ValueError('Overlapping tick labels')
    return {'all_text_inside_canvas': True, 'tick_labels_nonoverlapping': True,
            'all_curve_and_scatter_values_inside_limits': True}


def save(fig, stem, evidence):
    report = verify_text(fig)
    report.update(evidence)
    report.update({'figure_size_inches': list(fig.get_size_inches()),
                   'no_invented_error_bars': True, 'font': 'DejaVu Sans'})
    for suffix in ('png', 'pdf', 'svg', 'tiff'):
        fig.savefig(str(HERE / (stem + '.' + suffix)), dpi=450 if suffix == 'png' else 600,
                    bbox_inches=None, facecolor='white')
    plt.close(fig)
    (HERE / (stem + '_verification.json')).write_text(json.dumps(report, indent=2))
    print('Saved', stem, flush=True)


def main_figure(data):
    nm = data.read('noaa', 'cross_product_metrics.csv')
    im = data.read('ishii', 'cross_product_metrics.csv')
    ns = years_checked(select(data.read('noaa', 'matched_anomalies_and_annual_changes.csv'),
                              period='modern_common', depth_band='0-2000m'), 2005, 2023)
    ins = years_checked(select(data.read('ishii', 'matched_anomalies_and_annual_changes.csv'),
                               period='modern_common', depth_band='0-2000m'), 2005, 2023)
    np.testing.assert_allclose(ns.main_anomaly_anchored_mm, ins.main_anomaly_anchored_mm, atol=1e-12, rtol=0.)
    a_source = pd.DataFrame({'year': ns.year.to_numpy(), 'this_study_mm': ns.main_anomaly_anchored_mm.to_numpy(),
                             'noaa_mm': ns.noaa_anomaly_anchored_mm.to_numpy(), 'ishii_mm': ins.ishii_anomaly_anchored_mm.to_numpy()})
    a_source.to_csv(HERE / 'source_F08a_global_0_2000m.csv', index=False)
    rows, b_source = [], []
    for band in BANDS:
        n = one(nm, period='modern_common', depth_band=band)
        i = one(im, period='modern_common', depth_band=band)
        np.testing.assert_allclose(n.main_ols_trend_mm_per_year, i.main_ols_trend_mm_per_year, rtol=0., atol=1e-12)
        values = [n.main_ols_trend_mm_per_year, n.noaa_ols_trend_mm_per_year, i.ishii_ols_trend_mm_per_year]
        rows.append(values)
        for product, value in zip(('This study', 'NOAA', 'Ishii'), values):
            b_source.append({'depth_band': band, 'product': product, 'trend_mm_per_year': value,
                             'start_year': 2005, 'end_year': 2023, 'region': 'Global'})
    pd.DataFrame(b_source).to_csv(HERE / 'source_F08b_global_trends.csv', index=False)
    for column, value in zip(('this_study_mm', 'noaa_mm', 'ishii_mm'), rows[2]):
        check_trend(a_source.year, a_source[column], value)
    annual = data.read('en4', 'annual_thermal_expansion.csv')
    matched = select(annual, region='Global', depth_band='0-2000m')
    matched = matched[matched.target_year.between(2006, 2023)]
    pivot = matched.pivot(index='target_year', columns='product', values='sea_level_contribution_mm').sort_index()
    np.testing.assert_array_equal(pivot.index, np.arange(2006, 2024))
    residuals, _ = detrend(pivot[['IAP', 'EN4']].to_numpy().T, pivot.index.to_numpy())
    r = float(np.corrcoef(residuals)[0, 1])
    metric = one(data.read('en4', 'prespecified_period_comparison.csv'),
                 region='Global', depth_band='0-2000m', start_year=2005, end_year=2023)
    np.testing.assert_allclose(r, metric.detrended_annual_change_correlation, atol=1e-12, rtol=0.)
    c_source = pivot.reset_index()
    c_source['iap_detrended_annual_change_mm'] = residuals[0]
    c_source['en4_detrended_annual_change_mm'] = residuals[1]
    c_source.to_csv(HERE / 'source_F08c_uniform_algorithm_annual_scatter.csv', index=False)
    fig = plt.figure(figsize=(7.2, 6.1))
    a = fig.add_axes([.12, .62, .74, .27])
    b = fig.add_axes([.15, .14, .33, .31])
    c = fig.add_axes([.64, .14, .30, .31])
    series = [('This study', a_source.this_study_mm.to_numpy(), BLUE),
              ('NOAA', a_source.noaa_mm.to_numpy(), TEAL), ('Ishii', a_source.ishii_mm.to_numpy(), CORAL)]
    for label, values, color in series:
        a.plot(a_source.year, values, color=color, lw=1.6 if label == 'This study' else 1.1, zorder=3 if label == 'This study' else 2)
    top = end_labels(a, 2023, series, 1.7, 2023.6)
    a.set_xlim(2005, 2026.3)
    a.set_ylim(-1.5, max(26., top + 1.5))
    a.set_xticks([2005, 2010, 2015, 2020, 2023])
    a.set_ylabel('Cumulative change\nsince 2005 (mm)', labelpad=7.)
    a.set_xlabel('Year', labelpad=4.)
    clean_axis(a)
    dot_trends(b, rows, BAND_LABELS, (BLUE, TEAL, CORAL))
    fig.legend(handles=product_handles(('This study', 'NOAA', 'Ishii'), (BLUE, TEAL, CORAL)),
               loc='center', bbox_to_anchor=(.31, .505), ncol=3, fontsize=6.5,
               handletextpad=.35, columnspacing=1.1, handlelength=.8)
    limit = np.ceil(np.abs(residuals).max() * 1.15 * 2.) / 2.
    c.plot([-limit, limit], [-limit, limit], color='#a0a9ae', lw=.8, ls=(0, (3, 3)), zorder=1)
    c.scatter(residuals[0], residuals[1], s=20., color=TEAL, edgecolors='white', linewidths=.45, zorder=3)
    c.set_xlim(-limit, limit)
    c.set_ylim(-limit, limit)
    c.set_aspect('equal', adjustable='box')
    c.xaxis.set_major_locator(MaxNLocator(5))
    c.yaxis.set_major_locator(MaxNLocator(5))
    c.set_xlabel('IAP annual change (mm)', labelpad=5.)
    c.set_ylabel('EN4 annual change (mm)', labelpad=5.)
    c.text(.05, .95, '$r$ = {:.3f}\n$n$ = {}'.format(r, len(pivot)), transform=c.transAxes,
           va='top', fontsize=7., bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.})
    clean_axis(c, 'both')
    fig.text(.12, .925, 'Global | 0-2000 m | 2005-2023', fontsize=7.)
    fig.text(.64, .505, '1$^\circ$ common-domain recalculation', fontsize=6.8)
    fig.text(.64, .48, '0-2000 m | 2005-2023 | detrended', fontsize=6.5, color=GRAY)
    for label, x, y in (('a', .035, .925), ('b', .035, .505), ('c', .55, .505)):
        panel(fig, label, x, y)
    save(fig, STEMS[0], {'anomaly_states': 19, 'scatter_annual_pairs': 18,
                        'scatter_r_reproduces_archived_metric': r,
                        'trend_points_reproduce_archived_metrics': True,
                        'global_15arcsec_and_1degree_comparison_separated': True})


def en4_figure(data):
    states = years_checked(select(data.read('en4', 'matched_cumulative_anomalies.csv'),
                                   region='Global', depth_band='0-2000m'), 1940, 2023)
    metrics = select(data.read('en4', 'prespecified_period_comparison.csv'), region='Global', depth_band='0-2000m')
    periods = ((1940, 2023), (1955, 2023), (1993, 2023), (2005, 2023))
    rows, b_source = [], []
    for start, end in periods:
        metric = one(metrics, start_year=start, end_year=end)
        subset = states[states.year.between(start, end)]
        values = [metric.iap_ols_trend_mm_per_year, metric.en4_ols_trend_mm_per_year]
        check_trend(subset.year, subset.iap_cumulative_anomaly_mm, values[0])
        check_trend(subset.year, subset.en4_cumulative_anomaly_mm, values[1])
        rows.append(values)
        for product, value in zip(('IAP', 'EN4'), values):
            b_source.append({'start_year': start, 'end_year': end, 'product': product,
                             'trend_mm_per_year': value, 'region': 'Global', 'depth_band': '0-2000m'})
    weights = years_checked(select(data.read('en4', 'en4_observation_influence.csv'),
                                    region='Global', depth_band='0-2000m'), 1940, 2023)
    states.to_csv(HERE / 'source_S05a_en4_historical_states.csv', index=False)
    pd.DataFrame(b_source).to_csv(HERE / 'source_S05b_en4_period_trends.csv', index=False)
    weights.to_csv(HERE / 'source_S05c_en4_observation_weights.csv', index=False)
    fig = plt.figure(figsize=(7.2, 6.1))
    a = fig.add_axes([.12, .62, .74, .27])
    b = fig.add_axes([.15, .14, .33, .31])
    c = fig.add_axes([.64, .14, .30, .31])
    series = [('IAP', states.iap_cumulative_anomaly_mm.to_numpy(), BLUE),
              ('EN4', states.en4_cumulative_anomaly_mm.to_numpy(), TEAL)]
    for label, values, color in series:
        a.plot(states.year, values, color=color, lw=1.4)
    top = end_labels(a, 2023, series, 4., 2026., size=7.)
    a.set_xlim(1940, 2034)
    a.set_xticks([1940, 1960, 1980, 2000, 2023])
    minimum = min(float(values.min()) for _, values, _ in series)
    a.set_ylim(minimum - 2., top + 7.)
    a.set_ylabel('Cumulative change\nsince 1940 (mm)', labelpad=7.)
    a.set_xlabel('Year', labelpad=4.)
    clean_axis(a)
    dot_trends(b, rows, ['{}-{}'.format(*period) for period in periods], (BLUE, TEAL))
    fig.legend(handles=product_handles(('IAP', 'EN4'), (BLUE, TEAL)), loc='center',
               bbox_to_anchor=(.31, .505), ncol=2, fontsize=6.7, handlelength=.8, handletextpad=.4)
    c.plot(weights.year, weights.en4_t_weight_volume_weighted_mean, color=BLUE, lw=1.15, label='Temperature')
    c.plot(weights.year, weights.en4_s_weight_volume_weighted_mean, color=GRAY, lw=1.15, ls=(0, (3, 2)), label='Salinity')
    c.set_xlim(1940, 2023)
    c.set_ylim(0., 1.02)
    c.set_xticks([1940, 1960, 1980, 2000, 2020])
    c.set_yticks([0., .5, 1.])
    c.set_xlabel('Year', labelpad=5.)
    c.set_ylabel('EN4 observation weight', labelpad=5.)
    c.legend(loc='upper left', fontsize=6.3, handlelength=1.5, borderaxespad=.25)
    clean_axis(c)
    fig.text(.12, .925, '1$^\circ$ common domain | Global | 0-2000 m', fontsize=7.)
    fig.text(.64, .505, 'Volume-weighted observation influence', fontsize=6.8)
    for label, x, y in (('a', .035, .925), ('b', .035, .505), ('c', .55, .505)):
        panel(fig, label, x, y)
    save(fig, STEMS[1], {'historical_states': 84, 'trend_periods': 4,
                        'frozen_mask_shared_for_all_periods': True,
                        'observation_weights_not_uncertainty': True,
                        'trends_checked_against_state_curves': True})


def regional_figure(data):
    cop = data.read('copernicus', 'matched_anomalies_and_annual_changes.csv')
    cm = data.read('copernicus', 'cross_product_metrics.csv')
    observation = years_checked(select(cop, depth_band='0-2000m', reference_variable='thsl_mean_observations'), 2005, 2023)
    reanalysis = years_checked(select(cop, depth_band='0-2000m', reference_variable='thsl_mean_reanalysis'), 2005, 2023)
    np.testing.assert_allclose(observation.main_anomaly_anchored_mm, reanalysis.main_anomaly_anchored_mm, atol=1e-12, rtol=0.)
    cop_source = pd.DataFrame({'year': observation.year.to_numpy(), 'this_study_mm': observation.main_anomaly_anchored_mm.to_numpy(),
                               'observation_mean_mm': observation.copernicus_anomaly_anchored_mm.to_numpy(),
                               'reanalysis_mean_mm': reanalysis.copernicus_anomaly_anchored_mm.to_numpy()})
    cop_source.to_csv(HERE / 'source_S06a_copernicus_60S_60N_states.csv', index=False)
    cop_rows, cop_trends = [], []
    for band in BANDS:
        o = one(cm, depth_band=band, reference_variable='thsl_mean_observations')
        r = one(cm, depth_band=band, reference_variable='thsl_mean_reanalysis')
        np.testing.assert_allclose(o.main_ols_trend_mm_per_year, r.main_ols_trend_mm_per_year, atol=1e-12, rtol=0.)
        values = [o.main_ols_trend_mm_per_year, o.copernicus_ols_trend_mm_per_year, r.copernicus_ols_trend_mm_per_year]
        cop_rows.append(values)
        for product, value in zip(('This study', 'Observation mean', 'Reanalysis mean'), values):
            cop_trends.append({'product': product, 'depth_band': band, 'trend_mm_per_year': value,
                               'region': '60S-60N', 'start_year': 2005, 'end_year': 2023})
    pd.DataFrame(cop_trends).to_csv(HERE / 'source_S06b_copernicus_trends.csv', index=False)
    for column, value in zip(('this_study_mm', 'observation_mean_mm', 'reanalysis_mean_mm'), cop_rows[2]):
        check_trend(cop_source.year, cop_source[column], value)
    cs = years_checked(select(data.read('csiro', 'matched_anomalies_and_smoothed_differences.csv'),
                               period='interior_extended', main_smoothing='centered'), 1961, 2024)
    csm = data.read('csiro', 'cross_product_metrics.csv')
    csm = csm[csm.main_smoothing.eq('centered_three_year_complete_windows')].copy()
    periods = ((1961, 2003), (1961, 2023), (1993, 2023), (2005, 2023), (1961, 2024))
    cs_rows, cs_trends = [], []
    for start, end in periods:
        metric = one(csm, start_year=start, end_year=end)
        subset = cs[cs.year.between(start, end)]
        values = [metric.main_ols_trend_mm_per_year, metric.csiro_ols_trend_mm_per_year]
        check_trend(subset.year, subset.main_anchored_anomaly_mm, values[0])
        check_trend(subset.year, subset.csiro_anchored_anomaly_mm, values[1])
        cs_rows.append(values)
        for product, value in zip(('This study', 'CSIRO'), values):
            cs_trends.append({'product': product, 'trend_mm_per_year': value, 'start_year': start,
                              'end_year': end, 'region': '65S-65N', 'depth_band': '0-700m',
                              'smoothing': 'three_year_states; main_centered_assumption'})
    cs.to_csv(HERE / 'source_S06c_csiro_65S_65N_smoothed_states.csv', index=False)
    pd.DataFrame(cs_trends).to_csv(HERE / 'source_S06d_csiro_period_trends.csv', index=False)
    fig = plt.figure(figsize=(7.2, 6.8))
    a = fig.add_axes([.12, .62, .36, .27])
    b = fig.add_axes([.67, .62, .29, .27])
    c = fig.add_axes([.12, .13, .36, .27])
    d = fig.add_axes([.67, .13, .29, .27])
    for name, column, color in (('This study', 'this_study_mm', BLUE),
                                ('Observation mean', 'observation_mean_mm', TEAL),
                                ('Reanalysis mean', 'reanalysis_mean_mm', CORAL)):
        a.plot(cop_source.year, cop_source[column], color=color, lw=1.5 if name == 'This study' else 1.1)
    a.set_xlim(2005, 2023)
    a.set_xticks([2005, 2010, 2015, 2020, 2023])
    a.set_ylabel('Cumulative change\nsince 2005 (mm)', labelpad=7.)
    a.set_xlabel('Year', labelpad=4.)
    clean_axis(a)
    dot_trends(b, cop_rows, BAND_LABELS, (BLUE, TEAL, CORAL))
    fig.legend(handles=product_handles(('This study', 'Observation mean', 'Reanalysis mean'), (BLUE, TEAL, CORAL)),
               loc='center', bbox_to_anchor=(.54, .965), ncol=3, fontsize=6.6,
               handlelength=.8, handletextpad=.35, columnspacing=1.5)
    fig.text(.12, .925, '60$^\circ$S-60$^\circ$N | 0-2000 m', fontsize=7.)
    fig.text(.67, .925, '60$^\circ$S-60$^\circ$N | 2005-2023', fontsize=7.)
    c.plot(cs.year, cs.main_anchored_anomaly_mm, color=BLUE, lw=1.4)
    c.plot(cs.year, cs.csiro_anchored_anomaly_mm, color=CORAL, lw=1.1)
    c.set_xlim(1961, 2024)
    c.set_xticks([1961, 1980, 2000, 2024])
    c.set_ylabel('Cumulative change\nsince 1961 (mm)', labelpad=7.)
    c.set_xlabel('Year', labelpad=4.)
    clean_axis(c)
    dot_trends(d, cs_rows, ['{}-{}'.format(*period) for period in periods], (BLUE, CORAL))
    fig.legend(handles=product_handles(('This study', 'CSIRO'), (BLUE, CORAL)),
               loc='center', bbox_to_anchor=(.33, .475), ncol=2, fontsize=6.6,
               handlelength=.8, handletextpad=.35)
    fig.text(.12, .435, '65$^\circ$S-65$^\circ$N | 0-700 m | 3-year means', fontsize=6.8)
    fig.text(.67, .435, '65$^\circ$S-65$^\circ$N | 0-700 m', fontsize=6.8)
    for label, x, y in (('a', .035, .925), ('b', .55, .925), ('c', .035, .435), ('d', .55, .435)):
        panel(fig, label, x, y)
    save(fig, STEMS[2], {'copernicus_states': 19, 'csiro_states': 64,
                        'latitudes_not_mixed': True, 'csiro_smoothing_not_treated_as_annual_unsmoothed': True,
                        'paired_trends_checked_against_states': True,
                        'individual_copernicus_member_metrics_retained_in_table': True})


def full_table(data):
    records = []
    for product in ('noaa', 'ishii', 'copernicus', 'csiro', 'en4'):
        filename = 'prespecified_period_comparison.csv' if product == 'en4' else 'cross_product_metrics.csv'
        frame = data.read(product, filename)
        for _, row in frame.iterrows():
            comparator = 'IAP (1-degree common domain)' if product == 'en4' else 'This study (15-arcsecond integration)'
            reference = row.get('reference_label', product.upper() if product in ('noaa', 'en4', 'csiro') else 'Ishii')
            left = 'iap' if product == 'en4' else 'main'
            smoothing = row.get('main_smoothing', 'none; annual mean states')
            smooth = product == 'csiro'
            record = {
                'comparison_family': 'uniform_algorithm_common_domain' if product == 'en4' else 'published_product_comparison',
                'main_series': comparator, 'reference': reference,
                'reference_class': row.get('reference_class', 'temperature_salinity_reconstruction' if product == 'en4' else 'published_thermosteric_estimate'),
                'region': row.get('region', row.get('latitude_range', 'Global')),
                'depth_band': row.depth_band,
                'start_year': int(row.get('start_year', row.get('endpoint_start_year'))),
                'end_year': int(row.get('end_year', row.get('endpoint_end_year'))),
                'anomaly_state_count': int(row.anomaly_state_count),
                'change_count': int(row.get('annual_change_count', row.get('smoothed_state_difference_count'))),
                'main_trend_mm_per_year': row[left + '_ols_trend_mm_per_year'],
                'reference_trend_mm_per_year': row[product + '_ols_trend_mm_per_year'],
                'main_minus_reference_trend_mm_per_year': row[left + '_minus_' + product + '_trend_mm_per_year'],
                'detrended_anomaly_r': row.detrended_anomaly_correlation,
                'detrended_annual_change_r': np.nan if smooth else row.detrended_annual_change_correlation,
                'annual_change_rmse_mm': np.nan if smooth else row.annual_change_rmse_mm,
                'detrended_smoothed_state_difference_r': row.detrended_smoothed_state_difference_correlation if smooth else np.nan,
                'smoothed_state_difference_rmse_mm': row.smoothed_state_difference_rmse_mm if smooth else np.nan,
                'demeaned_anomaly_rmse_mm': row.demeaned_anomaly_rmse_mm,
                'main_smoothing': smoothing,
                'exact_common_product_support': product == 'en4',
                'depth_match': row.depth_match,
                'inference': 'descriptive; shared observations possible; not validation against independent truth',
                'source_metrics_file': str(FOLDERS[product] / filename),
                'source_period': row.get('period', '2005-2023'),
            }
            records.append(record)
    table = pd.DataFrame(records)
    assert len(table) == 79
    table.to_csv(HERE / 'Table_S01_Full_cross_product_metrics.csv', index=False, encoding='utf-8-sig')
    # Keep the complete original metrics, including all less favorable comparisons.
    sources = HERE / 'original_metric_snapshots'
    sources.mkdir()
    for product in ('noaa', 'ishii', 'copernicus', 'csiro', 'en4'):
        filename = 'prespecified_period_comparison.csv' if product == 'en4' else 'cross_product_metrics.csv'
        source = FOLDERS[product] / filename
        (sources / (product + '_' + filename)).write_bytes(source.read_bytes())
    print('Saved Table S01: {} comparisons'.format(len(table)), flush=True)


def main():
    for stem in STEMS:
        if any((HERE / (stem + '.' + suffix)).exists() for suffix in ('pdf', 'png', 'svg', 'tiff')):
            raise ValueError('Existing exports are protected from overwrite')
    if (HERE / 'Table_S01_Full_cross_product_metrics.csv').exists():
        raise ValueError('Existing table is protected from overwrite')
    style()
    data = Inputs()
    main_figure(data)
    en4_figure(data)
    regional_figure(data)
    full_table(data)
    data.verify()
    (HERE / 'input_sha256.json').write_text(json.dumps(data.signatures, indent=2))
    owner = ROOT.stat()
    for path in [HERE] + list(HERE.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    print('All inputs unchanged; preview package complete', flush=True)


if __name__ == '__main__':
    main()
