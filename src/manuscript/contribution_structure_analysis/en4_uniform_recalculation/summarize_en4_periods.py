#!/usr/bin/env python3
"""Summarize prespecified periods without recomputing or changing EN4/IAP fields."""
from manuscript_paths import workspace_root, input_path
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

pass  # Source imports are configured by the launcher.
import compare_noaa_thermosteric as statistics


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def period_metrics(rows, start, end):
    periods = [('full_period', start, end)]
    periods.extend((label, first, end) for label, first in
                   [('since_1955', 1955), ('since_1993', 1993), ('since_2005', 2005)]
                   if start < first and end - first >= 3)
    keys = sorted(set((row['region'], row['depth_band']) for row in rows))
    results = []
    for label, first, last in periods:
        for region, band in keys:
            selected = sorted((row for row in rows if row['region'] == region
                               and row['depth_band'] == band
                               and first <= int(row['year']) <= last),
                              key=lambda row: int(row['year']))
            years = np.asarray([int(row['year']) for row in selected])
            if not np.array_equal(years, np.arange(first, last + 1)):
                raise ValueError('Missing or duplicate years for {} {}'.format(region, band))
            iap = np.asarray([float(row['iap_cumulative_anomaly_mm']) for row in selected])
            en4 = np.asarray([float(row['en4_cumulative_anomaly_mm']) for row in selected])
            if not np.isfinite(iap).all() or not np.isfinite(en4).all():
                raise ValueError('Nonfinite cumulative contribution')
            result = {key.replace('noaa', 'en4').replace('main', 'iap'): value
                      for key, value in statistics.metrics(years, iap, en4).items()}
            result.update(period=label, start_year=first, end_year=last,
                          region=region, depth_band=band,
                          mask_scope='fixed_all_years_{}_{}'.format(start, end))
            results.append(result)
    return results


def summarize(directory, modern_reference=None, check_only=False):
    metadata = json.loads((directory / 'metadata.json').read_text())
    start, end = metadata['start_year'], metadata['end_year']
    rows = period_metrics(read_csv(directory / 'matched_cumulative_anomalies.csv'), start, end)
    original = read_csv(directory / 'uniform_comparison_metrics.csv')
    for row in [item for item in rows if item['period'] == 'full_period']:
        match = next(item for item in original if item['region'] == row['region']
                     and item['depth_band'] == row['depth_band'])
        for field in ('iap_ols_trend_mm_per_year', 'en4_ols_trend_mm_per_year',
                      'annual_change_rmse_mm'):
            np.testing.assert_allclose(row[field], float(match[field]), atol=1e-12, rtol=1e-12)
    if check_only:
        print('Verified {} period rows against saved full-period metrics'.format(len(rows)))
        return
    destinations = [directory / 'prespecified_period_comparison.csv', directory / 'PERIOD_REPORT.md']
    if modern_reference:
        destinations.append(directory / 'modern_common_mask_sensitivity.csv')
    if any(path.exists() for path in destinations):
        raise ValueError('Summary exists; refusing overwrite')
    lines = ['# Prespecified EN4 / IAP Time-Window Comparisons', '',
             'All windows below use the same frozen {}-{} common mask.'.format(start, end),
             'Trends are OLS slopes of cumulative annual volume-change contributions.',
             'Correlations are detrended adjacent-year changes, not cumulative correlations.', '',
             '| Period | Region | Depth | IAP mm/yr | EN4 mm/yr | IAP minus EN4 | Annual r |',
             '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        r = row['detrended_annual_change_correlation']
        lines.append('| {}-{} | {} | {} | {:.6f} | {:.6f} | {:.6f} | {} |'.format(
            row['start_year'], row['end_year'], row['region'], row['depth_band'],
            row['iap_ols_trend_mm_per_year'], row['en4_ols_trend_mm_per_year'],
            row['iap_minus_en4_trend_mm_per_year'], '{:.4f}'.format(r) if r != '' else 'undefined'))
    influence = read_csv(directory / 'en4_observation_influence.csv')
    lines.extend(['', '## Observation Influence Diagnostics', '',
                  'These are time averages of volume-weighted native EN4 observation weights.',
                  'Weights are not uncertainty intervals, significance tests or filtering criteria.', '',
                  '| Period | Region | Depth | Temperature weight mean | Salinity weight mean |',
                  '|---|---|---|---:|---:|'])
    for row in rows:
        selected = [item for item in influence if item['region'] == row['region']
                    and item['depth_band'] == row['depth_band']
                    and row['start_year'] <= int(item['year']) <= row['end_year']]
        if len(selected) != row['end_year'] - row['start_year'] + 1:
            raise ValueError('Observation influence annual records incomplete')
        means = [np.mean([float(item[field]) for item in selected]) for field in
                 ('en4_t_weight_volume_weighted_mean', 'en4_s_weight_volume_weighted_mean')]
        lines.append('| {}-{} | {} | {} | {:.4f} | {:.4f} |'.format(
            row['start_year'], row['end_year'], row['region'], row['depth_band'], *means))
    if modern_reference:
        reference = json.loads((modern_reference / 'metadata.json').read_text())
        if (reference['start_year'], reference['end_year']) != (2005, end):
            raise ValueError('Reference must be the completed 2005-{} comparison'.format(end))
        if reference['cache_version'] != metadata['cache_version']:
            raise ValueError('Different annual field processing conventions')
        for year in range(2005, end + 1):
            if (reference['annual_preparation'][str(year)]['annual_cache_sha256'] !=
                    metadata['annual_preparation'][str(year)]['annual_cache_sha256']):
                raise ValueError('Modern annual fields differ; cannot attribute changes to masks alone')
        if reference['normalization_areas_m2'] != metadata['normalization_areas_m2']:
            raise ValueError('Normalization differs between experiments')
        for source, digest in reference['input_hashes_unchanged'].items():
            if metadata['input_hashes_unchanged'].get(source) != digest:
                raise ValueError('Input provenance differs between experiments')
        old = read_csv(modern_reference / 'uniform_comparison_metrics.csv')
        sensitivity = []
        for row in [item for item in rows if item['start_year'] == 2005]:
            previous = next(item for item in old if item['region'] == row['region']
                            and item['depth_band'] == row['depth_band'])
            for product in ('iap', 'en4'):
                field = product + '_ols_trend_mm_per_year'
                sensitivity.append({'region': row['region'], 'depth_band': row['depth_band'],
                                    'product': product.upper(), 'start_year': 2005, 'end_year': end,
                                    'modern_only_mask_trend_mm_per_year': float(previous[field]),
                                    'long_period_mask_trend_mm_per_year': row[field],
                                    'long_mask_minus_modern_mask_mm_per_year': row[field] - float(previous[field]),
                                    'annual_fields_identical_verified': True})
        statistics.step1.write_csv(directory / 'modern_common_mask_sensitivity.csv', sensitivity)
        lines.extend(['', '## Modern Mask Sensitivity', '',
                      'modern_common_mask_sensitivity.csv compares 2005-{} using the original modern mask'.format(end),
                      'against the longer-period mask. Modern annual input caches are hash-identical.',
                      'This separates integration-support sensitivity from changes in annual input fields.'])
    lines.extend(['', '## Interpretation', '',
                  '- Historical low-observation-weight fields can be background-dominated; numerical completeness is not observational coverage.',
                  '- Use early periods as cross-product sensitivity, not independent truth or equal-confidence validation.',
                  '- Shared observations, a single g10 correction scheme, inherited IAP variable conventions and nominal layer boundaries remain limitations.',
                  '- No EN4 uncertainty propagation or 2000-6000m validation is performed.',
                  '- Original 15-arcsecond results and the completed modern comparison are unchanged.'])
    statistics.step1.write_csv(directory / 'prespecified_period_comparison.csv', rows)
    (directory / 'PERIOD_REPORT.md').write_text('\n'.join(lines) + '\n')
    owner = directory.stat()
    for path in destinations:
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o644)
    print('Saved {} comparisons in {}'.format(len(rows), directory))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--modern-reference', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    summarize(args.directory, args.modern_reference, args.check_only)
