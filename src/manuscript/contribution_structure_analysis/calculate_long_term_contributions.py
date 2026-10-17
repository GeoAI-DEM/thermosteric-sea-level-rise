#!/usr/bin/env python3
"""Step 1: descriptive long-term contributions from existing annual CSVs only."""

from manuscript_paths import workspace_root, input_path
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np


ROOT = workspace_root()
AREA = 361.9e12
BASINS = ('Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic')
EXCLUSIVE_DEPTHS = ('0-700m', '700-2000m', '2000-6000m')
DEPTHS = EXCLUSIVE_DEPTHS + ('0-2000m',)
PERIODS = (
    ('historical_full', 1940, 2025),
    ('historical_through_2023', 1940, 2023),
    ('recent_core', 1993, 2023),
    ('recent_extended', 1993, 2025),
    ('core_first_15_years', 1993, 2008),
    ('core_last_15_years', 2008, 2023),
    ('modern_overlap', 2005, 2023),
    ('historical_middle', 1955, 1993),
    ('historical_early', 1940, 1955),
    ('extended_first_15_years', 1995, 2010),
    ('extended_last_15_years', 2010, 2025),
)
COMPARISONS = (
    ('core_adjacent_15_years', 'core_first_15_years', 'core_last_15_years'),
    ('extended_adjacent_15_years', 'extended_first_15_years', 'extended_last_15_years'),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('No rows for ' + str(path))
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_table(path, depth_column, value_column, manifest):
    before = sha256(path)
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if depth_column not in reader.fieldnames or value_column not in reader.fieldnames:
            raise ValueError('Required columns missing from ' + str(path))
        rows = list(reader)
    if not rows:
        raise ValueError('Empty input: ' + str(path))
    depth = np.asarray([float(row[depth_column]) for row in rows])
    values = np.asarray([float(row[value_column]) for row in rows])
    if not np.isfinite(depth).all() or not np.isfinite(values).all():
        raise ValueError('Nonfinite statistics input: ' + str(path))
    manifest.append({'path': str(path), 'sha256': before, 'bytes': path.stat().st_size,
                     'value_column': value_column})
    return depth, values


def band_totals(depth, values):
    return {
        '0-700m': float(values[depth <= 700].sum(dtype=np.float64)),
        '700-2000m': float(values[(depth > 700) & (depth <= 2000)].sum(dtype=np.float64)),
        '2000-6000m': float(values[(depth > 2000) & (depth <= 6000)].sum(dtype=np.float64)),
        '0-2000m': float(values[depth <= 2000].sum(dtype=np.float64)),
        '0-6000m': float(values.sum(dtype=np.float64)),
    }


def load_annual(root, start, end, log):
    series, manifest = {}, []
    years = np.arange(start + 1, end + 1)
    for index, year in enumerate(years):
        directory = root / ('{}_vs_{}'.format(year, year - 1))
        z, volumes = load_table(directory / 'data.csv', 'depth_m', 'volume_change_km3', manifest)
        global_values = band_totals(z, volumes * 1e9 / AREA * 1000)
        entries = {('global', 'Global', '0-6000m'): global_values['0-6000m']}
        for band in DEPTHS:
            entries[('depth', 'Global', band)] = global_values[band]
        basin_values = {}
        for basin in BASINS:
            z, values = load_table(directory / (basin + '.csv'), 'depth_m',
                                   'layer_global_contribution_mm', manifest)
            totals = band_totals(z, values)
            basin_values[basin] = totals
            entries[('basin', basin, '0-6000m')] = totals['0-6000m']
            for band in DEPTHS:
                entries[('basin_depth', basin, band)] = totals[band]
        # Retain the arithmetic residual; this is not a new spatial/volume audit.
        for band in DEPTHS + ('0-6000m',):
            residual = global_values[band] - sum(basin_values[b][band] for b in BASINS)
            kind = 'basin' if band == '0-6000m' else 'basin_depth'
            entries[(kind, 'Unassigned', band)] = residual
        if not series:
            series = {key: [] for key in entries}
        if set(entries) != set(series):
            raise ValueError('Component availability differs in year ' + str(year))
        for key, value in entries.items():
            series[key].append(value)
        if index % 10 == 0 or year == years[-1]:
            log.info('Read annual CSVs %s/%s: %s relative to %s', index + 1, len(years), year, year - 1)
    return years, {key: np.asarray(value, dtype=np.float64) for key, value in series.items()}, manifest


def fit_cumulative(increments, endpoint_start):
    increments = np.asarray(increments, dtype=np.float64)
    if increments.ndim != 1 or increments.size < 2 or not np.isfinite(increments).all():
        raise ValueError('Need at least two finite annual increments.')
    years = np.arange(endpoint_start, endpoint_start + increments.size + 1)
    path = np.r_[0., np.cumsum(increments, dtype=np.float64)]
    centered = years - years.mean()
    slope = float(np.dot(centered, path - path.mean()) / np.dot(centered, centered))
    return {
        'annual_increment_count': len(increments),
        'cumulative_state_count': len(path),
        'cumulative_change_mm': float(path[-1]),
        'mean_annual_change_mm_per_year': float(path[-1] / len(increments)),
        'cumulative_ols_slope_mm_per_year': slope,
    }


def percentage(value, denominator, threshold):
    return float(value / denominator * 100) if abs(denominator) >= threshold else ''


def summarize_period(series, years, name, start, end):
    selected = (years > start) & (years <= end)
    if not np.array_equal(years[selected], np.arange(start + 1, end + 1)):
        raise ValueError('Period is not covered: {}-{}'.format(start, end))
    reference = fit_cumulative(series[('global', 'Global', '0-6000m')][selected], start)
    rows = []
    for (kind, region, band), values in series.items():
        metrics = fit_cumulative(values[selected], start)
        rows.append({
            'period': name, 'endpoint_start_year': start, 'endpoint_end_year': end,
            'first_annual_target_year': start + 1, 'last_annual_target_year': end,
            'component_type': kind, 'region': region, 'depth_band': band,
            'is_auxiliary_overlapping_depth_band': band == '0-2000m',
            **metrics,
            'global_cumulative_change_mm': reference['cumulative_change_mm'],
            'global_cumulative_ols_slope_mm_per_year': reference['cumulative_ols_slope_mm_per_year'],
            'cumulative_contribution_pct': percentage(metrics['cumulative_change_mm'], reference['cumulative_change_mm'], 1.0),
            'ols_trend_contribution_pct': percentage(metrics['cumulative_ols_slope_mm_per_year'],
                                                   reference['cumulative_ols_slope_mm_per_year'], 0.05),
            'statistical_inference': 'descriptive_only_no_uncertainty_or_significance',
        })
    return rows


def period_differences(rows):
    lookup = {(r['period'], r['component_type'], r['region'], r['depth_band']): r for r in rows}
    components = sorted({key[1:] for key in lookup})
    output = []
    for label, early, late in COMPARISONS:
        if not any(key[0] == early for key in lookup) or not any(key[0] == late for key in lookup):
            continue
        for key in components:
            a, b = lookup[(early,) + key], lookup[(late,) + key]
            share_difference = (b['ols_trend_contribution_pct'] - a['ols_trend_contribution_pct']
                                if isinstance(a['ols_trend_contribution_pct'], float)
                                and isinstance(b['ols_trend_contribution_pct'], float) else '')
            output.append({
                'comparison': label, 'early_period': early, 'late_period': late,
                'component_type': key[0], 'region': key[1], 'depth_band': key[2],
                'early_ols_slope_mm_per_year': a['cumulative_ols_slope_mm_per_year'],
                'late_ols_slope_mm_per_year': b['cumulative_ols_slope_mm_per_year'],
                'late_minus_early_ols_slope_mm_per_year': b['cumulative_ols_slope_mm_per_year'] - a['cumulative_ols_slope_mm_per_year'],
                'late_minus_early_mean_annual_change_mm_per_year': b['mean_annual_change_mm_per_year'] - a['mean_annual_change_mm_per_year'],
                'ols_contribution_change_percentage_points': share_difference,
                'statistical_inference': 'difference_not_yet_tested_for_significance',
            })
    return output


def make_report(out, period_rows, differences, metadata):
    lines = ['# Step 1: Long-Term Thermosteric Contributions', '',
             'Descriptive statistics only. No ensemble intervals, bootstrap, p-values or significance claims.', '',
             '## Period Convention', '',
             'A period 1993-2023 uses cumulative endpoints 1993 through 2023, including a',
             'zero baseline at 1993 and 30 annual increments from 1994 versus 1993 through',
             '2023 versus 2022. A 15-year rolling window contains 15 increments and 16 endpoints.',
             '', '## Global Summary', '',
             '| Endpoint years | Cumulative (mm) | Mean rate (mm/yr) | Cumulative OLS trend (mm/yr) |',
             '|---|---:|---:|---:|']
    for r in period_rows:
        if r['component_type'] == 'global':
            lines.append('| {}-{} | {:.6f} | {:.6f} | {:.6f} |'.format(
                r['endpoint_start_year'], r['endpoint_end_year'], r['cumulative_change_mm'],
                r['mean_annual_change_mm_per_year'], r['cumulative_ols_slope_mm_per_year']))
    for selected in ('historical_full', 'recent_core', 'recent_extended'):
        lines.extend(['', '## ' + selected, '',
            '| Component | Cumulative (mm) | OLS trend (mm/yr) | Trend contribution (%) |',
            '|---|---:|---:|---:|'])
        for r in period_rows:
            if r['period'] == selected and (r['component_type'] == 'basin'
                    or (r['component_type'] == 'depth' and r['depth_band'] != '0-2000m')):
                label = r['region'] if r['component_type'] == 'basin' else r['depth_band']
                pct = '{:.3f}'.format(r['ols_trend_contribution_pct']) if r['ols_trend_contribution_pct'] != '' else 'undefined'
                lines.append('| {} | {:.6f} | {:.6f} | {} |'.format(label,
                    r['cumulative_change_mm'], r['cumulative_ols_slope_mm_per_year'], pct))
    lines.extend(['', '## Predeclared Core Period Difference', '',
        '| Component | 1993-2008 trend | 2008-2023 trend | Late minus early (mm/yr) |',
        '|---|---:|---:|---:|'])
    for r in differences:
        if r['comparison'] == 'core_adjacent_15_years' and (
                r['component_type'] in ('global', 'basin') or
                (r['component_type'] == 'depth' and r['depth_band'] != '0-2000m')):
            label = r['depth_band'] if r['component_type'] == 'depth' else r['region']
            lines.append('| {} | {:.6f} | {:.6f} | {:.6f} |'.format(label,
                r['early_ols_slope_mm_per_year'], r['late_ols_slope_mm_per_year'],
                r['late_minus_early_ols_slope_mm_per_year']))
    lines.extend(['', '## Definitions and Limits', '',
        '- Global/depth central values come from the same data.csv layer volumes used by the main figures.',
        '- Basin values use layer_global_contribution_mm, not basin-mean sea level.',
        '- All volume-to-height conversion uses a fixed 361.9e12 m2 global ocean area.',
        '- Existing depth grouping is retained: depth <= 700, 700 < depth <= 2000, 2000 < depth <= 6000.',
        '- Unassigned is the arithmetic global-minus-five-basins residual, retained without a new audit.',
        '- 0-2000m is auxiliary and overlaps the other depth components; never add it again.',
        '- The signed OLS trend fraction is component trend / global trend, not an accuracy metric.',
        '- Negative contributions and percentages above 100 are possible when components offset one another.',
        '- Trend fractions are omitted when the absolute global trend is below 0.05 mm/yr;',
        '  cumulative fractions are omitted when absolute global cumulative change is below 1 mm.',
        '- A period mean rate (net change / elapsed years) is distinct from the OLS cumulative-path trend.',
        '- Overlapping rolling windows are not independent tests. No causal attribution is made.',
        '- This step does not rerun coordinates, layer replacements, geometry or spatial closure audits.',
        '- No temperature/salinity/DEM/NetCDF inputs or prior scientific results were modified.',
        '- Input SHA-256 hashes and unchanged-source verification are recorded.', '',
        'Run settings and source definitions: metadata.json. Complete results: the CSV files.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args):
    if args.end_year <= args.start_year or args.window_years < 2:
        raise ValueError('Invalid year range or rolling-window length.')
    if args.output_dir is None:
        target = ROOT / 'contribution_structure_analysis/step1_long_term_20261014'
        index = 2
        while target.exists():
            target = target.with_name('step1_long_term_20261014_{:02d}'.format(index))
            index += 1
    else:
        target = args.output_dir
        if target.exists() and any(target.iterdir()):
            raise ValueError('Choose an empty output directory; existing results will not be overwritten.')
    target.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('long_term_contributions')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(str(target / 'processing.log'))):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    years, series, manifest = load_annual(args.input_root, args.start_year, args.end_year, log)
    annual_rows, cumulative_rows = [], []
    for (kind, region, band), values in series.items():
        total = 0.
        cumulative_rows.append({'year': args.start_year, 'component_type': kind, 'region': region,
                               'depth_band': band, 'cumulative_since_baseline_mm': 0.})
        for year, value in zip(years, values):
            total += float(value)
            annual_rows.append({'base_year': int(year - 1), 'target_year': int(year),
                'component_type': kind, 'region': region, 'depth_band': band, 'annual_change_mm': float(value)})
            cumulative_rows.append({'year': int(year), 'component_type': kind, 'region': region,
                                   'depth_band': band, 'cumulative_since_baseline_mm': total})
    write_csv(target / 'annual_components.csv', annual_rows)
    write_csv(target / 'cumulative_components.csv', cumulative_rows)
    period_rows = []
    applicable = [(name, start, end) for name, start, end in PERIODS
                  if args.start_year <= start and end <= args.end_year]
    if not applicable:
        applicable = [('requested_range', args.start_year, args.end_year)]
    for name, start, end in applicable:
        period_rows.extend(summarize_period(series, years, name, start, end))
    write_csv(target / 'long_term_contributions.csv', period_rows)
    rolling_rows = []
    for start in range(args.start_year, args.end_year - args.window_years + 1):
        end = start + args.window_years
        rows = summarize_period(series, years, 'rolling_{}yr'.format(args.window_years), start, end)
        for row in rows:
            row['window_center_year'] = (start + end) / 2
        rolling_rows.extend(rows)
    if rolling_rows:
        write_csv(target / 'rolling_trends.csv', rolling_rows)
    differences = period_differences(period_rows)
    if differences:
        write_csv(target / 'period_trend_differences.csv', differences)
    unchanged = all(sha256(Path(r['path'])) == r['sha256'] for r in manifest)
    if not unchanged:
        raise RuntimeError('An input CSV changed while statistics were running.')
    write_csv(target / 'input_manifest.csv', manifest)
    metadata = {
        'analysis': 'step_1_descriptive_long_term_contributions', 'input_root': str(args.input_root),
        'endpoint_start_year': args.start_year, 'endpoint_end_year': args.end_year,
        'annual_pair_count': len(years), 'component_count': len(series),
        'periods': applicable, 'rolling_window_elapsed_years': args.window_years,
        'rolling_window_endpoint_count': args.window_years + 1,
        'predeclared_comparisons': COMPARISONS, 'global_area_m2': AREA,
        'input_file_count': len(manifest), 'inputs_hash_verified_unchanged': unchanged,
        'trend_method': 'OLS with intercept on cumulative path including zero period-start baseline',
        'period_rate_method': 'sum annual changes / elapsed years; separate from OLS trend',
        'trend_pct_min_absolute_global_slope_mm_per_year': 0.05,
        'cumulative_pct_min_absolute_global_change_mm': 1.0,
        'statistical_inference': 'none; no confidence intervals or significance tests',
        'no_requested_audits_repeated': True, 'step_2_3_4_performed': False,
        'script_sha256': sha256(Path(__file__)),
    }
    (target / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    make_report(target, period_rows, differences, metadata)
    owner = ROOT.stat()
    for path in [target] + list(target.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    log.info('Finished: %s; %s periods; %s rolling windows; all input hashes unchanged',
             target, len(applicable), len(rolling_rows) // len(series))
    print('RESULTS_DIRECTORY=' + str(target))
    for row in period_rows:
        if row['component_type'] == 'global':
            print('{}: cumulative={:.6f} mm; mean={:.6f} mm/yr; OLS={:.6f} mm/yr'.format(
                row['period'], row['cumulative_change_mm'], row['mean_annual_change_mm_per_year'],
                row['cumulative_ols_slope_mm_per_year']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'Results/yearly_1940_2025')
    parser.add_argument('--start-year', type=int, default=1940, help='Cumulative baseline endpoint year')
    parser.add_argument('--end-year', type=int, default=2025)
    parser.add_argument('--window-years', type=int, default=15, help='Elapsed years: 15 increments, 16 endpoints')
    parser.add_argument('--output-dir', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
