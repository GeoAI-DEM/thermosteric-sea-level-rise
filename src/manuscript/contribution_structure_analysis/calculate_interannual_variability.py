#!/usr/bin/env python3
"""Step 2: detrended interannual variance and covariance decomposition."""

from manuscript_paths import workspace_root, input_path
import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np

from calculate_long_term_contributions import (
    ROOT, BASINS, EXCLUSIVE_DEPTHS, COMPARISONS, sha256, write_csv,
)


GLOBAL = ('global', 'Global', '0-6000m')
MIN_VARIANCE = 1e-12


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def load_series(path):
    records = read_csv(path)
    keyed = {}
    all_years = set()
    for row in records:
        year = int(row['target_year'])
        if int(row['base_year']) != year - 1:
            raise ValueError('Only consecutive annual increments are supported.')
        key = (row['component_type'], row['region'], row['depth_band'])
        values = keyed.setdefault(key, {})
        if year in values:
            raise ValueError('Duplicate annual component: ' + repr((year, key)))
        values[year] = float(row['annual_change_mm'])
        all_years.add(year)
    years = np.asarray(sorted(all_years), dtype=int)
    if len(years) < 3 or not np.array_equal(years, np.arange(years[0], years[-1] + 1)):
        raise ValueError('Annual increment years must be contiguous.')
    series = {}
    for key, values in keyed.items():
        if set(values) != all_years:
            raise ValueError('Missing annual values for ' + repr(key))
        series[key] = np.asarray([values[year] for year in years], dtype=np.float64)
        if not np.isfinite(series[key]).all():
            raise ValueError('Nonfinite annual component: ' + repr(key))
    return years, series


def detrend(values, years):
    values = np.asarray(values, dtype=np.float64)
    years = np.asarray(years, dtype=np.float64)
    if values.shape[-1] != len(years) or len(years) < 3:
        raise ValueError('Need at least three annual values with matching years.')
    if not np.isfinite(values).all():
        raise ValueError('Detrending requires finite values.')
    x = years - years.mean()
    denominator = np.dot(x, x)
    if denominator <= 0:
        raise ValueError('Year coordinates must vary.')
    means = values.mean(axis=-1, keepdims=True)
    slopes = np.sum((values - means) * x, axis=-1, keepdims=True) / denominator
    residuals = values - means - slopes * x
    return residuals, np.squeeze(slopes, axis=-1)


def variance_decomposition(values, reference, years):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if values.ndim != 2 or reference.shape != (values.shape[1],):
        raise ValueError('Expected component by time matrix and a reference time series.')
    residuals, slopes = detrend(values, years)
    reference_residuals, reference_slope = detrend(reference, years)
    # Check only the algebra of the new statistic, not archived spatial data.
    scale = max(1., float(np.max(np.abs(reference))))
    if not np.allclose(values.sum(axis=0), reference, atol=1e-9 * scale, rtol=1e-10):
        raise ValueError('Components must be a mutually exclusive, exhaustive sum of the reference.')
    n = len(years)
    centered = residuals - residuals.mean(axis=1, keepdims=True)
    rr = reference_residuals - reference_residuals.mean()
    covariance = centered @ centered.T / (n - 1)
    reference_variance = float(np.dot(rr, rr) / (n - 1))
    reference_covariance = centered @ rr / (n - 1)
    own_variance = np.diag(covariance)
    std = np.sqrt(np.maximum(own_variance, 0.))
    reference_std = np.sqrt(max(reference_variance, 0.))
    fractions = (reference_covariance / reference_variance
                 if reference_variance >= MIN_VARIANCE else np.full(len(values), np.nan))
    correlation = np.divide(reference_covariance, std * reference_std,
                            out=np.full(len(values), np.nan), where=std * reference_std > 0)
    pair_correlation = np.divide(covariance, std[:, None] * std[None, :],
                                out=np.full_like(covariance, np.nan),
                                where=std[:, None] * std[None, :] > 0)
    diagonal_sum = float(own_variance.sum())
    qa = {
        'residual_sum_max_error_mm': float(np.max(np.abs(residuals.sum(axis=0) - reference_residuals))),
        'covariance_sum_minus_reference_variance_mm2': float(covariance.sum() - reference_variance),
        'fraction_sum_minus_one': float(fractions.sum() - 1) if reference_variance >= MIN_VARIANCE else None,
    }
    if abs(qa['covariance_sum_minus_reference_variance_mm2']) > 1e-9 * max(1., reference_variance):
        raise RuntimeError('Covariance decomposition identity failed.')
    if qa['fraction_sum_minus_one'] is not None and abs(qa['fraction_sum_minus_one']) > 1e-8:
        raise RuntimeError('Signed variance contributions do not sum to one.')
    return {'residuals': residuals, 'reference_residuals': reference_residuals,
            'slopes': slopes, 'reference_slope': float(reference_slope),
            'raw_std': np.std(values, axis=1, ddof=1), 'reference_raw_std': float(np.std(reference, ddof=1)),
            'covariance': covariance, 'pair_correlation': np.clip(pair_correlation, -1, 1),
            'reference_variance': reference_variance, 'reference_std': reference_std,
            'reference_covariance': reference_covariance, 'std': std,
            'fractions': fractions, 'correlation': np.clip(correlation, -1, 1),
            'diagonal_sum': diagonal_sum, 'qa': qa}


def groups():
    definitions = [
        ('global_by_basin', GLOBAL, [('basin', b, '0-6000m') for b in BASINS + ('Unassigned',)]),
        ('global_by_depth', GLOBAL, [('depth', 'Global', band) for band in EXCLUSIVE_DEPTHS]),
    ]
    for basin in BASINS + ('Unassigned',):
        definitions.append((basin + '_by_depth', ('basin', basin, '0-6000m'),
                            [('basin_depth', basin, band) for band in EXCLUSIVE_DEPTHS]))
    return definitions


def printable(value):
    return float(value) if np.isfinite(value) else ''


def analyze_period(series, years, name, start, end):
    mask = (years > start) & (years <= end)
    selected_years = years[mask]
    if not np.array_equal(selected_years, np.arange(start + 1, end + 1)):
        raise ValueError('Requested period has missing annual values.')
    common = {'period': name, 'endpoint_start_year': start, 'endpoint_end_year': end,
              'first_annual_target_year': start + 1, 'last_annual_target_year': end,
              'annual_increment_count': len(selected_years)}
    contributions, pairs, summaries, qa_rows = [], [], [], []
    for group, reference_key, component_keys in groups():
        data = np.stack([series[key][mask] for key in component_keys])
        result = variance_decomposition(data, series[reference_key][mask], selected_years)
        base = dict(common, decomposition=group, reference_region=reference_key[1],
                    reference_depth_band=reference_key[2])
        vv = result['reference_variance']
        diagonal = result['diagonal_sum']
        summaries.append(dict(base,
            reference_raw_std_mm=result['reference_raw_std'],
            reference_detrended_std_mm=result['reference_std'],
            reference_detrended_variance_mm2=vv,
            sum_component_variances_mm2=diagonal,
            twice_sum_cross_covariances_mm2=float(result['covariance'].sum() - diagonal),
            net_variance_to_diagonal_sum=vv / diagonal if diagonal >= MIN_VARIANCE else '',
            net_cancellation_pct_of_diagonal_sum=(diagonal - vv) / diagonal * 100 if diagonal >= MIN_VARIANCE else '',
            signed_contributions_sum_pct=float(result['fractions'].sum() * 100) if vv >= MIN_VARIANCE else '',
            variance_near_zero=vv < MIN_VARIANCE,
            statistical_inference='descriptive_only'))
        qa_rows.append(dict(base, **result['qa']))
        for i, (kind, region, band) in enumerate(component_keys):
            contributions.append(dict(base,
                component_type=kind, region=region, depth_band=band,
                raw_std_mm=float(result['raw_std'][i]),
                detrended_std_mm=float(result['std'][i]),
                component_variance_mm2=float(result['covariance'][i, i]),
                covariance_with_reference_mm2=float(result['reference_covariance'][i]),
                variance_contribution_pct=printable(result['fractions'][i] * 100),
                correlation_with_reference=printable(result['correlation'][i]),
                removed_linear_slope_of_annual_change_mm_per_calendar_year=float(result['slopes'][i]),
                reference_detrended_std_mm=result['reference_std'],
                reference_detrended_variance_mm2=vv,
                statistical_inference='descriptive_only'))
            for j in range(i, len(component_keys)):
                other = component_keys[j]
                factor = 1 if i == j else 2
                term = float(result['covariance'][i, j] * factor)
                pairs.append(dict(base,
                    component_a='|'.join(component_keys[i]), component_b='|'.join(other),
                    is_diagonal=i == j, covariance_mm2=float(result['covariance'][i, j]),
                    correlation=printable(result['pair_correlation'][i, j]),
                    term_in_total_variance_mm2=term,
                    term_contribution_to_reference_variance_pct=term / vv * 100 if vv >= MIN_VARIANCE else '',
                    statistical_inference='descriptive_only'))
    return contributions, pairs, summaries, qa_rows


def compare_periods(rows):
    table = {(r['period'], r['decomposition'], r['component_type'], r['region'], r['depth_band']): r for r in rows}
    output = []
    for label, early, late in COMPARISONS:
        for key, a in table.items():
            if key[0] != early:
                continue
            b = table.get((late,) + key[1:])
            if b is None:
                continue
            fraction_difference = (b['variance_contribution_pct'] - a['variance_contribution_pct']
                                   if a['variance_contribution_pct'] != '' and b['variance_contribution_pct'] != '' else '')
            output.append({
                'comparison': label, 'early_period': early, 'late_period': late,
                'decomposition': a['decomposition'], 'component_type': a['component_type'],
                'region': a['region'], 'depth_band': a['depth_band'],
                'early_detrended_std_mm': a['detrended_std_mm'], 'late_detrended_std_mm': b['detrended_std_mm'],
                'late_minus_early_detrended_std_mm': b['detrended_std_mm'] - a['detrended_std_mm'],
                'early_variance_contribution_pct': a['variance_contribution_pct'],
                'late_variance_contribution_pct': b['variance_contribution_pct'],
                'late_minus_early_contribution_percentage_points': fraction_difference,
                'statistical_inference': 'difference_not_yet_tested_for_significance'})
    return output


def contrast_long_term(rows, long_term):
    table = {(r['period'], r['component_type'], r['region'], r['depth_band']): r for r in long_term}
    output = []
    for r in rows:
        if r['decomposition'] not in ('global_by_basin', 'global_by_depth'):
            continue
        key = (r['period'], r['component_type'], r['region'], r['depth_band'])
        if key not in table:
            continue
        earlier = table[key]
        output.append({
            'period': r['period'], 'component_type': r['component_type'],
            'region': r['region'], 'depth_band': r['depth_band'],
            'cumulative_change_mm': float(earlier['cumulative_change_mm']),
            'long_term_ols_slope_mm_per_year': float(earlier['cumulative_ols_slope_mm_per_year']),
            'long_term_ols_trend_contribution_pct': earlier['ols_trend_contribution_pct'],
            'interannual_detrended_std_mm': r['detrended_std_mm'],
            'interannual_variance_contribution_pct': r['variance_contribution_pct'],
            'statistical_inference': 'descriptive_only_no_causal_attribution'})
    return output


def residual_records(series, years, periods):
    rows = []
    relevant = {GLOBAL}
    for group, reference, components in groups():
        relevant.add(reference)
        relevant.update(components)
    for name, start, end in periods:
        mask = (years > start) & (years <= end)
        for key in sorted(relevant):
            residual, slope = detrend(series[key][mask], years[mask])
            for year, value, res in zip(years[mask], series[key][mask], residual):
                rows.append({'period': name, 'endpoint_start_year': start, 'endpoint_end_year': end,
                    'target_year': int(year), 'component_type': key[0], 'region': key[1], 'depth_band': key[2],
                    'annual_change_mm': float(value), 'detrended_annual_change_mm': float(res)})
    return rows


def report(out, contributions, pairs, summaries):
    lines = ['# Step 2: Interannual Thermosteric Variability', '',
             'Descriptive annual-change statistics only; no ensemble uncertainty or significance tests.',
             '', '## Estimator', '',
             'Within each period/window, remove a fitted linear trend and intercept from every annual',
             'component and reference. Use sample covariances with denominator n-1. Signed variance',
             'contribution is Cov(component residual, reference residual) / Var(reference residual).',
             'This is not component variance divided by global variance and is not causal attribution.',
             '', '## Period Definition', '',
             'Endpoint period 1993-2023 uses annual increments 1994-2023, exactly matching Step 1.',
             'A 15-year rolling window uses 15 annual increments, not 15 cumulative endpoints.',
             'Each period/window is detrended separately; windows overlap and are not independent.', '']
    for period in ('historical_full', 'recent_core', 'core_first_15_years', 'core_last_15_years'):
        for group in ('global_by_basin', 'global_by_depth'):
            selected = [r for r in contributions if r['period'] == period and r['decomposition'] == group]
            if not selected:
                continue
            lines.extend(['## {}: {}'.format(period, group), '',
                          '| Component | Detrended SD (mm) | Global variance contribution (%) |',
                          '|---|---:|---:|'])
            for r in selected:
                label = r['region'] if group == 'global_by_basin' else r['depth_band']
                fraction = '{:.6f}'.format(r['variance_contribution_pct']) if r['variance_contribution_pct'] != '' else 'undefined'
                lines.append('| {} | {:.6f} | {} |'.format(label, r['detrended_std_mm'], fraction))
            summary = next(r for r in summaries if r['period'] == period and r['decomposition'] == group)
            lines.extend(['', 'Reference residual SD: {:.6f} mm.'.format(summary['reference_detrended_std_mm']),
                          'Net cancellation relative to sum of component variances: {}%.'.format(
                              '{:.6f}'.format(summary['net_cancellation_pct_of_diagonal_sum'])
                              if summary['net_cancellation_pct_of_diagonal_sum'] != '' else 'undefined'), ''])
    lines.extend(['## Interpretation', '',
        '- Positive signed contributions reinforce global variability; negative contributions offset it.',
        '- Contributions may exceed 100% when other components cancel them; never clip or renormalize.',
        '- A small contribution does not necessarily mean a small component SD.',
        '- Positive net-cancellation percentages mean covariance reduces the sum-of-variances benchmark.',
        '- Negative net-cancellation percentages mean net covariance amplification, not cancellation.',
        '- Five basins plus the existing Unassigned residual form the global basin decomposition.',
        '- Three exclusive depth ranges form the global depth decomposition. Do not add both systems together.',
        '- Within-basin depth decompositions use each basin total as the variance denominator, NOT global variance.',
        '- Covariance CSV stores the upper triangle: diagonal terms enter once, off-diagonal terms twice.',
        '- Constant/nearly constant references have undefined percentages; entries remain blank.',
        '- SD describes physical annual-change variability, not observational uncertainty.',
        '- Long-term shares and annual variance shares are different quantities; their difference is not a test.',
        '- No climate mechanism, significant change or robustness claim is established at this step.',
        '- No DEM, temperature/salinity, NetCDF or original yearly CSVs were read or recomputed.',
        '- Only Step 1 CSVs/metadata are read; previous outputs remain unchanged.',
        '', 'See the CSV files for every period, rolling window and basin-depth combination.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        default=ROOT / 'contribution_structure_analysis/step1_long_term_20261014')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--window-years', type=int, default=15)
    args = parser.parse_args()
    if args.window_years < 3:
        parser.error('At least three annual observations are needed.')
    inputs = [args.input_dir / filename for filename in
              ('annual_components.csv', 'long_term_contributions.csv', 'metadata.json')]
    manifest = [{'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in inputs]
    previous = json.loads(inputs[2].read_text(encoding='utf-8'))
    years, series = load_series(inputs[0])
    periods = [(name, int(start), int(end)) for name, start, end in previous['periods']]
    if args.output_dir is None:
        out = ROOT / 'contribution_structure_analysis/step2_interannual_20261014'
        number = 2
        while out.exists():
            out = out.with_name('step2_interannual_20261014_{:02d}'.format(number))
            number += 1
    else:
        out = args.output_dir
        if out.exists() and any(out.iterdir()):
            raise ValueError('Output directory must be empty; existing results are never overwritten.')
    out.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('interannual_variability')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(str(out / 'processing.log'))):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    contributions, pairs, summaries, qa = [], [], [], []
    for name, start, end in periods:
        a, b, c, checks = analyze_period(series, years, name, start, end)
        contributions.extend(a)
        pairs.extend(b)
        summaries.extend(c)
        qa.extend(checks)
        log.info('Period %s (%s-%s) complete: five basins, global depth, within-basin depth', name, start, end)
    write_csv(out / 'variability_contributions.csv', contributions)
    write_csv(out / 'component_covariance.csv', pairs)
    write_csv(out / 'decomposition_summary.csv', summaries)
    write_csv(out / 'detrended_annual_series.csv', residual_records(series, years, periods))
    differences = compare_periods(contributions)
    if differences:
        write_csv(out / 'period_variability_differences.csv', differences)
    write_csv(out / 'long_term_vs_variability.csv', contrast_long_term(contributions, read_csv(inputs[1])))
    rolling_contributions, rolling_pairs, rolling_summaries = [], [], []
    count = 0
    for start in range(int(years[0] - 1), int(years[-1] - args.window_years + 1)):
        end = start + args.window_years
        a, b, c, checks = analyze_period(series, years, 'rolling_{}yr'.format(args.window_years), start, end)
        for row in a + b + c:
            row['window_center_year'] = (start + end) / 2
        rolling_contributions.extend(a)
        rolling_pairs.extend(b)
        rolling_summaries.extend(c)
        qa.extend(checks)
        count += 1
    if rolling_contributions:
        write_csv(out / 'rolling_variability.csv', rolling_contributions)
        write_csv(out / 'rolling_covariance.csv', rolling_pairs)
        write_csv(out / 'rolling_decomposition_summary.csv', rolling_summaries)
    report(out, contributions, pairs, summaries)
    if not all(sha256(Path(item['path'])) == item['sha256'] for item in manifest):
        raise RuntimeError('A Step 1 input file changed during this run.')
    write_csv(out / 'input_manifest.csv', manifest)
    write_csv(out / 'statistic_identity_checks.csv', qa)
    max_fraction_error = max(abs(row['fraction_sum_minus_one']) for row in qa if row['fraction_sum_minus_one'] is not None)
    metadata = {
        'analysis': 'step_2_descriptive_interannual_variability', 'input_dir': str(args.input_dir),
        'input_hashes_verified_unchanged': True, 'periods': periods, 'period_count': len(periods),
        'rolling_window_count': count, 'rolling_elapsed_years': args.window_years,
        'detrending': 'OLS intercept and linear time slope fitted separately in every period/window',
        'variance_covariance_denominator': 'n-1', 'sd_units': 'mm of annual change, not uncertainty',
        'contribution_definition': 'Cov(component residual, reference residual)/Var(reference residual)',
        'minimum_reference_variance_mm2': MIN_VARIANCE,
        'group_definitions': groups(), 'max_fraction_sum_minus_one_abs': max_fraction_error,
        'max_covariance_sum_error_mm2': max(abs(r['covariance_sum_minus_reference_variance_mm2']) for r in qa),
        'statistic_identity_check_count': len(qa),
        'spatial_coordinate_depth_audits_repeated': False,
        'ensemble_uncertainty_or_significance_computed': False,
        'script_sha256': sha256(Path(__file__)),
        'shared_step1_module_sha256': sha256(Path(__file__).with_name('calculate_long_term_contributions.py')),
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    owner = ROOT.stat()
    for path in [out] + list(out.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    log.info('Finished %s; %s periods; %s rolling windows; max fraction sum error %.3g',
             out, len(periods), count, max_fraction_error)
    print('RESULTS_DIRECTORY=' + str(out))
    for row in contributions:
        if row['period'] == 'recent_core' and row['decomposition'] in ('global_by_basin', 'global_by_depth'):
            print('{} {} {}: SD={:.6f} mm; variance contribution={}%'.format(
                row['decomposition'], row['region'], row['depth_band'], row['detrended_std_mm'], row['variance_contribution_pct']))


if __name__ == '__main__':
    main()
