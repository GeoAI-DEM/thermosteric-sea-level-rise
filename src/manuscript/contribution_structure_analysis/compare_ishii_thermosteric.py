#!/usr/bin/env python3
"""Compare existing main thermosteric series with official Ishii v7.3.1 text."""
from manuscript_paths import workspace_root, input_path
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

import compare_noaa_thermosteric as common
import inspect_ishii_units as units


FILES = {'0-700m': 'Thermosteric_0-700.txt',
         '700-2000m': 'Thermosteric_700-2000.txt',
         '0-2000m': 'Thermosteric_0-2000.txt'}


def parse_ishii(path):
    values = np.loadtxt(str(path), dtype=np.float64, ndmin=2)
    if values.shape[1] != 3 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError('Expected three finite columns: year, anomaly, undocumented third column')
    years = values[:, 0].astype(int)
    if not np.array_equal(years, values[:, 0]):
        raise ValueError('Expected integer calendar years; do not round unknown timestamps')
    if not np.array_equal(years, np.arange(years.min(), years.max() + 1)):
        raise ValueError('Years must be unique, ascending and contiguous')
    if np.any(np.abs(values[:, 1:]) >= 999):
        raise ValueError('Missing-value sentinel or implausible unverified scale')
    return {'years': years, 'anomaly_mm': values[:, 1], 'third_column_raw': values[:, 2]}


def validate_depth_totals(references):
    upper, middle, total = [references[band] for band in ('0-700m', '700-2000m', '0-2000m')]
    if not np.array_equal(upper['years'], total['years']) or not np.array_equal(middle['years'], total['years']):
        raise ValueError('Depth series must have identical years')
    residual = upper['anomaly_mm'] + middle['anomaly_mm'] - total['anomaly_mm']
    if not np.allclose(residual, 0, atol=.0002, rtol=0):
        raise ValueError('Published depth totals do not sum within text rounding tolerance')
    return float(np.max(np.abs(residual)))


def validate_unit_checks(checks):
    # This tests the m-to-mm scale against same-source fields, not agreement to main/NOAA.
    for band in ('0-700m', '0-2000m'):
        check = checks[band]
        ratio = check['text_change_to_grid_change_ratio']
        if not 900 < ratio < 1100:
            raise ValueError('Text-to-grid change does not support mm units for ' + band)
    return 'mm supported by same-source annual-change scale check; not explicit text metadata'


def compare(state_years, primary, reference, start, end):
    required = np.arange(start, end + 1)
    if not np.isin(required, state_years).all() or not np.isin(required, reference['years']).all():
        raise ValueError('Period not fully covered by both products')
    main = primary[np.searchsorted(state_years, required)]
    external = reference['anomaly_mm'][np.searchsorted(reference['years'], required)]
    result = {key.replace('noaa', 'ishii'): value
              for key, value in common.metrics(required, main, external).items()}
    return required, main, external, result


def report(out, rows, unit_checks, max_residual):
    lines = ['# Ishii v7.3.1 Thermosteric Cross-Product Comparison', '',
             'Preliminary nominal-band comparison, not a fully harmonized accuracy test.', '',
             '## Matched Trends and Annual Variability', '',
             '| Depth | Endpoint years | Main OLS mm/yr | Ishii OLS mm/yr | Main minus Ishii | Detrended annual-change r | Annual-change RMSE mm |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append('| {} | {}-{} | {:.6f} | {:.6f} | {:.6f} | {} | {:.6f} |'.format(
            row['depth_band'], row['endpoint_start_year'], row['endpoint_end_year'],
            row['main_ols_trend_mm_per_year'], row['ishii_ols_trend_mm_per_year'],
            row['main_minus_ishii_trend_mm_per_year'],
            '{:.4f}'.format(row['detrended_annual_change_correlation'])
            if row['detrended_annual_change_correlation'] != '' else 'undefined',
            row['annual_change_rmse_mm']))
    lines.extend(['', '## Unit and File Checks', '',
                  '- Official latest directory examined: ' + units.BASE,
                  '- All three published annual text series cover 1955-2025, with integer calendar years.',
                  '- Text has no header. Columns 2 and 3 are preserved in raw snapshots.',
                  '- Official GRIB ctl specifies parameter 131 as thermosteric height in meters.',
                  '- GDAL does not recognize the local parameter table; ctl, not its guessed labels, defines the variable.',
                  '- Unit inference uses first two years of small official monthly GRIB prefixes, not main-series agreement.',
                  '- Annual DIFFERENCES are used because grid/text reference levels may differ.',
                  '- Text is treated as mm: its year-to-year changes are approximately 1000 times grid changes in m.',
                  '- Spatial averaging is not exactly reproduced; checks below do NOT certify identical masks or reference definitions.'])
    for band, check in unit_checks.items():
        lines.append('- {}: text 1956-1955 = {:.8f}; grid change = {:.10f} m; ratio = {:.6f}.'.format(
            band, check['annual_text_change'], check['annual_grid_change_m'],
            check['text_change_to_grid_change_ratio']))
    lines.extend([
        '- The third text column is retained as undocumented, NOT labeled 1-sigma, standard error or a confidence interval.',
        '- Published upper+middle minus total maximum absolute residual = {:.8f} mm (text rounding).'.format(max_residual),
        '- Middle band comes from the published file; its sum relationship does not create an independent third product.', '',
        '## Scope and Interpretation', '',
        '- Main inputs are unchanged adjacent-year 15-arcsecond-derived global integrals from Step 1.',
        '- Main anomalies are their cumulative sums, not a new direct endpoint calculation.',
        '- Main depth grouping retains its existing representative-depth conventions.',
        '- Ishii spatial mask, global normalization area, fixed/reference salinity treatment and EOS are not fully harmonized here.',
        '- Official grid has 360x180 one-degree cells, but that alone does not establish identical ocean coverage.',
        '- No new 15-arcsecond calculation, coordinate audit, bathymetry processing or mask changes.',
        '- Both curves anchored to first common year; anchored bias depends on that baseline.',
        '- Demeaned anomaly RMSE and detrended/annual correlations separate offset and common-trend effects.',
        '- No fitting, scaling or year selection is performed to maximize agreement.',
        '- Trends and RMSE are descriptive; no equivalence/significance claim or combined uncertainty is made.',
        '- Source observations can be shared across products, so this is not independent ground truth.',
        '- Global upper-ocean agreement cannot validate 2000-6000m or individual basin contributions.',
        '- First comparison includes historical sparse-observation years; modern and recent periods are also retained.', '',
        '## Reproducibility', '',
        '- Raw text, official documentation and small complete-message GRIB prefixes: raw_ishii/.',
        '- Hashes and retrieval records: downloads.json and metadata.json.',
        '- Shared alignment/statistics routines: compare_noaa_thermosteric.py (hash recorded).',
        '- Full metrics: cross_product_metrics.csv. Aligned annual data: matched_anomalies_and_annual_changes.csv.',
        '- NOAA/Ishii/main table on exactly shared periods: main_noaa_ishii_same_period.csv.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = common.step1.ROOT / 'contribution_structure_analysis'
    parser.add_argument('--step1-dir', type=Path, default=root / 'step1_long_term_20261014')
    parser.add_argument('--output-dir', type=Path, default=root / 'step4_ishii_comparison_20261014')
    parser.add_argument('--noaa-dir', type=Path, default=root / 'step4_noaa_comparison_20261014')
    args = parser.parse_args()
    out = args.output_dir
    if (out / 'metadata.json').exists() or (out / 'cross_product_metrics.csv').exists():
        raise ValueError('Completed comparison exists; choose a different output directory, no overwrite')
    out.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('ishii_comparison')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(str(out / 'processing.log'))):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    input_path = args.step1_dir / 'annual_components.csv'
    original_hash = common.step1.sha256(input_path)
    checks = units.prepare(out)
    unit_status = validate_unit_checks(checks)
    references = {band: parse_ishii(out / 'raw_ishii' / filename) for band, filename in FILES.items()}
    closure = validate_depth_totals(references)
    years, series = common.step2.load_series(input_path)
    state_years = np.r_[years[0] - 1, years]
    normalized, aligned, rows, joint = [], [], [], []
    noaa = {}
    noaa_hashes = {}
    for band, filename in common.FILES.items():
        path = args.noaa_dir / 'raw_noaa' / filename
        if path.exists():
            noaa[band] = common.parse_noaa(path)
            noaa_hashes[str(path)] = common.step1.sha256(path)
    if '0-700m' in noaa and '0-2000m' in noaa:
        noaa['700-2000m'] = common.derived_middle(noaa['0-700m'], noaa['0-2000m'])
    for band, reference in references.items():
        primary = np.r_[0., np.cumsum(series[('depth', 'Global', band)])]
        for index, year in enumerate(reference['years']):
            normalized.append({'year': int(year), 'depth_band': band,
                               'thermosteric_anomaly_mm': float(reference['anomaly_mm'][index]),
                               'third_column_raw_definition_unconfirmed': float(reference['third_column_raw'][index]),
                               'unit_status': unit_status, 'source': FILES[band]})
        for period, start, end in common.PERIODS:
            required = np.arange(start, end + 1)
            if not np.isin(required, reference['years']).all():
                continue
            required, m, r, result = compare(state_years, primary, reference, start, end)
            row = dict(period=period, depth_band=band, endpoint_start_year=start,
                       endpoint_end_year=end, **result)
            rows.append(row)
            ma, ra = m - m[0], r - r[0]
            for k, year in enumerate(required):
                aligned.append({'period': period, 'depth_band': band, 'year': int(year),
                                'main_anomaly_anchored_mm': float(ma[k]),
                                'ishii_anomaly_anchored_mm': float(ra[k]),
                                'main_annual_change_mm': float(m[k] - m[k - 1]) if k else '',
                                'ishii_annual_change_mm': float(r[k] - r[k - 1]) if k else ''})
            log.info('%s %d-%d main %.6f, Ishii %.6f mm/yr; detrended annual r=%s; RMSE %.6f mm',
                     band, start, end, result['main_ols_trend_mm_per_year'],
                     result['ishii_ols_trend_mm_per_year'], result['detrended_annual_change_correlation'],
                     result['annual_change_rmse_mm'])
            if band in noaa and np.isin(required, noaa[band]['years']).all():
                n = noaa[band]['anomaly_mm'][np.searchsorted(noaa[band]['years'], required)]
                nm = common.metrics(required, m, n)
                joint.append({'period': period, 'depth_band': band, 'start_year': start, 'end_year': end,
                              'main_trend_mm_per_year': result['main_ols_trend_mm_per_year'],
                              'ishii_trend_mm_per_year': result['ishii_ols_trend_mm_per_year'],
                              'noaa_trend_mm_per_year': nm['noaa_ols_trend_mm_per_year'],
                              'main_ishii_detrended_annual_r': result['detrended_annual_change_correlation'],
                              'main_noaa_detrended_annual_r': nm['detrended_annual_change_correlation'],
                              'main_ishii_annual_rmse_mm': result['annual_change_rmse_mm'],
                              'main_noaa_annual_rmse_mm': nm['annual_change_rmse_mm']})
    for name, data in [('ishii_normalized_series.csv', normalized),
                       ('matched_anomalies_and_annual_changes.csv', aligned),
                       ('cross_product_metrics.csv', rows), ('main_noaa_ishii_same_period.csv', joint)]:
        if data:
            common.step1.write_csv(out / name, data)
    if common.step1.sha256(input_path) != original_hash:
        raise RuntimeError('Main input changed during comparison')
    if any(common.step1.sha256(Path(path)) != digest for path, digest in noaa_hashes.items()):
        raise RuntimeError('NOAA snapshots changed during comparison')
    metadata = {'analysis': 'step4_ishii_v7.3.1_preliminary_global_consistency',
                'input': str(input_path), 'input_sha256': original_hash, 'main_input_unchanged': True,
                'source_base_url': units.BASE, 'external_version': '7.3.1', 'unit_status': unit_status,
                'unit_checks': checks, 'third_column_definition': 'unconfirmed_not_used_for_uncertainty',
                'external_depth_sum_max_residual_mm': closure,
                'comparison_count': len(rows), 'noaa_common_comparison_count': len(joint),
                'noaa_inputs_unchanged': True, 'noaa_input_hashes': noaa_hashes,
                'source_depth_years': {band: [int(r['years'][0]), int(r['years'][-1])] for band, r in references.items()},
                'full_spatial_mask_and_area_harmonization': False,
                'new_15arcsec_calculation': False, 'statistical_significance_test': False,
                'script_hashes': {p.name: common.step1.sha256(p) for p in
                                  [Path(__file__), Path(units.__file__), Path(common.__file__)]}}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    report(out, rows, checks, closure)
    owner = common.step1.ROOT.stat()
    for path in [out] + list(out.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    log.info('Complete: %s; original inputs unchanged', out)
    print('RESULTS_DIRECTORY=' + str(out))


if __name__ == '__main__':
    main()
