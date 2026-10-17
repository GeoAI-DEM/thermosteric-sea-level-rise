#!/usr/bin/env python3
"""Step 4 pilot: global main-series versus NOAA thermosteric anomalies."""

from manuscript_paths import workspace_root, input_path
import argparse
import csv
import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import calculate_long_term_contributions as step1
import calculate_interannual_variability as step2


PAGE = 'https://www.ncei.noaa.gov/access/global-ocean-heat-content/basin_tsl_data.html'
BASE = 'https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/3M_HEAT_CONTENT/DATA/basin/yearly_sl/'
FILES = {'0-700m': 'a-mm-w0-700m.dat', '0-2000m': 'a-mm-w0-2000m.dat'}
PERIODS = (('long_overlap', 1955, 2025), ('long_through_2023', 1955, 2023),
           ('recent_core', 1993, 2023), ('modern_common', 2005, 2023),
           ('modern_extended', 2005, 2025))


def fetch(url, destination, log):
    error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'ThermostericCrossProductCheck/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
                headers = dict(response.headers.items())
            if not payload:
                raise ValueError('Empty download.')
            destination.write_bytes(payload)
            return {'url': url, 'local_file': str(destination), 'bytes': len(payload),
                    'sha256': step1.sha256(destination), 'response_headers': headers,
                    'retrieved_utc': datetime.now(timezone.utc).isoformat()}
        except Exception as exc:
            error = exc
            log.warning('Download attempt %s/3 failed for %s: %s', attempt + 1, url, exc)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError('Official NOAA download failed: {}: {}'.format(url, error))


def parse_noaa(path):
    values = np.genfromtxt(str(path), names=True, dtype=np.float64, encoding='ascii')
    expected = ('YEAR', 'WO', 'WOse', 'NH', 'NHse', 'SH', 'SHse')
    if values.dtype.names != expected:
        raise ValueError('Unexpected NOAA header or nondata response: ' + str(path))
    values = np.atleast_1d(values)
    fractional = values['YEAR']
    years = np.floor(fractional).astype(int)
    if not np.allclose(fractional - years, .5, atol=1e-8):
        raise ValueError('NOAA annual timestamps must be year+0.5; refusing ambiguous dates.')
    if len(set(years)) != len(years) or not np.array_equal(years, np.arange(years.min(), years.max() + 1)):
        raise ValueError('NOAA annual values must be unique and contiguous.')
    if not np.isfinite(values['WO']).all() or np.any(np.abs(values['WO']) >= 999):
        raise ValueError('Missing/invalid global NOAA anomalies.')
    return {'years': years, 'anomaly_mm': values['WO'], 'standard_error_mm': values['WOse'],
            'time_decimal_year': fractional}


def correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if denominator <= 1e-14:
        return ''
    return float(np.clip(np.dot(a, b) / denominator, -1, 1))


def slope(values, years):
    x = years - years.mean()
    return float(np.dot(x, values - values.mean()) / np.dot(x, x))


def metrics(years, main, external):
    years, main, external = [np.asarray(v, dtype=np.float64) for v in (years, main, external)]
    if len(years) < 4 or not (len(main) == len(external) == len(years)):
        raise ValueError('Need at least four matched annual anomaly states.')
    if not np.array_equal(np.diff(years), np.ones(len(years) - 1)):
        raise ValueError('Annual states must be consecutive; do not difference across gaps.')
    main_anchor, ext_anchor = main - main[0], external - external[0]
    difference = main_anchor - ext_anchor
    centered_difference = (main - main.mean()) - (external - external.mean())
    main_residual, _ = step2.detrend(main, years)
    ext_residual, _ = step2.detrend(external, years)
    a, b = np.diff(main), np.diff(external)
    ar, _ = step2.detrend(a, years[1:])
    br, _ = step2.detrend(b, years[1:])
    main_trend, ext_trend = slope(main, years), slope(external, years)
    return {
        'anomaly_state_count': len(years), 'annual_change_count': len(years) - 1,
        'anchor_year': int(years[0]),
        'main_ols_trend_mm_per_year': main_trend,
        'noaa_ols_trend_mm_per_year': ext_trend,
        'main_minus_noaa_trend_mm_per_year': main_trend - ext_trend,
        'main_endpoint_mean_rate_mm_per_year': float((main[-1] - main[0]) / (years[-1] - years[0])),
        'noaa_endpoint_mean_rate_mm_per_year': float((external[-1] - external[0]) / (years[-1] - years[0])),
        'main_endpoint_change_mm': float(main_anchor[-1]), 'noaa_endpoint_change_mm': float(ext_anchor[-1]),
        'anchored_anomaly_bias_mm': float(difference.mean()),
        'anchored_anomaly_rmse_mm': float(np.sqrt(np.mean(difference**2))),
        'demeaned_anomaly_rmse_mm': float(np.sqrt(np.mean(centered_difference**2))),
        'raw_anomaly_correlation': correlation(main, external),
        'detrended_anomaly_correlation': correlation(main_residual, ext_residual),
        'annual_change_bias_mm': float((a - b).mean()),
        'annual_change_rmse_mm': float(np.sqrt(np.mean((a - b)**2))),
        'annual_change_correlation': correlation(a, b),
        'detrended_annual_change_correlation': correlation(ar, br),
        'main_annual_change_sd_mm': float(a.std(ddof=1)),
        'noaa_annual_change_sd_mm': float(b.std(ddof=1)),
        'main_detrended_annual_change_sd_mm': float(ar.std(ddof=1)),
        'noaa_detrended_annual_change_sd_mm': float(br.std(ddof=1)),
        'comparison_scope': 'cross_product_consistency_not_validation_against_truth',
        'depth_match': 'nominal_band_labels_not_exact_physical_layer_harmonization',
        'statistical_inference': 'descriptive_no_pvalues_or_confidence_intervals',
    }


def derived_middle(upper, total):
    years = np.intersect1d(upper['years'], total['years'])
    u = np.searchsorted(upper['years'], years)
    t = np.searchsorted(total['years'], years)
    return {'years': years, 'anomaly_mm': total['anomaly_mm'][t] - upper['anomaly_mm'][u],
            'standard_error_mm': np.full(len(years), np.nan),
            'time_decimal_year': years + .5}


def make_report(out, rows):
    lines = ['# Step 4 Pilot: NOAA Global Thermosteric Cross-Product Comparison', '',
        'This is a preliminary comparison of existing products, not a pass/fail accuracy test.', '',
        '## Modern Common Period: 2005-2023', '',
        '| Depth | Main OLS (mm/yr) | NOAA OLS (mm/yr) | Main minus NOAA | Detrended annual-change r | Annual-change RMSE (mm) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['period'] != 'modern_common':
            continue
        r = '{:.4f}'.format(row['detrended_annual_change_correlation']) if row['detrended_annual_change_correlation'] != '' else 'undefined'
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} | {} | {:.6f} |'.format(row['depth_band'],
            row['main_ols_trend_mm_per_year'], row['noaa_ols_trend_mm_per_year'],
            row['main_minus_noaa_trend_mm_per_year'], r, row['annual_change_rmse_mm']))
    lines.extend(['', '## All Available Comparison Periods', '',
        '| Depth | Endpoint years | Main OLS | NOAA OLS | Detrended anomaly r | Detrended annual-change r |',
        '|---|---|---:|---:|---:|---:|'])
    for row in rows:
        r1 = '{:.4f}'.format(row['detrended_anomaly_correlation']) if row['detrended_anomaly_correlation'] != '' else 'undefined'
        r2 = '{:.4f}'.format(row['detrended_annual_change_correlation']) if row['detrended_annual_change_correlation'] != '' else 'undefined'
        lines.append('| {} | {}-{} | {:.6f} | {:.6f} | {} | {} |'.format(row['depth_band'],
            row['endpoint_start_year'], row['endpoint_end_year'], row['main_ols_trend_mm_per_year'],
            row['noaa_ols_trend_mm_per_year'], r1, r2))
    lines.extend(['', '## Source and Definitions', '',
        '- Official NOAA documentation: ' + PAGE,
        '- NOAA YEAR=year+0.5 labels annual means; it is floored to the calendar year, never rounded.',
        '- NOAA WO is global thermosteric sea-level anomaly in mm; WOse is its supplied standard error.',
        '- Main anomalies are accumulated adjacent-year increments from Step 1, not a new direct endpoint rerun.',
        '- Both anomaly curves are anchored to zero at the first shared year of each period.',
        '- Anchored bias depends on that anchor. Demeaned RMSE is also reported as an offset-independent measure.',
        '- Mean-centered bias is deliberately NOT reported as validation: it would be zero by construction.',
        '- Trend rates fit the same annual anomaly states. Annual changes compare their consecutive differences.',
        '- Raw anomaly correlation can be inflated by common warming; detrended/annual metrics are reported separately.',
        '- 700-2000m NOAA anomalies are derived as 0-2000 minus 0-700, not a third independent product.',
        '- No derived middle-layer error bar is inferred: covariance between NOAA depth products is unavailable.',
        '- NOAA standard errors are retained, but not combined with main errors without temporal/product covariance.',
        '- Main depth groups retain existing representative-depth conventions; NOAA physical integration boundaries',
        '  and spatial coverage/EOS definitions have NOT been made identical. This is a nominal-band pilot.',
        '- No data are rescaled or selected to maximize agreement. No significance or equivalence test is claimed.',
        '- Shared source observations are possible; NOAA is not independent ground truth.',
        '- Basin comparisons are not performed: NOAA Atlantic includes Arctic and is incompatible with our five-basin masks.',
        '- No inference about 2000-6000m accuracy follows from upper-ocean agreement.',
        '- Original main results, temperature/salinity and DEM are unchanged. No new 15-arcsecond calculation.', '',
        'Raw official snapshots and download provenance are in raw_noaa/. Complete metrics: cross_product_metrics.csv.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step1-dir', type=Path, default=step1.ROOT / 'contribution_structure_analysis/step1_long_term_20261014')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    input_path = args.step1_dir / 'annual_components.csv'
    input_hash = step1.sha256(input_path)
    years, series = step2.load_series(input_path)
    state_years = np.r_[years[0] - 1, years]
    if args.output_dir is None:
        out = step1.ROOT / 'contribution_structure_analysis/step4_noaa_comparison_20261014'
        number = 2
        while out.exists():
            out = out.with_name('step4_noaa_comparison_20261014_{:02d}'.format(number))
            number += 1
    else:
        out = args.output_dir
        if out.exists() and any(out.iterdir()):
            raise ValueError('Output directory must be empty; no overwrite.')
    raw_dir = out / 'raw_noaa'
    raw_dir.mkdir(parents=True)
    log = logging.getLogger('noaa_comparison')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(str(out / 'processing.log'))):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    provenance, references = [], {}
    for band, filename in FILES.items():
        log.info('Downloading official NOAA global %s yearly anomalies', band)
        provenance.append(fetch(BASE + filename, raw_dir / filename, log))
        references[band] = parse_noaa(raw_dir / filename)
    provenance.append(fetch(PAGE, raw_dir / 'noaa_product_documentation.html', log))
    references['700-2000m'] = derived_middle(references['0-700m'], references['0-2000m'])
    normalized, aligned_rows, rows = [], [], []
    for band in ('0-700m', '0-2000m', '700-2000m'):
        reference = references[band]
        primary = np.r_[0., np.cumsum(series[('depth', 'Global', band)])]
        for j, year in enumerate(reference['years']):
            normalized.append({'year': int(year), 'noaa_decimal_year': float(reference['time_decimal_year'][j]),
                'depth_band': band, 'thermosteric_anomaly_mm': float(reference['anomaly_mm'][j]),
                'supplied_standard_error_mm': float(reference['standard_error_mm'][j])
                    if np.isfinite(reference['standard_error_mm'][j]) else '',
                'source': 'derived_difference_no_error_covariance' if band == '700-2000m' else FILES[band]})
        for period, start, end in PERIODS:
            required = np.arange(start, end + 1)
            if not np.isin(required, state_years).all() or not np.isin(required, reference['years']).all():
                continue
            main_index = np.searchsorted(state_years, required)
            ref_index = np.searchsorted(reference['years'], required)
            m, r = primary[main_index], reference['anomaly_mm'][ref_index]
            result = metrics(required, m, r)
            row = {'period': period, 'depth_band': band, 'endpoint_start_year': start, 'endpoint_end_year': end, **result}
            rows.append(row)
            ma, ra = m - m[0], r - r[0]
            mr, _ = step2.detrend(m, required)
            rr, _ = step2.detrend(r, required)
            for k, year in enumerate(required):
                aligned_rows.append({'period': period, 'depth_band': band, 'year': int(year),
                    'main_anomaly_anchored_mm': float(ma[k]), 'noaa_anomaly_anchored_mm': float(ra[k]),
                    'main_minus_noaa_anchored_mm': float(ma[k] - ra[k]),
                    'main_detrended_anomaly_mm': float(mr[k]), 'noaa_detrended_anomaly_mm': float(rr[k]),
                    'main_annual_change_mm': float(m[k] - m[k - 1]) if k else '',
                    'noaa_annual_change_mm': float(r[k] - r[k - 1]) if k else '',
                    'noaa_supplied_anomaly_standard_error_mm': float(reference['standard_error_mm'][ref_index[k]])
                        if np.isfinite(reference['standard_error_mm'][ref_index[k]]) else ''})
            log.info('%s %s-%s: main trend %.6f, NOAA %.6f mm/yr; annual detrended r=%s',
                     band, start, end, result['main_ols_trend_mm_per_year'], result['noaa_ols_trend_mm_per_year'],
                     result['detrended_annual_change_correlation'])
    step1.write_csv(out / 'noaa_normalized_series.csv', normalized)
    step1.write_csv(out / 'matched_anomalies_and_annual_changes.csv', aligned_rows)
    step1.write_csv(out / 'cross_product_metrics.csv', rows)
    if step1.sha256(input_path) != input_hash:
        raise RuntimeError('The main input changed during comparison.')
    metadata = {'analysis': 'step_4_preliminary_global_cross_product_consistency',
        'input': str(input_path), 'input_sha256': input_hash, 'main_input_unchanged': True,
        'downloads': provenance, 'units': 'mm thermosteric sea-level anomaly',
        'noaa_time_interpretation': 'annual year+0.5 converted by floor to calendar year',
        'anomaly_alignment': 'shared first-year anchor; centered RMSE also retained',
        'comparison_periods': PERIODS, 'comparison_count': len(rows),
        'noaa_depth_coverage': {band: [int(value['years'][0]), int(value['years'][-1])] for band, value in references.items()},
        'depth_and_mask_harmonization': 'nominal labels only; not identical physical integration masks or layer bounds',
        'noaa_basin_comparison': 'not performed; NOAA Atlantic includes Arctic',
        'uncertainty_or_equivalence_test': 'not performed; missing cross-product/temporal covariance',
        'noaa_is_independent_truth': False, 'new_15arcsec_calculations': False,
        'script_sha256': step1.sha256(Path(__file__))}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    make_report(out, rows)
    owner = step1.ROOT.stat()
    for path in [out] + list(out.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    log.info('Finished %s; %s comparisons; main input unchanged', out, len(rows))
    print('RESULTS_DIRECTORY=' + str(out))


if __name__ == '__main__':
    main()
