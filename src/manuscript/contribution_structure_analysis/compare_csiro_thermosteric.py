#!/usr/bin/env python3
"""Latitude-match existing 15-arcsecond maps to the Domingues 2026 v3 series."""
from manuscript_paths import workspace_root, input_path
import argparse
import csv
import io
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import netCDF4 as nc
import numpy as np

import compare_noaa_thermosteric as common
import prepare_csiro_reference as source


ROOT = common.step1.ROOT
VARIABLE = 'thermal_expansion_visual_filled_km3'
PERIODS = [('historical_classic', 1961, 2003), ('long_overlap', 1961, 2023),
           ('recent_core', 1993, 2023), ('modern_common', 2005, 2023),
           ('interior_extended', 1961, 2024)]


def parse_csiro(path):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    if 'Units: GOHC [J 1e22], GThSL [mm]' not in lines:
        raise ValueError('Unrecognized reference units')
    header = next((i for i, line in enumerate(lines) if line.startswith('Variables:')), None)
    if header is None:
        raise ValueError('Missing reference column description')
    expected = ['Year', 'GOHC_0-300m', 'GOHC_0-300m_error', 'GThSL_0-300m',
                'GThSL_0-300m_error', 'OHC_0-700m', 'GOHC_0-700m_error',
                'GThSL_0-700m', 'GThSL_0-700m_error']
    names = lines[header].split('[', 1)[1].rsplit(']', 1)[0].split()
    if names != expected:
        raise ValueError('Unexpected reference column order')
    values = np.loadtxt(io.StringIO('\n'.join(lines[header + 1:])), ndmin=2)
    if values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError('Reference must contain nine finite numeric columns')
    years = np.floor(values[:, 0]).astype(int)
    if not np.allclose(values[:, 0] - years, .5, atol=1e-8):
        raise ValueError('Reference timestamps must be year+0.5')
    if not np.array_equal(years, np.arange(years.min(), years.max() + 1)):
        raise ValueError('Reference years must be unique, ordered and consecutive')
    if np.any(values[:, [4, 8]] < 0) or np.any(np.abs(values[:, 1:]) >= 999):
        raise ValueError('Invalid uncertainty or missing-value sentinel')
    return {'years': years, 'anomaly_mm': values[:, 7], 'one_sigma_mm': values[:, 8],
            'anomaly_0_300_mm': values[:, 3], 'one_sigma_0_300_mm': values[:, 4]}


def smooth_three(years, values, mode='centered'):
    years, values = np.asarray(years), np.asarray(values, dtype=np.float64)
    if len(years) < 3 or len(values) != len(years) or not np.isfinite(values).all():
        raise ValueError('Need at least three finite annual states')
    if not np.array_equal(np.diff(years), np.ones(len(years) - 1)):
        raise ValueError('Cannot smooth across missing years')
    smoothed = np.convolve(values, np.ones(3) / 3., mode='valid')
    if mode == 'centered':
        return years[1:-1], smoothed
    if mode == 'trailing':
        return years[2:], smoothed
    raise ValueError('Unknown smoothing convention')


def signature(path):
    stat = path.stat()
    return {'path': str(path), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def latitude_total(latitudes, row_totals, lower=-65., upper=65.):
    selected = (np.asarray(latitudes) >= lower) & (np.asarray(latitudes) < upper)
    return float(np.asarray(row_totals, dtype=np.float64)[selected].sum(dtype=np.float64))


def load_core_volumes(path):
    result = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['latitude_band'] != '60S-60N' or row['depth_band'] != '0-700m':
                continue
            year = int(row['target_year'])
            if year in result or int(row['base_year']) != year - 1:
                raise ValueError('Duplicate core year or non-adjacent pair')
            if float(row['lat_lower_deg']) != -60 or float(row['lat_upper_deg']) != 60:
                raise ValueError('Wrong core latitude range')
            value = float(row['volume_change_km3'])
            if not np.isfinite(value):
                raise ValueError('Invalid core volume')
            result[year] = value
    if not all(year in result for year in range(1960, 2026)):
        raise ValueError('Core aggregation must cover every required year')
    return result


def aggregate_one(year, results_root, cache_dir, block_rows, core_volume=None, core_hash=None):
    path = results_root / '{0}_vs_{1}/{0}_700.nc'.format(year, year - 1)
    initial = signature(path)
    cache = cache_dir / '{}.json'.format(year)
    if cache.exists():
        record = json.loads(cache.read_text())
        if record.get('input_signature') != initial or record.get('bounds') != [-65., 65.]:
            raise ValueError('Cached source changed, refusing reuse: ' + str(cache))
        if 'core_source_sha256' in record and record['core_source_sha256'] != core_hash:
            raise ValueError('Cached core aggregation changed')
        return record, True
    volume = core_volume if core_volume is not None else 0.
    intervals = [(-65., -60.), (60., 65.)] if core_volume is not None else [(-65., 65.)]
    with nc.Dataset(str(path), 'r') as dataset:
        if int(dataset.base_year) != year - 1 or int(dataset.target_year) != year:
            raise ValueError('Unexpected annual pair in ' + str(path))
        if float(dataset.depth_limit_m) != 700:
            raise ValueError('Expected existing 700m output')
        lat = np.asarray(dataset.variables['lat'][:], dtype=np.float64)
        var = dataset.variables[VARIABLE]
        if var.dimensions != ('lat', 'lon'):
            raise ValueError('Expected two-dimensional lat/lon field')
        indices = np.flatnonzero((lat >= -65.) & (lat < 65.))
        if not len(indices):
            raise ValueError('No rows in requested latitude range')
        if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
            raise ValueError('Expected contiguous latitude subset')
        block_starts = set()
        for lower, upper in intervals:
            selected = np.flatnonzero((lat >= lower) & (lat < upper))
            block_starts.update(range(int(selected[0]) // block_rows * block_rows,
                                      int(selected[-1]) + 1, block_rows))
        for start in sorted(block_starts):
            stop = min(start + block_rows, var.shape[0])
            block = var[start:stop, :]
            field = np.asarray(np.ma.filled(block, 0.), dtype=np.float32)
            if np.isinf(field).any():
                raise ValueError('Infinite main data in ' + str(path))
            field[np.isnan(field)] = 0.
            row_totals = field.sum(axis=1, dtype=np.float64)
            volume += sum(latitude_total(lat[start:stop], row_totals, lower, upper)
                          for lower, upper in intervals)
    if signature(path) != initial:
        raise RuntimeError('Source map changed while reading ' + str(path))
    record = {'base_year': year - 1, 'target_year': year, 'depth_band': '0-700m',
              'bounds': [-65., 65.], 'volume_change_km3': volume,
              'input_signature': initial, 'input_unchanged': True,
              'latitude_rows': len(indices), 'aggregation': 'sum_original_cells_no_recalculation_or_extra_mask'}
    if core_volume is not None:
        record.update(core_60S_60N_volume_km3=core_volume, core_source_sha256=core_hash,
                      aggregation='reuse_exact_60S_60N_sum_plus_native_60_to_65_degree_strips')
    temporary = cache.with_suffix('.part')
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(cache)
    return record, False


def ocean_area(path):
    with nc.Dataset(str(path)) as dataset:
        lat = np.asarray(dataset.variables['lat'][:], dtype=np.float64)
        area = np.asarray(dataset.variables['ocean_area_m2'][:], dtype=np.float64)
        if dataset.variables['ocean_area_m2'].units != 'm2':
            raise ValueError('Unexpected area units')
        if not np.isfinite(area).all() or np.any(area < 0):
            raise ValueError('Invalid cached ocean area')
        total = latitude_total(lat, area.sum(axis=1, dtype=np.float64))
        return total, {key: getattr(dataset, key) for key in
                       ('source_dem', 'ocean_mask', 'earth_radius_m', 'area_formula')}


def compare_period(years, main, reference, period, start, end, convention):
    required = np.arange(start, end + 1)
    if not np.isin(required, years).all() or not np.isin(required, reference['years']).all():
        raise ValueError('Comparison period missing annual states')
    m = main[np.searchsorted(years, required)]
    r = reference['anomaly_mm'][np.searchsorted(reference['years'], required)]
    result = {key.replace('noaa', 'csiro').replace('annual_change', 'smoothed_state_difference'):
              value for key, value in common.metrics(required, m, r).items()}
    result.update({'period': period, 'start_year': start, 'end_year': end,
                   'depth_band': '0-700m', 'latitude_range': '65S-65N',
                   'main_smoothing': convention + '_three_year_complete_windows',
                   'external_smoothing': 'published_three_year_direction_and_endpoints_not_explicit',
                   'comparison_scope': 'latitude_matched_preliminary_product_consistency',
                   'exact_product_ocean_mask_match': False})
    return required, m, r, result


def make_report(out, rows, area, references):
    lines = ['# Domingues/CSIRO 2026 v3: Latitude-Matched Thermosteric Comparison', '',
             'DOI: https://doi.org/' + source.DOI, '',
             '## Primary Comparison: Centered Three-Year Mean', '',
             '| Endpoint years | Main trend mm/yr | Domingues trend mm/yr | Main minus external | Detrended anomaly r | Smoothed-difference r | Smoothed-difference RMSE mm |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['main_smoothing'].startswith('centered'):
            lines.append('| {}-{} | {:.6f} | {:.6f} | {:.6f} | {:.4f} | {:.4f} | {:.6f} |'.format(
                row['start_year'], row['end_year'], row['main_ols_trend_mm_per_year'],
                row['csiro_ols_trend_mm_per_year'], row['main_minus_csiro_trend_mm_per_year'],
                row['detrended_anomaly_correlation'], row['detrended_smoothed_state_difference_correlation'],
                row['smoothed_state_difference_rmse_mm']))
    lines.extend(['', '## Matching and Remaining Limitations', '',
        '- Published series covers 1960-2025, upper 700m, 65S-65N, units mm, with three-year smoothing.',
        '- Official error estimates are identified as one sigma by the archived metadata.',
        '- Main input: existing annual native-resolution 700m NetCDF maps, read only and summed in float64.',
        '- Reuse existing exact 60S-60N map sums; read only the [-65,-60) and [60,65) native-resolution strips.',
        '- This shortcut is checked against a complete 65S-65N read for 1960, without interpolating latitude bands.',
        '- Only cell centers in [-65,65) are summed; no interpolation from 60/70-degree bands.',
        '- No extra spatial filtering or thermodynamic recalculation is applied to main maps.',
        '- Near-global mean is regional volume change / fixed regional wet ocean area, not global contribution.',
        '- Fixed GEBCO wet ocean area for 65S-65N: {:.12e} m2.'.format(area),
        '- Area cache is the existing sum of native 15-arcsecond DEM pixel areas into 1-degree cells.',
        '- Both latitude limits lie on 1-degree cell edges, so no latitude clipping approximation is needed for that area sum.',
        '- Source reconstruction ocean mask, marginal-sea exclusions and exact ocean-area normalization are NOT fully reproduced.',
        '- Published metadata says three-year smoothing, but does not specify direction or endpoint treatment.',
        '- Primary assumption: equal-weight centered 3-year mean of cumulative main ANOMALY STATES.',
        '- Exclude published endpoint states 1960 and 2025; main centered windows have all three required years.',
        '- Trailing three-year means are a prespecified sensitivity, not a correlation-maximizing choice.',
        '- Same-period direction sensitivity: 1993-2023 and 2005-2023; retain both without picking best.',
        '- Differences between neighboring smoothed states are NOT unsmoothed interannual changes.',
        '- Correlations/RMSE cannot be compared directly with unsmoothed NOAA/Ishii annual statistics.',
        '- Keep baseline-dependent anchored RMSE and offset-independent demeaned RMSE separately.',
        '- Main uncertainties and external errors are NOT combined without their covariance.',
        '- No significance, equivalence, or deep-ocean accuracy claim; observations may be shared.',
        '- Existing main layer definitions, salinity treatment and EOS are not changed or fully harmonized.',
        '- Only 0-700m is compared; published 0-300m data are retained but not matched here.',
        '- Archived metadata contains inherited inconsistencies (e.g. 14 years versus 1993-2015); not silently corrected.', '',
        '## Files', '',
        '- main_65S_65N_0_700m_annual.csv: regional annual volumes and both regional/global normalizations.',
        '- csiro_normalized_series.csv: published 700m and 300m estimates and one-sigma errors.',
        '- matched_anomalies_and_smoothed_differences.csv: baseline-matched states and smoothed differences.',
        '- cross_product_metrics.csv: complete primary and smoothing-sensitivity statistics.',
        '- annual_cache/: resumable regional sums; input file size/mtime signatures retained.',
        '- metadata.json and downloads.json: units, areas, hashes, provenance and limitations.',
        '- raw_csiro/: official archive and unmodified extracted references.', '',
        'Original main code, temperature/salinity, DEM and NetCDF files are not modified.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT / 'contribution_structure_analysis'
    parser.add_argument('--output-dir', type=Path, default=root / 'step4_csiro_comparison_20261014')
    parser.add_argument('--results-root', type=Path, default=ROOT / 'Results/yearly_1940_2025')
    parser.add_argument('--area-cache', type=Path, default=ROOT / 'uncertainty_preprocessing/gebco_1deg_layer_volumes/gebco_1deg_layer_volume_0_6000m.nc')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--core-csv', type=Path, default=ROOT / 'figures/nested_latitude_15arcsec/Fig04_nested_latitude_15arcsec_annual_main.csv')
    parser.add_argument('--block-rows', type=int, default=128)
    parser.add_argument('--aggregate-end-year', type=int, default=2025, help='For a short IO pilot; omitting years produces aggregation only, not final comparison')
    args = parser.parse_args()
    out = args.output_dir
    if (out / 'metadata.json').exists():
        raise ValueError('Completed comparison exists; choose new output directory')
    if args.workers < 1 or args.block_rows < 1 or not 1960 <= args.aggregate_end_year <= 2025:
        raise ValueError('Invalid worker/block/year configuration')
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / 'annual_cache'
    cache_dir.mkdir(exist_ok=True)
    log = logging.getLogger('csiro_comparison')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(str(out / 'processing.log'))):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    source.prepare(out)
    reference_path = out / 'raw_csiro/gohc_gthsl_1960_2025_0_300m_0_700m.txt'
    reference = parse_csiro(reference_path)
    original_hashes = {str(path): common.step1.sha256(path) for path in
                       [reference_path, args.area_cache, args.core_csv, root / 'step1_long_term_20261014/annual_components.csv']}
    core_volumes = load_core_volumes(args.core_csv)
    core_hash = original_hashes[str(args.core_csv)]
    area, area_method = ocean_area(args.area_cache)
    log.info('65S-65N fixed wet area: %.12e m2; workers=%d', area, args.workers)
    records = []
    try:
        verification_dir = out / 'shortcut_verification'
        verification_dir.mkdir(exist_ok=True)
        full, _ = aggregate_one(1960, args.results_root, cache_dir, args.block_rows)
        if 'core_source_sha256' in full:
            raise ValueError('Need complete-read pilot for shortcut verification')
        shortcut, _ = aggregate_one(1960, args.results_root, verification_dir, args.block_rows,
                                    core_volumes[1960], core_hash)
        mismatch = shortcut['volume_change_km3'] - full['volume_change_km3']
        if abs(mismatch) > max(1e-5, abs(full['volume_change_km3']) * 1e-8):
            raise ValueError('Core-plus-strips differs from full read: {}'.format(mismatch))
        verification = {'year': 1960, 'complete_read_km3': full['volume_change_km3'],
                        'core_plus_strips_km3': shortcut['volume_change_km3'], 'difference_km3': mismatch}
        (out / 'shortcut_verification.json').write_text(json.dumps(verification, indent=2))
        log.info('Complete read vs core-plus-strips check: difference %.12g km3', mismatch)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(aggregate_one, year, args.results_root, cache_dir, args.block_rows,
                                   core_volumes[year], core_hash): year
                       for year in range(1960, args.aggregate_end_year + 1)}
            for future in as_completed(futures):
                record, reused = future.result()
                records.append(record)
                log.info('%d vs %d: %.6f km3, regional %.6f mm; %s (%d/%d)',
                         record['target_year'], record['base_year'], record['volume_change_km3'],
                         record['volume_change_km3'] * 1e12 / area, 'cached' if reused else 'aggregated',
                         len(records), len(futures))
        records.sort(key=lambda r: r['target_year'])
        annual = [{'base_year': r['base_year'], 'target_year': r['target_year'],
                   'depth_band': '0-700m', 'latitude_range': '65S-65N',
                   'volume_change_km3': r['volume_change_km3'], 'regional_ocean_area_m2': area,
                   'regional_mean_change_mm': r['volume_change_km3'] * 1e12 / area,
                   'global_contribution_mm': r['volume_change_km3'] / 361.9}
                  for r in records]
        common.step1.write_csv(out / 'main_65S_65N_0_700m_annual.csv', annual)
        if args.aggregate_end_year != 2025:
            log.info('Short aggregation pilot complete; no incomplete comparison produced')
            return
        years = np.array([r['target_year'] for r in annual])
        primary = np.cumsum([r['regional_mean_change_mm'] for r in annual])
        rows, aligned = [], []
        for convention in ('centered', 'trailing'):
            smooth_years, smoothed = smooth_three(years, primary, convention)
            periods = PERIODS if convention == 'centered' else [p for p in PERIODS if p[0] in ('recent_core', 'modern_common')]
            for period, start, end in periods:
                required, m, r, row = compare_period(smooth_years, smoothed, reference, period, start, end, convention)
                rows.append(row)
                for k, year in enumerate(required):
                    aligned.append({'period': period, 'main_smoothing': convention, 'year': int(year),
                                    'main_anchored_anomaly_mm': float(m[k] - m[0]),
                                    'csiro_anchored_anomaly_mm': float(r[k] - r[0]),
                                    'main_smoothed_state_difference_mm': float(m[k] - m[k - 1]) if k else '',
                                    'csiro_smoothed_state_difference_mm': float(r[k] - r[k - 1]) if k else '',
                                    'csiro_supplied_anomaly_one_sigma_mm': float(reference['one_sigma_mm'][np.searchsorted(reference['years'], year)])})
                log.info('%s %d-%d main=%.6f external=%.6f mm/yr; detrended smoothed-difference r=%.6f',
                         convention, start, end, row['main_ols_trend_mm_per_year'], row['csiro_ols_trend_mm_per_year'],
                         row['detrended_smoothed_state_difference_correlation'])
        normalized = [{'year': int(year), 'depth_band': band,
                       'anomaly_mm': float(reference[value][i]),
                       'supplied_one_sigma_mm': float(reference[error][i])}
                      for band, value, error in [('0-700m', 'anomaly_mm', 'one_sigma_mm'),
                                                 ('0-300m', 'anomaly_0_300_mm', 'one_sigma_0_300_mm')]
                      for i, year in enumerate(reference['years'])]
        for name, data in [('cross_product_metrics.csv', rows), ('csiro_normalized_series.csv', normalized),
                           ('matched_anomalies_and_smoothed_differences.csv', aligned)]:
            common.step1.write_csv(out / name, data)
        if any(common.step1.sha256(Path(path)) != value for path, value in original_hashes.items()):
            raise RuntimeError('Source changed during comparison')
        metadata = {'source_doi': source.DOI, 'source_version': '2026 v3', 'input_hashes_unchanged': original_hashes,
                    'main_map_signatures': [r['input_signature'] for r in records],
                    'all_read_main_maps_unchanged': True, 'new_thermodynamic_calculation': False,
                    'near_global_area_m2': area, 'area_method': area_method, 'latitude_range': [-65, 65],
                    'core_plus_strips_verification': verification,
                    'exact_reconstruction_mask_match': False,
                    'main_smoothing_primary': 'centered_three_year_complete_window',
                    'main_smoothing_sensitivity': 'trailing_three_year_complete_window',
                    'external_smoothing_direction': 'not_explicit_in_downloaded_metadata',
                    'external_endpoints_excluded': [1960, 2025], 'comparison_count': len(rows),
                    'script_hashes': {p.name: common.step1.sha256(p) for p in
                                      [Path(__file__), Path(source.__file__), Path(common.__file__)]}}
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        make_report(out, rows, area, reference)
        log.info('Finished: %s; originals unchanged', out)
        print('RESULTS_DIRECTORY=' + str(out))
    finally:
        owner = ROOT.stat()
        for path in [out] + list(out.rglob('*')):
            os.chown(str(path), owner.st_uid, owner.st_gid)
            path.chmod(0o755 if path.is_dir() else 0o644)


if __name__ == '__main__':
    main()
