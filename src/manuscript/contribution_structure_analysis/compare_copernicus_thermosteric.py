#!/usr/bin/env python3
"""Latitude-matched descriptive comparison with native Copernicus thermosteric OMI."""
from manuscript_paths import workspace_root, input_path
import argparse
import csv
import json
import logging
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np

import compare_noaa_thermosteric as common
import prepare_copernicus_reference as source


ROOT = common.step1.ROOT
SERIES = {
    'thsl_cora': ('CORA5.2', 'observation_based_reconstruction'),
    'thsl_armor': ('ARMOR-3D', 'observation_based_reconstruction'),
    'thsl_glor': ('GLORYS2V4', 'reanalysis'),
    'thsl_cglo': ('C-GLORS', 'reanalysis'),
    'thsl_oras': ('ORAS5', 'reanalysis'),
    'thsl_mean_observations': ('Observation Ensemble Mean', 'published_observation_ensemble_mean'),
    'thsl_mean_reanalysis': ('Reanalysis Ensemble Mean', 'published_reanalysis_ensemble_mean'),
    'thsl_bretherton': ('von Schuckmann / Bretherton', 'auxiliary_ohc_converted_reference'),
}
SPREADS = {
    'thsl_std_observations': 'published_interproduct_standard_deviation_not_main_measurement_error',
    'thsl_std_reanalysis': 'published_interproduct_standard_deviation_not_main_measurement_error',
    'thsl_bretherton_error': 'auxiliary_supplied_error_confidence_definition_not_assumed',
}
GROUPS = {
    'thsl_mean_observations': ('thsl_cora', 'thsl_armor'),
    'thsl_mean_reanalysis': ('thsl_glor', 'thsl_cglo', 'thsl_oras'),
}


def unit_factor(unit):
    if unit == 'm':
        return 1000.
    if unit == 'mm':
        return 1.
    raise ValueError('Unsupported or missing height unit: ' + repr(unit))


def json_value(value):
    if isinstance(value, np.ndarray):
        return [json_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def attribute_dict(obj):
    return {key: json_value(getattr(obj, key)) for key in obj.ncattrs()}


def finite_vector(variable, length):
    if variable.dimensions != ('time',) or variable.shape != (length,):
        raise ValueError('Expected one annual value per time for ' + variable.name)
    values = np.asarray(np.ma.filled(variable[:], np.nan), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(np.abs(values) >= 1e15):
        raise ValueError('Missing or invalid reference values: ' + variable.name)
    return values


def parse_reference(path):
    with nc.Dataset(str(path), 'r') as dataset:
        time = dataset.variables['time']
        calendar = getattr(time, 'calendar', 'standard')
        raw_times = np.asarray(time[:], dtype=np.float64)
        if raw_times.ndim != 1 or not np.isfinite(raw_times).all():
            raise ValueError('Expected finite annual time coordinates')
        dates = nc.num2date(raw_times, time.units, calendar=calendar)
        years = np.array([date.year for date in dates], dtype=int)
        if len(years) < 4 or not np.array_equal(years, np.arange(years.min(), years.max() + 1)):
            raise ValueError('Reference must have one ordered state per calendar year without gaps')
        metadata = {'global_attributes': attribute_dict(dataset), 'time_attributes': attribute_dict(time),
                    'native_timestamps': [date.isoformat() for date in dates],
                    'native_time_values': raw_times.tolist(), 'variable_attributes': {}}
        values, spreads = {}, {}
        for name in SERIES:
            variable = dataset.variables[name]
            if getattr(variable, 'standard_name', '') != 'thermosteric_change_in_mean_sea_level':
                raise ValueError('Unexpected declared physical variable: ' + name)
            metadata['variable_attributes'][name] = attribute_dict(variable)
            values[name] = finite_vector(variable, len(years)) * unit_factor(getattr(variable, 'units', None))
        for name in SPREADS:
            variable = dataset.variables[name]
            metadata['variable_attributes'][name] = attribute_dict(variable)
            raw = finite_vector(variable, len(years))
            if np.any(raw < 0):
                raise ValueError('Negative supplied spread or error: ' + name)
            unit = getattr(variable, 'units', None)
            if unit is None and name == 'thsl_bretherton_error':
                spreads[name] = {'raw_values': raw, 'raw_units': None, 'values_mm': None,
                                 'definition': SPREADS[name],
                                 'note': 'Native file omits units; preserve raw values, do not use in uncertainty arithmetic'}
            else:
                spreads[name] = {'raw_values': raw, 'raw_units': unit,
                                 'values_mm': raw * unit_factor(unit), 'definition': SPREADS[name]}
    return {'years': years, 'values_mm': values, 'spreads': spreads, 'metadata': metadata}


def verify_published_means(reference):
    result = {}
    for mean, members in GROUPS.items():
        calculated = np.mean([reference['values_mm'][name] for name in members], axis=0)
        difference = float(np.max(np.abs(calculated - reference['values_mm'][mean])))
        if difference > 1e-5:
            raise ValueError('Published mean does not match advertised members: ' + mean)
        result[mean] = difference
    return result


def derive_middle(upper, total):
    if not np.array_equal(upper['years'], total['years']):
        raise ValueError('Depth products must have identical calendar years for paired subtraction')
    if upper['metadata']['native_timestamps'] != total['metadata']['native_timestamps']:
        raise ValueError('Depth products must have matching native annual timestamps')
    return {'years': upper['years'].copy(),
            'values_mm': {name: total['values_mm'][name] - upper['values_mm'][name] for name in SERIES},
            'spreads': {}, 'metadata': {'derived_from': 'same_named_total_minus_upper',
                                      'native_timestamps': upper['metadata']['native_timestamps'],
                                      'uncertainty': 'not_derived_without_cross_depth_covariance'}}


def regional_area(path):
    with nc.Dataset(str(path), 'r') as dataset:
        latitude = np.asarray(dataset.variables['lat'][:], dtype=np.float64)
        variable = dataset.variables['ocean_area_m2']
        if variable.dimensions != ('lat', 'lon') or variable.units != 'm2':
            raise ValueError('Expected cached latitude/longitude ocean areas in m2')
        area = np.asarray(variable[:], dtype=np.float64)
        if not np.isfinite(area).all() or np.any(area < 0):
            raise ValueError('Invalid ocean area cache')
        selected = (latitude >= -60.) & (latitude < 60.)
        # Integer latitude boundaries coincide with edges of the existing 1-degree area cells.
        value = float(area[selected, :].sum(dtype=np.float64))
        if value <= 0:
            raise ValueError('Empty regional area')
        method = {key: json_value(getattr(dataset, key)) for key in
                  ('source_dem', 'ocean_mask', 'earth_radius_m', 'area_formula')}
    return value, method


def load_main(path, area):
    bands = {'0-700m': {}, '700-2000m': {}}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['latitude_band'] != '60S-60N' or row['depth_band'] not in bands:
                continue
            year, base = int(row['target_year']), int(row['base_year'])
            values = bands[row['depth_band']]
            if year in values or base != year - 1:
                raise ValueError('Duplicate main year or non-adjacent increment')
            if float(row['lat_lower_deg']) != -60. or float(row['lat_upper_deg']) != 60.:
                raise ValueError('Wrong main latitude bounds')
            value = float(row['volume_change_km3'])
            if not np.isfinite(value):
                raise ValueError('Nonfinite main volume')
            values[year] = value
    years = np.array(sorted(bands['0-700m']), dtype=int)
    if len(years) < 4 or not np.array_equal(years, np.arange(years.min(), years.max() + 1)):
        raise ValueError('Main annual increments have gaps')
    if sorted(bands['700-2000m']) != years.tolist():
        raise ValueError('Main depth bands must have identical years')
    volumes = {key: np.array([values[year] for year in years]) for key, values in bands.items()}
    volumes['0-2000m'] = volumes['0-700m'] + volumes['700-2000m']
    return {'increment_years': years, 'state_years': np.r_[years[0] - 1, years], 'volumes_km3': volumes,
            'annual_mm': {band: values * 1e12 / area for band, values in volumes.items()},
            'anomalies_mm': {band: np.r_[0., np.cumsum(values * 1e12 / area)] for band, values in volumes.items()}}


def compare_series(years, main, external):
    result = {key.replace('noaa', 'copernicus'): value
              for key, value in common.metrics(years, main, external).items()}
    result['comparison_scope'] = 'latitude_matched_descriptive_product_consistency_not_truth_validation'
    return result


def display_r(value):
    return '{:.4f}'.format(value) if value != '' else 'undefined'


def make_report(out, rows, area, versions, mean_checks):
    start, end = rows[0]['start_year'], rows[0]['end_year']
    lines = ['# Copernicus Thermosteric OMI: Latitude-Matched Comparison', '',
             'Period: {}-{}. Latitude: 60S-60N. Units: mm and mm/year.'.format(start, end), '',
             '## Published Ensemble Means', '',
             '| Depth | Reference | Main trend | External trend | Main minus external | Detrended annual-change r | Annual-change RMSE mm |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['reference_variable'] not in GROUPS:
            continue
        lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} | {} | {:.6f} |'.format(
            row['depth_band'], row['reference_label'], row['main_ols_trend_mm_per_year'],
            row['copernicus_ols_trend_mm_per_year'], row['main_minus_copernicus_trend_mm_per_year'],
            display_r(row['detrended_annual_change_correlation']), row['annual_change_rmse_mm']))
    lines.extend(['', '## Individual Members and Auxiliary Reference', '',
                  '| Depth | Reference | Type | External trend | Detrended annual-change r | Annual-change RMSE mm |',
                  '|---|---|---|---:|---:|---:|'])
    for row in rows:
        if row['reference_variable'] in GROUPS:
            continue
        lines.append('| {} | {} | {} | {:.6f} | {} | {:.6f} |'.format(
            row['depth_band'], row['reference_label'], row['reference_class'],
            row['copernicus_ols_trend_mm_per_year'], display_r(row['detrended_annual_change_correlation']),
            row['annual_change_rmse_mm']))
    lines.extend(['', '## Definitions and Limits', '',
        '- Credit: E.U. Copernicus Marine Service Information.',
        '- Official products: https://doi.org/10.48670/moi-00239 and https://doi.org/10.48670/moi-00240.',
        '- Product and dataset STAC snapshots, native files, PUM and QUID are archived in raw_copernicus/.',
        '- Versions: ' + json.dumps(versions, sort_keys=True),
        '- Native NetCDF calendar timestamps are June 30 annual labels, not instantaneous June observations.',
        '- STAC timestamps use January 1; native NetCDF CF dates determine comparison years.',
        '- No three-year smoothing is applied; these are annual means, unlike the Domingues comparison.',
        '- Height units explicitly read from each value variable: m converted to mm with factor 1000.',
        '- Main inputs are previously summed native 15-arcsecond 60S-60N map volumes, not new TEOS-10 calculations.',
        '- Main regional mean = regional volume change / fixed regional DEM wet area.',
        '- Fixed regional area = {:.12e} m2 from the existing native-DEM-derived 1-degree area cache.'.format(area),
        '- Original global_contribution_mm is NOT used as a regional average.',
        '- Main 0-2000m annual volume is the sum of the existing nonoverlapping upper and middle bands.',
        '- Main anomalies are cumulative adjacent-year increments, not a new direct reference-year calculation.',
        '- Both curves are anchored to their first shared year; anchored bias depends on that choice.',
        '- Demeaned anomaly RMSE, detrended anomaly r and detrended annual-change r are retained separately.',
        '- Published observation mean contains CORA5.2 and ARMOR-3D; reanalysis mean contains GLORYS2V4, C-GLORS and ORAS5.',
        '- Individual members and their means are NOT independent validation samples; raw observations can also be shared with IAP.',
        '- Published means checked against their named members, maximum residuals in mm: ' + json.dumps(mean_checks, sort_keys=True),
        '- Regional latitude limits match, but ocean masks, native meshes, shallow-cell treatment and exact area normalization do not fully match.',
        '- Official product is declared thermosteric; native long names mention temperature AND salinity. Source T/S decomposition and EOS were not reconstructed here.',
        '- Main layer definitions and salinity/EOS treatment are unchanged; this is not strict same-algorithm validation.',
        '- Middle-band external anomaly = SAME member 0-2000m minus 0-700m; no new independent third depth product.',
        '- Do not subtract supplied standard deviations or combine them in quadrature: cross-depth covariance is unavailable.',
        '- Group standard deviations represent interproduct spread, not main measurement uncertainty or confidence bounds.',
        '- Auxiliary thsl_bretherton is identified as a time series by PUM/QUID and standard_name, although native long_name incorrectly starts with Error.',
        '- QUID describes this auxiliary as OHC-converted thermosteric height; it is separately classified and is not an independent CORA product.',
        '- Native thsl_bretherton_error omits units; preserve its raw values without conversion or uncertainty arithmetic.',
        '- No significance, equivalence, or 2000-6000m accuracy claim. No rescaling or period selection to maximize agreement.', '',
        '## Files', '',
        '- cross_product_metrics.csv: all member/mean/depth descriptive metrics.',
        '- main_60S_60N_annual.csv: main regional annual changes and native-volume sums.',
        '- copernicus_normalized_series.csv: all height series in mm.',
        '- copernicus_supplied_spreads.csv: separately preserved native spread/error columns.',
        '- matched_anomalies_and_annual_changes.csv: baseline-matched states and annual differences.',
        '- reference_variable_metadata.json: original variable attributes and calendar coordinates.',
        '- input_manifest.csv, downloads.json, metadata.json: source hashes and provenance.', '',
        'All original main results, code, DEM, temperature/salinity and prior comparisons are unchanged.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args):
    out = args.output_dir
    if (out / 'metadata.json').exists() or (out / 'cross_product_metrics.csv').exists():
        raise ValueError('Completed comparison exists; choose another output directory, no overwrite')
    out.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('copernicus_comparison')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(str(out / 'processing.log'))):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    files, versions = source.prepare(out)
    paths = [args.main_csv, args.area_cache] + list(files.values())
    before = {str(p): common.step1.sha256(p) for p in paths}
    area, area_method = regional_area(args.area_cache)
    main = load_main(args.main_csv, area)
    references = {band: parse_reference(path) for band, path in files.items()}
    mean_checks = {band: verify_published_means(ref) for band, ref in references.items()}
    references['700-2000m'] = derive_middle(references['0-700m'], references['0-2000m'])
    required = np.arange(args.start_year, args.end_year + 1)
    if len(required) < 4 or not np.isin(required, main['state_years']).all():
        raise ValueError('Need at least four fully covered annual states')
    rows, normalized, spread_rows, aligned, main_rows = [], [], [], [], []
    for band in ('0-700m', '700-2000m', '0-2000m'):
        reference = references[band]
        if not np.isin(required, reference['years']).all():
            raise ValueError('Reference does not fully cover specified comparison period')
        m = main['anomalies_mm'][band][np.searchsorted(main['state_years'], required)]
        for name, (label, family) in SERIES.items():
            values = reference['values_mm'][name]
            for j, year in enumerate(reference['years']):
                normalized.append({'year': int(year), 'depth_band': band, 'reference_variable': name,
                                   'reference_label': label, 'reference_class': family,
                                   'anomaly_mm': float(values[j]),
                                   'derived_middle': band == '700-2000m'})
            r = values[np.searchsorted(reference['years'], required)]
            result = compare_series(required, m, r)
            result.update(start_year=args.start_year, end_year=args.end_year, depth_band=band,
                          latitude_range='60S-60N', reference_variable=name,
                          reference_label=label, reference_class=family,
                          exact_product_mask_match=False, primary_comparison=family != 'auxiliary_ohc_converted_reference')
            rows.append(result)
            for j, year in enumerate(required):
                aligned.append({'year': int(year), 'depth_band': band, 'reference_variable': name,
                                'main_anomaly_anchored_mm': float(m[j] - m[0]),
                                'copernicus_anomaly_anchored_mm': float(r[j] - r[0]),
                                'main_annual_change_mm': float(m[j] - m[j - 1]) if j else '',
                                'copernicus_annual_change_mm': float(r[j] - r[j - 1]) if j else ''})
            log.info('%s %s: main trend %.6f, external %.6f mm/yr; detrended annual r=%s; RMSE %.6f mm',
                     band, label, result['main_ols_trend_mm_per_year'], result['copernicus_ols_trend_mm_per_year'],
                     result['detrended_annual_change_correlation'], result['annual_change_rmse_mm'])
        for name, spread in reference['spreads'].items():
            for j, year in enumerate(reference['years']):
                spread_rows.append({'year': int(year), 'depth_band': band, 'reference_variable': name,
                                    'raw_value': float(spread['raw_values'][j]), 'raw_units': spread['raw_units'] or '',
                                    'value_mm': float(spread['values_mm'][j]) if spread['values_mm'] is not None else '',
                                    'definition': spread['definition'], 'used_for_main_uncertainty': False})
        for j, year in enumerate(main['increment_years']):
            main_rows.append({'base_year': int(year) - 1, 'target_year': int(year), 'depth_band': band,
                              'latitude_range': '60S-60N', 'volume_change_km3': float(main['volumes_km3'][band][j]),
                              'regional_ocean_area_m2': area, 'regional_mean_change_mm': float(main['annual_mm'][band][j]),
                              'global_contribution_mm': float(main['volumes_km3'][band][j] / 361.9)})
    for name, records in [('cross_product_metrics.csv', rows), ('copernicus_normalized_series.csv', normalized),
                           ('copernicus_supplied_spreads.csv', spread_rows), ('main_60S_60N_annual.csv', main_rows),
                           ('matched_anomalies_and_annual_changes.csv', aligned)]:
        common.step1.write_csv(out / name, records)
    if any(common.step1.sha256(Path(path)) != digest for path, digest in before.items()):
        raise RuntimeError('Input changed during comparison')
    manifest = [{'path': path, 'sha256': digest, 'unchanged_after_comparison': True} for path, digest in before.items()]
    common.step1.write_csv(out / 'input_manifest.csv', manifest)
    (out / 'reference_variable_metadata.json').write_text(
        json.dumps({band: ref['metadata'] for band, ref in references.items()}, indent=2), encoding='utf-8')
    metadata = {'analysis': 'step4_copernicus_latitude_matched_thermosteric_consistency',
                'versions': versions, 'start_year': args.start_year, 'end_year': args.end_year,
                'comparison_count': len(rows), 'primary_comparison_count': sum(r['primary_comparison'] for r in rows),
                'latitude_range': [-60., 60.], 'regional_ocean_area_m2': area, 'area_method': area_method,
                'main_input': str(args.main_csv), 'original_input_hashes': before, 'all_inputs_unchanged': True,
                'published_mean_residuals_mm': mean_checks, 'new_15arcsec_thermodynamic_calculation': False,
                'new_large_netcdf_reads': False, 'main_normalization': 'regional_volume_divided_by_regional_DEM_ocean_area',
                'height_unit_conversion': 'native m to mm factor 1000; native spread without units retained raw',
                'reference_time_source': 'native_CF_calendar_not_STAC_timestamp', 'smoothing': 'none',
                'exact_spatial_mask_and_eos_harmonization': False, 'statistical_significance_test': False,
                'source_credit': 'E.U. Copernicus Marine Service Information',
                'script_sha256': {p.name: common.step1.sha256(p) for p in
                                  [Path(__file__), Path(source.__file__), Path(common.__file__)]}}
    make_report(out, rows, area, versions, mean_checks)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    log.info('Finished %s; %d comparisons; originals unchanged', out, len(rows))
    print('RESULTS_DIRECTORY=' + str(out))


def main_cli():
    root = ROOT / 'contribution_structure_analysis'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=root / 'step4_copernicus_comparison_20261015')
    parser.add_argument('--main-csv', type=Path, default=ROOT / 'figures/nested_latitude_15arcsec/Fig04_nested_latitude_15arcsec_annual_main.csv')
    parser.add_argument('--area-cache', type=Path, default=ROOT / 'uncertainty_preprocessing/gebco_1deg_layer_volumes/gebco_1deg_layer_volume_0_6000m.nc')
    parser.add_argument('--start-year', type=int, default=2005)
    parser.add_argument('--end-year', type=int, default=2023)
    args = parser.parse_args()
    try:
        run(args)
    finally:
        if args.output_dir.exists():
            owner = ROOT.stat()
            for path in [args.output_dir] + list(args.output_dir.rglob('*')):
                os.chown(str(path), owner.st_uid, owner.st_gid)
                path.chmod(0o755 if path.is_dir() else 0o644)


if __name__ == '__main__':
    main_cli()
