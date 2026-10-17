#!/usr/bin/env python3
"""Recompute IAP and EN4 on shared 1-degree wet volumes and a fixed common mask."""
from manuscript_paths import workspace_root, input_path
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import netCDF4 as nc
import numpy as np

pass  # Source imports are configured by the launcher.
import compare_noaa_thermosteric as statistics

import en4_uniform_core as core
import prepare_en4_uniform_reference as download


ROOT = workspace_root()


def read_volumes(path):
    with nc.Dataset(str(path), 'r') as dataset:
        all_depth = np.asarray(dataset.variables['depth_std'][:], dtype=np.float64)
        keep = np.flatnonzero(all_depth <= 2000.)
        volume = np.asarray(dataset.variables['layer_ocean_volume_m3'][:len(keep)], dtype=np.float64)
        thickness = np.asarray(dataset.variables['layer_thickness_m'][:len(keep)], dtype=np.float64)
        lat = np.asarray(dataset.variables['lat'][:], dtype=np.float64)
        lon = np.asarray(dataset.variables['lon'][:], dtype=np.float64)
        area = np.asarray(dataset.variables['ocean_area_m2'][:], dtype=np.float64)
        attrs = {key: str(getattr(dataset, key)) for key in dataset.ncattrs()}
    if not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError('Invalid existing volume weights')
    near_global = np.broadcast_to(((lat >= -60.) & (lat < 60.))[:, None], area.shape)
    regions = {'Global': np.ones(area.shape, dtype=bool), '60S-60N': near_global}
    denominators = {'Global': 361.9e12, '60S-60N': float(area[near_global].sum(dtype=np.float64))}
    return all_depth[keep], thickness, lat, lon, volume, regions, denominators, attrs


def sum_region_depth(values, depth, group, region):
    return float(values[core.group_mask(depth, group)][:, region].sum(dtype=np.float64))


def write_report(phase_dir, annual_rows, coverage, metrics, start, end, pilot):
    title = 'EN4 / IAP 1-degree Uniform-Formula Pilot' if pilot else 'EN4 / IAP 1-degree Uniform-Formula Comparison'
    lines = ['# ' + title, '', 'Years: {}-{}. EN4 version: 4.2.2, g10.'.format(start, end), '',
             'This is a separate cross-product sensitivity experiment, not a replacement for original 15-arcsecond results.', '',
             '## Fixed Common-Domain Coverage', '',
             '| Region | Nominal depth | Possible wet volume km3 | Common wet volume km3 | Common fraction |',
             '|---|---|---:|---:|---:|']
    for row in coverage:
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.2f}% |'.format(
            row['region'], row['depth_band'], row['possible_volume_km3'], row['common_volume_km3'],
            row['common_volume_fraction'] * 100.))
    if pilot:
        lines.extend(['', '## One Annual Change', '', '| Region | Depth | IAP mm | EN4 mm | EN4 minus IAP mm |',
                      '|---|---|---:|---:|---:|'])
        for iap in [row for row in annual_rows if row['product'] == 'IAP']:
            en4 = next(row for row in annual_rows if row['product'] == 'EN4' and row['region'] == iap['region']
                       and row['depth_band'] == iap['depth_band'] and row['target_year'] == iap['target_year'])
            lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} |'.format(
                iap['region'], iap['depth_band'], iap['sea_level_contribution_mm'], en4['sea_level_contribution_mm'],
                en4['sea_level_contribution_mm'] - iap['sea_level_contribution_mm']))
        lines.append('\nThe pilot mask uses only these two years. The final mask will intersect every annual state from 2005 through 2023.')
    else:
        lines.extend(['', '## Matched Anomaly Trends', '',
                      '| Region | Nominal depth | IAP OLS mm/yr | EN4 OLS mm/yr | IAP minus EN4 | Detrended annual r | Annual RMSE mm |',
                      '|---|---|---:|---:|---:|---:|---:|'])
        for row in metrics:
            r = row['detrended_annual_change_correlation']
            r = '{:.4f}'.format(r) if r != '' else 'undefined'
            lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} | {} | {:.6f} |'.format(
                row['region'], row['depth_band'], row['iap_ols_trend_mm_per_year'],
                row['en4_ols_trend_mm_per_year'], row['iap_minus_en4_trend_mm_per_year'], r,
                row['annual_change_rmse_mm']))
    lines.extend(['', '## Calculation Contract', '',
        '- Identical existing DEM-derived 1-degree wet-cell volumes for IAP and EN4. No new DEM aggregation.',
        '- Formula retained: dV = V * (alpha0 + alpha1)/2 * (CT1 - CT0).',
        '- Shared GSW CT_from_t and alpha calls; depth in meters used numerically as dbar, matching the original main approximation.',
        '- IAP annual temperature is read in degC; IAP salinity is retained as declared absolute salinity in g/kg.',
        '- The IAP absolute-salinity label is an input assumption inherited from the annual files, not independently established from raw profiles here.',
        '- EN4 monthly values are potential temperature and practical salinity. netCDF4 first decodes kelvin, then subtract 273.15.',
        '- EN4 SP -> SA_from_SP at native longitude/latitude and the shared depth-as-pressure convention.',
        '- EN4 potential temperature -> CT_from_pt -> t_from_CT before equal-month annual averaging.',
        '- Annual averaging is float64 sum/count over finite monthly values, identical in weighting to np.nanmean across 12 months.',
        '- No day-count weighting, no small-island filling, no nearest-depth replacement, and no smoothing for either input.',
        '- Periodic linear interpolation uses actual per-variable longitudes, with strict finite interpolation support.',
        '- Latitude is not extrapolated. EN4 top approximately 5m is held constant toward the shallowest main depth (1m); no deep extrapolation.',
        '- Vertical mapping uses native EN4 levels bracketing 2000m, not only levels shallower than 2000m.',
        '- Final common mask intersects finite T and SA for BOTH products over ALL selected years and V > 0.',
        '- A single frozen mask is used for every pair; pair-by-pair support changes cannot generate the trend.',
        '- Missing cells are not filled or replaced with zero-temperature seawater.',
        '- These are common-domain volume changes divided by a FIXED regional area, not complete-coverage global estimates.',
        '- Global normalization retains 361.9e12 m2. 60S-60N normalization uses fixed regional DEM wet area.',
        '- Nominal depth grouping follows original representative-depth bins: <=700 and >700 through <=2000.',
        '- Original layer thicknesses/volumes are retained, NOT newly clipped to exact 700/2000m boundaries.',
        '- The result remains the original average-alpha/CT-difference formulation, not a newly salinity-held-fixed density decomposition.',
        '- Main and EN4 may share original observations; this is not validation against independent truth.',
        '- EN4 observation-weight diagnostics are retained without clipping or filtering; low weights indicate background influence, not measurement certainty.',
        '- Native EN4 files have inherited global date attributes; only native CF time is used to identify each month.',
        '- No EN4 error fields are propagated here, and no 2000-6000m estimate is fabricated.',
        '- g10 was chosen before results. Other XBT correction versions have not been tested; this is not an EN4-version ensemble.', '',
        '## Saved Outputs', '',
        '- annual_thermal_expansion.csv: each adjacent-year pair, product, region and depth band.',
        '- layer_thermal_expansion.csv: layer-by-layer contributions.',
        '- common_domain_coverage.csv and fixed_common_mask.npz: reproducible frozen support.',
        '- en4_observation_influence.csv: volume-weighted observation weights and complete-month fractions.',
        '- uniform_comparison_metrics.csv and matched_cumulative_anomalies.csv: final descriptive comparison (not generated for a two-year pilot).',
        '- Annual inputs, raw downloads, per-year metadata/logs and processing caches are retained. No automatic deletion.', '',
        'Original main code, original IAP annual files, DEM, original NetCDF results and earlier comparisons are unchanged.'])
    (phase_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args):
    out = args.output_dir
    phase_dir = out / 'pilot_{}_{}'.format(args.start_year, args.end_year) if args.pilot_only else out
    if (phase_dir / 'metadata.json').exists():
        raise ValueError('Completed phase exists; choose a new output directory rather than overwrite')
    for path in (out, phase_dir, out / 'annual_cache', out / 'annual_processing'):
        path.mkdir(parents=True, exist_ok=True)
    log = download.log_setup(phase_dir / 'processing.log')
    started = time.monotonic()
    urls = download.prepare_index(out)
    volume_hash = download.sha256(args.volume_cache)
    depth, thickness, lat, lon, volumes, regions, denominators, volume_metadata = read_volumes(args.volume_cache)
    log.info('Range %d-%d; %d annual states; %d layers; workers=%d; phase=%s', args.start_year,
             args.end_year, args.end_year - args.start_year + 1, len(depth), args.workers,
             'pilot' if args.pilot_only else 'final')
    caches = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for year in range(args.start_year, args.end_year + 1):
            filename = 'EN.4.2.2.analyses.{}.{}.zip'.format(args.correction, year)
            if filename not in urls:
                raise ValueError('Official EN4 index has no annual analysis for ' + str(year))
            future = pool.submit(core.year_worker, year, args.correction, urls[filename], str(out),
                                 str(args.annual_dir), depth, lat, lon)
            futures[future] = year
        for future in as_completed(futures):
            year = futures[future]
            caches[year] = Path(future.result())
            log.info('Annual state %d ready (%d/%d)', year, len(caches), len(futures))
    years = np.arange(args.start_year, args.end_year + 1)
    states = {}
    annual_metadata = {}
    for year in years:
        with np.load(str(caches[year])) as archive:
            states[int(year)] = {key: archive[key] for key in archive.files}
        annual_metadata[int(year)] = json.loads(caches[year].with_suffix('.json').read_text())
    mask = core.common_mask(states.values(), volumes)
    np.savez_compressed(str(phase_dir / 'fixed_common_mask.npz'), mask=mask, depth=depth, lat=lat, lon=lon)
    coverage = []
    for region_name, region in regions.items():
        for group in core.GROUPS:
            possible = sum_region_depth(volumes, depth, group, region)
            common = sum_region_depth(np.where(mask, volumes, 0.), depth, group, region)
            coverage.append({'region': region_name, 'depth_band': group, 'possible_volume_km3': possible / 1e9,
                             'common_volume_km3': common / 1e9, 'common_volume_fraction': common / possible})
            log.info('%s %s fixed common volume coverage: %.3f%%', region_name, group, 100. * common / possible)
    annual_rows, layer_rows, influence_rows = [], [], []
    for year in years:
        state = states[int(year)]
        for region_name, region in regions.items():
            for group in core.GROUPS:
                k = core.group_mask(depth, group)
                weighted_volume = np.where(mask[k], volumes[k], 0.)[:, region]
                total = weighted_volume.sum(dtype=np.float64)
                diagnostics = {'year': int(year), 'region': region_name, 'depth_band': group,
                               'complete_12_month_volume_fraction': float((weighted_volume * state['en4_complete_year'][k][:, region]).sum() / total)}
                for name in ('en4_t_weight', 'en4_s_weight'):
                    values = state[name][k][:, region]
                    valid = np.isfinite(values)
                    denominator = weighted_volume[valid].sum(dtype=np.float64)
                    diagnostics[name + '_volume_weighted_mean'] = float((weighted_volume[valid] * values[valid]).sum() / denominator)
                influence_rows.append(diagnostics)
    for base, target in zip(years[:-1], years[1:]):
        for product, prefix in [('IAP', 'iap'), ('EN4', 'en4')]:
            a, b = states[int(base)], states[int(target)]
            changes = core.layer_changes(a[prefix + '_t'], a[prefix + '_sa'], b[prefix + '_t'], b[prefix + '_sa'],
                                         depth, volumes, mask)
            for region_name, region in regions.items():
                layer_sums = changes[:, region].sum(axis=1, dtype=np.float64)
                for k, z in enumerate(depth):
                    layer_rows.append({'base_year': int(base), 'target_year': int(target), 'product': product,
                                       'region': region_name, 'depth_std_m': float(z), 'layer_thickness_m': float(thickness[k]),
                                       'volume_change_km3': float(layer_sums[k] / 1e9),
                                       'sea_level_contribution_mm': float(layer_sums[k] / denominators[region_name] * 1000.)})
                for group in core.GROUPS:
                    change = float(layer_sums[core.group_mask(depth, group)].sum(dtype=np.float64))
                    annual_rows.append({'base_year': int(base), 'target_year': int(target), 'product': product,
                                        'region': region_name, 'depth_band': group, 'volume_change_km3': change / 1e9,
                                        'normalization_area_m2': denominators[region_name],
                                        'sea_level_contribution_mm': change / denominators[region_name] * 1000.,
                                        'domain': 'fixed_both_products_all_years_finite_mask'})
            log.info('%s %d vs %d: common Global 0-2000m %.6f mm', product, target, base,
                     next(row['sea_level_contribution_mm'] for row in annual_rows
                          if row['product'] == product and row['target_year'] == int(target)
                          and row['region'] == 'Global' and row['depth_band'] == '0-2000m'))
        statistics.step1.write_csv(phase_dir / 'annual_thermal_expansion.csv', annual_rows)
    metrics, matched = [], []
    if not args.pilot_only:
        if len(years) < 4:
            raise ValueError('Final comparison requires at least four annual anomaly states')
        for region_name in regions:
            for group in core.GROUPS:
                trajectories = {}
                for product in ('IAP', 'EN4'):
                    changes = [next(row['sea_level_contribution_mm'] for row in annual_rows
                                    if row['product'] == product and row['region'] == region_name
                                    and row['depth_band'] == group and row['target_year'] == int(year)) for year in years[1:]]
                    trajectories[product] = np.r_[0., np.cumsum(changes)]
                result = {key.replace('noaa', 'en4').replace('main', 'iap'): value
                          for key, value in statistics.metrics(years, trajectories['IAP'], trajectories['EN4']).items()}
                result.update(region=region_name, depth_band=group, start_year=args.start_year, end_year=args.end_year,
                              comparison_scope='shared_1degree_weights_formula_fixed_common_domain_not_original_15arcsec_validation')
                metrics.append(result)
                for i, year in enumerate(years):
                    matched.append({'year': int(year), 'region': region_name, 'depth_band': group,
                                    'iap_cumulative_anomaly_mm': float(trajectories['IAP'][i]),
                                    'en4_cumulative_anomaly_mm': float(trajectories['EN4'][i])})
                log.info('%s %s IAP trend %.6f, EN4 trend %.6f mm/yr; detrended annual r=%s', region_name, group,
                         result['iap_ols_trend_mm_per_year'], result['en4_ols_trend_mm_per_year'],
                         result['detrended_annual_change_correlation'])
    for filename, rows in [('layer_thermal_expansion.csv', layer_rows), ('common_domain_coverage.csv', coverage),
                           ('en4_observation_influence.csv', influence_rows), ('uniform_comparison_metrics.csv', metrics),
                           ('matched_cumulative_anomalies.csv', matched)]:
        if rows:
            statistics.step1.write_csv(phase_dir / filename, rows)
    original_inputs = {str(args.volume_cache): volume_hash}
    for metadata in annual_metadata.values():
        for record in metadata['iap_inputs'].values():
            original_inputs[record['path']] = record['sha256']
    if any(download.sha256(Path(path)) != digest for path, digest in original_inputs.items()):
        raise RuntimeError('Original input changed during independent comparison')
    statistics.step1.write_csv(phase_dir / 'original_input_manifest.csv',
                              [{'path': path, 'sha256': digest, 'unchanged': True} for path, digest in original_inputs.items()])
    metadata = {'phase': 'pilot_pair_domain' if args.pilot_only else 'full_period_fixed_common_domain',
                'start_year': args.start_year, 'end_year': args.end_year, 'annual_state_count': len(years),
                'annual_change_count': len(years) - 1, 'en4_correction': args.correction,
                'depth_reference_count': len(depth), 'pressure': 'numeric_depth_m_as_dbar_main_compatibility',
                'common_domain': 'finite_IAP_and_EN4_T_SA_in_every_selected_year_and_wet_volume',
                'input_hashes_unchanged': original_inputs, 'normalization_areas_m2': denominators,
                'coverage': coverage, 'volume_metadata': volume_metadata,
                'gsw_version': core.gsw.__version__, 'cache_version': core.CACHE_VERSION,
                'annual_preparation': annual_metadata, 'raw_downloads_and_caches_retained': True,
                'original_code_or_results_modified': False, 'new_15arcsec_calculation': False,
                'elapsed_seconds': time.monotonic() - started,
                'script_hashes': {p.name: download.sha256(p) for p in
                                  [Path(__file__), Path(core.__file__), Path(download.__file__), Path(statistics.__file__)]}}
    write_report(phase_dir, annual_rows, coverage, metrics, args.start_year, args.end_year, args.pilot_only)
    (phase_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    log.info('Completed %s in %.1f minutes; original inputs unchanged', phase_dir, metadata['elapsed_seconds'] / 60.)
    print('RESULTS_DIRECTORY=' + str(phase_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-year', type=int, default=2005)
    parser.add_argument('--end-year', type=int, default=2023)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--correction', choices=['g10'], default='g10')
    parser.add_argument('--annual-dir', type=Path, default=input_path('THERMOSTERIC_ANNUAL_DIR', 'inputs/annual_averages'))
    parser.add_argument('--volume-cache', type=Path, default=ROOT / 'uncertainty_preprocessing/gebco_1deg_layer_volumes/gebco_1deg_layer_volume_0_6000m.nc')
    parser.add_argument('--output-dir', type=Path, default=(workspace_root() / 'contribution_structure_analysis/en4_uniform_recalculation') / 'results_2005_2023_g10_20261015')
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    if args.end_year <= args.start_year or args.workers < 1:
        parser.error('Need adjacent annual states and positive workers')
    try:
        run(args)
    finally:
        if args.output_dir.exists():
            owner = ROOT.stat()
            for p in [args.output_dir] + list(args.output_dir.rglob('*')):
                os.chown(str(p), owner.st_uid, owner.st_gid)
                p.chmod(0o755 if p.is_dir() else 0o644)


if __name__ == '__main__':
    main()
