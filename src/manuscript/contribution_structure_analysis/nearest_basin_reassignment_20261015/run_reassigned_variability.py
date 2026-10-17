#!/usr/bin/env python3
"""Reassign spatial gaps, then recompute descriptive variability, retaining provenance."""
from manuscript_paths import workspace_root, input_path
import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_nearest_ocean_cells as geometry

HERE = (workspace_root() / 'contribution_structure_analysis/nearest_basin_reassignment_20261015')
ROOT = HERE.parent.parent
SOURCE = ROOT / 'contribution_structure_analysis'
PERIOD = 'recent_extended'
BASINS = ('Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic')
LIMIT_MM = 5e-7


def worker(year):
    path = HERE / ('correction_{}.json'.format(year))
    source = geometry.YEARS / ('{}_vs_{}'.format(year, year - 1)) / ('{}_6000.nc'.format(year))
    if path.exists():
        value = json.loads(path.read_text())
        stat = source.stat()
        if (value['source_nc_bytes'], value['source_nc_mtime_ns']) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError('Cached correction source changed')
        return value
    return geometry.sample_year(year, HERE / 'nearest_assignments.npz', logging.getLogger('nearest_run'))


def residual(values, years):
    x = years - years.mean()
    return values - values.mean() - x * np.dot(values - values.mean(), x) / np.dot(x, x)


def run(args):
    output = HERE / 'reassigned_statistics_1993_2025'
    if (output / 'metadata.json').exists():
        raise ValueError('Completed statistics exist; refusing overwrite')
    output.mkdir(exist_ok=True)
    log = logging.getLogger('nearest_run')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(str(HERE / 'run.log'))])
    if not (HERE / 'geometry.json').exists() or not (HERE / 'nearest_assignments.npz').exists():
        raise ValueError('Prepare and verify geometry first')
    annual = pd.read_csv(SOURCE / 'step1_long_term_20261014/annual_components.csv')
    annual = annual[annual.target_year.between(1994, 2025)].copy()
    years = np.arange(1994, 2026)

    def values(kind, region, band):
        rows = annual[(annual.component_type == kind) & (annual.region == region)
                      & (annual.depth_band == band)].sort_values('target_year')
        np.testing.assert_array_equal(rows.target_year, years)
        return rows.annual_change_mm.to_numpy(dtype=float)

    global_values = values('global', 'Global', '0-6000m')
    old = np.stack([values('basin', name, '0-6000m') for name in BASINS])
    expected = global_values - old.sum(axis=0)
    pilot = worker(2025)
    pilot_error = pilot['spatial_unassigned_total_mm'] - expected[-1]
    log.info('2025 pilot: arithmetic residual %.12g; spatial gap %.12g; difference %.12g mm',
             expected[-1], pilot['spatial_unassigned_total_mm'], pilot_error)
    if abs(pilot_error) > LIMIT_MM:
        raise ValueError('Pilot spatial gaps do not explain arithmetic residual; refusing arbitrary allocation')
    records = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, int(year)): int(year) for year in years}
        for future in as_completed(futures):
            year = futures[future]
            record = future.result()
            difference = record['spatial_unassigned_total_mm'] - expected[year - 1994]
            if abs(difference) > LIMIT_MM:
                raise ValueError('{} has a nonspatial residual: {:.12g} mm; not allocated'.format(year, difference))
            records[year] = record
            log.info('Verified nearest reassignment %d (%d/32), roundoff remainder %.12g mm', year, len(records), difference)
    corrections = np.asarray([[records[int(year)]['basin_corrections_mm'][name] for year in years]
                              for name in BASINS])
    corrected = old + corrections
    numerical_remainder = global_values - corrected.sum(axis=0)
    if np.max(np.abs(numerical_remainder)) > LIMIT_MM:
        raise ValueError('Unlocated residual exceeds original float32 map precision allowance')
    global_residual = residual(global_values, years)
    basin_residuals = np.stack([residual(row, years) for row in corrected])
    variance = np.var(global_residual, ddof=1)
    covariance = np.cov(basin_residuals, ddof=1)
    cross_global = np.asarray([np.cov(row, global_residual, ddof=1)[0, 1] for row in basin_residuals])
    common = {'period': PERIOD, 'endpoint_start_year': 1993, 'endpoint_end_year': 2025,
              'first_annual_target_year': 1994, 'last_annual_target_year': 2025, 'annual_increment_count': 32,
              'reference_region': 'Global', 'reference_depth_band': '0-6000m', 'statistical_inference': 'descriptive_only'}
    contribution_rows = []
    for k, name in enumerate(BASINS):
        contribution_rows.append(dict(common, decomposition='global_by_basin', component_type='basin',
                                      region=name, depth_band='0-6000m',
                                      variance_contribution_pct=float(100 * cross_global[k] / variance),
                                      covariance_with_reference_mm2=float(cross_global[k]),
                                      component_variance_mm2=float(covariance[k, k]),
                                      detrended_std_mm=float(np.sqrt(covariance[k, k]))))
    original_contributions = pd.read_csv(SOURCE / 'step2_interannual_20261014/variability_contributions.csv')
    original_depth = original_contributions[(original_contributions.period == PERIOD)
                                            & (original_contributions.decomposition == 'global_by_depth')]
    contribution_rows += original_depth.to_dict('records')
    pair_rows = []
    for i, a in enumerate(BASINS):
        for j in range(i, len(BASINS)):
            value = float(covariance[i, j])
            term = value if i == j else 2 * value
            pair_rows.append(dict(common, decomposition='global_by_basin',
                                  component_a='basin|{}|0-6000m'.format(a),
                                  component_b='basin|{}|0-6000m'.format(BASINS[j]),
                                  is_diagonal=i == j, covariance_mm2=value,
                                  term_in_total_variance_mm2=term,
                                  term_contribution_to_reference_variance_pct=100 * term / variance))
    residual_rows = []
    for year, value in zip(years, global_residual):
        residual_rows.append(dict(common, target_year=int(year), component_type='global', region='Global',
                                  depth_band='0-6000m', detrended_annual_change_mm=float(value)))
    for k, name in enumerate(BASINS):
        for year, value in zip(years, basin_residuals[k]):
            residual_rows.append(dict(common, target_year=int(year), component_type='basin', region=name,
                                      depth_band='0-6000m', detrended_annual_change_mm=float(value)))
    original_residuals = pd.read_csv(SOURCE / 'step2_interannual_20261014/detrended_annual_series.csv')
    residual_rows += original_residuals[(original_residuals.period == PERIOD)
                                       & original_residuals.component_type.eq('depth')].to_dict('records')
    diagonal = float(np.trace(covariance))
    summary = dict(common, decomposition='global_by_basin', reference_detrended_variance_mm2=float(variance),
                   reference_detrended_std_mm=float(np.sqrt(variance)), sum_component_variances_mm2=diagonal,
                   twice_sum_cross_covariances_mm2=float(covariance.sum() - diagonal),
                   net_cancellation_pct_of_diagonal_sum=float((diagonal - variance) / diagonal * 100),
                   signed_contributions_sum_pct=float(cross_global.sum() / variance * 100))
    pd.DataFrame(contribution_rows).to_csv(output / 'variability_contributions.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(output / 'component_covariance.csv', index=False)
    pd.DataFrame(residual_rows).to_csv(output / 'detrended_annual_series.csv', index=False)
    pd.DataFrame([summary]).to_csv(output / 'decomposition_summary.csv', index=False)
    corrected_rows, closure_rows = [], []
    for i, year in enumerate(years):
        for k, name in enumerate(BASINS):
            corrected_rows.append({'base_year': int(year - 1), 'target_year': int(year), 'basin': name,
                                   'original_global_contribution_mm': float(old[k, i]),
                                   'nearest_spatial_correction_mm': float(corrections[k, i]),
                                   'reassigned_global_contribution_mm': float(corrected[k, i])})
        closure_rows.append({'target_year': int(year), 'original_global_mm': float(global_values[i]),
                             'original_five_basin_sum_mm': float(old[:, i].sum()),
                             'arithmetic_residual_mm': float(expected[i]),
                             'spatial_reassigned_sum_mm': float(corrections[:, i].sum()),
                             'unlocated_float32_roundoff_mm': float(numerical_remainder[i])})
    pd.DataFrame(corrected_rows).to_csv(output / 'corrected_annual_basin_contributions.csv', index=False)
    pd.DataFrame(closure_rows).to_csv(output / 'allocation_closure.csv', index=False)
    metadata = {'method': 'only original unassigned wet cells assigned to nearest classified wet center by great-circle distance',
                'endpoint_years': [1993, 2025], 'annual_target_years': [1994, 2025],
                'global_main_series_changed': False, 'depth_series_changed': False,
                'original_basin_labels_changed': False, 'no_renormalization_or_proportional_allocation': True,
                'remaining_numerical_remainder_max_abs_mm': float(np.max(np.abs(numerical_remainder))),
                'numerical_acceptance_limit_mm': LIMIT_MM,
                'variance_allocation_sum_pct': summary['signed_contributions_sum_pct'],
                'variance_identity_roundoff_mm2': float(variance - covariance.sum()),
                'geometry_metadata': json.loads((HERE / 'geometry.json').read_text()),
                'original_netcdf_or_csv_modified': False}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    log.info('Completed five-basin statistics; variance fractions sum %.12g%%; maximum roundoff %.12g mm',
             metadata['variance_allocation_sum_pct'], metadata['remaining_numerical_remainder_max_abs_mm'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    try:
        run(args)
    finally:
        owner = ROOT.stat()
        for path in [HERE] + list(HERE.rglob('*')):
            os.chown(str(path), owner.st_uid, owner.st_gid)
            path.chmod(0o755 if path.is_dir() else 0o644)
