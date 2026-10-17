#!/usr/bin/env python3
"""Joint residual-block bootstrap for the completed Figure 07 definitions."""
from manuscript_paths import workspace_root, input_path
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = (workspace_root() / 'contribution_structure_analysis/step3_nearest_20261015')
ANALYSIS = HERE.parent
ROOT = ANALYSIS.parent
pass  # Source imports are configured by the launcher.
from calculate_interannual_variability import detrend, MIN_VARIANCE
from calculate_uncertainty_robustness import moving_indices

BASINS = ('Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic')
BANDS = ('0-700m', '700-2000m', '2000-6000m')
COMPONENTS = BASINS + BANDS
SOURCE = ANALYSIS / 'nearest_basin_reassignment_20261015/reassigned_statistics_1993_2025'
PERIODS = ((1993, 2025), (1995, 2025), (2000, 2025), (2005, 2025), (1993, 2020))
REPLICATES = 5000
SEED = 20261015


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def shares(values):
    values = np.asarray(values, dtype=float)
    residuals, _ = detrend(values, np.arange(values.shape[-1]))
    reference = residuals[..., 0, :]
    variance = np.sum(reference ** 2, axis=-1) / (values.shape[-1] - 1)
    covariance = np.sum(residuals[..., 1:, :] * reference[..., None, :], axis=-1) / (values.shape[-1] - 1)
    return np.divide(100. * covariance, variance[..., None],
                     out=np.full_like(covariance, np.nan), where=variance[..., None] >= MIN_VARIANCE)


def draws(values, count, block, seed):
    residuals, _ = detrend(values, np.arange(values.shape[-1]))
    fitted = values - residuals
    indices = moving_indices(values.shape[-1], count, block, np.random.RandomState(seed))
    sampled = residuals[:, indices].transpose(1, 0, 2)
    return shares(fitted[None, :, :] + sampled), indices


def main():
    output = HERE / 'statistics'
    if output.exists():
        raise ValueError('Existing statistics are protected from overwrite')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(str(HERE / 'processing.log'))])
    files = [ANALYSIS / 'step1_long_term_20261014/annual_components.csv',
             SOURCE / 'corrected_annual_basin_contributions.csv', SOURCE / 'variability_contributions.csv',
             SOURCE / 'metadata.json']
    signatures = {str(path): sha256(path) for path in files}
    annual = pd.read_csv(files[0])
    basins = pd.read_csv(files[1])
    years = np.arange(1994, 2026)

    def annual_vector(kind, band):
        rows = annual[annual.component_type.eq(kind) & annual.region.eq('Global') & annual.depth_band.eq(band)
                      & annual.target_year.between(1994, 2025)].sort_values('target_year')
        np.testing.assert_array_equal(rows.target_year, years)
        return rows.annual_change_mm.to_numpy(dtype=float)

    vectors = [annual_vector('global', '0-6000m')]
    for basin in BASINS:
        rows = basins[basins.basin.eq(basin)].sort_values('target_year')
        np.testing.assert_array_equal(rows.target_year, years)
        vectors.append(rows.reassigned_global_contribution_mm.to_numpy(dtype=float))
    vectors += [annual_vector('depth', band) for band in BANDS]
    values = np.stack(vectors)
    assert np.isfinite(values).all()
    assert np.max(np.abs(values[0] - values[1:6].sum(axis=0))) < 5e-7
    assert np.max(np.abs(values[0] - values[6:].sum(axis=0))) < 5e-7
    point = shares(values)
    reference = pd.read_csv(files[2])
    expected = np.r_[reference[reference.decomposition.eq('global_by_basin')].set_index('region').loc[list(BASINS)].variance_contribution_pct,
                     reference[reference.decomposition.eq('global_by_depth')].set_index('depth_band').loc[list(BANDS)].variance_contribution_pct]
    np.testing.assert_allclose(point, expected, rtol=0., atol=1e-10)
    output.mkdir()
    records = []
    for block in (2, 3, 5):
        samples, indices = draws(values, REPLICATES, block, SEED + block)
        np.savez_compressed(str(output / ('bootstrap_block{}.npz'.format(block))),
                            variance_shares_pct=samples, sampled_year_indices=indices,
                            component_names=np.asarray(COMPONENTS), target_years=years)
        for i, name in enumerate(COMPONENTS):
            finite = samples[:, i][np.isfinite(samples[:, i])]
            if len(finite) < .95 * REPLICATES:
                raise ValueError('Insufficient finite draws for {}'.format(name))
            low, high = np.percentile(finite, [2.5, 97.5])
            records.append({'component': name, 'kind': 'basin' if i < 5 else 'depth',
                            'endpoint_start_year': 1993, 'endpoint_end_year': 2025,
                            'annual_increment_count': len(years), 'block_years': block,
                            'point_share_pct': point[i], 'lower_2p5_pct': low, 'upper_97p5_pct': high,
                            'finite_draw_count': len(finite), 'requested_draw_count': REPLICATES,
                            'interval_type': 'conditional_time_sampling_pointwise_95pct_percentile',
                            'seed': SEED + block})
        logging.info('Block %d years: %d joint replicates complete', block, REPLICATES)
    pd.DataFrame(records).to_csv(output / 'joint_bootstrap_intervals.csv', index=False)
    sensitivities = []
    for start, end in PERIODS:
        mask = (years > start) & (years <= end)
        np.testing.assert_array_equal(years[mask], np.arange(start + 1, end + 1))
        for i, value in enumerate(shares(values[:, mask])):
            sensitivities.append({'component': COMPONENTS[i], 'kind': 'basin' if i < 5 else 'depth',
                                  'endpoint_start_year': start, 'endpoint_end_year': end,
                                  'annual_increment_count': int(mask.sum()), 'share_pct': value,
                                  'statistical_inference': 'point_sensitivity_not_independent_test'})
    pd.DataFrame(sensitivities).to_csv(output / 'period_sensitivity.csv', index=False)
    pd.DataFrame(values.T, columns=('Global',) + COMPONENTS).assign(target_year=years).to_csv(output / 'annual_input_series.csv', index=False)
    assert signatures == {str(path): sha256(path) for path in files}
    metadata = {'scope': 'five-basin and global-depth allocations matching Figure07; not basin-by-depth statistics',
                'endpoint_years': [1993, 2025], 'annual_target_years': [1994, 2025],
                'replicates_per_block': REPLICATES, 'block_lengths_years': [2, 3, 5],
                'base_seed': SEED, 'method': 'joint circular residual blocks; restore fitted intercept/linear trend; re-detrend each replicate',
                'same_year_indices_all_components': True,
                'intervals': 'pointwise approximate 95% percentile conditional time-sampling intervals; not observation-error intervals',
                'no_simultaneous_or_significance_claim': True,
                'periods': PERIODS, 'no_clipping_or_renormalization': True,
                'observational_temperature_member_spread_included': False,
                'assumption': 'approximately stationary residuals within the analysis period',
                'main_heat_expansion_recomputed': False, 'source_hashes_unchanged': True,
                'input_sha256': signatures, 'figure07_point_agreement_max_abs_pct': float(np.max(np.abs(point - expected))),
                'basin_annual_closure_max_abs_mm': float(np.max(np.abs(values[0] - values[1:6].sum(axis=0)))),
                'depth_annual_closure_max_abs_mm': float(np.max(np.abs(values[0] - values[6:].sum(axis=0))))}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    owner = ROOT.stat()
    for path in [HERE] + list(HERE.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    logging.info('All statistics and retained bootstrap draws saved')


if __name__ == '__main__':
    main()
