#!/usr/bin/env python3
"""Step 3: separate temperature-member spread and joint residual block bootstrap."""

from manuscript_paths import workspace_root, input_path
import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

import calculate_long_term_contributions as step1
import calculate_interannual_variability as step2


ROOT = step1.ROOT
BASINS = step1.BASINS + ('Unassigned',)
DEPTHS = step1.EXCLUSIVE_DEPTHS
ARCHIVE = ROOT / 'uncertainty_analysis/full_spatiotemporal_uncertainty/results_1940_2023_lon_aligned'
MIN_VARIANCE = step2.MIN_VARIANCE


def clean(value):
    return float(value) if np.isfinite(value) else ''


def metric_key(record):
    return tuple(record.get(key, '') for key in
                 ('metric', 'decomposition', 'component_type', 'region', 'depth_band', 'other_component'))


class StatisticsModel:
    """Map disjoint basin/depth atoms to all descriptive metrics, in batches."""

    def __init__(self, upper_only=False):
        self.upper_only = upper_only
        depths = ('0-700m',) if upper_only else DEPTHS
        self.atoms = [(basin, band) for basin in BASINS for band in depths]
        self.total_band = '0-700m' if upper_only else '0-6000m'
        keys = [('global', 'Global', self.total_band)]
        if not upper_only:
            keys += [('depth', 'Global', band) for band in DEPTHS + ('0-2000m',)]
        keys += [('basin', basin, self.total_band) for basin in BASINS]
        if not upper_only:
            keys += [('basin_depth', basin, band) for basin in BASINS for band in DEPTHS + ('0-2000m',)]
        self.keys = keys
        self.index = {key: i for i, key in enumerate(keys)}
        weights = np.zeros((len(keys), len(self.atoms)))
        for i, (kind, region, band) in enumerate(keys):
            for j, (basin, atom_band) in enumerate(self.atoms):
                region_ok = region == 'Global' or region == basin
                depth_ok = band == self.total_band or band == atom_band or (
                    band == '0-2000m' and atom_band in ('0-700m', '700-2000m'))
                weights[i, j] = region_ok and depth_ok
        self.weights = weights
        self.groups = [('upper_by_basin' if upper_only else 'global_by_basin', 0,
                        [self.index[('basin', basin, self.total_band)] for basin in BASINS])]
        if not upper_only:
            self.groups.append(('global_by_depth', 0, [self.index[('depth', 'Global', band)] for band in DEPTHS]))
            self.groups += [(basin + '_by_depth', self.index[('basin', basin, self.total_band)],
                             [self.index[('basin_depth', basin, band)] for band in DEPTHS]) for basin in BASINS]
        self.descriptors = None

    def evaluate(self, atoms):
        atoms = np.asarray(atoms, dtype=np.float64)
        if atoms.shape[-2] != len(self.atoms) or atoms.shape[-1] < 3 or not np.isfinite(atoms).all():
            raise ValueError('Need finite disjoint atoms with at least three annual increments.')
        values = np.einsum('ka,...at->...kt', self.weights, atoms, optimize=True)
        n = values.shape[-1]
        t = np.arange(n, dtype=np.float64)
        residual, _ = step2.detrend(values, t)
        residual -= residual.mean(axis=-1, keepdims=True)
        endpoint_t = np.arange(n + 1) - n / 2
        trend_weights = np.cumsum(endpoint_t[::-1])[::-1][1:] / np.dot(endpoint_t, endpoint_t)
        trend = np.sum(values * trend_weights, axis=-1)
        mean_rate = values.mean(axis=-1)
        cumulative = values.sum(axis=-1)
        denominator = trend[..., 0]
        shares = np.divide(trend * 100, denominator[..., None],
                           out=np.full_like(trend, np.nan), where=np.abs(denominator[..., None]) >= .05)
        cumulative_shares = np.divide(cumulative * 100, cumulative[..., :1],
                           out=np.full_like(cumulative, np.nan), where=np.abs(cumulative[..., :1]) >= 1.)
        columns, descriptions = [], []

        def add(metric, decomposition, key, array, other=''):
            columns.append(np.asarray(array))
            descriptions.append({'metric': metric, 'decomposition': decomposition,
                'component_type': key[0], 'region': key[1], 'depth_band': key[2], 'other_component': other})

        for i, key in enumerate(self.keys):
            for metric, data in (('cumulative_change_mm', cumulative), ('mean_rate_mm_per_year', mean_rate),
                                 ('ols_trend_mm_per_year', trend), ('ols_trend_share_pct', shares),
                                 ('cumulative_share_pct', cumulative_shares)):
                add(metric, 'long_term', key, data[..., i])
        for name, ref, indices in self.groups:
            r = residual[..., indices, :]
            reference = residual[..., ref, :]
            covariance = np.einsum('...it,...jt->...ij', r, r) / (n - 1)
            own = np.diagonal(covariance, axis1=-2, axis2=-1)
            variance = np.sum(reference * reference, axis=-1) / (n - 1)
            cov_reference = np.sum(r * reference[..., None, :], axis=-1) / (n - 1)
            fraction = np.divide(cov_reference * 100, variance[..., None],
                                 out=np.full_like(cov_reference, np.nan), where=variance[..., None] >= MIN_VARIANCE)
            std = np.sqrt(np.maximum(own, 0))
            for i, component in enumerate(indices):
                add('detrended_sd_mm', name, self.keys[component], std[..., i])
                add('variance_share_pct', name, self.keys[component], fraction[..., i])
            add('reference_detrended_sd_mm', name, self.keys[ref], np.sqrt(np.maximum(variance, 0)))
            diagonal = own.sum(axis=-1)
            cancellation = np.divide((diagonal - variance) * 100, diagonal,
                                     out=np.full_like(variance, np.nan), where=diagonal >= MIN_VARIANCE)
            add('net_cancellation_pct', name, self.keys[ref], cancellation)
            if name in ('global_by_basin', 'global_by_depth', 'upper_by_basin'):
                for i, component in enumerate(indices):
                    for j in range(i + 1, len(indices)):
                        other = '|'.join(self.keys[indices[j]])
                        add('pair_covariance_mm2', name, self.keys[component], covariance[..., i, j], other)
                        product = std[..., i] * std[..., j]
                        correlation = np.divide(covariance[..., i, j], product,
                                                out=np.full_like(product, np.nan), where=product > 0)
                        add('pair_correlation', name, self.keys[component], np.clip(correlation, -1, 1), other)
        if self.descriptors is None:
            self.descriptors = descriptions
        elif descriptions != self.descriptors:
            raise RuntimeError('Statistic ordering changed between runs.')
        return np.stack(columns, axis=-1)


def moving_indices(n, replicates, block_length, rng):
    if not 1 <= block_length <= n:
        raise ValueError('Invalid block length.')
    blocks = int(np.ceil(n / block_length))
    starts = rng.randint(0, n, size=(replicates, blocks))
    return ((starts[:, :, None] + np.arange(block_length)) % n).reshape(replicates, -1)[:, :n]


def seed_for(seed, name, block):
    digest = hashlib.sha256('{}:{}'.format(name, block).encode('ascii')).hexdigest()
    return (seed + int(digest[:8], 16)) % (2**32)


def bootstrap_draws(atoms, model, count, block, seed):
    residual, _ = step2.detrend(atoms, np.arange(atoms.shape[-1]))
    fitted = atoms - residual
    indices = moving_indices(atoms.shape[-1], count, block, np.random.RandomState(seed))
    sampled = residual[:, indices].transpose(1, 0, 2)
    return model.evaluate(fitted[None] + sampled)


def distribution_rows(draws, point, model, context, label):
    rows = []
    for i, descriptor in enumerate(model.descriptors):
        values = draws[:, i]
        values = values[np.isfinite(values)]
        record = dict(context, **descriptor)
        record.update({'distribution_type': label, 'point_estimate': clean(point[i]),
                       'valid_draw_count': len(values), 'requested_draw_count': len(draws),
                       'finite_draw_fraction': len(values) / len(draws)})
        if len(values):
            low, high = np.percentile(values, [2.5, 97.5])
            record.update({'distribution_mean': float(values.mean()),
                'distribution_sd': float(values.std(ddof=1)) if len(values) > 1 else '',
                'lower_2p5': float(low), 'upper_97p5': float(high),
                'fraction_positive': float(np.mean(values > 0)),
                'fraction_negative': float(np.mean(values < 0)),
                'percentile_interval_excludes_zero': bool(low > 0 or high < 0),
                'interval_reliable_draw_fraction': len(values) >= .95 * len(draws)})
        else:
            record.update({key: '' for key in ('distribution_mean', 'distribution_sd', 'lower_2p5',
                'upper_97p5', 'fraction_positive', 'fraction_negative',
                'percentile_interval_excludes_zero', 'interval_reliable_draw_fraction')})
        rows.append(record)
    return rows


def simultaneous_intervals(draws, point, descriptors, metric, context):
    selected = []
    for i, d in enumerate(descriptors):
        primary = (d['component_type'] == 'basin' and d['region'] in step1.BASINS) or (
                   d['component_type'] == 'depth' and d['depth_band'] in DEPTHS)
        if metric == 'variance_share_pct':
            chosen = primary and d['decomposition'] in ('global_by_basin', 'global_by_depth')
        else:
            chosen = primary or d['component_type'] == 'global'
        if d['metric'] == metric and chosen:
            selected.append(i)
    matrix = draws[:, selected]
    valid = np.isfinite(matrix).all(axis=1)
    if valid.sum() < 50:
        raise ValueError('Too few common finite draws for simultaneous intervals.')
    matrix = matrix[valid]
    se = matrix.std(axis=0, ddof=1)
    safe = se > 1e-14
    deviations = np.divide(np.abs(matrix - point[selected]), se,
                           out=np.zeros_like(matrix), where=safe)
    critical = float(np.percentile(deviations.max(axis=1), 95))
    rows = []
    for k, i in enumerate(selected):
        low, high = point[i] - critical * se[k], point[i] + critical * se[k]
        rows.append(dict(context, **descriptors[i],
            point_difference=clean(point[i]), simultaneous_lower=float(low), simultaneous_upper=float(high),
            bootstrap_standard_error=float(se[k]), family_metric=metric, family_size=len(selected),
            family_critical_max_standardized_deviation=critical,
            common_finite_draw_count=int(valid.sum()),
            common_finite_draw_fraction=float(valid.mean()),
            simultaneous_interval_excludes_zero=bool(low > 0 or high < 0),
            method='approximate_fixed_SE_bootstrap_max_standardized_deviation_95pct'))
    return rows


def period_atoms(atoms, years, start, end):
    mask = (years > start) & (years <= end)
    if not np.array_equal(years[mask], np.arange(start + 1, end + 1)):
        raise ValueError('Incomplete period {}-{}'.format(start, end))
    return atoms[..., mask]


def build_member_atoms(archive):
    global_rows = step2.read_csv(archive / 'member_results_global.csv')
    basin_rows = step2.read_csv(archive / 'member_results_basin.csv')
    mapping = {}
    years = set()
    members = set()
    for row in global_rows + basin_rows:
        if row['depth_band'] not in DEPTHS:
            continue
        year = int(row['target_year'])
        if int(row['base_year']) != year - 1:
            raise ValueError('Nonannual member pair.')
        key = (row['member'], year, row['region'], row['depth_band'])
        if key in mapping:
            raise ValueError('Duplicate member record: ' + repr(key))
        # Normalize absolute heights to the SAME area used by the central statistics.
        mapping[key] = float(row['volume_change_km3']) * 1e9 / step1.AREA * 1000
        years.add(year)
        members.add(row['member'])
    years = np.asarray(sorted(years), dtype=int)
    if not np.array_equal(years, np.arange(years[0], years[-1] + 1)):
        raise ValueError('Member years are not contiguous.')
    results = {}
    for label, depths in (('full_depth_common_members', DEPTHS), ('upper_only_available_members', ('0-700m',))):
        accepted, rejected, data = [], [], []
        for member in sorted(members):
            required = [(member, int(year), region, band) for year in years
                        for region in ('Global',) + step1.BASINS for band in depths]
            missing = [key for key in required if key not in mapping or not np.isfinite(mapping[key])]
            if missing:
                rejected.append({'member': member, 'missing_count': len(missing)})
                continue
            array = []
            for basin in BASINS:
                for band in depths:
                    values = []
                    for year in years:
                        if basin == 'Unassigned':
                            value = mapping[(member, int(year), 'Global', band)] - sum(
                                mapping[(member, int(year), b, band)] for b in step1.BASINS)
                        else:
                            value = mapping[(member, int(year), basin, band)]
                        values.append(value)
                    array.append(values)
            accepted.append(member)
            data.append(array)
        if len(accepted) < 2:
            raise ValueError('Fewer than two complete members for ' + label)
        results[label] = {'members': accepted, 'rejected': rejected, 'atoms': np.asarray(data, dtype=np.float64)}
    return years, results


def member_summary(values, model, context, members):
    records, summary = [], []
    for i, descriptor in enumerate(model.descriptors):
        sample = values[:, i]
        finite = sample[np.isfinite(sample)]
        for member, value in zip(members, sample):
            records.append(dict(context, **descriptor, member=member, value=clean(value)))
        record = dict(context, **descriptor,
            available_member_count=len(members), finite_member_count=len(finite),
            member_mean=float(finite.mean()) if len(finite) else '',
            member_sd=float(finite.std(ddof=1)) if len(finite) > 1 else '',
            member_min=float(finite.min()) if len(finite) else '',
            member_max=float(finite.max()) if len(finite) else '',
            member_p05=float(np.percentile(finite, 5)) if len(finite) else '',
            member_p95=float(np.percentile(finite, 95)) if len(finite) else '',
            positive_member_fraction=float(np.mean(finite > 0)) if len(finite) else '',
            uncertainty_scope='temperature_member_spread_not_confidence_interval_or_total_uncertainty',
            central_15arcsec_error_bar_transfer='not_performed')
        summary.append(record)
    return records, summary


def ensemble_analysis(archive, periods, out, log):
    years, sets = build_member_atoms(archive)
    records, summaries, difference_records, difference_summaries = [], [], [], []
    provenance = {}
    for label, dataset in sets.items():
        model = StatisticsModel(upper_only=label.startswith('upper'))
        available = [(name, start, end) for name, start, end in periods
                     if start >= years[0] - 1 and end <= years[-1]]
        period_values = {}
        for name, start, end in available:
            values = model.evaluate(period_atoms(dataset['atoms'], years, start, end))
            period_values[name] = values
            context = {'ensemble_set': label, 'period': name, 'endpoint_start_year': start,
                       'endpoint_end_year': end, 'source_grid': '1_degree',
                       'depth_convention': 'archive_labels_not_rebinned_to_main'}
            r, s = member_summary(values, model, context, dataset['members'])
            records.extend(r)
            summaries.extend(s)
        for comparison, early, late in step1.COMPARISONS:
            if early not in period_values or late not in period_values:
                continue
            # Pair each member with itself across periods; never pair independently drawn members.
            differences = period_values[late] - period_values[early]
            context = {'ensemble_set': label, 'comparison': comparison, 'early_period': early,
                       'late_period': late, 'source_grid': '1_degree',
                       'depth_convention': 'archive_labels_not_rebinned_to_main'}
            r, s = member_summary(differences, model, context, dataset['members'])
            difference_records.extend(r)
            difference_summaries.extend(s)
        provenance[label] = {'members': dataset['members'], 'rejected': dataset['rejected'],
                             'period_count': len(available)}
        log.info('Ensemble %s: %s complete members; %s periods', label, len(dataset['members']), len(available))
    step1.write_csv(out / 'ensemble_member_statistics.csv', records)
    step1.write_csv(out / 'ensemble_statistic_spread.csv', summaries)
    step1.write_csv(out / 'ensemble_member_period_differences.csv', difference_records)
    step1.write_csv(out / 'ensemble_period_difference_spread.csv', difference_summaries)
    return provenance


def validate_point_reproduction(model, point_values, long_term, variability):
    largest_trend, largest_share = 0., 0.
    descriptors = {metric_key(d): i for i, d in enumerate(model.descriptors)}
    for row in long_term:
        if row['period'] not in point_values:
            continue
        d = {'metric': 'ols_trend_mm_per_year', 'decomposition': 'long_term',
             'component_type': row['component_type'], 'region': row['region'], 'depth_band': row['depth_band']}
        error = abs(point_values[row['period']][descriptors[metric_key(d)]] - float(row['cumulative_ols_slope_mm_per_year']))
        largest_trend = max(largest_trend, error)
    for row in variability:
        if row['period'] not in point_values or row['decomposition'] not in ('global_by_basin', 'global_by_depth'):
            continue
        if row['variance_contribution_pct'] == '':
            continue
        d = {'metric': 'variance_share_pct', 'decomposition': row['decomposition'],
             'component_type': row['component_type'], 'region': row['region'], 'depth_band': row['depth_band']}
        error = abs(point_values[row['period']][descriptors[metric_key(d)]] - float(row['variance_contribution_pct']))
        largest_share = max(largest_share, error)
    if largest_trend > 1e-9 or largest_share > 1e-7:
        raise RuntimeError('Step 3 point statistics differ from Step 1/2.')
    return {'max_trend_reproduction_error_mm_per_year': largest_trend,
            'max_variance_share_reproduction_error_percentage_points': largest_share}


def robustness_table(rows, blocks):
    groups = {}
    for row in rows:
        key = (row['comparison'],) + metric_key(row)
        groups.setdefault(key, []).append(row)
    output = []
    for key, entries in groups.items():
        complete = {r['block_length_years'] for r in entries} == set(blocks)
        same_side = (all(r['simultaneous_lower'] > 0 for r in entries)
                     or all(r['simultaneous_upper'] < 0 for r in entries))
        output.append({'comparison': key[0], **{k: entries[0][k] for k in
            ('metric', 'decomposition', 'component_type', 'region', 'depth_band', 'other_component')},
            'point_difference': entries[0]['point_difference'],
            'block_lengths_tested': '|'.join(str(r['block_length_years']) for r in entries),
            'minimum_simultaneous_lower': min(r['simultaneous_lower'] for r in entries),
            'maximum_simultaneous_upper': max(r['simultaneous_upper'] for r in entries),
            'all_blocks_exclude_zero_same_direction': bool(complete and same_side),
            'statistical_scope': 'approximate_simultaneous_intervals_within_each_prespecified_family'})
    return output


def make_report(out, model, point_values, bootstrap_rows, robust, provenance, metadata):
    lines = ['# Step 3: Uncertainty and Robustness', '',
        'Temperature-member spread and time-sampling bootstrap intervals are reported SEPARATELY.',
        'No total observational uncertainty or strict 15-arcsecond main-value +/- member interval is claimed.', '',
        '## Main-Series Bootstrap', '',
        '- Basis: the 18 disjoint five-basin-plus-residual by three-depth annual series from Step 1.',
        '- Joint circular residual blocks resample the SAME years across all 18 atoms.',
        '- Fit an annual linear trend/intercept, resample its residuals, and add back fitted annual values.',
        '- Recompute cumulative trends and detrended variance shares for every replicate.',
        '- This assumes approximately stationary residuals within each period.',
        '- Earlier/later period samples are generated independently within strata; boundary-spanning',
        '  temporal dependence is not modeled. Differences preserve component covariance within each stratum.',
        '- Percentile 2.5-97.5 intervals are approximate, not exact coverage guarantees.',
        '- Simultaneous difference intervals use the 95th percentile of the maximum standardized',
        '  bootstrap deviation, with fixed bootstrap SEs. They are approximate and family-specific.',
        '- The two prespecified families contain 8 global variance shares and 9 absolute trends.',
        '- Multiple block lengths and windows are sensitivity analyses, not independent discoveries.', '',
        '## Recent-Core Point Values and Block-3 Percentile Intervals', '',
        '| Component | Point variance share (%) | Approximate 95% interval (%) |',
        '|---|---:|---:|']
    for row in bootstrap_rows:
        if row['period'] == 'recent_core' and row['block_length_years'] == 3 and row['metric'] == 'variance_share_pct':
            if row['decomposition'] in ('global_by_basin', 'global_by_depth'):
                name = row['depth_band'] if row['component_type'] == 'depth' else row['region']
                lines.append('| {} | {:.4f} | [{:.4f}, {:.4f}] |'.format(name,
                    row['point_estimate'], row['lower_2p5'], row['upper_97p5']))
    lines.extend(['', '## Core Period Difference: All Block Lengths', '',
                  '| Component / metric | Point late-early | Union of simultaneous intervals | All exclude zero? |',
                  '|---|---:|---:|---|'])
    for row in robust:
        if row['comparison'] != 'core_adjacent_15_years':
            continue
        name = row['depth_band'] if row['component_type'] == 'depth' else row['region']
        lines.append('| {} / {} | {:.4f} | [{:.4f}, {:.4f}] | {} |'.format(name, row['metric'],
            row['point_difference'], row['minimum_simultaneous_lower'], row['maximum_simultaneous_upper'],
            row['all_blocks_exclude_zero_same_direction']))
    lines.extend(['', '## Archived Temperature Members', '',
        '- Full-depth cross-component comparisons use the same {} complete members.'.format(
            len(provenance['full_depth_common_members']['members'])),
        '- Upper-only additional analysis uses {} complete members.'.format(
            len(provenance['upper_only_available_members']['members'])),
        '- Member trajectories remain intact; later-minus-earlier is computed within the same member.',
        '- No member resampling, forced narrowing or recentering onto the main estimate is performed.',
        '- Archived 1-degree volume changes are normalized to the same global-area denominator.',
        '- Salinity is fixed in the archive. Spread does not cover salinity, DEM or all model errors.',
        '- Member p05-p95 is an empirical ensemble range, NOT a confidence interval.',
        '- The archive depth labels are used as-is; boundary conventions are not corrected from band totals.',
        '- Therefore no strict matching of individual archived depth bands to main-series bands is claimed.',
        '- Data stop at 2023; no invented 2024-2025 member paths are generated.', '',
        '## Additional Sensitivities', '',
        '- Complete rolling point statistics are saved for 10/15/20-year spans (or CLI choices).',
        '- Recent-core alternatives include 1995-2023, 2000-2023, 2005-2023 and 1993-2020.',
        '- These settings are fixed for this run, not claimed as a formal preregistration.',
        '- Residual SD is physical interannual variability, not observational uncertainty.',
        '- A variance contribution is an allocation, not a causal driver.',
        '- Source hashes are verified unchanged. No new downloads or 15-arcsecond calculations occur.', '',
        'See metadata.json, the CSV tables, and bootstrap_draws/ for exact settings and reproducibility.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step1-dir', type=Path, default=ROOT / 'contribution_structure_analysis/step1_long_term_20261014')
    parser.add_argument('--step2-dir', type=Path, default=ROOT / 'contribution_structure_analysis/step2_interannual_20261014')
    parser.add_argument('--ensemble-dir', type=Path, default=ARCHIVE)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--block-lengths', nargs='+', type=int, default=[2, 3, 5])
    parser.add_argument('--window-years', nargs='+', type=int, default=[10, 15, 20])
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.replicates < 200 or min(args.block_lengths) < 1 or min(args.window_years) < 3:
        parser.error('Need >=200 replicates, positive blocks and windows >=3 years.')
    paths = [args.step1_dir / p for p in ('annual_components.csv', 'long_term_contributions.csv', 'metadata.json')]
    paths += [args.step2_dir / 'variability_contributions.csv']
    paths += [args.ensemble_dir / p for p in ('member_results_global.csv', 'member_results_basin.csv', 'metadata.json')]
    manifest = [{'path': str(p), 'sha256': step1.sha256(p), 'bytes': p.stat().st_size} for p in paths]
    previous = json.loads((args.step1_dir / 'metadata.json').read_text())
    archive_metadata = json.loads((args.ensemble_dir / 'metadata.json').read_text())
    years, series = step2.load_series(args.step1_dir / 'annual_components.csv')
    model = StatisticsModel()
    atoms = np.stack([series[('basin_depth', basin, band)] for basin, band in model.atoms])
    periods = [(name, int(start), int(end)) for name, start, end in previous['periods']]
    for name, start, end in (('sensitivity_1995_2023', 1995, 2023), ('sensitivity_2000_2023', 2000, 2023),
                             ('sensitivity_1993_2020', 1993, 2020)):
        if years[0] - 1 <= start and end <= years[-1]:
            periods.append((name, start, end))
    if args.output_dir is None:
        out = ROOT / 'contribution_structure_analysis/step3_uncertainty_20261014'
        number = 2
        while out.exists():
            out = out.with_name('step3_uncertainty_20261014_{:02d}'.format(number))
            number += 1
    else:
        out = args.output_dir
        if out.exists() and any(out.iterdir()):
            raise ValueError('Choose a new empty output directory.')
    out.mkdir(parents=True, exist_ok=True)
    draws_dir = out / 'bootstrap_draws'
    draws_dir.mkdir()
    log = logging.getLogger('uncertainty_robustness')
    log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(str(out / 'processing.log'))):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    point_values, point_rows, intervals = {}, [], []
    period_draws = {}
    needed = {name for _, early, late in step1.COMPARISONS for name in (early, late)}
    for name, start, end in periods:
        selected = period_atoms(atoms, years, start, end)
        point = model.evaluate(selected)
        point_values[name] = point
        for descriptor, value in zip(model.descriptors, point):
            point_rows.append({'period': name, 'endpoint_start_year': start, 'endpoint_end_year': end,
                               **descriptor, 'value': clean(value), 'source': '15arcsec_main_series_statistics'})
        for block in args.block_lengths:
            draws = bootstrap_draws(selected, model, args.replicates, block, seed_for(args.seed, name, block))
            context = {'period': name, 'endpoint_start_year': start, 'endpoint_end_year': end,
                       'block_length_years': block, 'source': '15arcsec_main_series_time_sampling'}
            intervals.extend(distribution_rows(draws, point, model, context, 'joint_circular_residual_block_bootstrap'))
            np.savez_compressed(str(draws_dir / ('{}_block{}.npz'.format(name, block))),
                                statistics=draws, point=point,
                                metric_keys=np.asarray([json.dumps(metric_key(d)) for d in model.descriptors]))
            if name in needed:
                period_draws[(name, block)] = draws
        log.info('Main bootstrap %s (%s-%s), %s replicates per block, blocks=%s complete',
                 name, start, end, args.replicates, args.block_lengths)
    reproduction = validate_point_reproduction(model, point_values,
        step2.read_csv(args.step1_dir / 'long_term_contributions.csv'),
        step2.read_csv(args.step2_dir / 'variability_contributions.csv'))
    step1.write_csv(out / 'main_point_statistics.csv', point_rows)
    step1.write_csv(out / 'main_block_bootstrap_intervals.csv', intervals)
    difference_intervals, simultaneous = [], []
    for comparison, early, late in step1.COMPARISONS:
        if early not in point_values or late not in point_values:
            continue
        point = point_values[late] - point_values[early]
        for block in args.block_lengths:
            draws = period_draws[(late, block)] - period_draws[(early, block)]
            context = {'comparison': comparison, 'early_period': early, 'late_period': late,
                       'block_length_years': block}
            difference_intervals.extend(distribution_rows(draws, point, model, context,
                                         'stratified_joint_residual_block_bootstrap_difference'))
            for metric in ('variance_share_pct', 'ols_trend_mm_per_year'):
                simultaneous.extend(simultaneous_intervals(draws, point, model.descriptors, metric, context))
    step1.write_csv(out / 'main_period_difference_intervals.csv', difference_intervals)
    step1.write_csv(out / 'main_simultaneous_difference_intervals.csv', simultaneous)
    robust = robustness_table(simultaneous, args.block_lengths)
    step1.write_csv(out / 'block_length_robustness.csv', robust)
    provenance = ensemble_analysis(args.ensemble_dir, periods, out, log)
    sensitivities = []
    for span in args.window_years:
        for start in range(int(years[0] - 1), int(years[-1] - span + 1)):
            end = start + span
            values = model.evaluate(period_atoms(atoms, years, start, end))
            for descriptor, value in zip(model.descriptors, values):
                sensitivities.append({'window_elapsed_years': span, 'endpoint_start_year': start,
                    'endpoint_end_year': end, 'window_center_year': (start + end) / 2,
                    **descriptor, 'value': clean(value), 'inference': 'point_sensitivity_not_independent_test'})
    step1.write_csv(out / 'window_length_point_sensitivity.csv', sensitivities)
    if not all(step1.sha256(Path(p['path'])) == p['sha256'] for p in manifest):
        raise RuntimeError('An input file changed during Step 3.')
    step1.write_csv(out / 'input_manifest.csv', manifest)
    metadata = {'analysis': 'step_3_separate_temperature_spread_and_time_sampling',
        'replicates_per_period_block': args.replicates, 'block_lengths_years': args.block_lengths,
        'rolling_window_lengths_years': args.window_years, 'seed': args.seed, 'periods': periods,
        'bootstrap': 'joint circular residual blocks; annual linear fitted means/trends retained',
        'period_difference_assumption': 'independent residual resampling within early/late strata; no boundary dependence model',
        'simultaneous_intervals': 'approximate fixed-bootstrap-SE max-standardized-deviation intervals; 8 variance shares / 9 trends',
        'families_are_separate': True, 'confidence_and_member_spread_combined': False,
        'member_spread_transferred_to_15arcsec_central_value': False,
        'ensemble_archive_depth_binning': archive_metadata.get('depth_binning', 'not_declared_in_archive_metadata'),
        'ensemble_depth_policy': 'use archived band totals as-is; no strict boundary pairing with main asserted',
        'ensemble_sets': provenance, 'ensemble_salt_treatment': archive_metadata.get('salinity'),
        'member_absolute_height_conversion_area_m2': step1.AREA,
        'member_draw_quantiles_are_confidence_intervals': False,
        'point_reproduction_checks': reproduction, 'input_hashes_verified_unchanged': True,
        'new_downloads_or_15arcsec_recalculations': False,
        'minimum_reference_variance_mm2': MIN_VARIANCE,
        'metric_descriptors': model.descriptors,
        'script_sha256': step1.sha256(Path(__file__))}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    make_report(out, model, point_values, intervals, robust, provenance, metadata)
    owner = ROOT.stat()
    for path in [out] + list(out.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    log.info('Completed %s; reproduction=%s; inputs unchanged', out, reproduction)
    print('RESULTS_DIRECTORY=' + str(out))


if __name__ == '__main__':
    main()
