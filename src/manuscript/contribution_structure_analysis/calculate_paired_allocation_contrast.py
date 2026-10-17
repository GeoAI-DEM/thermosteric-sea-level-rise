#!/usr/bin/env python3
"""Reuse Step 3 joint draws to compare growth shares with variance shares."""

from manuscript_paths import workspace_root, input_path
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import calculate_long_term_contributions as step1
import calculate_interannual_variability as step2
import calculate_uncertainty_robustness as step3


def contrasts(statistics, descriptors):
    lookup = {(d['component_type'], d['region'], d['depth_band']): i
              for i, d in enumerate(descriptors) if d['metric'] == 'ols_trend_share_pct'}
    values, paired_descriptors = [], []
    for i, descriptor in enumerate(descriptors):
        if descriptor['metric'] != 'variance_share_pct' or descriptor['decomposition'] not in (
                'global_by_basin', 'global_by_depth', 'upper_by_basin'):
            continue
        key = (descriptor['component_type'], descriptor['region'], descriptor['depth_band'])
        if key not in lookup:
            raise ValueError('Matching long-term share is missing.')
        values.append(statistics[..., i] - statistics[..., lookup[key]])
        paired_descriptors.append(dict(descriptor, metric='variance_minus_trend_share_pp'))
    return np.stack(values, axis=-1), paired_descriptors


def ensemble_contrasts(path, out):
    rows = step2.read_csv(path)
    long_term = {}
    for row in rows:
        if row['metric'] == 'ols_trend_share_pct':
            key = tuple(row[k] for k in ('ensemble_set', 'period', 'member', 'component_type', 'region', 'depth_band'))
            long_term[key] = row['value']
    output, summary, grouped = [], [], {}
    for row in rows:
        if row['metric'] != 'variance_share_pct' or row['decomposition'] not in (
                'global_by_basin', 'global_by_depth', 'upper_by_basin'):
            continue
        key = tuple(row[k] for k in ('ensemble_set', 'period', 'member', 'component_type', 'region', 'depth_band'))
        previous = long_term.get(key, '')
        value = float(row['value']) - float(previous) if row['value'] != '' and previous != '' else np.nan
        record = {k: row[k] for k in ('ensemble_set', 'period', 'member', 'decomposition',
                                      'component_type', 'region', 'depth_band')}
        record.update({'variance_minus_ols_trend_share_percentage_points': step3.clean(value),
                       'source_grid': '1_degree_archive', 'strict_main_value_pairing': 'not_asserted'})
        output.append(record)
        group = tuple(record[k] for k in ('ensemble_set', 'period', 'decomposition', 'component_type', 'region', 'depth_band'))
        grouped.setdefault(group, []).append(value)
    for key, values in grouped.items():
        finite = np.asarray(values)[np.isfinite(values)]
        summary.append(dict(zip(('ensemble_set', 'period', 'decomposition', 'component_type', 'region', 'depth_band'), key),
            available_member_count=len(values), finite_member_count=len(finite),
            member_mean=float(finite.mean()) if len(finite) else '',
            member_sd=float(finite.std(ddof=1)) if len(finite) > 1 else '',
            member_p05=float(np.percentile(finite, 5)) if len(finite) else '',
            member_p95=float(np.percentile(finite, 95)) if len(finite) else '',
            positive_member_fraction=float(np.mean(finite > 0)) if len(finite) else '',
            range_type='empirical_temperature_member_spread_not_confidence_interval'))
    step1.write_csv(out / 'ensemble_paired_allocation_contrast.csv', output)
    step1.write_csv(out / 'ensemble_paired_allocation_spread.csv', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step3-dir', type=Path,
        default=step1.ROOT / 'contribution_structure_analysis/step3_uncertainty_20261014')
    args = parser.parse_args()
    metadata_path = args.step3_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    descriptors = metadata['metric_descriptors']
    out = args.step3_dir / 'paired_allocation_contrast'
    if out.exists():
        raise ValueError('Paired-contrast output exists; no overwrite is allowed.')
    out.mkdir()
    percentile_rows, simultaneous, manifest = [], [], []
    manifest.append({'path': str(metadata_path), 'sha256': step1.sha256(metadata_path)})
    for period, start, end in metadata['periods']:
        for block in metadata['block_lengths_years']:
            path = args.step3_dir / 'bootstrap_draws' / ('{}_block{}.npz'.format(period, block))
            manifest.append({'path': str(path), 'sha256': step1.sha256(path)})
            with np.load(str(path), allow_pickle=False) as saved:
                expected = np.asarray([json.dumps(step3.metric_key(d)) for d in descriptors])
                if not np.array_equal(saved['metric_keys'], expected):
                    raise ValueError('Saved bootstrap statistic ordering differs from metadata.')
                draws, paired = contrasts(saved['statistics'], descriptors)
                point, _ = contrasts(saved['point'], descriptors)
            context = {'comparison': 'allocation_' + period, 'period': period,
                       'endpoint_start_year': start, 'endpoint_end_year': end, 'block_length_years': block}
            percentile_rows.extend(step3.distribution_rows(draws, point, SimpleNamespace(descriptors=paired),
                                  context, 'paired_same_draw_variance_minus_trend_share'))
            simultaneous.extend(step3.simultaneous_intervals(draws, point, paired,
                                'variance_minus_trend_share_pp', context))
    step1.write_csv(out / 'main_paired_contrast_intervals.csv', percentile_rows)
    step1.write_csv(out / 'main_paired_contrast_simultaneous_intervals.csv', simultaneous)
    robust = step3.robustness_table(simultaneous, metadata['block_lengths_years'])
    step1.write_csv(out / 'paired_contrast_block_robustness.csv', robust)
    member_path = args.step3_dir / 'ensemble_member_statistics.csv'
    manifest.append({'path': str(member_path), 'sha256': step1.sha256(member_path)})
    ensemble_contrasts(member_path, out)
    if not all(step1.sha256(Path(r['path'])) == r['sha256'] for r in manifest):
        raise RuntimeError('A source changed during paired allocation contrast.')
    step1.write_csv(out / 'input_manifest.csv', manifest)
    lines = ['# Paired Long-Term versus Interannual Allocation Contrast', '',
        'For each SAME joint bootstrap draw, subtract the long-term cumulative-OLS trend',
        'share from the annual-variance share. The difference is in percentage points.',
        'This avoids treating two correlated estimates as independent.', '',
        'Both quantities use global denominators for the main comparison, but describe',
        'different allocations; this is not a physical rate difference or causal effect.',
        'Approximate simultaneous intervals use a family of 8 components per period.',
        'No combined correction across every sensitivity period is claimed.', '',
        '## Recent Core: 1993-2023', '',
        '| Component | Variance share minus trend share (pp) | Union of block-length simultaneous intervals | All blocks exclude zero? |',
        '|---|---:|---:|---|']
    for row in robust:
        if row['comparison'] != 'allocation_recent_core':
            continue
        name = row['depth_band'] if row['component_type'] == 'depth' else row['region']
        lines.append('| {} | {:.4f} | [{:.4f}, {:.4f}] | {} |'.format(name,
            row['point_difference'], row['minimum_simultaneous_lower'], row['maximum_simultaneous_upper'],
            row['all_blocks_exclude_zero_same_direction']))
    lines.extend(['', 'Archived ensemble contrasts are paired within each member and saved separately.',
                  'They retain archived grid/depth definitions and are NOT transferred to main estimates.',
                  'All inputs and original bootstrap draws were verified unchanged.'])
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    owner = step1.ROOT.stat()
    for path in [out] + list(out.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    print('PAIRED_CONTRAST_DIRECTORY=' + str(out))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
