#!/usr/bin/env python3
"""Plot five-basin variability after verified nearest-cell reassignment."""
from manuscript_paths import workspace_root, input_path
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = (workspace_root() / 'figures/step2_interannual_2025_nearest_20261015')
ROOT = HERE.parent.parent
SOURCE = ROOT / 'contribution_structure_analysis/nearest_basin_reassignment_20261015/reassigned_statistics_1993_2025'
pass  # Source imports are configured by the launcher.
import plot_interannual_variability_2025 as original


def main():
    stem = 'Figure_interannual_variability_1993_2025_nearest_basins'
    if any((HERE / (stem + '.' + suffix)).exists() for suffix in ('png', 'pdf', 'svg', 'tiff')):
        raise ValueError('Existing figure is protected from overwrite')
    metadata = json.loads((SOURCE / 'metadata.json').read_text())
    if metadata['remaining_numerical_remainder_max_abs_mm'] > metadata['numerical_acceptance_limit_mm']:
        raise ValueError('Unexplained residual cannot be hidden')
    names = original.BASINS[:5]
    contributions = pd.read_csv(SOURCE / 'variability_contributions.csv')
    basins = contributions[contributions.decomposition.eq('global_by_basin')].set_index('region').loc[list(names)].reset_index()
    depth = contributions[contributions.decomposition.eq('global_by_depth')].set_index('depth_band').loc[list(original.BANDS)].reset_index()
    pairs = pd.read_csv(SOURCE / 'component_covariance.csv')
    pairs = pairs[~pairs.is_diagonal].copy()
    if len(basins) != 5 or len(pairs) != 10:
        raise ValueError('Expected five assigned basins and ten unique pairs')
    pairs['a'] = pairs.component_a.str.split(r'\|').str[1]
    pairs['b'] = pairs.component_b.str.split(r'\|').str[1]
    if pairs.a.eq('Unassigned').any() or pairs.b.eq('Unassigned').any():
        raise ValueError('Unassigned component was not actually reassigned')
    pairs = pairs.assign(magnitude=pairs.term_contribution_to_reference_variance_pct.abs()).sort_values('magnitude', ascending=False)
    original.style()
    fig = plt.figure(figsize=(7.2, 4.8))
    a = fig.add_axes([.13, .52, .295, .35])
    b = fig.add_axes([.13, .13, .295, .23])
    c = fig.add_axes([.75, .13, .215, .74])
    original.contribution_axis(a, list(names), basins.variance_contribution_pct.to_numpy(), original.BASIN_COLORS[:5])
    original.contribution_axis(b, ['0-700 m', '700-2000 m', '2000-6000 m'],
                               depth.variance_contribution_pct.to_numpy(), original.DEPTH_COLORS)
    a.set_title('1993-2025', loc='left', fontsize=7.6, fontweight='bold', pad=9)
    for y, (_, row) in enumerate(pairs.iterrows()):
        value = float(row.term_contribution_to_reference_variance_pct)
        color = original.OFFSET if value < 0 else original.REINFORCE
        c.plot([0, value], [y, y], color=color, linewidth=2., solid_capstyle='round', zorder=3)
        c.scatter([value], [y], s=14, color=color, edgecolors='white', linewidths=.4, zorder=4)
        c.text(value + .9, y, '{:+.2f}'.format(value), va='center', ha='left', fontsize=6.1,
               color='#30383d', zorder=5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .45})
    c.set_yticks(np.arange(10))
    c.set_yticklabels(['{} $\\times$ {}'.format(row.a, row.b) for _, row in pairs.iterrows()])
    c.set_ylim(9.6, -.6)
    c.set_xlim(-36., 8.)
    c.set_xticks([-30, -20, -10, 0])
    c.tick_params(axis='y', length=0, pad=5)
    c.axvline(0, color='#9ca6ad', linewidth=.7, zorder=1)
    c.grid(axis='x', color='#e8eaec', linewidth=.5, zorder=0)
    c.set_xlabel('Covariance contribution (%)\n$2\\,\\mathrm{Cov}(i,j)/\\mathrm{Var}(G)$', labelpad=4, fontsize=6.4)
    c.legend(handles=[Line2D([], [], color=original.OFFSET, marker='o', ms=3., lw=1., label='Offsetting'),
                      Line2D([], [], color=original.REINFORCE, marker='o', ms=3., lw=1., label='Reinforcing')],
              loc='lower left', bbox_to_anchor=(-.02, 1.045), ncol=2, frameon=False,
              fontsize=6., handlelength=1.1, handletextpad=.35, columnspacing=.8, borderaxespad=0.)
    for x, y, label in ((.045, .905, 'a'), (.045, .395, 'b'), (.525, .905, 'c')):
        fig.text(x, y, label, fontweight='bold', fontsize=10.)
    qa = original.check_layout(fig)
    for suffix in ('png', 'pdf', 'svg', 'tiff'):
        fig.savefig(str(HERE / (stem + '.' + suffix)), dpi=450 if suffix == 'png' else 600,
                    facecolor='white', bbox_inches=None)
    plt.close(fig)
    basins.to_csv(HERE / 'source_five_basin_variance.csv', index=False)
    depth.to_csv(HERE / 'source_depth_variance_unchanged.csv', index=False)
    pairs.to_csv(HERE / 'source_ten_basin_covariances.csv', index=False)
    qa.update({'actual_spatial_reassignment': True, 'no_proportional_residual_allocation': True,
               'no_forced_renormalization': True, 'basin_variance_share_sum_pct': float(basins.variance_contribution_pct.sum()),
               'source_reassignment_metadata': metadata})
    (HERE / 'verification.json').write_text(json.dumps(qa, indent=2))
    owner = ROOT.stat()
    for path in [HERE] + list(HERE.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    print('Saved five-basin figure:', HERE)


if __name__ == '__main__':
    main()
