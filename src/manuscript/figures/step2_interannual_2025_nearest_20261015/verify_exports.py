#!/usr/bin/env python3
"""Independently check the reassigned figure data and exported files."""
from manuscript_paths import workspace_root, input_path
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = (workspace_root() / 'figures/step2_interannual_2025_nearest_20261015')
ROOT = HERE.parent.parent
SOURCE = ROOT / 'contribution_structure_analysis/nearest_basin_reassignment_20261015/reassigned_statistics_1993_2025'
OLD = ROOT / 'contribution_structure_analysis/step2_interannual_20261014'
STEM = 'Figure_interannual_variability_1993_2025_nearest_basins'


def main():
    basins = pd.read_csv(HERE / 'source_five_basin_variance.csv')
    depths = pd.read_csv(HERE / 'source_depth_variance_unchanged.csv')
    pairs = pd.read_csv(HERE / 'source_ten_basin_covariances.csv')
    assert set(basins.region) == {'Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic'}
    assert len(basins) == 5 and len(depths) == 3 and len(pairs) == 10
    assert not pairs[['a', 'b']].eq('Unassigned').any().any()
    original = pd.read_csv(OLD / 'variability_contributions.csv')
    original = original[original.period.eq('recent_extended') & original.decomposition.eq('global_by_depth')]
    pd.testing.assert_frame_equal(depths.set_index('depth_band').sort_index()[original.columns.drop('depth_band')],
                                  original.set_index('depth_band').sort_index(), check_exact=True)
    closure = pd.read_csv(SOURCE / 'allocation_closure.csv')
    corrected = pd.read_csv(SOURCE / 'corrected_annual_basin_contributions.csv')
    assert np.array_equal(closure.target_year.to_numpy(), np.arange(1994, 2026))
    assert len(corrected) == 160
    assert corrected.groupby('target_year').size().eq(5).all()
    sums = corrected.groupby('target_year').reassigned_global_contribution_mm.sum()
    remainder = closure.set_index('target_year').original_global_mm - sums
    assert np.abs(remainder).max() < 5e-7
    assert np.isclose(basins.variance_contribution_pct.sum(), 100., rtol=0., atol=1e-6)
    old_residuals = pd.read_csv(OLD / 'detrended_annual_series.csv')
    old_global = old_residuals[old_residuals.period.eq('recent_extended') & old_residuals.component_type.eq('global')]
    new_residuals = pd.read_csv(SOURCE / 'detrended_annual_series.csv')
    new_global = new_residuals[new_residuals.component_type.eq('global')]
    np.testing.assert_allclose(new_global.sort_values('target_year').detrended_annual_change_mm,
                               old_global.sort_values('target_year').detrended_annual_change_mm,
                               rtol=0., atol=1e-14)
    pdf = HERE / (STEM + '.pdf')
    contents = subprocess.check_output(['pdftotext', str(pdf), '-']).decode('utf-8')
    assert 'Unassigned' not in contents
    assert all(name in contents for name in basins.region)
    fonts = subprocess.check_output(['pdffonts', str(pdf)]).decode('utf-8')
    assert 'TrueType' in fonts and 'Type 3' not in fonts
    font_rows = fonts.splitlines()[2:]
    assert font_rows and all(row.split()[-5] == 'yes' for row in font_rows)
    svg = ET.parse(str(HERE / (STEM + '.svg')))
    text_nodes = svg.findall('.//{http://www.w3.org/2000/svg}text')
    assert text_nodes
    with Image.open(str(HERE / (STEM + '.png'))) as png:
        dimensions = png.size
        assert np.asarray(png.convert('RGB')).std() > 10.
    with Image.open(str(HERE / (STEM + '.tiff'))) as tiff:
        tiff_dpi = tuple(float(value) for value in tiff.info['dpi'])
        assert all(abs(value - 600.) < .01 for value in tiff_dpi)
    report = {'passed': True, 'basins': 5, 'depth_bands': 3, 'basin_pairs': 10,
              'annual_changes': 32, 'global_series_unchanged': True, 'depth_statistics_unchanged': True,
              'no_unassigned_in_figure': True, 'pdf_fonts_embedded': True,
              'svg_editable_text_nodes': len(text_nodes), 'png_dimensions': dimensions,
              'tiff_dpi': tiff_dpi, 'max_annual_closure_remainder_mm': float(np.abs(remainder).max())}
    (HERE / 'export_qa.json').write_text(json.dumps(report, indent=2))
    owner = ROOT.stat()
    for path in [HERE] + list(HERE.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
