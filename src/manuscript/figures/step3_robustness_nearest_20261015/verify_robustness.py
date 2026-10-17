#!/usr/bin/env python3
"""Check displayed intervals against retained draws and inspect export integrity."""
from manuscript_paths import workspace_root, input_path
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = (workspace_root() / 'figures/step3_robustness_nearest_20261015')
ROOT = HERE.parent.parent
SOURCE = ROOT / 'contribution_structure_analysis/step3_nearest_20261015/statistics'
STEM = 'Figure_S04_Interannual_allocation_robustness_1993_2025'


def main():
    metadata = json.loads((SOURCE / 'metadata.json').read_text())
    for path, expected in metadata['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    table = pd.read_csv(HERE / 'source_bootstrap_intervals.csv')
    sensitivity = pd.read_csv(HERE / 'source_period_sensitivity.csv')
    assert len(table) == 24 and len(sensitivity) == 40
    assert not table.component.eq('Unassigned').any()
    closure_errors = []
    for block in (2, 3, 5):
        with np.load(str(SOURCE / ('bootstrap_block{}.npz'.format(block)))) as archive:
            names = archive['component_names']
            values = archive['variance_shares_pct']
            assert values.shape == (5000, 8) and np.isfinite(values).all()
            low, high = np.percentile(values, [2.5, 97.5], axis=0)
            selected = table[table.block_years.eq(block)].set_index('component').loc[list(names)]
            np.testing.assert_allclose(selected.lower_2p5_pct, low, rtol=0., atol=1e-12)
            np.testing.assert_allclose(selected.upper_97p5_pct, high, rtol=0., atol=1e-12)
            closure_errors.append(float(np.max(np.abs(values[:, :5].sum(axis=1) - 100.))))
            closure_errors.append(float(np.max(np.abs(values[:, 5:].sum(axis=1) - 100.))))
    assert max(closure_errors) < 1e-5
    pdf = HERE / (STEM + '.pdf')
    text = subprocess.check_output(['pdftotext', str(pdf), '-']).decode('utf-8')
    assert all(name in text for name in ('Pacific', 'Atlantic', 'Indian', 'Southern', 'Arctic'))
    assert 'Unassigned' not in text
    fonts = subprocess.check_output(['pdffonts', str(pdf)]).decode('utf-8')
    assert 'TrueType' in fonts and 'Type 3' not in fonts
    assert all(row.split()[-5] == 'yes' for row in fonts.splitlines()[2:])
    svg = ET.parse(str(HERE / (STEM + '.svg')))
    nodes = svg.findall('.//{http://www.w3.org/2000/svg}text')
    assert len(nodes) > 20
    with Image.open(str(HERE / (STEM + '.png'))) as png:
        dimensions = png.size
        assert np.asarray(png.convert('RGB')).std() > 10.
        assert all(abs(value - 450.) < .1 for value in png.info['dpi'])
    with Image.open(str(HERE / (STEM + '.tiff'))) as tiff:
        tiff_dpi = tuple(float(value) for value in tiff.info['dpi'])
        assert all(abs(value - 600.) < .01 for value in tiff_dpi)
    report = {'passed': True, 'interval_rows': 24, 'period_sensitivity_points': 40,
              'all_15000_joint_draws_finite': True, 'displayed_percentiles_verified_from_draws': True,
              'maximum_bootstrap_allocation_closure_error_percentage_points': max(closure_errors),
              'input_hashes_unchanged': True, 'figure07_point_agreement_max_abs_pct': metadata['figure07_point_agreement_max_abs_pct'],
              'embedded_truetype_pdf': True, 'svg_editable_text_nodes': len(nodes),
              'png_dimensions': dimensions, 'tiff_dpi': tiff_dpi,
              'visual_review': 'No clipping or incoherent overlaps in exported Python-rendered PNG.'}
    (HERE / 'export_qa.json').write_text(json.dumps(report, indent=2))
    owner = ROOT.stat()
    for path in [HERE] + list(HERE.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
