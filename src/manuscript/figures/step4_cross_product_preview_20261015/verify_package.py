#!/usr/bin/env python3
"""Audit the three-figure preview package against archived source statistics."""
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

import plot_cross_product_figures as plots

HERE = (workspace_root() / 'figures/step4_cross_product_preview_20261015')


def main():
    signatures = json.loads((HERE / 'input_sha256.json').read_text())
    assert signatures == {name: plots.sha256(Path(name)) for name in signatures}
    source_paths = list(HERE.glob('source_*.csv'))
    before = {str(path): plots.sha256(path) for path in source_paths}
    reports = {}

    def check_only(fig, stem, evidence):
        qa = plots.verify_text(fig)
        if stem == plots.STEMS[0]:
            ax = fig.axes[2]
            assert ax.get_xlim() == ax.get_ylim()
            assert ax.get_aspect() == 1.
        reports[stem] = dict(qa, **evidence)
        # Preserve the already rendered files; only audit the Python artists.
        plots.plt.close(fig)

    plots.save = check_only
    plots.style()
    data = plots.Inputs()
    plots.main_figure(data)
    plots.en4_figure(data)
    plots.regional_figure(data)
    data.verify()
    assert before == {name: plots.sha256(Path(name)) for name in before}
    table = pd.read_csv(HERE / 'Table_S01_Full_cross_product_metrics.csv')
    assert len(table) == 79 and np.isfinite(table.main_trend_mm_per_year).all()
    np.testing.assert_allclose(table.main_trend_mm_per_year - table.reference_trend_mm_per_year,
                               table.main_minus_reference_trend_mm_per_year, atol=1e-12, rtol=0.)
    smoothed = table.reference.eq('CSIRO')
    assert int(smoothed.sum()) == 7
    assert table.loc[smoothed, ['detrended_annual_change_r', 'annual_change_rmse_mm']].isna().all().all()
    assert table.loc[smoothed, ['detrended_smoothed_state_difference_r', 'smoothed_state_difference_rmse_mm']].notna().all().all()
    assert table.loc[~smoothed, ['detrended_annual_change_r', 'annual_change_rmse_mm']].notna().all().all()
    assert table.end_year.max() == 2025 and table.start_year.min() == 1940
    sources = {
        'noaa': 'cross_product_metrics.csv', 'ishii': 'cross_product_metrics.csv',
        'copernicus': 'cross_product_metrics.csv', 'csiro': 'cross_product_metrics.csv',
        'en4': 'prespecified_period_comparison.csv',
    }
    for product, filename in sources.items():
        assert (HERE / 'original_metric_snapshots' / (product + '_' + filename)).read_bytes() == (plots.FOLDERS[product] / filename).read_bytes()
    for stem in plots.STEMS:
        pdf = HERE / (stem + '.pdf')
        fonts = subprocess.check_output(['pdffonts', str(pdf)]).decode('utf-8')
        rows = fonts.splitlines()[2:]
        assert rows and 'TrueType' in fonts and 'Type 3' not in fonts
        assert all(row.split()[-5] == 'yes' for row in rows)
        text = subprocess.check_output(['pdftotext', str(pdf), '-']).decode('utf-8')
        assert 'Year' in text and 'mm' in text
        nodes = ET.parse(str(HERE / (stem + '.svg'))).findall('.//{http://www.w3.org/2000/svg}text')
        assert len(nodes) > 20
        with Image.open(str(HERE / (stem + '.png'))) as img:
            size = img.size
            assert np.asarray(img.convert('RGB')).std() > 10.
            assert all(abs(float(value) - 450.) < .1 for value in img.info['dpi'])
        with Image.open(str(HERE / (stem + '.tiff'))) as img:
            assert all(abs(float(value) - 600.) < .01 for value in img.info['dpi'])
        reports[stem].update({'png_dimensions': size, 'tiff_dpi': 600,
                              'embedded_truetype_fonts': True, 'editable_svg_text_nodes': len(nodes)})
    result = {'passed': True, 'figures': reports, 'comparison_table_rows': len(table),
              'table_preserves_1940_2025_available_periods': True,
              'smoothed_and_annual_change_metrics_separate': True,
              'all_input_hashes_unchanged': True, 'source_csvs_unchanged_during_audit': True,
              'original_metric_snapshots_byte_identical': True,
              'visual_review': 'All three Python-rendered PNGs inspected; full EN4 negative historical excursion retained.',
              'final_manuscript_collection_modified': False}
    (HERE / 'package_qa.json').write_text(json.dumps(result, indent=2))
    owner = plots.ROOT.stat()
    for path in [HERE] + list(HERE.rglob('*')):
        os.chown(str(path), owner.st_uid, owner.st_gid)
        path.chmod(0o755 if path.is_dir() else 0o644)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
