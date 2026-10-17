"""Regression tests for Ishii-specific parsing and alignment."""
from manuscript_paths import workspace_root, input_path
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compare_ishii_thermosteric as ishii
import inspect_ishii_units as units


class IshiiComparisonTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'series.txt'
            path.write_text(text)
            return ishii.parse_ishii(path)

    def test_no_header_negative_anomaly_preserved(self):
        r = self.parse('1955 -12.2585 2.9\n1956 -9.5 2.8\n1957 -8 2.7\n1958 -7 2.6\n')
        self.assertAlmostEqual(r['anomaly_mm'][0], -12.2585)
        self.assertEqual(r['third_column_raw'][0], 2.9)

    def test_fractional_year_not_rounded(self):
        with self.assertRaises(ValueError):
            self.parse('1955.5 1 2\n1956.5 2 2\n1957.5 3 2\n1958.5 4 2\n')

    def test_missing_year_rejected(self):
        with self.assertRaises(ValueError):
            self.parse('1955 1 2\n1956 2 2\n1958 3 2\n1959 4 2\n')

    def test_sentinel_rejected(self):
        with self.assertRaises(ValueError):
            self.parse('1955 99999 2\n1956 2 2\n1957 3 2\n1958 4 2\n')

    def test_band_totals_and_rounding(self):
        refs = {key: {'years': np.arange(4), 'anomaly_mm': np.full(4, v)}
                for key, v in [('0-700m', 1.), ('700-2000m', 2.), ('0-2000m', 3.00005)]}
        self.assertAlmostEqual(ishii.validate_depth_totals(refs), .00005)
        refs['0-2000m']['anomaly_mm'][1] = 5
        with self.assertRaises(ValueError):
            ishii.validate_depth_totals(refs)

    def test_units_checked_against_source_not_main(self):
        checks = {band: {'text_change_to_grid_change_ratio': r}
                  for band, r in [('0-700m', 984.93), ('0-2000m', 1044.39)]}
        self.assertIn('mm supported', ishii.validate_unit_checks(checks))
        checks['0-700m']['text_change_to_grid_change_ratio'] = 1.
        with self.assertRaises(ValueError):
            ishii.validate_unit_checks(checks)

    def test_comparison_offset_and_endpoint_counts(self):
        years = np.arange(1990, 2000)
        primary = (years - 1990.) * 2
        ref = {'years': years, 'anomaly_mm': primary + 100}
        _, _, _, result = ishii.compare(years, primary, ref, 1993, 1998)
        self.assertEqual(result['annual_change_count'], 5)
        self.assertAlmostEqual(result['ishii_ols_trend_mm_per_year'], 2.)
        self.assertAlmostEqual(result['anchored_anomaly_rmse_mm'], 0.)
        with self.assertRaises(ValueError):
            ishii.compare(years, primary, ref, 1989, 1998)

    def test_incomplete_grib_prefix_retains_only_complete_messages(self):
        message = b'GRIB' + (12).to_bytes(3, 'big') + b'\x01' + b'7777'
        self.assertEqual(units.first_complete_grib_messages(message * 2 + message[:9]), message * 2)
        with self.assertRaises(ValueError):
            units.first_complete_grib_messages(b'HTML')


if __name__ == '__main__':
    unittest.main()
