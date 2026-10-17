from manuscript_paths import workspace_root, input_path
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compare_noaa_thermosteric as calc


class NoaaComparisonTests(unittest.TestCase):
    def test_noaa_year_is_floored_not_rounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.dat'
            path.write_text('YEAR WO WOse NH NHse SH SHse\n2005.500 -4.948 2.7 0 0 0 0\n2006.500 1 1 0 0 0 0\n', encoding='ascii')
            result = calc.parse_noaa(path)
            np.testing.assert_array_equal(result['years'], [2005, 2006])
            self.assertEqual(result['anomaly_mm'][0], -4.948)

    def test_linear_series_with_constant_offset_align(self):
        years = np.arange(2005, 2011)
        main = 2 * np.arange(6.) + 100
        reference = 2 * np.arange(6.) - 10
        result = calc.metrics(years, main, reference)
        self.assertAlmostEqual(result['main_ols_trend_mm_per_year'], 2)
        self.assertAlmostEqual(result['anchored_anomaly_bias_mm'], 0)
        self.assertAlmostEqual(result['anchored_anomaly_rmse_mm'], 0)
        self.assertEqual(result['annual_change_count'], 5)

    def test_anchored_bias_is_not_forced_to_zero(self):
        years = np.arange(2005, 2010)
        result = calc.metrics(years, np.arange(5.), 2 * np.arange(5.))
        self.assertAlmostEqual(result['anchored_anomaly_bias_mm'], -2)
        self.assertAlmostEqual(result['annual_change_bias_mm'], -1)
        self.assertAlmostEqual(result['main_minus_noaa_trend_mm_per_year'], -1)

    def test_year_gaps_cannot_be_differenced_as_annual(self):
        with self.assertRaises(ValueError):
            calc.metrics(np.array([2005, 2006, 2008, 2009]), np.arange(4.), np.arange(4.))

    def test_constant_correlation_is_undefined(self):
        self.assertEqual(calc.correlation(np.ones(10), np.arange(10.)), '')

    def test_correlation_is_invariant_to_offset(self):
        a = np.sin(np.arange(10.))
        self.assertAlmostEqual(calc.correlation(a + 100, a - 10), 1)

    def test_middle_layer_se_is_not_invented(self):
        upper = {'years': np.array([2005, 2006]), 'anomaly_mm': np.array([10., 11.]),
                 'standard_error_mm': np.array([1., 1.])}
        total = {'years': np.array([2005, 2006]), 'anomaly_mm': np.array([13., 15.]),
                 'standard_error_mm': np.array([2., 2.])}
        result = calc.derived_middle(upper, total)
        np.testing.assert_array_equal(result['anomaly_mm'], [3, 4])
        self.assertTrue(np.isnan(result['standard_error_mm']).all())

    def test_raw_correlation_does_not_replace_detrended_metric(self):
        years = np.arange(2005, 2025)
        x = np.arange(20.)
        result = calc.metrics(years, 10 * x + np.sin(x), 10 * x - np.sin(x))
        self.assertGreater(result['raw_anomaly_correlation'], .99)
        self.assertLess(result['detrended_anomaly_correlation'], -.99)


if __name__ == '__main__':
    unittest.main()
