from manuscript_paths import workspace_root, input_path
import unittest

import numpy as np

import calculate_interannual_variability as calc


class InterannualTests(unittest.TestCase):
    def setUp(self):
        self.years = np.arange(1994, 2024)
        self.signal = np.sin(np.arange(30) * 1.7) + np.cos(np.arange(30) * .4)

    def test_linear_detrend_removes_mean_and_slope(self):
        residual, slope = calc.detrend(7 + .2 * self.years, self.years)
        np.testing.assert_allclose(residual, 0, atol=1e-12)
        self.assertAlmostEqual(float(slope), .2)

    def test_detrending_commutes_with_addition(self):
        data = np.stack([self.signal, np.cos(np.arange(30))])
        a, _ = calc.detrend(data, self.years)
        b, _ = calc.detrend(data.sum(axis=0), self.years)
        np.testing.assert_allclose(a.sum(axis=0), b, atol=1e-14)

    def test_reinforcing_components(self):
        data = np.stack([self.signal, 2 * self.signal])
        result = calc.variance_decomposition(data, data.sum(axis=0), self.years)
        np.testing.assert_allclose(result['fractions'], [1 / 3, 2 / 3], atol=1e-14)

    def test_negative_and_above_100_percent_are_retained(self):
        data = np.stack([2 * self.signal, -self.signal])
        result = calc.variance_decomposition(data, data.sum(axis=0), self.years)
        np.testing.assert_allclose(result['fractions'], [2, -1], atol=1e-14)

    def test_sample_covariance_and_off_diagonal_factor(self):
        data = np.stack([self.signal, np.cos(np.arange(30))])
        result = calc.variance_decomposition(data, data.sum(axis=0), self.years)
        expected = np.cov(result['residuals'], ddof=1)
        np.testing.assert_allclose(result['covariance'], expected)
        reconstructed = np.trace(expected) + 2 * expected[0, 1]
        self.assertAlmostEqual(reconstructed, result['reference_variance'])

    def test_zero_reference_has_undefined_fractions(self):
        data = np.stack([self.signal, -self.signal])
        result = calc.variance_decomposition(data, data.sum(axis=0), self.years)
        self.assertTrue(np.isnan(result['fractions']).all())
        self.assertEqual(result['reference_std'], 0.)

    def test_zero_component_is_supported(self):
        data = np.stack([self.signal, np.zeros(30)])
        result = calc.variance_decomposition(data, data.sum(axis=0), self.years)
        np.testing.assert_allclose(result['fractions'], [1, 0])
        self.assertTrue(np.isnan(result['correlation'][1]))

    def test_nonexclusive_components_are_rejected(self):
        with self.assertRaises(ValueError):
            calc.variance_decomposition(np.stack([self.signal, self.signal]), self.signal, self.years)

    def test_shifted_years_do_not_change_statistics(self):
        data = np.stack([self.signal, np.cos(np.arange(30))])
        a = calc.variance_decomposition(data, data.sum(axis=0), self.years)
        b = calc.variance_decomposition(data, data.sum(axis=0), self.years + 100)
        np.testing.assert_allclose(a['fractions'], b['fractions'])

    def test_depth_groups_exclude_overlapping_auxiliary_band(self):
        definitions = calc.groups()
        depth = next(row for row in definitions if row[0] == 'global_by_depth')
        self.assertEqual(len(depth[2]), 3)
        self.assertNotIn(('depth', 'Global', '0-2000m'), depth[2])
        basins = next(row for row in definitions if row[0] == 'global_by_basin')
        self.assertEqual(len(basins[2]), 6)
        self.assertIn(('basin', 'Southern', '0-6000m'), basins[2])
        self.assertIn(('basin', 'Arctic', '0-6000m'), basins[2])


if __name__ == '__main__':
    unittest.main()
