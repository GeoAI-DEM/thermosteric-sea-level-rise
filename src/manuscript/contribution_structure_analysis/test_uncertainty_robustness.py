from manuscript_paths import workspace_root, input_path
import unittest

import numpy as np

import calculate_long_term_contributions as step1
import calculate_interannual_variability as step2
import calculate_uncertainty_robustness as calc


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.model = calc.StatisticsModel()
        self.atoms = np.random.RandomState(12).normal(size=(18, 30)) * .1 + .05

    def descriptor_index(self, metric, kind, region, depth, group):
        return next(i for i, d in enumerate(self.model.descriptors)
                    if (d['metric'], d['component_type'], d['region'], d['depth_band'], d['decomposition'])
                    == (metric, kind, region, depth, group))

    def test_closed_form_cumulative_trend_matches_step1(self):
        values = self.model.evaluate(self.atoms)
        k = self.descriptor_index('ols_trend_mm_per_year', 'global', 'Global', '0-6000m', 'long_term')
        expected = step1.fit_cumulative(self.atoms.sum(axis=0), 1993)['cumulative_ols_slope_mm_per_year']
        self.assertAlmostEqual(values[k], expected, places=12)

    def test_vectorized_variance_matches_step2(self):
        values = self.model.evaluate(self.atoms)
        data = np.stack([self.atoms[i * 3:(i + 1) * 3].sum(axis=0) for i in range(6)])
        expected = step2.variance_decomposition(data, data.sum(axis=0), np.arange(30))
        for i, basin in enumerate(calc.BASINS):
            k = self.descriptor_index('variance_share_pct', 'basin', basin, '0-6000m', 'global_by_basin')
            self.assertAlmostEqual(values[k], expected['fractions'][i] * 100, places=10)

    def test_batch_and_individual_evaluation_match(self):
        batch = np.stack([self.atoms, 2 * self.atoms])
        a = self.model.evaluate(batch)
        b = np.stack([self.model.evaluate(data) for data in batch])
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_signed_and_above_100_shares_are_not_clipped(self):
        signal = np.sin(np.arange(30))
        atoms = np.zeros((18, 30))
        atoms[0] = 2 * signal
        atoms[3] = -signal
        values = self.model.evaluate(atoms)
        pacific = self.descriptor_index('variance_share_pct', 'basin', 'Pacific', '0-6000m', 'global_by_basin')
        atlantic = self.descriptor_index('variance_share_pct', 'basin', 'Atlantic', '0-6000m', 'global_by_basin')
        self.assertAlmostEqual(values[pacific], 200)
        self.assertAlmostEqual(values[atlantic], -100)

    def test_circular_blocks_are_in_range_and_contiguous(self):
        indices = calc.moving_indices(15, 100, 3, np.random.RandomState(0))
        self.assertEqual(indices.shape, (100, 15))
        self.assertTrue(((indices >= 0) & (indices < 15)).all())
        chunks = indices.reshape(100, 5, 3)
        np.testing.assert_array_equal(np.diff(chunks, axis=-1) % 15, np.ones((100, 5, 2)))

    def test_resampling_is_reproducible(self):
        a = calc.bootstrap_draws(self.atoms, self.model, 50, 3, 9)
        b = calc.bootstrap_draws(self.atoms, self.model, 50, 3, 9)
        np.testing.assert_allclose(a, b, equal_nan=True)

    def test_joint_resampling_preserves_component_relations(self):
        atoms = np.zeros((18, 30))
        atoms[0] = np.sin(np.arange(30))
        atoms[3] = -.5 * atoms[0]
        draws = calc.bootstrap_draws(atoms, self.model, 50, 3, 7)
        k = self.descriptor_index('variance_share_pct', 'basin', 'Pacific', '0-6000m', 'global_by_basin')
        np.testing.assert_allclose(draws[:, k], 200, atol=1e-9)

    def test_simultaneous_families_have_expected_sizes(self):
        point = self.model.evaluate(self.atoms)
        draws = calc.bootstrap_draws(self.atoms, self.model, 100, 3, 22)
        for metric, expected in (('variance_share_pct', 8), ('ols_trend_mm_per_year', 9)):
            rows = calc.simultaneous_intervals(draws - point, np.zeros_like(point),
                                               self.model.descriptors, metric, {})
            self.assertEqual(len(rows), expected)
            self.assertTrue(all(row['family_size'] == expected for row in rows))

    def test_15_year_period_selects_15_increments(self):
        years = np.arange(1993, 2024)
        values = np.arange(31)[None]
        selected = calc.period_atoms(values, years, 1993, 2008)
        self.assertEqual(selected.shape, (1, 15))
        np.testing.assert_array_equal(selected[0], np.arange(1, 16))

    def test_upper_model_has_only_upper_atoms(self):
        model = calc.StatisticsModel(upper_only=True)
        self.assertEqual(len(model.atoms), 6)
        model.evaluate(np.ones((6, 20)))
        self.assertTrue(all(d['depth_band'] == '0-700m' for d in model.descriptors))


if __name__ == '__main__':
    unittest.main()
