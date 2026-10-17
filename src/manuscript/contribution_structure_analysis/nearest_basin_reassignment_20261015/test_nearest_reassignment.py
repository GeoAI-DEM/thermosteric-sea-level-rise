"""Focused tests for spherical distance and unchanged residual estimation."""
from manuscript_paths import workspace_root, input_path
import unittest

import numpy as np
from scipy.spatial import cKDTree

import prepare_nearest_ocean_cells as geometry
from run_reassigned_variability import residual


class NearestTests(unittest.TestCase):
    def test_dateline_is_periodic(self):
        tree = cKDTree(geometry.xyz(np.array([0., 0.]), np.array([-179.9, 170.])))
        distance, index = tree.query(geometry.xyz(np.array([0.]), np.array([179.9])))
        self.assertEqual(int(index[0]), 0)
        km = 2 * np.arcsin(distance[0] / 2) * 6371.
        self.assertAlmostEqual(float(km), 22.2389853, places=5)

    def test_high_latitude_distance_is_not_raw_degree_distance(self):
        tree = cKDTree(geometry.xyz(np.array([80., 80.5]), np.array([1., 0.])))
        _, index = tree.query(geometry.xyz(np.array([80.]), np.array([0.])))
        self.assertEqual(int(index[0]), 0)

    def test_residual_matches_intercept_and_linear_least_squares(self):
        years = np.arange(1994, 2026)
        values = np.sin(np.arange(32)) + .02 * years
        design = np.column_stack([np.ones(32), years - years.mean()])
        fitted = design @ np.linalg.lstsq(design, values, rcond=None)[0]
        np.testing.assert_allclose(residual(values, years), values - fitted, atol=1e-13)

    def test_residual_does_not_remove_an_annual_anomaly(self):
        years = np.arange(1994, 2026)
        values = np.zeros(32)
        values[10] = 1.
        result = residual(values, years)
        self.assertGreater(result[10], .9)
        self.assertAlmostEqual(float(result.sum()), 0.)


if __name__ == '__main__':
    unittest.main()
