from manuscript_paths import workspace_root, input_path
import unittest
import numpy as np
from calculate_figure07_robustness import shares, draws


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.parts = np.random.RandomState(11).normal(size=(5, 32))
        self.values = np.vstack([self.parts.sum(axis=0), self.parts])

    def test_point_closure_and_scale(self):
        self.assertAlmostEqual(shares(self.values).sum(), 100., places=11)
        np.testing.assert_allclose(shares(self.values), shares(100. * self.values), atol=1e-12)

    def test_joint_bootstrap_closes_and_reproduces(self):
        a, indices = draws(self.values, 200, 3, 17)
        b, repeated = draws(self.values, 200, 3, 17)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(indices, repeated)
        np.testing.assert_allclose(a.sum(axis=1), 100., atol=1e-10)

    def test_circular_blocks(self):
        _, indices = draws(self.values, 100, 5, 20)
        for start in range(0, 30, 5):
            np.testing.assert_array_equal(np.diff(indices[:, start:start + 5], axis=1) % 32,
                                          np.ones((100, 4), dtype=int))

    def test_linear_trend_invariance_and_zero_variance(self):
        offsets = np.arange(6)[:, None] + np.arange(32)[None, :] * np.arange(1, 7)[:, None]
        np.testing.assert_allclose(shares(self.values), shares(self.values + offsets), atol=1e-11)
        self.assertTrue(np.isnan(shares(np.zeros_like(self.values))).all())


if __name__ == '__main__':
    unittest.main()
