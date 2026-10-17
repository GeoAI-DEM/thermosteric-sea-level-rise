import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'src/uncertainty'
sys.path.insert(0, str(SOURCE))
import iapv4_le_1deg_pilot as pilot

spec = importlib.util.spec_from_file_location('public_uncertainty', str(SOURCE / 'full_spatiotemporal/run_full_uncertainty.py'))
uncertainty = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uncertainty)


class UncertaintyCoordinateTests(unittest.TestCase):
    def test_longitude_permutation(self):
        source = np.broadcast_to(np.arange(360.), (2, 3, 360)).copy()
        actual = pilot.align_member_longitude(source)
        np.testing.assert_array_equal(actual[..., :180], source[..., 180:])
        np.testing.assert_array_equal(actual[..., 180:], source[..., :180])
        self.assertEqual(actual.sum(), source.sum())

    def test_reject_wrong_longitude_count(self):
        with self.assertRaises(ValueError):
            pilot.align_member_longitude(np.zeros((2, 359)))

    def test_boundary_conventions(self):
        for depth, legacy, compatible in ((700., False, True), (2000., False, True)):
            lower = 0. if depth == 700. else 700.
            self.assertEqual(uncertainty.depth_in_band(depth, lower, depth, 'legacy'), legacy)
            self.assertEqual(uncertainty.depth_in_band(depth, lower, depth, 'main_compatible'), compatible)
        self.assertTrue(uncertainty.depth_in_band(6000., 2000., 6000.1, 'legacy'))

    def test_latitude_masks_broadcast_over_longitudes(self):
        lat = np.arange(-89.5, 90., 1.)
        lon = np.arange(-179.5, 180., 1.)
        volume = np.ones((1, 180, 360))
        masks = uncertainty.region_masks(lat, lon, volume, np.ones((180, 360)))
        self.assertTrue(all(mask.shape == (180, 360) for mask in masks.values()))
        self.assertEqual(sum(masks[name].sum() for name, _, _ in uncertainty.LAT10), 180 * 360)


if __name__ == '__main__':
    unittest.main()
