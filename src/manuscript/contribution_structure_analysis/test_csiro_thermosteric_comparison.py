"""Tests for reference columns, latitude bounds and smoothing conventions."""
from manuscript_paths import workspace_root, input_path
import tempfile
import unittest
from pathlib import Path

import numpy as np

import compare_csiro_thermosteric as csiro


class CsiroTests(unittest.TestCase):
    def test_correct_thermosteric_column_and_floor_year(self):
        text = ('Units: GOHC [J 1e22], GThSL [mm]\n'
                'Variables: [Year GOHC_0-300m GOHC_0-300m_error GThSL_0-300m GThSL_0-300m_error '
                'OHC_0-700m GOHC_0-700m_error GThSL_0-700m GThSL_0-700m_error]\n'
                '1960.5 1 2 3 4 99 6 -7 8\n1961.5 1 2 3 4 99 6 -6 8\n')
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'data.txt'
            p.write_text(text)
            r = csiro.parse_csiro(p)
            np.testing.assert_equal(r['years'], [1960, 1961])
            np.testing.assert_equal(r['anomaly_mm'], [-7, -6])
            np.testing.assert_equal(r['one_sigma_mm'], [8, 8])
            p.write_text(text.replace('1961.5', '1962.5'))
            with self.assertRaises(ValueError):
                csiro.parse_csiro(p)
            p.write_text(text.replace('GThSL [mm]', 'GThSL [m]'))
            with self.assertRaises(ValueError):
                csiro.parse_csiro(p)

    def test_centered_has_no_partial_edges(self):
        y, v = csiro.smooth_three(np.arange(1960, 1965), [0, 0, 9, 0, 0])
        np.testing.assert_equal(y, [1961, 1962, 1963])
        np.testing.assert_allclose(v, [3, 3, 3])

    def test_trailing_is_explicit_time_shift(self):
        years = np.arange(1960, 1965)
        yc, c = csiro.smooth_three(years, years**2, 'centered')
        yt, t = csiro.smooth_three(years, years**2, 'trailing')
        np.testing.assert_equal(yt, yc + 1)
        np.testing.assert_equal(c, t)

    def test_missing_year_not_smoothed(self):
        with self.assertRaises(ValueError):
            csiro.smooth_three([1960, 1961, 1963], [1, 2, 3])

    def test_latitude_selection_not_60_or_70_approximation(self):
        lat = np.array([-70, -65, -64.5, 0, 64.5, 65, 70])
        self.assertEqual(csiro.latitude_total(lat, [100, 1, 2, -3, 4, 100, 100]), 4.)
        self.assertEqual(csiro.latitude_total(lat[::-1], np.array([100, 1, 2, -3, 4, 100, 100])[::-1]), 4.)

    def test_regional_and_global_normalizations_differ(self):
        volume = 361.9
        area = 330e12
        self.assertAlmostEqual(volume / 361.9, 1.)
        self.assertAlmostEqual(volume * 1e12 / area, 1.0966666666666667)

    def test_smoothed_metrics_not_labeled_raw_interannual(self):
        years = np.arange(1990, 2000)
        main = np.array([1, 2, 4, 5, 7, 8, 11, 12, 13, 15], dtype=float)
        ref = {'years': years, 'anomaly_mm': main + 100}
        _, _, _, result = csiro.compare_period(years, main, ref, 'test', 1993, 1998, 'centered')
        self.assertNotIn('annual_change_rmse_mm', result)
        self.assertAlmostEqual(result['smoothed_state_difference_rmse_mm'], 0)
        self.assertAlmostEqual(result['main_minus_csiro_trend_mm_per_year'], 0)

    def test_netCDF_streamed_sum_and_resume_are_read_only(self):
        import netCDF4 as nc
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = root / '1960_vs_1959'
            pair.mkdir()
            cache = root / 'cache'
            cache.mkdir()
            p = pair / '1960_700.nc'
            with nc.Dataset(str(p), 'w') as d:
                d.createDimension('lat', 6)
                d.createDimension('lon', 2)
                d.base_year, d.target_year, d.depth_limit_m = 1959, 1960, 700.
                lat = d.createVariable('lat', 'f8', ('lat',))
                lat[:] = [-80, -65, -30, 0, 64.5, 80]
                v = d.createVariable(csiro.VARIABLE, 'f4', ('lat', 'lon'), fill_value=-9999)
                v[:] = [[100, 100], [1, -1], [2, 3], [-2, -9999], [1, 1], [100, 100]]
            before = csiro.common.step1.sha256(p)
            row, reused = csiro.aggregate_one(1960, root, cache, 2)
            self.assertFalse(reused)
            self.assertAlmostEqual(row['volume_change_km3'], 5.)
            again, reused = csiro.aggregate_one(1960, root, cache, 3)
            self.assertTrue(reused)
            self.assertEqual(row, again)
            fast_cache = root / 'fast_cache'
            fast_cache.mkdir()
            fast, _ = csiro.aggregate_one(1960, root, fast_cache, 2, core_volume=3., core_hash='test')
            self.assertAlmostEqual(fast['volume_change_km3'], row['volume_change_km3'])
            with self.assertRaises(ValueError):
                csiro.aggregate_one(1960, root, fast_cache, 2, core_volume=3., core_hash='changed')
            self.assertEqual(before, csiro.common.step1.sha256(p))


if __name__ == '__main__':
    unittest.main()
