"""Tests for thermodynamic type adaptation, interpolation and fixed support."""
from manuscript_paths import workspace_root, input_path
import tempfile
import unittest
from pathlib import Path

import gsw
import netCDF4 as nc
import numpy as np

import en4_uniform_core as core


class UniformTests(unittest.TestCase):
    def test_auto_decoded_kelvin_not_double_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'temperature.nc'
            with nc.Dataset(str(path), 'w') as dataset:
                dataset.createDimension('sample', 2)
                variable = dataset.createVariable('temperature', 'f8', ('sample',))
                variable.units, variable.add_offset, variable.scale_factor = 'kelvin', 273.15, 1.
                variable[:] = [283.15, 288.15]
            with nc.Dataset(str(path)) as dataset:
                v = dataset.variables['temperature']
                np.testing.assert_allclose(core.decoded_temperature_c(v, v[:]), [10., 15.])

    def test_unknown_temperature_unit_rejected(self):
        class Variable:
            units = 'fahrenheit'
        with self.assertRaises(ValueError):
            core.decoded_temperature_c(Variable(), np.array([60.]))

    def test_iap_explicit_in_situ_celsius_alias_preserves_values(self):
        class Variable:
            units = 'degree Celsius (in situ)'
        values = np.array([-1., 10., 30.])
        np.testing.assert_equal(core.decoded_temperature_c(Variable(), values), values)

    def test_potential_to_in_situ_to_CT_roundtrip(self):
        sp, pt, p = np.array([34., 35., 36.]), np.array([5., 10., 15.]), np.array([100., 1000., 2000.])
        sa = gsw.SA_from_SP(sp, p, np.array([10., 100., 200.]), np.array([-30., 0., 30.]))
        ct = gsw.CT_from_pt(sa, pt)
        t = gsw.t_from_CT(sa, ct, p)
        np.testing.assert_allclose(gsw.CT_from_t(sa, t, p), ct, atol=1e-10)
        self.assertGreater(float(np.max(np.abs(t - pt))), .01)

    def test_equal_month_nanmean(self):
        stack = np.array([[1., np.nan], [3., 2.], [8., 4.]])
        total, count = np.zeros(2), np.zeros(2, dtype=np.uint8)
        for values in stack:
            core.accumulate(total, count, values)
        np.testing.assert_allclose(core.divide_mean(total, count), np.nanmean(stack, axis=0))

    def test_periodic_longitude_no_clamped_seam(self):
        values = np.array([[[0., 10., 20., 30.], [0., 10., 20., 30.]]])
        result = core.horizontal(values, np.array([-1., 1.]), np.array([0., 90., 180., 270.]),
                                 np.array([0.]), np.array([-45., 45.]))
        np.testing.assert_allclose(result[0, 0], [15., 5.])

    def test_exact_nodes_do_not_inherit_zero_weight_missing_neighbor(self):
        values = np.array([[[3., np.nan], [3., np.nan]]])
        result = core.horizontal(values, np.array([-1., 1.]), np.array([0., 1.]),
                                 np.array([0.]), np.array([0., .5]))
        self.assertAlmostEqual(result[0, 0, 0], 3.)
        self.assertTrue(np.isnan(result[0, 0, 1]))

    def test_latitude_outside_source_is_missing(self):
        result = core.horizontal(np.ones((1, 2, 2)), np.array([-1., 1.]), np.array([0., 180.]),
                                 np.array([-2., 0., 2.]), np.array([0.]))
        self.assertTrue(np.isnan(result[0, 0, 0]))
        self.assertEqual(result[0, 1, 0], 1.)
        self.assertTrue(np.isnan(result[0, 2, 0]))

    def test_vertical_surface_hold_and_no_deep_extrapolation(self):
        values = np.array([[[10.]], [[20.]], [[30.]]])
        mapped = core.vertical(values, np.array([5., 15., 25.]), np.array([1., 5., 10., 25.]))
        np.testing.assert_allclose(mapped[:, 0, 0], [10., 10., 15., 30.])
        with self.assertRaises(ValueError):
            core.vertical(values, np.array([5., 15., 25.]), np.array([26.]))

    def test_vertical_exact_finite_endpoint_with_missing_neighbor(self):
        values = np.array([[[np.nan]], [[20.]], [[30.]]])
        mapped = core.vertical(values, np.array([5., 15., 25.]), np.array([10., 15., 20.]))
        self.assertTrue(np.isnan(mapped[0, 0, 0]))
        np.testing.assert_allclose(mapped[1:, 0, 0], [20., 25.])

    def test_static_mask_intersects_all_products_and_years(self):
        volumes = np.ones((1, 1, 3))
        states = [{key: np.ones_like(volumes) for key in core.FIELDS} for _ in range(3)]
        states[0]['en4_t'][0, 0, 0] = np.nan
        states[2]['iap_sa'][0, 0, 1] = np.nan
        np.testing.assert_equal(core.common_mask(states, volumes), [[[False, False, True]]])

    def test_legacy_thermal_formula_and_identical_input_zero(self):
        depth = np.array([100.])
        t0, t1 = np.array([[[10.]]]), np.array([[[11.]]])
        s0 = s1 = np.array([[[35.]]])
        volume, mask = np.array([[[1e9]]]), np.ones((1, 1, 1), dtype=bool)
        result = core.layer_changes(t0, s0, t1, s1, depth, volume, mask)
        ct0, ct1 = gsw.CT_from_t(35., 10., 100.), gsw.CT_from_t(35., 11., 100.)
        expected = 1e9 * (gsw.alpha(35., ct0, 100.) + gsw.alpha(35., ct1, 100.)) / 2. * (ct1 - ct0)
        self.assertAlmostEqual(result[0, 0, 0], expected)
        np.testing.assert_equal(core.layer_changes(t0, s0, t0, s0, depth, volume, mask), 0.)

    def test_main_depth_group_boundaries_preserved(self):
        depth = np.array([1., 700., 750., 2000.])
        upper, middle, total = [core.group_mask(depth, group) for group in core.GROUPS]
        np.testing.assert_equal(upper, [True, True, False, False])
        np.testing.assert_equal(middle, [False, False, True, True])
        np.testing.assert_equal(upper | middle, total)
        self.assertFalse(np.any(upper & middle))


if __name__ == '__main__':
    unittest.main()
