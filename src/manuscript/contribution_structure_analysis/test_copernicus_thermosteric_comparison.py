"""Focused tests for Copernicus units, dates, member identities and regional means."""
from manuscript_paths import workspace_root, input_path
import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np

import compare_copernicus_thermosteric as comparison


def fixture(path, years=(2005, 2006, 2007, 2008), units='m', missing=False):
    with nc.Dataset(str(path), 'w') as dataset:
        dataset.createDimension('time', len(years))
        time = dataset.createVariable('time', 'f8', ('time',))
        time.units = 'days since 1950-01-01'
        time.calendar = 'gregorian'
        time[:] = nc.date2num([datetime(year, 6, 30) for year in years], time.units, time.calendar)
        for i, name in enumerate(comparison.SERIES):
            variable = dataset.createVariable(name, 'f8', ('time',), fill_value=1e20)
            variable.units = units
            variable.standard_name = 'thermosteric_change_in_mean_sea_level'
            variable[:] = np.arange(len(years)) * .001 + .0001 * i
            if missing and name == 'thsl_cora':
                variable[1] = 1e20
        for name in comparison.SPREADS:
            variable = dataset.createVariable(name, 'f8', ('time',))
            if name != 'thsl_bretherton_error':
                variable.units = units
            variable[:] = .0001


class CopernicusTests(unittest.TestCase):
    def test_height_units_explicit_not_inferred_from_size(self):
        self.assertEqual(comparison.unit_factor('m'), 1000.)
        self.assertEqual(comparison.unit_factor('mm'), 1.)
        with self.assertRaises(ValueError):
            comparison.unit_factor(None)
        with self.assertRaises(ValueError):
            comparison.unit_factor('cm')

    def test_native_calendar_and_value_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'reference.nc'
            fixture(p)
            reference = comparison.parse_reference(p)
            np.testing.assert_equal(reference['years'], [2005, 2006, 2007, 2008])
            np.testing.assert_allclose(reference['values_mm']['thsl_cora'], [0, 1, 2, 3])
            self.assertTrue(reference['metadata']['native_timestamps'][0].startswith('2005-06-30'))
            self.assertNotIn('thsl_std_observations', reference['values_mm'])

    def test_missing_value_not_treated_as_real_anomaly(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'reference.nc'
            fixture(p, missing=True)
            with self.assertRaises(ValueError):
                comparison.parse_reference(p)

    def test_duplicate_and_missing_years_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'reference.nc'
            for years in ((2005, 2005, 2006, 2007), (2005, 2006, 2008, 2009)):
                fixture(p, years=years)
                with self.assertRaises(ValueError):
                    comparison.parse_reference(p)

    def test_undocumented_auxiliary_error_units_not_assumed(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'reference.nc'
            fixture(p)
            reference = comparison.parse_reference(p)
            error = reference['spreads']['thsl_bretherton_error']
            self.assertIsNone(error['values_mm'])
            self.assertIsNone(error['raw_units'])
            np.testing.assert_allclose(error['raw_values'], .0001)

    def test_middle_pairing_preserves_identity_no_invented_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'reference.nc'
            fixture(p)
            upper, total = comparison.parse_reference(p), comparison.parse_reference(p)
            for name in total['values_mm']:
                total['values_mm'][name] += 5.
            middle = comparison.derive_middle(upper, total)
            for name in middle['values_mm']:
                np.testing.assert_allclose(middle['values_mm'][name], 5.)
            self.assertEqual(middle['spreads'], {})
            total['years'][0] = 2004
            with self.assertRaises(ValueError):
                comparison.derive_middle(upper, total)

    def test_published_group_means_checked_not_replaced(self):
        values = {name: np.arange(4, dtype=float) + i for i, name in enumerate(comparison.SERIES)}
        for mean, members in comparison.GROUPS.items():
            values[mean] = np.mean([values[name] for name in members], axis=0)
        self.assertEqual(max(comparison.verify_published_means({'values_mm': values}).values()), 0.)
        values['thsl_mean_observations'][0] += 1.
        with self.assertRaises(ValueError):
            comparison.verify_published_means({'values_mm': values})

    def test_main_uses_regional_area_and_nonoverlap_depth_sum(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'main.csv'
            columns = ['base_year', 'target_year', 'latitude_band', 'lat_lower_deg',
                       'lat_upper_deg', 'depth_band', 'volume_change_km3', 'global_contribution_mm']
            with p.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for year in range(2001, 2006):
                    for band, volume in [('0-700m', 2.), ('700-2000m', 3.)]:
                        writer.writerow(dict(base_year=year - 1, target_year=year, latitude_band='60S-60N',
                                             lat_lower_deg=-60, lat_upper_deg=60, depth_band=band,
                                             volume_change_km3=volume, global_contribution_mm=99999))
            main = comparison.load_main(p, area=1e12)
            np.testing.assert_allclose(main['annual_mm']['0-2000m'], 5.)
            np.testing.assert_allclose(main['anomalies_mm']['0-2000m'], [0, 5, 10, 15, 20, 25])
            np.testing.assert_equal(main['state_years'], np.arange(2000, 2006))

    def test_area_uses_existing_wet_pixels_not_entire_latitude_belt(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'area.nc'
            with nc.Dataset(str(p), 'w') as dataset:
                dataset.createDimension('lat', 6)
                dataset.createDimension('lon', 2)
                dataset.createVariable('lat', 'f8', ('lat',))[:] = [-60.5, -59.5, -.5, .5, 59.5, 60.5]
                area = dataset.createVariable('ocean_area_m2', 'f8', ('lat', 'lon'))
                area.units = 'm2'
                area[:] = [[100, 100], [1, 0], [2, 0], [3, 0], [4, 0], [100, 100]]
                dataset.source_dem = 'test'
                dataset.ocean_mask = 'elevation < 0'
                dataset.earth_radius_m = 6371000.
                dataset.area_formula = 'test'
            result, _ = comparison.regional_area(p)
            self.assertEqual(result, 10.)

    def test_common_statistics_offset_invariance_and_endpoint_counts(self):
        years = np.arange(2005, 2024)
        values = np.arange(19, dtype=float)**2 * .01
        result = comparison.compare_series(years, values, values + 20)
        self.assertEqual(result['annual_change_count'], 18)
        self.assertAlmostEqual(result['main_minus_copernicus_trend_mm_per_year'], 0.)
        self.assertAlmostEqual(result['anchored_anomaly_rmse_mm'], 0.)


if __name__ == '__main__':
    unittest.main()
