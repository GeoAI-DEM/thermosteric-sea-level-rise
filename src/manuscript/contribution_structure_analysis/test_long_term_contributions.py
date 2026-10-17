from manuscript_paths import workspace_root, input_path
import unittest

import numpy as np

import calculate_long_term_contributions as calc


class LongTermContributionTests(unittest.TestCase):
    def test_constant_rate_and_baseline(self):
        metrics = calc.fit_cumulative(np.full(15, 2.), 1993)
        self.assertEqual(metrics['annual_increment_count'], 15)
        self.assertEqual(metrics['cumulative_state_count'], 16)
        self.assertEqual(metrics['cumulative_change_mm'], 30.)
        self.assertAlmostEqual(metrics['mean_annual_change_mm_per_year'], 2.)
        self.assertAlmostEqual(metrics['cumulative_ols_slope_mm_per_year'], 2.)

    def test_mean_and_ols_are_distinct(self):
        metrics = calc.fit_cumulative(np.array([0., 0., 10., 0.]), 2000)
        self.assertNotEqual(metrics['mean_annual_change_mm_per_year'], metrics['cumulative_ols_slope_mm_per_year'])

    def test_ols_is_additive_for_common_period(self):
        a = np.array([1., -2., 3., 7., -1.])
        b = np.array([2., 4., -3., 1., 8.])
        trend = lambda x: calc.fit_cumulative(x, 2000)['cumulative_ols_slope_mm_per_year']
        self.assertAlmostEqual(trend(a + b), trend(a) + trend(b), places=12)

    def test_same_increment_different_absolute_years(self):
        a = calc.fit_cumulative(np.arange(15.), 1940)
        b = calc.fit_cumulative(np.arange(15.), 2008)
        self.assertEqual(a, b)

    def test_depth_boundaries_match_current_figures(self):
        depth = np.array([1., 700., 750., 2000., 2100., 6000.])
        totals = calc.band_totals(depth, np.arange(1., 7.))
        self.assertEqual(totals['0-700m'], 3.)
        self.assertEqual(totals['700-2000m'], 7.)
        self.assertEqual(totals['2000-6000m'], 11.)
        self.assertEqual(totals['0-2000m'], 10.)
        self.assertEqual(sum(totals[b] for b in calc.EXCLUSIVE_DEPTHS), totals['0-6000m'])

    def test_endpoint_period_uses_following_years(self):
        years = np.arange(1993, 2009)
        series = {('global', 'Global', '0-6000m'): np.r_[999., np.ones(15)]}
        rows = calc.summarize_period(series, years, 'test', 1993, 2008)
        self.assertEqual(rows[0]['cumulative_change_mm'], 15.)
        self.assertEqual(rows[0]['first_annual_target_year'], 1994)
        self.assertEqual(rows[0]['ols_trend_contribution_pct'], 100.)

    def test_signed_percentages_and_small_denominator(self):
        self.assertEqual(calc.percentage(-1, 2, .05), -50.)
        self.assertEqual(calc.percentage(3, 2, .05), 150.)
        self.assertEqual(calc.percentage(1, .001, .05), '')

    def test_trend_differences_have_expected_direction(self):
        rows = []
        for period, slope, share in (('core_first_15_years', 1., 30.), ('core_last_15_years', 2., 40.)):
            rows.append({'period': period, 'component_type': 'basin', 'region': 'Pacific',
                'depth_band': '0-6000m', 'cumulative_ols_slope_mm_per_year': slope,
                'mean_annual_change_mm_per_year': slope, 'ols_trend_contribution_pct': share})
        result = calc.period_differences(rows)[0]
        self.assertEqual(result['late_minus_early_ols_slope_mm_per_year'], 1.)
        self.assertEqual(result['ols_contribution_change_percentage_points'], 10.)


if __name__ == '__main__':
    unittest.main()
