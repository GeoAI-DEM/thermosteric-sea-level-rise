"""Regression checks for saved-trajectory period selection and reanchoring."""
from manuscript_paths import workspace_root, input_path
import unittest
import tempfile
from pathlib import Path

import numpy as np

import summarize_en4_periods as summary


class PeriodTests(unittest.TestCase):
    def test_existing_csv_writer_header_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'series.csv'
            expected = [{'year': '2005', 'region': 'Global'}]
            summary.statistics.step1.write_csv(path, expected)
            self.assertEqual(summary.read_csv(path), expected)

    def rows(self, first=1940, last=2023):
        return [{'region': region, 'depth_band': band, 'year': year,
                 'iap_cumulative_anomaly_mm': 2. * (year - first) + 100.,
                 'en4_cumulative_anomaly_mm': 3. * (year - first) - 20.}
                for region in ('Global', '60S-60N')
                for band in ('0-700m', '700-2000m', '0-2000m')
                for year in range(first, last + 1)]

    def test_all_four_prespecified_periods_and_trends(self):
        results = summary.period_metrics(self.rows(), 1940, 2023)
        self.assertEqual(len(results), 24)
        self.assertEqual({row['start_year'] for row in results}, {1940, 1955, 1993, 2005})
        for row in results:
            np.testing.assert_allclose(row['iap_ols_trend_mm_per_year'], 2.)
            np.testing.assert_allclose(row['en4_ols_trend_mm_per_year'], 3.)
            self.assertEqual(row['annual_change_count'], 2023 - row['start_year'])
            self.assertEqual(row['anchor_year'], row['start_year'])

    def test_modern_run_has_no_duplicate_periods(self):
        results = summary.period_metrics(self.rows(2005), 2005, 2023)
        self.assertEqual(len(results), 6)
        self.assertEqual({row['period'] for row in results}, {'full_period'})

    def test_missing_or_duplicate_year_rejected(self):
        rows = self.rows()
        with self.assertRaises(ValueError):
            summary.period_metrics(rows[1:], 1940, 2023)
        with self.assertRaises(ValueError):
            summary.period_metrics(rows + [rows[0]], 1940, 2023)

    def test_nonfinite_input_rejected(self):
        rows = self.rows()
        rows[0]['en4_cumulative_anomaly_mm'] = float('nan')
        with self.assertRaises(ValueError):
            summary.period_metrics(rows, 1940, 2023)


if __name__ == '__main__':
    unittest.main()
