#!/usr/bin/env python3
"""Build a pre-2021 performance-weighted forecast ensemble.

The ensemble combines the established 10-year uncertainty-weighted persistence
forecast with the global sparse Bayesian dynamic model.  The BSTS specification
and its non-negative stacking weight are selected only from rolling origins
before 2021, then frozen for the 2021-2025 holdout and 2026-2030 projection.
"""

from __future__ import print_function
from manuscript_paths import workspace_root, input_path

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = (workspace_root() / 'short_term_projection')
BASELINE_DIR = ROOT / "results"
BSTS_DIR = ROOT / "sparse_bsts_results"
OUTPUT_DIR = ROOT / "performance_weighted_ensemble_results"
SELECTION_CUTOFF_ORIGIN = 2020
RANDOM_SEED = 20260925


def rmse(values):
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def select_bsts_and_weight():
    details = pd.read_csv(BSTS_DIR / "hindcast_predictions.csv")
    selection = details[details["origin_year"] < SELECTION_CUTOFF_ORIGIN].copy()
    bsts_scores = (
        selection[selection["method"].str.startswith("sparse_bsts_")]
        .groupby("method")["error_mm_per_year"]
        .apply(rmse)
        .sort_values()
    )
    selected_bsts = str(bsts_scores.index[0])
    keys = ["origin_year", "target_year"]
    baseline = selection[selection["method"] == "weighted_mean_10y"].set_index(keys)
    bsts = selection[selection["method"] == selected_bsts].set_index(keys)
    joined = baseline.join(bsts, lsuffix="_baseline", rsuffix="_bsts", how="inner")
    baseline_error = joined["error_mm_per_year_baseline"].to_numpy(dtype=float)
    bsts_error = joined["error_mm_per_year_bsts"].to_numpy(dtype=float)
    difference = bsts_error - baseline_error
    denominator = float(np.dot(difference, difference))
    bsts_weight = 0.0 if denominator <= 1.0e-12 else float(
        np.clip(-np.dot(baseline_error, difference) / denominator, 0.0, 1.0)
    )
    return details, selected_bsts, bsts_weight, joined


def evaluate_frozen_weight(details, selected_bsts, bsts_weight):
    keys = ["origin_year", "target_year"]
    baseline = details[details["method"] == "weighted_mean_10y"].set_index(keys)
    bsts = details[details["method"] == selected_bsts].set_index(keys)
    joined = baseline.join(bsts, lsuffix="_baseline", rsuffix="_bsts", how="inner").reset_index()
    joined["ensemble_predicted_mm_per_year"] = (
        (1.0 - bsts_weight) * joined["predicted_mm_per_year_baseline"]
        + bsts_weight * joined["predicted_mm_per_year_bsts"]
    )
    joined["ensemble_error_mm_per_year"] = (
        joined["ensemble_predicted_mm_per_year"] - joined["actual_mm_per_year_baseline"]
    )
    score_rows = []
    for label, subset in (
        ("selection_pre_2021", joined[joined["origin_year"] < SELECTION_CUTOFF_ORIGIN]),
        ("holdout_2021_2025", joined[joined["origin_year"] == SELECTION_CUTOFF_ORIGIN]),
        ("all_hindcasts", joined),
    ):
        for method, column in (
            ("weighted_mean_10y", "error_mm_per_year_baseline"),
            ("global_sparse_bsts", "error_mm_per_year_bsts"),
            ("performance_weighted_ensemble", "ensemble_error_mm_per_year"),
        ):
            values = subset[column].to_numpy(dtype=float)
            score_rows.append(
                {
                    "period": label,
                    "method": method,
                    "n": len(values),
                    "rmse_mm_per_year": rmse(values),
                    "mae_mm_per_year": float(np.mean(np.abs(values))),
                    "bias_mm_per_year": float(np.mean(values)),
                }
            )
    return joined, pd.DataFrame(score_rows)


def mixture_summary(paths, years):
    cumulative = np.cumsum(paths, axis=1)
    annual_rows = []
    cumulative_rows = []
    for index, year in enumerate(years):
        annual_rows.append(
            {
                "year": int(year),
                "comparison": "{} vs {}".format(year, year - 1),
                "central_mm_per_year": float(np.median(paths[:, index])),
                "one_sigma_mm_per_year": float(np.std(paths[:, index], ddof=1)),
                "p05_mm_per_year": float(np.percentile(paths[:, index], 5)),
                "p95_mm_per_year": float(np.percentile(paths[:, index], 95)),
            }
        )
        cumulative_rows.append(
            {
                "through_year": int(year),
                "central_cumulative_mm": float(np.median(cumulative[:, index])),
                "one_sigma_cumulative_mm": float(np.std(cumulative[:, index], ddof=1)),
                "p05_cumulative_mm": float(np.percentile(cumulative[:, index], 5)),
                "p95_cumulative_mm": float(np.percentile(cumulative[:, index], 95)),
            }
        )
    return pd.DataFrame(annual_rows), pd.DataFrame(cumulative_rows), cumulative


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    details, selected_bsts, bsts_weight, selection_pairs = select_bsts_and_weight()
    evaluation, scores = evaluate_frozen_weight(details, selected_bsts, bsts_weight)

    baseline = np.load(BASELINE_DIR / "projection_samples.npz")
    bsts = np.load(BSTS_DIR / "sparse_bsts_samples.npz")
    years = baseline["years"].astype(int)
    if not np.array_equal(years, bsts["years"].astype(int)):
        raise ValueError("Baseline and BSTS forecast years do not match.")
    baseline_paths = baseline["annual_mm_per_year"]
    bsts_paths = bsts["annual_mm_per_year"]
    if baseline_paths.shape != bsts_paths.shape:
        raise ValueError("Baseline and BSTS sample dimensions do not match.")
    rng = np.random.RandomState(RANDOM_SEED)
    choose_bsts = rng.uniform(size=baseline_paths.shape[0]) < bsts_weight
    ensemble_paths = baseline_paths.copy()
    ensemble_paths[choose_bsts, :] = bsts_paths[choose_bsts, :]
    annual, cumulative, cumulative_paths = mixture_summary(ensemble_paths, years)

    selection_pairs.reset_index().to_csv(OUTPUT_DIR / "selection_pairs_pre_2021.csv", index=False)
    evaluation.to_csv(OUTPUT_DIR / "hindcast_ensemble_predictions.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "hindcast_ensemble_scores.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "ensemble_projection_2026_2030.csv", index=False)
    cumulative.to_csv(OUTPUT_DIR / "ensemble_cumulative_2026_2030.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "ensemble_projection_samples.npz",
        annual_mm_per_year=ensemble_paths,
        cumulative_mm=cumulative_paths,
        years=years,
    )
    metadata = {
        "model": "Pre-2021 performance-weighted predictive mixture",
        "baseline_method": "weighted_mean_10y",
        "bsts_method": selected_bsts,
        "baseline_weight": 1.0 - bsts_weight,
        "bsts_weight": bsts_weight,
        "selection_origins": "Origins before 2020 only",
        "holdout_origin": 2020,
        "holdout_targets": [2021, 2022, 2023, 2024, 2025],
        "mixture_sampling": "Each Monte Carlo path is drawn from the baseline or BSTS predictive distribution using the fixed model weight.",
        "warning": "The holdout score is reported separately; no post-hoc reweighting is performed.",
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print("Selected BSTS method: {}".format(selected_bsts))
    print("Weights: baseline={:.6f}, BSTS={:.6f}".format(1.0 - bsts_weight, bsts_weight))
    print(scores.to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print("\nPerformance-weighted ensemble projection:")
    print(annual.to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print("\nSaved to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
