#!/usr/bin/env python3
"""Forecast three thermosteric depth bands and reconcile them to total output.

Each band is forecast independently by an observation-constrained ten-year
mean plus a compact Bayesian dynamic-regression model with AR(1) residuals.
The dynamic component is deliberately simpler than the global approximate
BSTS: it avoids the unstable ARD backend on the shorter, depth-resolved
series. Band samples are then reconciled to the existing 0-6000 m
performance-weighted predictive distribution so all three bands sum exactly
to total thermosteric sea-level change in every Monte Carlo draw.
"""

from __future__ import print_function
from manuscript_paths import workspace_root, input_path

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from run_explainable_dynamic_projection import SAMPLES, expected_inputs, fit_oni_ar1, load_model_data
from run_global_projection import fit_recent_mean, mean_point_forecast, mean_samples
from run_sparse_bsts_projection import selected_raw_features, standardize


ROOT = (workspace_root() / 'short_term_projection')
V3_ROOT = ROOT.parent
ANNUAL_RESULTS = V3_ROOT / "Results" / "yearly_1940_2025"
UNCERTAINTY_FILE = (
    V3_ROOT
    / "uncertainty_analysis"
    / "full_spatiotemporal_uncertainty"
    / "results_1940_2023"
    / "uncertainty_summary_global.csv"
)
TOTAL_SAMPLES_FILE = ROOT / "performance_weighted_ensemble_results" / "ensemble_projection_samples.npz"
OUTPUT_DIR = ROOT / "depth_band_ensemble_results"
PAIR_PATTERN = re.compile(r"^(?P<target>\d{4})_vs_(?P<base>\d{4})$")
DEPTH_BANDS = ("0-700m", "700-2000m", "2000-6000m")
HINDCAST_ORIGINS = (1995, 2000, 2005, 2010, 2015, 2020)
FORECAST_YEARS = np.arange(2026, 2031)
RANDOM_SEED = 20260926
DYNAMIC_DRIVERS = ("log_co2", "oni_lag1")


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def relative_sigma(sigma):
    """Retain WLS relative weights while stabilising the legacy ARD backend."""
    scale = max(float(np.median(sigma)), 1.0e-8)
    return np.maximum(np.asarray(sigma, dtype=float) / scale, 0.05)


def fit_dynamic(values, sigma, co2, oni):
    features, mean, scale = standardize(selected_raw_features(co2, oni, DYNAMIC_DRIVERS))
    design = np.column_stack([np.ones(len(values)), features])
    model = BayesianRidge(fit_intercept=False)
    model.fit(design / sigma[:, None], values / sigma)
    residual = values - design.dot(model.coef_)
    denominator = float(np.dot(residual[:-1], residual[:-1]))
    phi = 0.0 if denominator <= 1.0e-12 else float(np.dot(residual[1:], residual[:-1]) / denominator)
    phi = float(np.clip(phi, -0.95, 0.95))
    innovations = residual[1:] - phi * residual[:-1]
    return {
        "model": model,
        "coefficients": model.coef_,
        "feature_mean": mean,
        "feature_scale": scale,
        "residual_last": float(residual[-1]),
        "phi": phi,
        "innovation_sd": max(float(np.std(innovations, ddof=1)), 0.01),
    }


def dynamic_design(co2, oni, fit, previous_co2, previous_oni):
    features = selected_raw_features(co2, oni, DYNAMIC_DRIVERS, previous_co2, previous_oni)
    features, _, _ = standardize(features, fit["feature_mean"], fit["feature_scale"])
    return np.column_stack([np.ones(len(features)), features])


def dynamic_point(fit, future_co2, future_oni, previous_co2, previous_oni):
    design = dynamic_design(future_co2, future_oni, fit, previous_co2, previous_oni)
    residual = fit["residual_last"]
    output = []
    for row in design:
        residual = fit["phi"] * residual
        output.append(float(row.dot(fit["coefficients"]) + residual))
    return np.asarray(output)


def dynamic_samples(fit, future_co2, co2_history, oni_history, rng):
    oni_beta, oni_covariance, oni_sd = fit_oni_ar1(oni_history)
    paths = np.empty((SAMPLES, len(FORECAST_YEARS)), dtype=float)
    coefficient_covariance = np.asarray(fit["model"].sigma_, dtype=float)
    coefficient_covariance = (coefficient_covariance + coefficient_covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(coefficient_covariance)
    coefficient_covariance = eigenvectors.dot(
        np.diag(np.clip(eigenvalues, 1.0e-10, None))
    ).dot(eigenvectors.T)
    for row in range(SAMPLES):
        coefficients = rng.multivariate_normal(fit["coefficients"], coefficient_covariance)
        oni_parameters = rng.multivariate_normal(oni_beta, oni_covariance)
        previous_co2 = float(co2_history[-1])
        previous_oni = float(oni_history[-1])
        oni = previous_oni
        residual = rng.normal(fit["residual_last"], fit["innovation_sd"])
        for column in range(len(FORECAST_YEARS)):
            oni = oni_parameters[0] + oni_parameters[1] * oni + rng.normal(0.0, oni_sd)
            design = dynamic_design([future_co2[column]], [oni], fit, previous_co2, previous_oni)[0]
            residual = fit["phi"] * residual + rng.normal(0.0, fit["innovation_sd"])
            paths[row, column] = float(design.dot(coefficients) + residual)
            previous_co2 = float(future_co2[column])
            previous_oni = oni
    return paths


def band_name(depth_m):
    if depth_m <= 700.0:
        return "0-700m"
    if depth_m <= 2000.0:
        return "700-2000m"
    return "2000-6000m"


def read_band_history():
    records = []
    for directory in ANNUAL_RESULTS.iterdir():
        match = PAIR_PATTERN.match(directory.name)
        path = directory / "layers_basin.csv"
        if not match or not path.is_file():
            continue
        frame = pd.read_csv(path)
        frame["depth_band"] = frame["depth_m"].map(band_name)
        grouped = frame.groupby("depth_band")["layer_sea_level_rise_mm"].sum()
        for band in DEPTH_BANDS:
            records.append(
                {
                    "year": int(match.group("target")),
                    "depth_band": band,
                    "thermosteric_mm_per_year": float(grouped.get(band, 0.0)),
                }
            )
    history = pd.DataFrame(records).sort_values(["depth_band", "year"])
    expected = len(DEPTH_BANDS)
    if not (history.groupby("year").size() == expected).all():
        raise ValueError("Some annual output directories are missing depth-band data.")
    return history


def band_uncertainty(band, years):
    summary = pd.read_csv(UNCERTAINTY_FILE)
    subset = summary[summary["depth_band"] == band].copy()
    values = subset.set_index("target_year")["ensemble_sd_mm"]
    sigma = np.asarray([values.get(year, np.nan) for year in years], dtype=float)
    fallback = float(subset[(subset["target_year"] >= 2019) & (subset["target_year"] <= 2023)]["ensemble_sd_mm"].median())
    missing = ~np.isfinite(sigma)
    sigma[missing] = fallback
    return sigma, fallback, [int(year) for year in np.asarray(years)[missing]]


def prepare_band_data(band, full_history, drivers):
    history = full_history[full_history["depth_band"] == band].drop(columns=["depth_band"])
    data = history.merge(drivers, on="year", how="inner").sort_values("year").reset_index(drop=True)
    data = data[data["year"] >= 1960].reset_index(drop=True)
    years = data["year"].to_numpy(dtype=int)
    sigma, fallback, filled_years = band_uncertainty(band, years)
    data["observation_sigma_mm"] = sigma
    return data, fallback, filled_years


def hindcast(data):
    years = data["year"].to_numpy(dtype=int)
    values = data["thermosteric_mm_per_year"].to_numpy(dtype=float)
    sigma = data["observation_sigma_mm"].to_numpy(dtype=float)
    co2 = data["co2_ppm"].to_numpy(dtype=float)
    oni = data["oni"].to_numpy(dtype=float)
    records = []
    for origin in HINDCAST_ORIGINS:
        position = np.where(years == origin)[0]
        if len(position) != 1:
            continue
        endpoint = int(position[0]) + 1
        horizon = min(5, len(years) - endpoint)
        if horizon == 0:
            continue
        target_years = years[endpoint : endpoint + horizon]
        candidates = {
            "weighted_mean_10y": mean_point_forecast(fit_recent_mean(values[:endpoint], sigma[:endpoint]), horizon)
        }
        future_co2, future_oni = expected_inputs(years[:endpoint], co2[:endpoint], oni[:endpoint], target_years)
        fit = fit_dynamic(
            values[:endpoint], relative_sigma(sigma[:endpoint]), co2[:endpoint], oni[:endpoint]
        )
        candidates["bayesian_dynamic"] = dynamic_point(
            fit, future_co2, future_oni, co2[endpoint - 1], oni[endpoint - 1]
        )
        for method, prediction in candidates.items():
            for step, value in enumerate(prediction):
                actual = float(values[endpoint + step])
                records.append(
                    {
                        "method": method,
                        "origin_year": origin,
                        "target_year": int(target_years[step]),
                        "lead_years": step + 1,
                        "predicted_mm_per_year": float(value),
                        "actual_mm_per_year": actual,
                        "error_mm_per_year": float(value - actual),
                    }
                )
    return pd.DataFrame(records)


def selection_and_scores(details):
    selection = details[details["origin_year"] < 2020]
    selected_dynamic = "bayesian_dynamic"
    keys = ["origin_year", "target_year"]
    baseline = selection[selection["method"] == "weighted_mean_10y"].set_index(keys)
    dynamic = selection[selection["method"] == selected_dynamic].set_index(keys)
    pairs = baseline.join(dynamic, lsuffix="_baseline", rsuffix="_dynamic", how="inner")
    baseline_error = pairs["error_mm_per_year_baseline"].to_numpy(dtype=float)
    dynamic_error = pairs["error_mm_per_year_dynamic"].to_numpy(dtype=float)
    difference = dynamic_error - baseline_error
    denominator = float(np.dot(difference, difference))
    weight = 0.0 if denominator <= 1.0e-12 else float(np.clip(
        -np.dot(baseline_error, difference) / denominator, 0.0, 1.0
    ))
    all_baseline = details[details["method"] == "weighted_mean_10y"].set_index(keys)
    all_dynamic = details[details["method"] == selected_dynamic].set_index(keys)
    evaluation = all_baseline.join(all_dynamic, lsuffix="_baseline", rsuffix="_dynamic", how="inner").reset_index()
    evaluation["ensemble_error_mm_per_year"] = (
        (1.0 - weight) * evaluation["error_mm_per_year_baseline"]
        + weight * evaluation["error_mm_per_year_dynamic"]
    )
    score_rows = []
    for period, subset in (
        ("selection_pre_2021", evaluation[evaluation["origin_year"] < 2020]),
        ("holdout_2021_2025", evaluation[evaluation["origin_year"] == 2020]),
        ("all_hindcasts", evaluation),
    ):
        for method, column in (
            ("weighted_mean_10y", "error_mm_per_year_baseline"),
            ("bayesian_dynamic", "error_mm_per_year_dynamic"),
            ("performance_weighted_ensemble", "ensemble_error_mm_per_year"),
        ):
            error = subset[column].to_numpy(dtype=float)
            score_rows.append({
                "period": period, "method": method, "n": len(error),
                "rmse_mm_per_year": rmse(error),
                "mae_mm_per_year": float(np.mean(np.abs(error))),
                "bias_mm_per_year": float(np.mean(error)),
            })
    return selected_dynamic, weight, evaluation, pd.DataFrame(score_rows)


def forecast_band(data, dynamic_weight, seed):
    years = data["year"].to_numpy(dtype=int)
    values = data["thermosteric_mm_per_year"].to_numpy(dtype=float)
    sigma = data["observation_sigma_mm"].to_numpy(dtype=float)
    co2 = data["co2_ppm"].to_numpy(dtype=float)
    oni = data["oni"].to_numpy(dtype=float)
    baseline_fit = fit_recent_mean(values, sigma)
    baseline_paths = mean_samples(baseline_fit, len(FORECAST_YEARS), SAMPLES, np.random.RandomState(seed))
    dynamic_fit = fit_dynamic(values, relative_sigma(sigma), co2, oni)
    future_co2, _ = expected_inputs(years, co2, oni, FORECAST_YEARS)
    dynamic_paths = dynamic_samples(
        dynamic_fit, future_co2, co2, oni, np.random.RandomState(seed + 1)
    )
    selector = np.random.RandomState(seed + 2).uniform(size=SAMPLES) < dynamic_weight
    paths = baseline_paths.copy()
    paths[selector, :] = dynamic_paths[selector, :]
    return paths


def summarise(paths, band):
    rows = []
    for index, year in enumerate(FORECAST_YEARS):
        values = paths[:, index]
        rows.append({
            "depth_band": band,
            "year": int(year),
            "comparison": "{} vs {}".format(year, year - 1),
            "central_mm_per_year": float(np.median(values)),
            "one_sigma_mm_per_year": float(np.std(values, ddof=1)),
            "p05_mm_per_year": float(np.percentile(values, 5)),
            "p95_mm_per_year": float(np.percentile(values, 95)),
        })
    return rows


def reconcile(raw_paths, total_paths):
    """Minimum-variance reconciliation under independent band forecast errors."""
    stacked = np.stack([raw_paths[band] for band in DEPTH_BANDS], axis=1)
    reconciled = stacked.copy()
    for column in range(stacked.shape[2]):
        variance = np.var(stacked[:, :, column], axis=0, ddof=1)
        weights = variance / variance.sum() if variance.sum() > 0.0 else np.full(3, 1.0 / 3.0)
        residual = total_paths[:, column] - stacked[:, :, column].sum(axis=1)
        reconciled[:, :, column] += residual[:, None] * weights[None, :]
    return {band: reconciled[:, index, :] for index, band in enumerate(DEPTH_BANDS)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = read_band_history()
    drivers, _, _ = load_model_data()
    drivers = drivers[["year", "co2_ppm", "oni"]]
    raw_paths = {}
    weights = []
    all_hindcasts = []
    all_scores = []
    for index, band in enumerate(DEPTH_BANDS):
        data, fallback, filled_years = prepare_band_data(band, history, drivers)
        details = hindcast(data)
        selected, weight, evaluation, scores = selection_and_scores(details)
        raw_paths[band] = forecast_band(data, weight, RANDOM_SEED + index * 10)
        data.to_csv(OUTPUT_DIR / "input_{}.csv".format(band), index=False)
        details.assign(depth_band=band).to_csv(OUTPUT_DIR / "hindcasts_all_{}.csv".format(band), index=False)
        evaluation.assign(depth_band=band).to_csv(OUTPUT_DIR / "hindcast_ensemble_{}.csv".format(band), index=False)
        all_hindcasts.append(evaluation.assign(depth_band=band))
        all_scores.append(scores.assign(depth_band=band))
        weights.append({
            "depth_band": band,
            "selected_dynamic_method": selected,
            "baseline_weight": 1.0 - weight,
            "dynamic_weight": weight,
            "fallback_sigma_mm": fallback,
            "fallback_sigma_years": ";".join(map(str, filled_years)),
        })
    total = np.load(TOTAL_SAMPLES_FILE)
    total_paths = total["annual_mm_per_year"]
    if not np.array_equal(total["years"].astype(int), FORECAST_YEARS) or total_paths.shape[0] != SAMPLES:
        raise ValueError("Existing 0-6000m ensemble samples do not match this forecast configuration.")
    reconciled = reconcile(raw_paths, total_paths)
    raw_rows = []
    reconciled_rows = []
    for band in DEPTH_BANDS:
        raw_rows.extend(summarise(raw_paths[band], band))
        reconciled_rows.extend(summarise(reconciled[band], band))
    total_check = total_paths - sum(reconciled[band] for band in DEPTH_BANDS)
    pd.DataFrame(weights).to_csv(OUTPUT_DIR / "band_model_weights.csv", index=False)
    pd.concat(all_hindcasts, ignore_index=True).to_csv(OUTPUT_DIR / "band_hindcast_ensemble_predictions.csv", index=False)
    pd.concat(all_scores, ignore_index=True).to_csv(OUTPUT_DIR / "band_hindcast_ensemble_scores.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(OUTPUT_DIR / "depth_band_projection_raw_2026_2030.csv", index=False)
    pd.DataFrame(reconciled_rows).to_csv(OUTPUT_DIR / "depth_band_projection_reconciled_2026_2030.csv", index=False)
    np.savez_compressed(
        OUTPUT_DIR / "depth_band_projection_samples_reconciled.npz",
        years=FORECAST_YEARS,
        band_0_700m=reconciled["0-700m"],
        band_700_2000m=reconciled["700-2000m"],
        band_2000_6000m=reconciled["2000-6000m"],
        total_0_6000m=total_paths,
    )
    metadata = {
        "method": "Independent band performance-weighted ensembles reconciled to the existing total predictive mixture",
        "bands": list(DEPTH_BANDS),
        "weight_selection": "Depth-band dynamic-model weights use origins before 2020 only; origin 2020 is a frozen holdout.",
        "reconciliation": "Per-year minimum-variance additive adjustment using raw band predictive variances; each draw sums exactly to total 0-6000m prediction.",
        "max_total_closure_error_mm_per_year": float(np.max(np.abs(total_check))),
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(pd.DataFrame(weights).to_string(index=False, float_format=lambda value: "{:.4f}".format(value)))
    print("\nReconciled depth-band projections:")
    print(pd.DataFrame(reconciled_rows).to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print("\nMaximum total closure error: {:.3e} mm/yr".format(np.max(np.abs(total_check))))
    print("Saved to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
