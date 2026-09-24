#!/usr/bin/env python3
"""Run a sparse Bayesian dynamic-regression (approximate BSTS) projection.

The model uses automatic relevance determination (ARD) to shrink irrelevant
drivers and an AR(1) state for unresolved annual variability. It is an
approximate BSTS implementation because Stan/PyMC/R bsts is unavailable in the
current environment; all approximation details are emitted in metadata.
"""

from __future__ import print_function

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ARDRegression

from run_explainable_dynamic_projection import (
    SAMPLES,
    expected_inputs,
    fit_oni_ar1,
    load_model_data,
)
from run_global_projection import (
    FORECAST_END,
    FORECAST_START,
    RANDOM_SEED,
    fit_recent_linear,
    fit_recent_mean,
    linear_point_forecast,
    mean_point_forecast,
)
import run_explainable_dynamic_projection as dynamic_projection
import run_global_projection as global_projection


OUTPUT_DIR = None
RAW_FEATURE_NAMES = ("log_co2", "log_co2_squared", "co2_growth", "oni", "oni_lag1")
# Specifications are fixed before evaluating the historical hindcasts.  CO2
# growth is a noisy annual proxy for slow forcing, but two sensitivity
# specifications retain it so its empirical hindcast value is tested directly.
FEATURE_SPECS = {
    "co2_level": ("log_co2",),
    "co2_level_oni": ("log_co2", "oni"),
    "co2_level_oni_lag": ("log_co2", "oni_lag1"),
    "co2_level_oni_both": ("log_co2", "oni", "oni_lag1"),
    "co2_nonlinear_oni_lag": ("log_co2", "log_co2_squared", "oni_lag1"),
    "co2_level_growth_oni_lag": ("log_co2", "co2_growth", "oni_lag1"),
    "co2_nonlinear_growth_oni_both": (
        "log_co2", "log_co2_squared", "co2_growth", "oni", "oni_lag1"
    ),
}


def raw_features(co2, oni, previous_co2=None, previous_oni=None):
    co2 = np.asarray(co2, dtype=float)
    oni = np.asarray(oni, dtype=float)
    if previous_co2 is None:
        previous_co2 = co2[0]
    if previous_oni is None:
        previous_oni = oni[0]
    prior_co2 = np.r_[float(previous_co2), co2[:-1]]
    prior_oni = np.r_[float(previous_oni), oni[:-1]]
    return np.column_stack(
        [
            np.log(co2),
            np.log(co2) ** 2,
            co2 - prior_co2,
            oni,
            prior_oni,
        ]
    )


def standardize(features, mean=None, scale=None):
    if mean is None:
        mean = features.mean(axis=0)
    if scale is None:
        scale = features.std(axis=0, ddof=1)
        scale[scale < 1.0e-10] = 1.0
    return (features - mean) / scale, mean, scale


def selected_raw_features(co2, oni, driver_names, previous_co2=None, previous_oni=None):
    raw = raw_features(co2, oni, previous_co2, previous_oni)
    indexes = [RAW_FEATURE_NAMES.index(name) for name in driver_names]
    return raw[:, indexes]


def fit_sparse_bsts(values, sigma, co2, oni, driver_names):
    features, mean, scale = standardize(selected_raw_features(co2, oni, driver_names))
    design = np.column_stack([np.ones(len(values)), features])
    # Weighted ARD: the supplied historical ensemble uncertainty is the
    # observation scale, while ARD determines which covariates remain active.
    weighted_design = design / sigma[:, None]
    weighted_values = values / sigma
    model = ARDRegression(
        n_iter=3000,
        tol=1.0e-5,
        threshold_lambda=10000.0,
        fit_intercept=False,
        compute_score=True,
    )
    model.fit(weighted_design, weighted_values)
    fitted = design.dot(model.coef_)
    residual = values - fitted
    if len(residual) < 3:
        raise ValueError("Not enough records for an AR(1) dynamic residual.")
    denominator = float(np.dot(residual[:-1], residual[:-1]))
    phi = 0.0 if denominator <= 1.0e-12 else float(np.dot(residual[1:], residual[:-1]) / denominator)
    phi = float(np.clip(phi, -0.95, 0.95))
    innovations = residual[1:] - phi * residual[:-1]
    innovation_sd = max(float(np.std(innovations, ddof=1)), 0.05)
    return {
        "model": model,
        "driver_names": tuple(driver_names),
        "feature_mean": mean,
        "feature_scale": scale,
        "coefficients": model.coef_,
        "residual_last": float(residual[-1]),
        "phi": phi,
        "innovation_sd": innovation_sd,
        "training_residual_sd": float(np.std(residual, ddof=1)),
    }


def feature_design(co2, oni, fit, previous_co2=None, previous_oni=None):
    features = selected_raw_features(
        co2, oni, fit["driver_names"], previous_co2, previous_oni
    )
    standardized, _, _ = standardize(features, fit["feature_mean"], fit["feature_scale"])
    return np.column_stack([np.ones(len(features)), standardized])


def point_forecast(fit, future_co2, future_oni, previous_co2, previous_oni):
    design = feature_design(future_co2, future_oni, fit, previous_co2, previous_oni)
    residual = fit["residual_last"]
    forecast = []
    contributions = []
    for row in design:
        residual = fit["phi"] * residual
        pieces = row * fit["coefficients"]
        forecast.append(float(pieces.sum() + residual))
        record = dict(zip(("intercept",) + fit["driver_names"], pieces))
        record["ar1_residual_mm_per_year"] = residual
        contributions.append(record)
    return np.asarray(forecast), pd.DataFrame(contributions)


def draw_coefficients(fit, rng):
    model = fit["model"]
    coefficients = model.coef_.copy()
    active = np.flatnonzero(model.lambda_ < model.threshold_lambda)
    if len(active) and model.sigma_.shape == (len(active), len(active)):
        coefficients[active] = rng.multivariate_normal(coefficients[active], model.sigma_)
    return coefficients


def sample_forecast(fit, future_co2, co2_history, oni_history, future_years, rng):
    oni_beta, oni_covariance, oni_sd = fit_oni_ar1(oni_history)
    paths = np.empty((SAMPLES, len(future_years)), dtype=float)
    for index in range(SAMPLES):
        coefficients = draw_coefficients(fit, rng)
        oni_parameters = rng.multivariate_normal(oni_beta, oni_covariance)
        oni = float(oni_history[-1])
        previous_co2 = float(co2_history[-1])
        previous_oni = float(oni_history[-1])
        residual = rng.normal(fit["residual_last"], fit["innovation_sd"])
        for column, _year in enumerate(future_years):
            oni = oni_parameters[0] + oni_parameters[1] * oni + rng.normal(0.0, oni_sd)
            design = feature_design(
                [future_co2[column]], [oni], fit, previous_co2, previous_oni
            )[0]
            residual = fit["phi"] * residual + rng.normal(0.0, fit["innovation_sd"])
            paths[index, column] = float(design.dot(coefficients) + residual)
            previous_co2 = float(future_co2[column])
            previous_oni = oni
    return paths


def evaluate(values, sigma, years, co2, oni):
    records = []
    for origin in (1995, 2000, 2005, 2010, 2015, 2020):
        endpoint = np.where(years == origin)[0]
        if len(endpoint) != 1:
            continue
        endpoint = int(endpoint[0]) + 1
        horizon = min(5, len(values) - endpoint)
        if horizon == 0:
            continue
        train = slice(0, endpoint)
        target_years = years[endpoint : endpoint + horizon]
        candidates = {
            "weighted_mean_10y": mean_point_forecast(fit_recent_mean(values[train], sigma[train]), horizon),
            "weighted_linear_20y": linear_point_forecast(fit_recent_linear(values[train], sigma[train]), horizon),
        }
        predicted_co2, predicted_oni = expected_inputs(years[train], co2[train], oni[train], target_years)
        for spec_name, driver_names in FEATURE_SPECS.items():
            try:
                fit = fit_sparse_bsts(
                    values[train], sigma[train], co2[train], oni[train], driver_names
                )
                candidates["sparse_bsts_{}".format(spec_name)] = point_forecast(
                    fit, predicted_co2, predicted_oni, co2[train][-1], oni[train][-1]
                )[0]
            except Exception as error:
                print("Sparse BSTS hindcast failure for {} at {}: {}".format(spec_name, origin, error))
        for method, prediction in candidates.items():
            for step, value in enumerate(prediction):
                actual = values[endpoint + step]
                records.append(
                    {
                        "method": method,
                        "origin_year": int(origin),
                        "target_year": int(target_years[step]),
                        "lead_years": int(step + 1),
                        "predicted_mm_per_year": float(value),
                        "actual_mm_per_year": float(actual),
                        "error_mm_per_year": float(value - actual),
                    }
                )
    details = pd.DataFrame(records)
    scores = details.groupby("method").agg(
        n=("error_mm_per_year", "size"),
        rmse_mm_per_year=("error_mm_per_year", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
        mae_mm_per_year=("error_mm_per_year", lambda x: float(np.mean(np.abs(np.asarray(x))))),
        bias_mm_per_year=("error_mm_per_year", "mean"),
    ).reset_index().sort_values("rmse_mm_per_year")
    return details, scores


def summarise(paths, years):
    cumulative = np.cumsum(paths, axis=1)
    annual = []
    cumulative_rows = []
    for index, year in enumerate(years):
        annual.append(
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
    return pd.DataFrame(annual), pd.DataFrame(cumulative_rows), cumulative


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-results", type=Path, required=True)
    parser.add_argument("--uncertainty-results", type=Path, required=True)
    parser.add_argument("--driver-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "sparse_bsts_projection")
    args = parser.parse_args()
    global_projection.MAIN_RESULTS = args.main_results.resolve()
    global_projection.UNCERTAINTY_RESULTS = args.uncertainty_results.resolve()
    dynamic_projection.DATA_DIR = args.driver_data_dir.resolve()
    dynamic_projection.CO2_FILE = dynamic_projection.DATA_DIR / "noaa_mlo_co2_annmean.txt"
    dynamic_projection.ONI_FILE = dynamic_projection.DATA_DIR / "noaa_cpc_oni.txt"
    OUTPUT_DIR = args.output_dir.resolve()
    if not global_projection.MAIN_RESULTS.is_dir():
        raise FileNotFoundError("Annual results directory not found: {}".format(global_projection.MAIN_RESULTS))
    if not (global_projection.UNCERTAINTY_RESULTS / "uncertainty_summary_global.csv").is_file():
        raise FileNotFoundError("Missing uncertainty_summary_global.csv in {}".format(global_projection.UNCERTAINTY_RESULTS))
    if not dynamic_projection.CO2_FILE.is_file() or not dynamic_projection.ONI_FILE.is_file():
        raise FileNotFoundError("Missing NOAA CO2 or ONI input under {}".format(dynamic_projection.DATA_DIR))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, fallback_sigma, fallback_years = load_model_data()
    # Lagged ONI is defined from 1960 onward.
    data = data[data["year"] >= 1960].reset_index(drop=True)
    years = data["year"].to_numpy(dtype=int)
    values = data["thermosteric_mm_per_year"].to_numpy(dtype=float)
    sigma = data["observation_sigma_mm"].to_numpy(dtype=float)
    co2 = data["co2_ppm"].to_numpy(dtype=float)
    oni = data["oni"].to_numpy(dtype=float)

    details, scores = evaluate(values, sigma, years, co2, oni)
    bsts_scores = scores[scores["method"].str.startswith("sparse_bsts_")].copy()
    selected_method = str(bsts_scores.iloc[0]["method"])
    selected_spec = selected_method.replace("sparse_bsts_", "", 1)
    selected_drivers = FEATURE_SPECS[selected_spec]
    fit = fit_sparse_bsts(values, sigma, co2, oni, selected_drivers)
    future_years = np.arange(FORECAST_START, FORECAST_END + 1)
    future_co2, expected_oni = expected_inputs(years, co2, oni, future_years)
    deterministic, contributions = point_forecast(
        fit, future_co2, expected_oni, co2[-1], oni[-1]
    )
    paths = sample_forecast(
        fit, future_co2, co2, oni, future_years, np.random.RandomState(RANDOM_SEED + 2)
    )
    annual, cumulative, cumulative_paths = summarise(paths, future_years)
    contributions.insert(0, "year", future_years)
    contributions.insert(1, "expected_co2_ppm", future_co2)
    contributions.insert(2, "expected_oni", expected_oni)
    contributions.insert(3, "deterministic_forecast_mm_per_year", deterministic)

    coefficient_table = pd.DataFrame(
        {
            "feature": ("intercept",) + selected_drivers,
            "coefficient": fit["coefficients"],
            "ard_precision_lambda": fit["model"].lambda_,
            "selected_by_ard": fit["model"].lambda_ < fit["model"].threshold_lambda,
        }
    )
    data.to_csv(OUTPUT_DIR / "model_input_1960_2025.csv", index=False)
    details.to_csv(OUTPUT_DIR / "hindcast_predictions.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "hindcast_scores.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "sparse_bsts_projection_2026_2030.csv", index=False)
    cumulative.to_csv(OUTPUT_DIR / "sparse_bsts_cumulative_2026_2030.csv", index=False)
    contributions.to_csv(OUTPUT_DIR / "driver_contributions_2026_2030.csv", index=False)
    coefficient_table.to_csv(OUTPUT_DIR / "ard_feature_selection.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "sparse_bsts_samples.npz", annual_mm_per_year=paths, cumulative_mm=cumulative_paths, years=future_years)
    metadata = {
        "model": "Approximate BSTS: uncertainty-weighted ARD sparse regression plus AR(1) dynamic residual",
        "training_period": [int(years.min()), int(years.max())],
        "candidate_driver_specifications": {
            name: list(drivers) for name, drivers in FEATURE_SPECS.items()
        },
        "selected_bsts_specification": selected_spec,
        "selected_bsts_drivers": list(selected_drivers),
        "co2_future_assumption": "Linear extrapolation fitted to the latest 10 annual NOAA CO2 values",
        "oni_future_assumption": "AR(1) stochastic simulation of annual ONI",
        "direct_ensemble_uncertainty_years": [1941, 2023],
        "fallback_observation_sigma_years": fallback_years,
        "fallback_observation_sigma_mm": fallback_sigma,
        "monte_carlo_samples": SAMPLES,
        "random_seed": RANDOM_SEED + 2,
        "best_hindcast_method_across_all_candidates": str(scores.iloc[0]["method"]),
        "best_hindcast_bsts_method": selected_method,
        "limitation": "Approximate BSTS due to unavailable full Bayesian state-space runtime; ARD is used for Bayesian sparsity and AR(1) for the dynamic state.",
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(scores.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    print("\nSparse BSTS projection:")
    print(annual.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    print("\nSaved to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
