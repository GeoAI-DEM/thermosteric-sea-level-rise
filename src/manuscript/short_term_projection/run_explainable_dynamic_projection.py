#!/usr/bin/env python3
"""Explainable nonlinear dynamic-regression projection for 2026-2030.

This is an empirical, observation-constrained short-term model. It does not
use CMIP temperature or salinity fields and does not replace the 15-arc-second
historical central estimate with an ensemble mean.
"""

from __future__ import print_function
from manuscript_paths import workspace_root, input_path

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from run_global_projection import (
    FORECAST_END,
    FORECAST_START,
    RANDOM_SEED,
    fit_recent_linear,
    fit_recent_mean,
    linear_point_forecast,
    mean_point_forecast,
    read_main_history,
    read_observation_sigma,
)


ROOT = (workspace_root() / 'short_term_projection')
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "explainable_dynamic_results"
CO2_FILE = DATA_DIR / "noaa_mlo_co2_annmean.txt"
ONI_FILE = DATA_DIR / "noaa_cpc_oni.txt"
SAMPLES = 10000


def load_co2():
    frame = pd.read_csv(CO2_FILE, comment="#", delim_whitespace=True, names=["year", "co2_ppm", "unc_ppm"])
    frame = frame[(frame["co2_ppm"] > 0.0) & np.isfinite(frame["co2_ppm"])]
    return frame[["year", "co2_ppm"]]


def load_oni():
    frame = pd.read_csv(ONI_FILE, delim_whitespace=True)
    frame.columns = [str(column).lower() for column in frame.columns]
    annual = frame.groupby("yr", as_index=False)["anom"].mean()
    annual.columns = ["year", "oni"]
    return annual


def load_model_data():
    history = read_main_history()
    sigma, fallback, fallback_years = read_observation_sigma(history)
    history["observation_sigma_mm"] = sigma
    inputs = load_co2().merge(load_oni(), on="year", how="inner")
    data = history.merge(inputs, on="year", how="inner").sort_values("year").reset_index(drop=True)
    if data["year"].min() > 1960 or data["year"].max() < 2025:
        raise ValueError("The common CO2/ONI/history period is unexpectedly short.")
    return data, fallback, fallback_years


def make_features(co2_ppm, oni, log_co2_mean, log_co2_std):
    z = (np.log(np.asarray(co2_ppm, dtype=float)) - log_co2_mean) / log_co2_std
    return np.column_stack([np.ones_like(z), z, z ** 2, np.asarray(oni, dtype=float)])


def unpack(theta):
    phi = 0.995 * np.tanh(theta[0])
    q_ar = np.exp(theta[1])
    obs_scale = np.exp(theta[2])
    transition = np.diag([phi, 1.0, 1.0, 1.0, 1.0])
    covariance = np.diag([q_ar, 0.0, 0.0, 0.0, 0.0])
    return transition, covariance, obs_scale


def filter_model(values, sigma, features, theta, return_state=False):
    """Kalman filter with an AR(1) state and static interpretable coefficients."""
    transition, process_covariance, obs_scale = unpack(theta)
    state = np.zeros(5, dtype=float)
    covariance = np.diag([10.0, 1.0e4, 1.0e4, 1.0e4, 1.0e4])
    nll = 0.0
    latest_components = None

    for value, uncertainty, feature in zip(values, sigma, features):
        observation = np.concatenate([[1.0], feature])
        variance = float(observation.dot(covariance).dot(observation) + (obs_scale * uncertainty) ** 2)
        if not np.isfinite(variance) or variance <= 0.0:
            return np.inf if not return_state else None
        innovation = value - float(observation.dot(state))
        gain = covariance.dot(observation) / variance
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, observation.dot(covariance))
        covariance = (covariance + covariance.T) * 0.5
        nll += 0.5 * (np.log(2.0 * np.pi * variance) + innovation ** 2 / variance)
        latest_components = observation * state
        state = transition.dot(state)
        covariance = transition.dot(covariance).dot(transition.T) + process_covariance
        covariance = (covariance + covariance.T) * 0.5

    if return_state:
        # State/covariance immediately after the last observation, not its forecast.
        transition_inv = np.diag([1.0 / transition[0, 0], 1.0, 1.0, 1.0, 1.0])
        state = transition_inv.dot(state)
        covariance = transition_inv.dot(covariance - process_covariance).dot(transition_inv.T)
        covariance = (covariance + covariance.T) * 0.5
        return nll, state, covariance, latest_components
    return nll


def fit_hybrid(values, sigma, co2, oni):
    log_co2 = np.log(np.asarray(co2, dtype=float))
    log_co2_mean = float(np.mean(log_co2))
    log_co2_std = float(np.std(log_co2, ddof=1))
    features = make_features(co2, oni, log_co2_mean, log_co2_std)
    bounds = [(-3.0, 3.0), (-10.0, 4.0), (-2.0, 2.0)]
    starts = [np.array([0.0, -0.5, 0.0]), np.array([0.5, -1.5, -0.2]), np.array([-0.5, 0.0, 0.2])]
    fits = []
    for start in starts:
        result = minimize(
            lambda theta: filter_model(values, sigma, features, theta),
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1.0e-10},
        )
        if result.success and np.isfinite(result.fun):
            fits.append(result)
    if not fits:
        raise RuntimeError("Hybrid dynamic-regression optimisation did not converge.")
    result = min(fits, key=lambda item: item.fun)
    _, state, covariance, _ = filter_model(values, sigma, features, result.x, return_state=True)
    return {
        "theta": result.x,
        "result": result,
        "state": state,
        "covariance": covariance,
        "mean": log_co2_mean,
        "std": log_co2_std,
        "bounds": bounds,
    }


def fit_co2_forecast(years, co2):
    use = min(10, len(years))
    x = np.asarray(years[-use:], dtype=float)
    y = np.asarray(co2[-use:], dtype=float)
    design = np.column_stack([np.ones(use), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design.dot(beta)
    variance = max(float(np.sum(residual ** 2) / max(1, use - 2)), 1.0e-6)
    covariance = variance * np.linalg.pinv(design.T.dot(design))
    return beta, covariance, float(np.sqrt(variance))


def fit_oni_ar1(oni):
    y = np.asarray(oni, dtype=float)
    design = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    beta = np.linalg.lstsq(design, y[1:], rcond=None)[0]
    residual = y[1:] - design.dot(beta)
    variance = max(float(np.sum(residual ** 2) / max(1, len(residual) - 2)), 1.0e-6)
    covariance = variance * np.linalg.pinv(design.T.dot(design))
    return beta, covariance, float(np.sqrt(variance))


def expected_inputs(train_years, train_co2, train_oni, target_years):
    co2_beta, _, _ = fit_co2_forecast(train_years, train_co2)
    co2 = co2_beta[0] + co2_beta[1] * np.asarray(target_years, dtype=float)
    oni_beta, _, _ = fit_oni_ar1(train_oni)
    oni = []
    previous = float(train_oni[-1])
    for _ in target_years:
        previous = oni_beta[0] + oni_beta[1] * previous
        oni.append(previous)
    return co2, np.asarray(oni)


def hybrid_point_forecast(fit, future_co2, future_oni):
    features = make_features(future_co2, future_oni, fit["mean"], fit["std"])
    transition, _, _ = unpack(fit["theta"])
    state = fit["state"].copy()
    output = []
    contributions = []
    for feature in features:
        state = transition.dot(state)
        observation = np.concatenate([[1.0], feature])
        output.append(float(observation.dot(state)))
        contributions.append(
            {
                "ar1_residual_mm_per_year": float(state[0]),
                "intercept_mm_per_year": float(state[1]),
                "co2_linear_mm_per_year": float(feature[1] * state[2]),
                "co2_quadratic_mm_per_year": float(feature[2] * state[3]),
                "oni_mm_per_year": float(feature[3] * state[4]),
            }
        )
    return np.asarray(output), pd.DataFrame(contributions)


def hybrid_samples(fit, values, sigma, co2, oni, future_years, rng):
    sample_count = SAMPLES
    try:
        theta_cov = np.asarray(fit["result"].hess_inv.todense(), dtype=float)
    except AttributeError:
        theta_cov = np.asarray(fit["result"].hess_inv, dtype=float)
    theta_cov = (theta_cov + theta_cov.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(theta_cov)
    theta_cov = eigenvectors.dot(np.diag(np.clip(eigenvalues, 1.0e-8, 1.0))).dot(eigenvectors.T)
    theta_draws = rng.multivariate_normal(fit["theta"], theta_cov, size=sample_count)
    lower = np.asarray([item[0] for item in fit["bounds"]])
    upper = np.asarray([item[1] for item in fit["bounds"]])
    theta_draws = np.clip(theta_draws, lower, upper)

    co2_beta, co2_cov, co2_sd = fit_co2_forecast(np.arange(1959, 1959 + len(co2)), co2)
    oni_beta, oni_cov, oni_sd = fit_oni_ar1(oni)
    paths = np.empty((sample_count, len(future_years)), dtype=float)

    for row, theta in enumerate(theta_draws):
        features = make_features(co2, oni, fit["mean"], fit["std"])
        _, state, covariance, _ = filter_model(values, sigma, features, theta, return_state=True)
        transition, process_covariance, _ = unpack(theta)
        state = rng.multivariate_normal(state, covariance)
        co2_parameters = rng.multivariate_normal(co2_beta, co2_cov)
        oni_parameters = rng.multivariate_normal(oni_beta, oni_cov)
        previous_oni = float(oni[-1])
        for column, year in enumerate(future_years):
            future_co2 = co2_parameters[0] + co2_parameters[1] * year + rng.normal(0.0, co2_sd)
            previous_oni = oni_parameters[0] + oni_parameters[1] * previous_oni + rng.normal(0.0, oni_sd)
            feature = make_features([future_co2], [previous_oni], fit["mean"], fit["std"])[0]
            state = transition.dot(state) + rng.multivariate_normal(np.zeros(5), process_covariance)
            paths[row, column] = float(np.concatenate([[1.0], feature]).dot(state))
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
        future_years = years[endpoint : endpoint + horizon]
        try:
            fit = fit_hybrid(values[train], sigma[train], co2[train], oni[train])
            future_co2, future_oni = expected_inputs(years[train], co2[train], oni[train], future_years)
            prediction, _ = hybrid_point_forecast(fit, future_co2, future_oni)
            candidates = {"explainable_dynamic_state_space": prediction}
        except Exception as error:
            print("Hybrid hindcast failure at {}: {}".format(origin, error))
            candidates = {}
        candidates["weighted_mean_10y"] = mean_point_forecast(fit_recent_mean(values[train], sigma[train]), horizon)
        candidates["weighted_linear_20y"] = linear_point_forecast(fit_recent_linear(values[train], sigma[train]), horizon)
        for method, estimate in candidates.items():
            for step, predicted in enumerate(estimate):
                actual = values[endpoint + step]
                records.append(
                    {
                        "method": method,
                        "origin_year": int(origin),
                        "target_year": int(future_years[step]),
                        "lead_years": int(step + 1),
                        "predicted_mm_per_year": float(predicted),
                        "actual_mm_per_year": float(actual),
                        "error_mm_per_year": float(predicted - actual),
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
    data, fallback_sigma, fallback_years = load_model_data()
    years = data["year"].to_numpy(dtype=int)
    values = data["thermosteric_mm_per_year"].to_numpy(dtype=float)
    sigma = data["observation_sigma_mm"].to_numpy(dtype=float)
    co2 = data["co2_ppm"].to_numpy(dtype=float)
    oni = data["oni"].to_numpy(dtype=float)

    details, scores = evaluate(values, sigma, years, co2, oni)
    fit = fit_hybrid(values, sigma, co2, oni)
    future_years = np.arange(FORECAST_START, FORECAST_END + 1)
    expected_co2, expected_oni = expected_inputs(years, co2, oni, future_years)
    point_forecast, contributions = hybrid_point_forecast(fit, expected_co2, expected_oni)
    rng = np.random.RandomState(RANDOM_SEED + 1)
    paths = hybrid_samples(fit, values, sigma, co2, oni, future_years, rng)
    annual, cumulative, cumulative_paths = summarise(paths, future_years)
    contributions.insert(0, "year", future_years)
    contributions.insert(1, "expected_co2_ppm", expected_co2)
    contributions.insert(2, "expected_oni", expected_oni)
    contributions.insert(3, "point_forecast_mm_per_year", point_forecast)

    data.to_csv(OUTPUT_DIR / "model_input_1959_2025.csv", index=False)
    details.to_csv(OUTPUT_DIR / "hindcast_predictions.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "hindcast_scores.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "hybrid_projection_2026_2030.csv", index=False)
    cumulative.to_csv(OUTPUT_DIR / "hybrid_cumulative_projection_2026_2030.csv", index=False)
    contributions.to_csv(OUTPUT_DIR / "driver_contributions_2026_2030.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "hybrid_projection_samples.npz", annual_mm_per_year=paths, cumulative_mm=cumulative_paths, years=future_years)
    metadata = {
        "training_period": [int(years.min()), int(years.max())],
        "outcome": "15-arc-second global 0-6000 m annual thermosteric sea-level change",
        "model": "Quadratic additive log-CO2 response + linear ONI response + AR(1) state-space residual",
        "co2_source": "NOAA GML Mauna Loa annual mean CO2",
        "oni_source": "NOAA CPC Oceanic Nino Index, annual mean",
        "future_co2_assumption": "Linear extrapolation fitted to the latest 10 annual CO2 values",
        "future_oni_assumption": "AR(1) stochastic simulation fitted to annual ONI",
        "direct_ensemble_uncertainty_years": [1941, 2023],
        "fallback_observation_sigma_years": fallback_years,
        "fallback_observation_sigma_mm": fallback_sigma,
        "forecast_years": [FORECAST_START, FORECAST_END],
        "monte_carlo_samples": SAMPLES,
        "random_seed": RANDOM_SEED + 1,
        "interpretation": "Empirical observation-constrained projection; driver contributions are associative, not causal attribution.",
        "best_hindcast_method": str(scores.iloc[0]["method"]),
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(scores.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    print("\nHybrid projection:")
    print(annual.to_string(index=False, float_format=lambda x: "{:.3f}".format(x)))
    print("\nSaved to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
