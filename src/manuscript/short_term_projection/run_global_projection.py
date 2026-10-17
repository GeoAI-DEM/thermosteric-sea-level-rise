#!/usr/bin/env python3
"""Observation-constrained 2026-2030 global thermosteric projection.

The central history is the existing 15-arc-second 0-6000 m annual result.
IAPv4-LE spread is used only as observation uncertainty; ensemble means never
replace the 15-arc-second central values.
"""

from __future__ import print_function
from manuscript_paths import workspace_root, input_path

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = (workspace_root() / 'short_term_projection')
V3_ROOT = ROOT.parent
MAIN_RESULTS = V3_ROOT / "Results" / "yearly_1940_2025"
UNCERTAINTY_RESULTS = (
    V3_ROOT
    / "uncertainty_analysis"
    / "full_spatiotemporal_uncertainty"
    / "results_1940_2023"
)
OUTPUT_DIR = ROOT / "results"
PAIR_PATTERN = re.compile(r"^(?P<target>\d{4})_vs_(?P<base>\d{4})$")
FORECAST_START = 2026
FORECAST_END = 2030
MONTE_CARLO_SAMPLES = 10000
RANDOM_SEED = 20260922


def read_main_history():
    records = []
    for pair_dir in sorted(MAIN_RESULTS.iterdir()):
        match = PAIR_PATTERN.match(pair_dir.name)
        summary = pair_dir / "summary_basin.csv"
        if not match or not summary.is_file():
            continue
        frame = pd.read_csv(summary)
        if frame.empty or "global_sea_level_rise_mm" not in frame.columns:
            continue
        records.append(
            {
                "base_year": int(match.group("base")),
                "year": int(match.group("target")),
                "thermosteric_mm_per_year": float(frame.iloc[0]["global_sea_level_rise_mm"]),
            }
        )
    history = pd.DataFrame(records).sort_values("year").drop_duplicates("year")
    expected = np.arange(history["year"].min(), history["year"].max() + 1)
    if not np.array_equal(history["year"].to_numpy(), expected):
        raise ValueError("The annual 15-arc-second result sequence is not contiguous.")
    return history.reset_index(drop=True)


def read_observation_sigma(history):
    path = UNCERTAINTY_RESULTS / "uncertainty_summary_global.csv"
    summary = pd.read_csv(path)
    subset = summary[summary["depth_band"] == "0-6000m"].copy()
    sigma_by_year = subset.set_index("target_year")["ensemble_sd_mm"]
    observed = np.asarray([sigma_by_year.get(year, np.nan) for year in history["year"]], dtype=float)
    recent = subset[(subset["target_year"] >= 2019) & (subset["target_year"] <= 2023)]["ensemble_sd_mm"]
    fallback = float(np.median(recent))
    missing = ~np.isfinite(observed)
    observed[missing] = fallback
    return observed, fallback, history.loc[missing, "year"].astype(int).tolist()


def unpack_parameters(theta):
    q_level = np.exp(theta[0])
    q_slope = np.exp(theta[1])
    phi = 0.995 * np.tanh(theta[2])
    observation_scale = np.exp(theta[3])
    transition = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, phi]])
    process_covariance = np.diag([q_level, q_slope, np.exp(theta[4])])
    return transition, process_covariance, observation_scale


def kalman_filter(values, observation_sigma, theta, return_states=False):
    """Filter a local linear trend plus AR(1) variability state model."""
    transition, process_covariance, observation_scale = unpack_parameters(theta)
    observation_matrix = np.array([1.0, 0.0, 1.0])
    state = np.array([values[0], 0.0, 0.0])
    covariance = np.diag([100.0, 25.0, 25.0])
    negative_log_likelihood = 0.0
    filtered = []

    for value, sigma in zip(values, observation_sigma):
        variance = float(observation_matrix.dot(covariance).dot(observation_matrix) + (observation_scale * sigma) ** 2)
        if not np.isfinite(variance) or variance <= 0.0:
            return np.inf if not return_states else None
        innovation = value - float(observation_matrix.dot(state))
        gain = covariance.dot(observation_matrix) / variance
        state = state + gain * innovation
        covariance = covariance - np.outer(gain, observation_matrix.dot(covariance))
        covariance = (covariance + covariance.T) * 0.5
        negative_log_likelihood += 0.5 * (np.log(2.0 * np.pi * variance) + innovation ** 2 / variance)
        if return_states:
            filtered.append((state.copy(), covariance.copy()))
        state = transition.dot(state)
        covariance = transition.dot(covariance).dot(transition.T) + process_covariance
        covariance = (covariance + covariance.T) * 0.5

    if return_states:
        return negative_log_likelihood, filtered[-1][0], filtered[-1][1]
    return negative_log_likelihood


def fit_state_space(values, observation_sigma):
    bounds = [(-12.0, 4.0), (-15.0, 2.0), (-3.0, 3.0), (-2.0, 2.0), (-12.0, 4.0)]
    starts = [
        np.array([-1.5, -5.5, 0.0, 0.0, -1.5]),
        np.array([-3.0, -7.0, 0.5, -0.2, -1.0]),
        np.array([-0.5, -4.0, -0.5, 0.1, -2.0]),
    ]
    fits = []
    for start in starts:
        result = minimize(
            lambda theta: kalman_filter(values, observation_sigma, theta),
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1.0e-10},
        )
        if result.success and np.isfinite(result.fun):
            fits.append(result)
    if not fits:
        raise RuntimeError("State-space optimisation did not converge.")
    best = min(fits, key=lambda item: item.fun)
    _, state, covariance = kalman_filter(values, observation_sigma, best.x, return_states=True)
    return {"theta": best.x, "result": best, "state": state, "covariance": covariance, "bounds": bounds}


def state_space_point_forecast(fit, steps):
    transition, _, _ = unpack_parameters(fit["theta"])
    observation_matrix = np.array([1.0, 0.0, 1.0])
    state = fit["state"].copy()
    values = []
    for _ in range(steps):
        state = transition.dot(state)
        values.append(float(observation_matrix.dot(state)))
    return np.asarray(values)


def state_space_samples(fit, values, observation_sigma, steps, sample_count, rng):
    """Laplace-approximation parameter draws plus state/process uncertainty."""
    result = fit["result"]
    try:
        parameter_covariance = np.asarray(result.hess_inv.todense(), dtype=float)
    except AttributeError:
        parameter_covariance = np.asarray(result.hess_inv, dtype=float)
    parameter_covariance = (parameter_covariance + parameter_covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(parameter_covariance)
    parameter_covariance = eigenvectors.dot(np.diag(np.clip(eigenvalues, 1.0e-8, 4.0))).dot(eigenvectors.T)
    parameter_draws = rng.multivariate_normal(fit["theta"], parameter_covariance, size=sample_count)
    lower = np.asarray([item[0] for item in fit["bounds"]])
    upper = np.asarray([item[1] for item in fit["bounds"]])
    parameter_draws = np.clip(parameter_draws, lower, upper)
    observation_matrix = np.array([1.0, 0.0, 1.0])
    paths = np.empty((sample_count, steps), dtype=float)

    for index, theta in enumerate(parameter_draws):
        _, state, covariance = kalman_filter(values, observation_sigma, theta, return_states=True)
        transition, process_covariance, _ = unpack_parameters(theta)
        state = rng.multivariate_normal(state, covariance)
        for step in range(steps):
            state = transition.dot(state) + rng.multivariate_normal(np.zeros(3), process_covariance)
            paths[index, step] = float(observation_matrix.dot(state))
    return paths


def fit_recent_linear(values, sigma, window=20):
    use = min(window, len(values))
    y = values[-use:]
    x = np.arange(use, dtype=float)
    weights = 1.0 / np.maximum(sigma[-use:], 0.05) ** 2
    design = np.column_stack([np.ones(use), x])
    information = design.T.dot(weights[:, None] * design)
    beta = np.linalg.solve(information, design.T.dot(weights * y))
    residual = y - design.dot(beta)
    residual_sd = max(float(np.sqrt(np.average(residual ** 2, weights=weights))), 0.05)
    covariance = residual_sd ** 2 * np.linalg.pinv(information)
    return {"beta": beta, "covariance": covariance, "residual_sd": residual_sd, "x_end": use - 1}


def linear_point_forecast(fit, steps):
    x = fit["x_end"] + np.arange(1, steps + 1)
    return fit["beta"][0] + fit["beta"][1] * x


def linear_samples(fit, steps, sample_count, rng):
    beta = rng.multivariate_normal(fit["beta"], fit["covariance"], size=sample_count)
    x = fit["x_end"] + np.arange(1, steps + 1)
    mean = beta[:, [0]] + beta[:, [1]] * x[None, :]
    return mean + rng.normal(0.0, fit["residual_sd"], size=(sample_count, steps))


def fit_recent_mean(values, sigma, window=10):
    use = min(window, len(values))
    y = values[-use:]
    weights = 1.0 / np.maximum(sigma[-use:], 0.05) ** 2
    mean = float(np.average(y, weights=weights))
    residual_sd = max(float(np.sqrt(np.average((y - mean) ** 2, weights=weights))), 0.05)
    mean_sd = residual_sd / np.sqrt(use)
    return {"mean": mean, "residual_sd": residual_sd, "mean_sd": mean_sd}


def mean_point_forecast(fit, steps):
    return np.full(steps, fit["mean"], dtype=float)


def mean_samples(fit, steps, sample_count, rng):
    means = rng.normal(fit["mean"], fit["mean_sd"], size=(sample_count, 1))
    return means + rng.normal(0.0, fit["residual_sd"], size=(sample_count, steps))


METHODS = {
    "state_space": (fit_state_space, state_space_point_forecast, state_space_samples),
    "weighted_linear_20y": (fit_recent_linear, linear_point_forecast, linear_samples),
    "weighted_mean_10y": (fit_recent_mean, mean_point_forecast, mean_samples),
}


def hindcast(values, sigma, years):
    records = []
    origins = [year for year in (1995, 2000, 2005, 2010, 2015, 2020) if year in years]
    for method_name, functions in METHODS.items():
        fitter, point_forecaster, _ = functions
        for origin in origins:
            endpoint = int(np.where(years == origin)[0][0]) + 1
            horizon = min(5, len(values) - endpoint)
            if horizon == 0:
                continue
            try:
                fit = fitter(values[:endpoint], sigma[:endpoint])
                prediction = point_forecaster(fit, horizon)
            except Exception as error:
                print("Hindcast failure: {} at {}: {}".format(method_name, origin, error))
                continue
            for step in range(horizon):
                actual = values[endpoint + step]
                records.append(
                    {
                        "method": method_name,
                        "origin_year": origin,
                        "target_year": int(years[endpoint + step]),
                        "lead_years": step + 1,
                        "predicted_mm_per_year": float(prediction[step]),
                        "actual_mm_per_year": float(actual),
                        "error_mm_per_year": float(prediction[step] - actual),
                    }
                )
    details = pd.DataFrame(records)
    scores = details.groupby("method").agg(
        n=("error_mm_per_year", "size"),
        rmse_mm_per_year=("error_mm_per_year", lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2)))),
        mae_mm_per_year=("error_mm_per_year", lambda values: float(np.mean(np.abs(np.asarray(values))))),
        bias_mm_per_year=("error_mm_per_year", "mean"),
    ).reset_index()
    return details, scores


def select_method(scores):
    ranked = scores.sort_values("rmse_mm_per_year").reset_index(drop=True)
    best = ranked.iloc[0]
    return str(best["method"]), "Selected the lowest-RMSE rolling-hindcast method."


def write_projection(paths, years):
    cumulative = np.cumsum(paths, axis=1)
    annual_rows = []
    cumulative_rows = []
    for column, year in enumerate(years):
        values = paths[:, column]
        totals = cumulative[:, column]
        annual_rows.append(
            {
                "year": int(year),
                "comparison": "{} vs {}".format(year, year - 1),
                "central_mm_per_year": float(np.median(values)),
                "one_sigma_mm_per_year": float(np.std(values, ddof=1)),
                "p05_mm_per_year": float(np.percentile(values, 5)),
                "p95_mm_per_year": float(np.percentile(values, 95)),
            }
        )
        cumulative_rows.append(
            {
                "through_year": int(year),
                "period": "2026-{} relative to end-2025".format(year),
                "central_cumulative_mm": float(np.median(totals)),
                "one_sigma_cumulative_mm": float(np.std(totals, ddof=1)),
                "p05_cumulative_mm": float(np.percentile(totals, 5)),
                "p95_cumulative_mm": float(np.percentile(totals, 95)),
            }
        )
    return pd.DataFrame(annual_rows), pd.DataFrame(cumulative_rows), cumulative


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history = read_main_history()
    sigma, fallback_sigma, filled_years = read_observation_sigma(history)
    years = history["year"].to_numpy(dtype=int)
    values = history["thermosteric_mm_per_year"].to_numpy(dtype=float)

    details, scores = hindcast(values, sigma, years)
    selected, selection_reason = select_method(scores)
    fitter, _, sampler = METHODS[selected]
    fit = fitter(values, sigma)
    forecast_years = np.arange(FORECAST_START, FORECAST_END + 1)
    rng = np.random.RandomState(RANDOM_SEED)
    if selected == "state_space":
        paths = sampler(fit, values, sigma, len(forecast_years), MONTE_CARLO_SAMPLES, rng)
    else:
        paths = sampler(fit, len(forecast_years), MONTE_CARLO_SAMPLES, rng)
    annual, cumulative, cumulative_paths = write_projection(paths, forecast_years)

    history.assign(observation_sigma_mm=sigma).to_csv(OUTPUT_DIR / "input_history.csv", index=False)
    details.to_csv(OUTPUT_DIR / "hindcast_predictions.csv", index=False)
    scores.sort_values("rmse_mm_per_year").to_csv(OUTPUT_DIR / "hindcast_scores.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "projection_2026_2030.csv", index=False)
    cumulative.to_csv(OUTPUT_DIR / "cumulative_projection_2026_2030.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "projection_samples.npz", annual_mm_per_year=paths, cumulative_mm=cumulative_paths, years=forecast_years)

    metadata = {
        "central_input": "15-arc-second global 0-6000 m annual thermosteric result",
        "central_input_years": [int(years.min()), int(years.max())],
        "uncertainty_input": "IAPv4-LE global 0-6000 m ensemble standard deviation",
        "uncertainty_years_direct": [1941, 2023],
        "uncertainty_fallback_years": filled_years,
        "uncertainty_fallback_sigma_mm": fallback_sigma,
        "forecast_years": [int(forecast_years.min()), int(forecast_years.max())],
        "candidate_methods": list(METHODS.keys()),
        "selected_method": selected,
        "selection_reason": selection_reason,
        "rolling_origins": [1995, 2000, 2005, 2010, 2015, 2020],
        "monte_carlo_samples": MONTE_CARLO_SAMPLES,
        "random_seed": RANDOM_SEED,
        "interpretation": "Empirical observation-constrained short-term projection; not a CMIP scenario projection.",
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print("Selected method: {}".format(selected))
    print(selection_reason)
    print(annual.to_string(index=False, float_format=lambda value: "{:.3f}".format(value)))
    print("Saved results to {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
