#!/usr/bin/env python3
"""Build calibrated 2024-2025 uncertainty proxies from IAPv4-LE spread.

This script does not create new IAPv4-LE ensemble members. It estimates a
tail proxy only because the available ensemble record ends in 2023. For each
depth band, four simple error-variance models are evaluated using blocked
rolling-origin hindcasts within the modern 2005-2023 observing era. Their
weights and multiplicative calibration are fitted separately for one- and
two-year forecast horizons.

The resulting values must be labelled "calibrated extrapolated uncertainty"
or "proxy 1-sigma". They are not observed IAPv4-LE ensemble uncertainty.
"""

from __future__ import print_function

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = None
OUTPUT_FILE = None
VALIDATION_FILE = None
METADATA_FILE = None

DEPTH_BANDS = ("0-700m", "700-2000m", "2000-6000m", "0-6000m")
MODERN_START_YEAR = 2005
ROLLING_WINDOW_YEARS = 5
TARGET_YEARS = (2024, 2025)
CANDIDATES = (
    "rms5_variance_persistence",
    "logmean5",
    "log_ar1",
    "damped_log_linear_10y",
)


def rms_sigma(values):
    """RMS is the locally constant variance forecast expressed as 1 sigma."""
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


def candidate_forecast(sigma, horizon, candidate):
    """Forecast annual 1-sigma spread at one or more future horizons."""
    sigma = np.asarray(sigma, dtype=float)
    log_sigma = np.log(sigma)
    if candidate == "rms5_variance_persistence":
        return np.full(horizon, rms_sigma(sigma[-ROLLING_WINDOW_YEARS:]))
    if candidate == "logmean5":
        return np.full(horizon, np.exp(np.mean(log_sigma[-ROLLING_WINDOW_YEARS:])))
    if candidate == "log_ar1":
        design = np.column_stack([np.ones(len(log_sigma) - 1), log_sigma[:-1]])
        coefficients = np.linalg.lstsq(design, log_sigma[1:], rcond=None)[0]
        output = []
        state = float(log_sigma[-1])
        for _ in range(horizon):
            state = float(coefficients[0] + coefficients[1] * state)
            output.append(np.exp(state))
        return np.asarray(output)
    if candidate == "damped_log_linear_10y":
        window = min(10, len(log_sigma))
        x = np.arange(window, dtype=float)
        coefficients = np.linalg.lstsq(
            np.column_stack([np.ones(window), x]), log_sigma[-window:], rcond=None
        )[0]
        # A half-strength future slope prevents an unvalidated linear trend
        # from dominating a two-year tail extrapolation.
        future_x = window - 1 + 0.5 * np.arange(1, horizon + 1, dtype=float)
        return np.exp(coefficients[0] + coefficients[1] * future_x)
    raise ValueError("Unknown candidate: {}".format(candidate))


def fit_horizon_ensemble(modern, horizon):
    """Fit horizon-specific candidate weights from strict rolling origins."""
    sigma = modern["ensemble_sd_mm"].to_numpy(dtype=float)
    start = max(ROLLING_WINDOW_YEARS + 2, 7)
    validation_rows = []
    for endpoint in range(start, len(sigma) - horizon + 1):
        actual = float(sigma[endpoint + horizon - 1])
        for candidate in CANDIDATES:
            predicted = float(candidate_forecast(sigma[:endpoint], horizon, candidate)[-1])
            validation_rows.append({
                "origin_year": int(modern.iloc[endpoint - 1]["target_year"]),
                "target_year": int(modern.iloc[endpoint + horizon - 1]["target_year"]),
                "horizon_years": horizon,
                "candidate": candidate,
                "hindcast_sigma_mm": predicted,
                "observed_iapv4le_sigma_mm": actual,
            })
    validation = pd.DataFrame(validation_rows)
    scores = validation.groupby("candidate").apply(
        lambda group: float(np.sqrt(np.mean(np.square(
            np.log(group["hindcast_sigma_mm"].to_numpy(dtype=float))
            - np.log(group["observed_iapv4le_sigma_mm"].to_numpy(dtype=float))
        ))))
    )
    inverse_variance = 1.0 / np.maximum(np.square(scores), 1.0e-12)
    weights = inverse_variance / inverse_variance.sum()
    weight_map = {candidate: float(weights[candidate]) for candidate in CANDIDATES}
    validation["candidate_weight"] = validation["candidate"].map(weight_map)
    ensemble = validation.groupby(["origin_year", "target_year", "horizon_years"], as_index=False).apply(
        lambda group: pd.Series({
            "ensemble_hindcast_sigma_mm": float(np.sum(
                group["candidate_weight"].to_numpy(dtype=float)
                * group["hindcast_sigma_mm"].to_numpy(dtype=float)
            )),
            "observed_iapv4le_sigma_mm": float(group["observed_iapv4le_sigma_mm"].iloc[0]),
        })
    ).reset_index(drop=True)
    log_ratio = np.log(
        ensemble["observed_iapv4le_sigma_mm"].to_numpy(dtype=float)
        / ensemble["ensemble_hindcast_sigma_mm"].to_numpy(dtype=float)
    )
    calibration_factor = float(np.exp(np.mean(log_ratio)))
    residual_log_sd = float(np.std(log_ratio, ddof=1))
    return validation, scores.to_dict(), weight_map, calibration_factor, residual_log_sd


def build_band_proxy(frame, band):
    series = frame.loc[frame["depth_band"] == band, ["target_year", "ensemble_sd_mm"]].copy()
    series = series.dropna().sort_values("target_year").reset_index(drop=True)
    modern = series[series["target_year"] >= MODERN_START_YEAR].reset_index(drop=True)
    if len(modern) < ROLLING_WINDOW_YEARS + 3:
        raise ValueError("Insufficient modern IAPv4-LE uncertainty records for {}.".format(band))

    rows = []
    validations = []
    source_sigma = modern["ensemble_sd_mm"].to_numpy(dtype=float)
    for horizon, year in enumerate(TARGET_YEARS, start=1):
        validation, scores, weights, calibration_factor, residual_log_sd = fit_horizon_ensemble(modern, horizon)
        weighted_raw = sum(
            weights[candidate] * candidate_forecast(source_sigma, horizon, candidate)[-1]
            for candidate in CANDIDATES
        )
        proxy_sigma = weighted_raw * calibration_factor
        validation["depth_band"] = band
        validation["rolling_log_rmse"] = validation["candidate"].map(scores)
        validation["calibration_factor"] = calibration_factor
        validations.append(validation)
        rows.append({
            "target_year": year,
            "depth_band": band,
            "proxy_sigma_mm": proxy_sigma,
            "weighted_raw_sigma_mm": weighted_raw,
            "rolling_calibration_factor": calibration_factor,
            "rolling_log_residual_sd": residual_log_sd,
            "validation_n": int(validation["origin_year"].nunique()),
            "candidate_log_rmse": json.dumps(scores, sort_keys=True),
            "candidate_weights": json.dumps(weights, sort_keys=True),
            "method": "horizon_specific_performance_weighted_sigma_forecast_with_rolling_origin_log_scale_calibration",
            "status": "calibrated_extrapolated_proxy_not_iapv4le_ensemble",
        })
    return pd.DataFrame(rows), pd.concat(validations, ignore_index=True)


def main():
    global INPUT_FILE, OUTPUT_FILE, VALIDATION_FILE, METADATA_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="uncertainty_summary_global.csv through 2023")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "tail_uncertainty_proxy")
    args = parser.parse_args()
    INPUT_FILE = args.input_file.resolve()
    output_dir = args.output_dir.resolve()
    OUTPUT_FILE = output_dir / "tail_uncertainty_proxy_2024_2025.csv"
    VALIDATION_FILE = output_dir / "tail_uncertainty_proxy_rolling_validation.csv"
    METADATA_FILE = output_dir / "tail_uncertainty_proxy_metadata.json"
    if not INPUT_FILE.is_file():
        raise FileNotFoundError("Missing IAPv4-LE uncertainty input: {}".format(INPUT_FILE))
    source = pd.read_csv(INPUT_FILE)
    outputs = []
    validations = []
    for band in DEPTH_BANDS:
        proxy, validation = build_band_proxy(source, band)
        outputs.append(proxy)
        validations.append(validation)
    output = pd.concat(outputs, ignore_index=True)
    validation = pd.concat(validations, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    validation.to_csv(VALIDATION_FILE, index=False)
    metadata = {
        "purpose": "Sensitivity-only tail uncertainty proxy for 2024 and 2025 after IAPv4-LE member results end in 2023.",
        "input": str(INPUT_FILE),
        "modern_calibration_period": "2005-2023",
        "variance_forecast": "Horizon-specific performance-weighted ensemble of five-year RMS persistence, five-year log mean, log-AR(1), and damped ten-year log-linear candidates",
        "calibration": "Multiplicative geometric-mean correction from rolling-origin hindcasts on the log-sigma scale, fitted separately for each horizon",
        "interpretation": "Not an observed IAPv4-LE uncertainty. Do not describe as an ensemble-derived uncertainty.",
        "recommended_label": "calibrated extrapolated uncertainty (proxy 1-sigma)",
    }
    with METADATA_FILE.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(output.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
    print("\nWrote:", OUTPUT_FILE)
    print("Wrote:", VALIDATION_FILE)


if __name__ == "__main__":
    main()
