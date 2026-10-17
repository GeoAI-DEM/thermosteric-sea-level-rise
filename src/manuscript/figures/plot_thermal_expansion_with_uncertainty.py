#!/usr/bin/env python3
"""Plot historical 15-arc-second results with uncertainty and 2026-2030 projections.

The central estimates always come from Results/yearly_1940_2025.  The
IAPv4-LE ensemble is used only to quantify historical spread, so its ensemble
mean never replaces the 15-arc-second central estimate.  The short-term
forecast is kept visually and statistically separate from the historical
record: dashed lines and hatched bars show its predictive median and the
shading/error bars show its predictive +/-1 sigma distribution.
"""

from __future__ import print_function
from manuscript_paths import workspace_root, input_path

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


FIGURE_DIR = (workspace_root() / 'figures')
V3_ROOT = FIGURE_DIR.parent
MAIN_RESULTS = V3_ROOT / "Results" / "yearly_1940_2025"
UNCERTAINTY_DIR = (
    V3_ROOT
    / "uncertainty_analysis"
    / "full_spatiotemporal_uncertainty"
    / "results_1940_2023"
)
PROJECTION_DIR = V3_ROOT / "short_term_projection"
TOTAL_PROJECTION_FILE = PROJECTION_DIR / "performance_weighted_ensemble_results" / "ensemble_projection_2026_2030.csv"
TOTAL_PROJECTION_SAMPLES = PROJECTION_DIR / "performance_weighted_ensemble_results" / "ensemble_projection_samples.npz"
DEPTH_PROJECTION_FILE = PROJECTION_DIR / "depth_band_ensemble_results" / "depth_band_projection_reconciled_2026_2030.csv"
DEPTH_PROJECTION_SAMPLES = PROJECTION_DIR / "depth_band_ensemble_results" / "depth_band_projection_samples_reconciled.npz"
TAIL_UNCERTAINTY_FILE = (
    V3_ROOT
    / "uncertainty_analysis"
    / "tail_uncertainty_proxy"
    / "tail_uncertainty_proxy_2024_2025.csv"
)

DEPTH_BANDS = ("0-700m", "700-2000m", "2000-6000m", "0-6000m")
DEPTH_LABELS = {
    "0-700m": "0-700 m",
    "700-2000m": "700-2000 m",
    "2000-6000m": "2000-6000 m",
    "0-6000m": "0-6000 m",
}
DEPTH_COLORS = {
    "0-700m": "#1f5a9d",
    "700-2000m": "#2b9c94",
    "2000-6000m": "#7956a7",
}
PROJECTION_COLOR = "#c65b3d"
TAIL_PROXY_COLOR = "#6d6d6d"
PAIR_PATTERN = re.compile(r"^(?P<target>\d{4})_vs_(?P<base>\d{4})$")


def depth_band_volumes(data):
    """Sum layer volume changes using the same non-overlapping depth limits."""
    depth = data["depth_m"].to_numpy(dtype=float)
    change = data["volume_change_km3"].to_numpy(dtype=float)
    return {
        # Match the exact non-overlapping boundaries used by the depth-band
        # forecast so the observed and projected lines join without a bin-edge shift.
        "0-700m": change[depth <= 700.0].sum(),
        "700-2000m": change[(depth > 700.0) & (depth <= 2000.0)].sum(),
        "2000-6000m": change[(depth > 2000.0) & (depth <= 6000.0)].sum(),
        "0-6000m": change.sum(),
    }


def read_summary_area(pair_dir, total_volume_km3):
    """Derive the exact km3/mm conversion used by the 15-arc-second run."""
    candidates = [pair_dir / "summary_basin.csv"] + sorted(pair_dir.glob("summary_*.csv"))
    for path in candidates:
        if not path.is_file():
            continue
        summary = pd.read_csv(path)
        if summary.empty:
            continue
        row = summary.iloc[0]
        for column in ("global_sea_level_rise_mm", "sea_level_rise_mm"):
            if column in summary.columns:
                value = float(row[column])
                if np.isfinite(value) and abs(value) > 1.0e-12:
                    return total_volume_km3 / value
    raise ValueError("Cannot derive the global km3/mm conversion from {}".format(pair_dir))


def load_main_results():
    """Read annual 15-arc-second layer results and convert volumes to mm."""
    records = []
    conversions = []
    for pair_dir in sorted(MAIN_RESULTS.iterdir()):
        match = PAIR_PATTERN.match(pair_dir.name)
        data_path = pair_dir / "data.csv"
        if not match or not data_path.is_file():
            continue
        volumes = depth_band_volumes(pd.read_csv(data_path))
        conversion = read_summary_area(pair_dir, volumes["0-6000m"])
        conversions.append(conversion)
        record = {
            "base_year": int(match.group("base")),
            "target_year": int(match.group("target")),
            "km3_per_mm": conversion,
        }
        for band, value in volumes.items():
            record[band + "_km3"] = value
            record[band + "_mm"] = value / conversion
        records.append(record)

    if not records:
        raise FileNotFoundError("No annual results found under {}".format(MAIN_RESULTS))
    frame = pd.DataFrame(records).sort_values("target_year").reset_index(drop=True)
    reference_conversion = float(np.median(conversions))
    if not np.allclose(conversions, reference_conversion, rtol=1.0e-8, atol=1.0e-8):
        raise ValueError("The 15-arc-second runs do not use a common global area conversion.")
    return frame, reference_conversion


def load_uncertainty():
    summary_path = UNCERTAINTY_DIR / "uncertainty_summary_global.csv"
    member_path = UNCERTAINTY_DIR / "member_results_global.csv"
    if not summary_path.is_file() or not member_path.is_file():
        raise FileNotFoundError("Expected uncertainty CSV files under {}".format(UNCERTAINTY_DIR))
    return pd.read_csv(summary_path), pd.read_csv(member_path)


def load_projections():
    """Load the reconciled predictive distributions used only beyond 2025."""
    required = (
        TOTAL_PROJECTION_FILE,
        TOTAL_PROJECTION_SAMPLES,
        DEPTH_PROJECTION_FILE,
        DEPTH_PROJECTION_SAMPLES,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing projection outputs: {}".format("; ".join(missing)))

    total = pd.read_csv(TOTAL_PROJECTION_FILE).sort_values("year").reset_index(drop=True)
    depth = pd.read_csv(DEPTH_PROJECTION_FILE).sort_values(["depth_band", "year"]).reset_index(drop=True)
    with np.load(TOTAL_PROJECTION_SAMPLES) as archive:
        total_years = archive["years"].astype(int)
        total_samples = archive["annual_mm_per_year"].astype(float)
    with np.load(DEPTH_PROJECTION_SAMPLES) as archive:
        depth_years = archive["years"].astype(int)
        depth_samples = {
            "0-700m": archive["band_0_700m"].astype(float),
            "700-2000m": archive["band_700_2000m"].astype(float),
            "2000-6000m": archive["band_2000_6000m"].astype(float),
        }
    if not np.array_equal(total_years, depth_years):
        raise ValueError("Global and depth-band projection years do not match.")
    if not np.array_equal(total["year"].to_numpy(dtype=int), total_years):
        raise ValueError("Global projection summary and samples do not match.")
    for band in DEPTH_BANDS[:3]:
        band_years = depth.loc[depth["depth_band"] == band, "year"].to_numpy(dtype=int)
        if not np.array_equal(band_years, total_years):
            raise ValueError("Depth-band projection summary is incomplete for {}.".format(band))
    return {
        "years": total_years,
        "total": total,
        "total_samples": total_samples,
        "depth": depth,
        "depth_samples": depth_samples,
    }


def load_tail_uncertainty():
    if not TAIL_UNCERTAINTY_FILE.is_file():
        raise FileNotFoundError("Missing calibrated tail uncertainty proxy: {}".format(TAIL_UNCERTAINTY_FILE))
    tail = pd.read_csv(TAIL_UNCERTAINTY_FILE).sort_values(["depth_band", "target_year"]).reset_index(drop=True)
    expected_years = np.array([2024, 2025], dtype=int)
    for band in DEPTH_BANDS:
        years = tail.loc[tail["depth_band"] == band, "target_year"].to_numpy(dtype=int)
        if not np.array_equal(years, expected_years):
            raise ValueError("Tail proxy is incomplete for {}.".format(band))
    return tail


def tail_sigma(tail, band):
    subset = tail[tail["depth_band"] == band].sort_values("target_year")
    return (
        subset["target_year"].to_numpy(dtype=int),
        subset["proxy_sigma_mm"].to_numpy(dtype=float),
    )


def cumulative_tail_sigma(cumulative_sigma_mm, years, annual_tail_sigma, tail_years):
    """Extend the 2023 cumulative spread with proxy annual errors.

    The extension assumes independent tail-year proxy errors because the
    unavailable IAPv4-LE members cannot supply their temporal covariance.
    """
    anchor_year = int(tail_years.min()) - 1
    anchor_index = np.where(years == anchor_year)[0]
    if len(anchor_index) != 1 or not np.isfinite(cumulative_sigma_mm[anchor_index[0]]):
        raise ValueError("Cannot anchor cumulative tail uncertainty at {}.".format(anchor_year))
    anchor_sigma = float(cumulative_sigma_mm[anchor_index[0]])
    return np.sqrt(anchor_sigma ** 2 + np.cumsum(np.square(annual_tail_sigma)))


def annual_sigma(summary, years, band):
    subset = summary[summary["depth_band"] == band]
    values = subset.set_index("target_year")["ensemble_sd_mm"]
    return np.asarray([values.get(year, np.nan) for year in years], dtype=float)


def cumulative_sigma(member_results, years, band):
    """Calculate spread after each member's annual anomalies are accumulated."""
    subset = member_results[member_results["depth_band"] == band]
    table = subset.pivot(index="target_year", columns="member", values="global_contribution_mm")
    table = table.reindex([year for year in years if year <= int(table.index.max())])
    if table.isna().any().any():
        raise ValueError("Incomplete member time series for {}".format(band))
    sigma = table.cumsum(axis=0).std(axis=1, ddof=1)
    output = np.full(len(years), np.nan, dtype=float)
    for index, year in enumerate(years):
        if year in sigma.index:
            output[index] = float(sigma.loc[year])
    return output


def save_all(fig, stem):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            FIGURE_DIR / (stem + "." + suffix),
            dpi=450 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def mark_projection_start(axis):
    axis.axvline(2025.5, color="0.35", linestyle=(0, (2, 2)), linewidth=0.8, zorder=1)


def add_compact_figure_legend(fig, axis):
    """Keep uncertainty labels outside the data area."""
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.890),
        ncol=5,
        frameon=False,
        fontsize=6.8,
        handlelength=1.45,
        columnspacing=0.85,
        handletextpad=0.4,
    )


def plot_global(main, summary, member_results, projections, tail):
    years = main["target_year"].to_numpy(dtype=int)
    central = main["0-6000m_mm"].to_numpy(dtype=float)
    sigma = annual_sigma(summary, years, "0-6000m")
    cumulative = np.cumsum(central)
    cumulative_sigma_mm = cumulative_sigma(member_results, years, "0-6000m")
    forecast_years = projections["years"]
    forecast = projections["total"]
    forecast_samples = projections["total_samples"]
    tail_years, tail_annual_sigma = tail_sigma(tail, "0-6000m")
    tail_positions = np.searchsorted(years, tail_years)
    if not np.array_equal(years[tail_positions], tail_years):
        raise ValueError("Historical central record does not contain both tail-proxy years.")

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), sharex=True, gridspec_kw={"hspace": 0.24})
    positive = np.where(central >= 0.0, "#2b9c94", "#ce6d54")
    axes[0].bar(years, central, color=positive, width=0.78, linewidth=0, zorder=2, label="15 arcsec estimate")
    valid = np.isfinite(sigma)
    axes[0].errorbar(years[valid], central[valid], yerr=sigma[valid], fmt="none", ecolor="0.25", elinewidth=0.55, capsize=1.5, alpha=0.72, zorder=3, label="IAPv4-LE 1$\\sigma$")
    axes[0].errorbar(tail_years, central[tail_positions], yerr=tail_annual_sigma, fmt="none", ecolor=TAIL_PROXY_COLOR, elinewidth=0.85, capsize=1.9, zorder=4, label="2024-25 proxy 1$\\sigma$")
    axes[0].bar(forecast_years, forecast["central_mm_per_year"], width=0.78, color=PROJECTION_COLOR, alpha=0.72, hatch="//", edgecolor=PROJECTION_COLOR, linewidth=0.55, zorder=2, label="2026-30 forecast")
    axes[0].errorbar(forecast_years, forecast["central_mm_per_year"], yerr=forecast["one_sigma_mm_per_year"], fmt="none", ecolor=PROJECTION_COLOR, elinewidth=0.75, capsize=1.7, zorder=4, label="Forecast 1$\\sigma$")
    axes[0].axhline(0.0, color="0.25", linewidth=0.75)
    mark_projection_start(axes[0])
    axes[0].set_ylabel("Annual thermosteric\nchange (mm)")
    axes[0].text(-0.09, 1.03, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=13)

    axes[1].plot(years, cumulative, color="#1f5a9d", marker="o", markersize=2.7, linewidth=1.8, label="15 arcsec estimate")
    valid = np.isfinite(cumulative_sigma_mm)
    axes[1].fill_between(years[valid], cumulative[valid] - cumulative_sigma_mm[valid], cumulative[valid] + cumulative_sigma_mm[valid], color="#7ca6d0", alpha=0.34, linewidth=0, label="IAPv4-LE 1$\\sigma$")
    tail_cumulative_sigma = cumulative_tail_sigma(cumulative_sigma_mm, years, tail_annual_sigma, tail_years)
    tail_cumulative_years = np.r_[tail_years[0] - 1, tail_years]
    tail_cumulative_values = np.r_[cumulative[tail_positions[0] - 1], cumulative[tail_positions]]
    tail_cumulative_spread = np.r_[cumulative_sigma_mm[tail_positions[0] - 1], tail_cumulative_sigma]
    axes[1].fill_between(tail_cumulative_years, tail_cumulative_values - tail_cumulative_spread, tail_cumulative_values + tail_cumulative_spread, color=TAIL_PROXY_COLOR, alpha=0.18, linewidth=0, label="2024-25 proxy 1$\\sigma$")
    # Prediction samples are conditional on the 2025 central estimate. Add
    # the independent historical-plus-tail proxy offset so cumulative forecast
    # intervals begin continuously from the 2025 uncertainty envelope.
    tail_offset = np.random.RandomState(20260928).normal(
        0.0, tail_cumulative_sigma[-1], size=(forecast_samples.shape[0], 1)
    )
    forecast_cumulative_paths = cumulative[-1] + tail_offset + np.cumsum(forecast_samples, axis=1)
    forecast_cumulative = np.median(forecast_cumulative_paths, axis=0)
    forecast_cumulative_sigma = np.std(forecast_cumulative_paths, axis=0, ddof=1)
    axes[1].plot(np.r_[years[-1], forecast_years], np.r_[cumulative[-1], forecast_cumulative], color=PROJECTION_COLOR, linestyle="-", marker="o", markersize=2.1, linewidth=1.5, label="2026-30 forecast")
    axes[1].fill_between(np.r_[years[-1], forecast_years], np.r_[cumulative[-1] - tail_cumulative_sigma[-1], forecast_cumulative - forecast_cumulative_sigma], np.r_[cumulative[-1] + tail_cumulative_sigma[-1], forecast_cumulative + forecast_cumulative_sigma], color=PROJECTION_COLOR, alpha=0.20, linewidth=0, label="Forecast 1$\\sigma$")
    axes[1].axhline(0.0, color="0.25", linewidth=0.75)
    mark_projection_start(axes[1])
    axes[1].set_ylabel("Cumulative change\nsince 1940 (mm)")
    axes[1].set_xlabel("Year")
    axes[1].text(-0.09, 1.03, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=13)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(years.min() - 0.8, forecast_years.max() + 0.8)
    axes[1].set_xticks(np.arange(1940, 2031, 10))
    fig.subplots_adjust(top=0.87, bottom=0.08)
    add_compact_figure_legend(fig, axes[1])
    save_all(fig, "Fig01_global_annual_and_cumulative_with_projection_and_tail_proxy")


def plot_depth_annual(main, summary, km3_per_mm, projections, tail):
    years = main["target_year"].to_numpy(dtype=int)
    forecast_years = projections["years"]
    fig, axes = plt.subplots(3, 1, figsize=(11.2, 10.0), sharex=True, gridspec_kw={"hspace": 0.12})
    for panel, band in enumerate(DEPTH_BANDS[:3]):
        values = main[band + "_km3"].to_numpy(dtype=float)
        sigma_km3 = annual_sigma(summary, years, band) * km3_per_mm
        forecast_samples = projections["depth_samples"][band] * km3_per_mm
        forecast_median = np.median(forecast_samples, axis=0)
        forecast_sigma = np.std(forecast_samples, axis=0, ddof=1)
        tail_years, tail_annual_sigma = tail_sigma(tail, band)
        tail_positions = np.searchsorted(years, tail_years)
        colors = np.where(values >= 0.0, "#267f3f", "#bd4744")
        axes[panel].bar(years, values, width=0.78, color=colors, linewidth=0, zorder=2)
        valid = np.isfinite(sigma_km3)
        axes[panel].errorbar(years[valid], values[valid], yerr=sigma_km3[valid], fmt="none", ecolor="0.18", elinewidth=0.42, capsize=1.25, alpha=0.62, zorder=3)
        axes[panel].errorbar(tail_years, values[tail_positions], yerr=tail_annual_sigma * km3_per_mm, fmt="none", ecolor=TAIL_PROXY_COLOR, elinewidth=0.85, capsize=1.9, zorder=4)
        axes[panel].bar(forecast_years, forecast_median, width=0.78, color=PROJECTION_COLOR, alpha=0.72, hatch="//", edgecolor=PROJECTION_COLOR, linewidth=0.55, zorder=2)
        axes[panel].errorbar(forecast_years, forecast_median, yerr=forecast_sigma, fmt="none", ecolor=PROJECTION_COLOR, elinewidth=0.75, capsize=1.7, zorder=4)
        axes[panel].axhline(0.0, color="0.25", linewidth=0.72)
        mark_projection_start(axes[panel])
        axes[panel].set_title(DEPTH_LABELS[band], loc="left", fontsize=11, pad=7)
        axes[panel].text(-0.10, 1.03, chr(ord("a") + panel), transform=axes[panel].transAxes, fontweight="bold", fontsize=13)
        axes[panel].spines["top"].set_visible(False)
        axes[panel].spines["right"].set_visible(False)
    axes[1].set_ylabel("Annual expansion\n(km$^3$ yr$^{-1}$)")
    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(np.arange(1940, 2031, 10))
    axes[-1].set_xlim(years.min() - 0.8, forecast_years.max() + 0.8)
    fig.text(0.5, 0.005, "IAPv4-LE through 2023; gray: 2024-25 proxy; hatched orange: 2026-30 forecast.", ha="center", fontsize=7.4, color="0.32")
    save_all(fig, "FigA_annual_changes_by_depth_class_with_projection_and_tail_proxy")


def plot_depth_cumulative(main, member_results, km3_per_mm, projections, tail):
    """Build the complete Fig. 08 composite.

    Panel a retains the three historical depth-band time series and adds the
    clearly separated uncertainty proxy and forecast.  Panels b and c remain
    historical summaries, matching the original Fig. 08 definitions rather
    than mixing a short forecast into period-average statistics.
    """
    years = main["target_year"].to_numpy(dtype=int)
    forecast_years = projections["years"]
    fig = plt.figure(figsize=(13.0, 8.2))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(1.38, 1.0),
        hspace=0.34,
        wspace=0.32,
        left=0.09,
        right=0.98,
        bottom=0.11,
        top=0.84,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
    ]
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    period_axis = fig.add_subplot(grid[:2, 1])
    share_axis = fig.add_subplot(grid[2, 1])

    for panel, band in enumerate(DEPTH_BANDS[:3]):
        cumulative = np.cumsum(main[band + "_km3"].to_numpy(dtype=float))
        sigma_km3 = cumulative_sigma(member_results, years, band) * km3_per_mm
        color = DEPTH_COLORS[band]
        tail_years, tail_annual_sigma = tail_sigma(tail, band)
        tail_positions = np.searchsorted(years, tail_years)
        tail_cumulative_sigma = cumulative_tail_sigma(
            sigma_km3, years, tail_annual_sigma * km3_per_mm, tail_years
        )
        tail_cumulative_years = np.r_[tail_years[0] - 1, tail_years]
        tail_cumulative_values = np.r_[cumulative[tail_positions[0] - 1], cumulative[tail_positions]]
        tail_cumulative_spread = np.r_[sigma_km3[tail_positions[0] - 1], tail_cumulative_sigma]
        forecast_paths = projections["depth_samples"][band] * km3_per_mm
        tail_offset = np.random.RandomState(20260930 + panel).normal(
            0.0, tail_cumulative_sigma[-1], size=(forecast_paths.shape[0], 1)
        )
        forecast_cumulative_paths = cumulative[-1] + tail_offset + np.cumsum(forecast_paths, axis=1)
        forecast_cumulative = np.median(forecast_cumulative_paths, axis=0)
        forecast_cumulative_sigma = np.std(forecast_cumulative_paths, axis=0, ddof=1)
        axes[panel].plot(years, cumulative, color=color, linewidth=2.0, label="Historical estimate")
        valid = np.isfinite(sigma_km3)
        axes[panel].fill_between(years[valid], cumulative[valid] - sigma_km3[valid], cumulative[valid] + sigma_km3[valid], color=color, alpha=0.24, linewidth=0, label="IAPv4-LE 1$\\sigma$" if panel == 0 else None)
        axes[panel].fill_between(tail_cumulative_years, tail_cumulative_values - tail_cumulative_spread, tail_cumulative_values + tail_cumulative_spread, color=TAIL_PROXY_COLOR, alpha=0.18, linewidth=0, label="2024-25 proxy 1$\\sigma$" if panel == 0 else None)
        axes[panel].plot(np.r_[years[-1], forecast_years], np.r_[cumulative[-1], forecast_cumulative], color=PROJECTION_COLOR, linestyle="-", marker="o", markersize=2.0, linewidth=1.5, label="2026-30 forecast" if panel == 0 else None)
        axes[panel].fill_between(np.r_[years[-1], forecast_years], np.r_[cumulative[-1] - tail_cumulative_sigma[-1], forecast_cumulative - forecast_cumulative_sigma], np.r_[cumulative[-1] + tail_cumulative_sigma[-1], forecast_cumulative + forecast_cumulative_sigma], color=PROJECTION_COLOR, alpha=0.20, linewidth=0, label="Forecast 1$\\sigma$" if panel == 0 else None)
        axes[panel].axhline(0.0, color="0.35", linewidth=0.7)
        mark_projection_start(axes[panel])
        axes[panel].text(
            0.015,
            1.010,
            DEPTH_LABELS[band],
            transform=axes[panel].transAxes,
            ha="left",
            va="bottom",
            color=color,
            fontsize=8.8,
            fontweight="bold",
        )
        axes[panel].spines["top"].set_visible(False)
        axes[panel].spines["right"].set_visible(False)
    axes[0].text(-0.12, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Cumulative expansion\n(10$^3$ km$^3$)")
    volume_formatter = FuncFormatter(lambda value, _position: "{:.0f}".format(value / 1000.0))
    for axis in axes:
        axis.yaxis.set_major_formatter(volume_formatter)
    axes[-1].set_xlabel("Year")
    axes[-1].set_xticks(np.arange(1940, 2031, 10))
    axes[-1].set_xlim(years.min() - 0.8, forecast_years.max() + 0.8)

    # Historical period means, preserved from the original Fig. 08.
    periods = (
        (1941, 1956),
        (1957, 1970),
        (1971, 1992),
        (1993, 2004),
        (2005, 2025),
    )
    positions = np.arange(len(periods), dtype=float)
    width = 0.23
    for index, band in enumerate(DEPTH_BANDS[:3]):
        values = []
        for start, end in periods:
            subset = main.loc[
                (main["target_year"] >= start) & (main["target_year"] <= end),
                band + "_km3",
            ]
            values.append(float(subset.mean()) / 1000.0)
        bars = period_axis.bar(
            positions + (index - 1) * width,
            values,
            width=width,
            color=DEPTH_COLORS[band],
            linewidth=0,
            zorder=2,
        )
        for bar, value in zip(bars, values):
            offset = 0.010 if value >= 0.0 else -0.010
            period_axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + offset,
                "{:.3f}".format(value),
                ha="center",
                va="bottom" if value >= 0.0 else "top",
                fontsize=5.8,
                color=DEPTH_COLORS[band],
                rotation=90,
            )
    period_axis.axhline(0.0, color="0.25", linewidth=0.75, zorder=1)
    period_axis.set_xlim(-0.60, len(periods) - 0.40)
    period_axis.set_ylim(-0.075, 0.50)
    period_axis.set_xticks(positions)
    period_axis.set_xticklabels(["{}-{}".format(start, end) for start, end in periods], rotation=32, ha="right", fontsize=7)
    period_axis.set_ylabel("Mean annual expansion\n(10$^3$ km$^3$ yr$^{-1}$)")
    period_axis.text(-0.13, 1.04, "b", transform=period_axis.transAxes, fontweight="bold", fontsize=13)
    period_axis.legend(
        handles=[Patch(facecolor=DEPTH_COLORS[band], edgecolor="none") for band in DEPTH_BANDS[:3]],
        labels=[DEPTH_LABELS[band] for band in DEPTH_BANDS[:3]],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        handlelength=1.4,
        columnspacing=1.1,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    period_axis.spines["top"].set_visible(False)
    period_axis.spines["right"].set_visible(False)

    # Contribution shares use the same 1941-2025 historical integration as
    # the original Fig. 08. Forecast years are intentionally excluded.
    historical = main.loc[(main["target_year"] >= 1941) & (main["target_year"] <= 2025)]
    contributions = np.array(
        [historical[band + "_km3"].sum() for band in DEPTH_BANDS[:3]], dtype=float
    )
    shares = contributions / contributions.sum() * 100.0
    left = 0.0
    segment_centres = []
    for index, (band, share) in enumerate(zip(DEPTH_BANDS[:3], shares)):
        share_axis.barh(0, share, left=left, height=0.34, color=DEPTH_COLORS[band], linewidth=0)
        segment_centres.append(left + share / 2.0)
        share_axis.text(
            left + share / 2.0,
            0,
            "{:.2f}%".format(share),
            ha="center",
            va="center",
            color="white",
            fontsize=8.3,
            fontweight="bold",
        )
        left += share
    # Keep labels outside the colored blocks so they stay single-line and do
    # not compete with the contribution percentages.
    label_layout = (
        (0.47, "bottom"),
        (-0.47, "top"),
        (0.47, "bottom"),
    )
    for band, centre, (label_y, vertical_alignment) in zip(DEPTH_BANDS[:3], segment_centres, label_layout):
        line_end = label_y - 0.07 if label_y > 0.0 else label_y + 0.07
        share_axis.plot([centre, centre], [0.19 if label_y > 0.0 else -0.19, line_end], color=DEPTH_COLORS[band], linewidth=1.05, clip_on=False)
        share_axis.text(
            centre,
            label_y,
            DEPTH_LABELS[band],
            ha="center",
            va=vertical_alignment,
            color=DEPTH_COLORS[band],
            fontsize=7.4,
            fontweight="bold",
            clip_on=False,
        )
    share_axis.set_xlim(0.0, 100.0)
    share_axis.set_ylim(-0.62, 0.62)
    share_axis.set_axis_off()
    share_axis.text(-0.13, 1.04, "c", transform=share_axis.transAxes, fontweight="bold", fontsize=13)
    share_box = share_axis.get_position()
    share_axis.set_position([
        share_box.x0,
        share_box.y0 - 0.035,
        share_box.width,
        share_box.height,
    ])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.32, 0.890),
        ncol=5,
        frameon=False,
        fontsize=6.7,
        handlelength=1.45,
        columnspacing=0.75,
        handletextpad=0.4,
    )
    save_all(fig, "Fig08_three_depth_evolution_with_projection_and_tail_proxy")


def main():
    main_results, km3_per_mm = load_main_results()
    summary, member_results = load_uncertainty()
    projections = load_projections()
    tail = load_tail_uncertainty()
    years = main_results["target_year"].to_numpy(dtype=int)
    expected = np.arange(years.min(), years.max() + 1)
    if not np.array_equal(years, expected):
        raise ValueError("Annual 15-arc-second results are not contiguous.")
    if int(main_results["target_year"].max()) + 1 != int(projections["years"].min()):
        raise ValueError("Projection must start immediately after the historical record.")
    plot_global(main_results, summary, member_results, projections, tail)
    plot_depth_annual(main_results, summary, km3_per_mm, projections, tail)
    plot_depth_cumulative(main_results, member_results, km3_per_mm, projections, tail)
    print("Saved projection-overlay figures in {}".format(FIGURE_DIR))


if __name__ == "__main__":
    main()
