#!/usr/bin/env python3
"""Create a publication-ready historical ocean-basin thermosteric figure.

Panel a: cumulative 1941-2025 contribution to global mean sea level by basin.
Panel b: 2005-2025 mean annual basin-by-depth contribution.
Panel c: cumulative 1941-2025 depth composition within each basin.

All values are global-mean sea-level-equivalent contributions (mm), calculated
with the common global ocean area; they are additive across the five basin
masks.  No regional forecast is shown.
"""

from __future__ import print_function
from manuscript_paths import workspace_root, input_path

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.linewidth": 0.65,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


FIGURE_DIR = (workspace_root() / 'figures')
V3_ROOT = FIGURE_DIR.parent
RESULTS_ROOT = V3_ROOT / "Results" / "yearly_1940_2025"
UNCERTAINTY_MEMBER_FILE = (
    V3_ROOT
    / "uncertainty_analysis"
    / "full_spatiotemporal_uncertainty"
    / "results_1940_2023"
    / "member_results_basin.csv"
)
PAIR_PATTERN = re.compile(r"^(?P<target>\d{4})_vs_(?P<base>\d{4})$")
BASINS = ("Pacific", "Atlantic", "Indian", "Southern", "Arctic")
DEPTH_BANDS = ("0-700 m", "700-2000 m", "2000-6000 m")
DEPTH_COLORS = {
    "0-700 m": "#1f5a9d",
    "700-2000 m": "#2b9c94",
    "2000-6000 m": "#7956a7",
}
BASIN_COLORS = {
    "Pacific": "#2c6aa8",
    "Atlantic": "#d2694b",
    "Indian": "#2a9d8f",
    "Southern": "#8064a2",
    "Arctic": "#6f6f6f",
}


def band_name(depth_m):
    if depth_m <= 700.0:
        return "0-700 m"
    if depth_m <= 2000.0:
        return "700-2000 m"
    return "2000-6000 m"


def load_basin_history():
    """Read native basin CSV files and verify additive closure every year."""
    records = []
    closure = []
    for directory in sorted(RESULTS_ROOT.iterdir()):
        match = PAIR_PATTERN.match(directory.name)
        global_file = directory / "layers_basin.csv"
        if not match or not global_file.is_file():
            continue
        year = int(match.group("target"))
        global_total = float(pd.read_csv(global_file)["layer_sea_level_rise_mm"].sum())
        basin_total = 0.0
        for basin in BASINS:
            path = directory / (basin + ".csv")
            if not path.is_file():
                raise FileNotFoundError("Missing {}".format(path))
            frame = pd.read_csv(path)
            frame["depth_band"] = frame["depth_m"].map(band_name)
            grouped = frame.groupby("depth_band")["layer_global_contribution_mm"].sum()
            for band in DEPTH_BANDS:
                value = float(grouped.get(band, 0.0))
                records.append({
                    "year": year,
                    "basin": basin,
                    "depth_band": band,
                    "contribution_to_gmsl_mm": value,
                })
                basin_total += value
        closure.append({
            "year": year,
            "global_total_mm": global_total,
            "five_basin_total_mm": basin_total,
            "unassigned_residual_mm": global_total - basin_total,
        })

    annual = pd.DataFrame(records).sort_values(["basin", "year", "depth_band"])
    closure = pd.DataFrame(closure).sort_values("year")
    expected_years = np.arange(annual.year.min(), annual.year.max() + 1)
    if not np.array_equal(np.sort(annual.year.unique()), expected_years):
        raise ValueError("Basin history is not contiguous.")
    if closure["unassigned_residual_mm"].abs().max() > 0.01:
        raise ValueError("Five-basin closure exceeds 0.01 mm/yr.")
    return annual, closure


def load_cumulative_basin_sigma(years):
    """Calculate each basin's cumulative 1-sigma spread from member paths.

    Annual member anomalies are accumulated within each member before the
    ensemble standard deviation is evaluated. This retains temporal covariance
    and prevents annual standard deviations from being combined as independent.
    """
    if not UNCERTAINTY_MEMBER_FILE.is_file():
        raise FileNotFoundError("Missing {}".format(UNCERTAINTY_MEMBER_FILE))
    members = pd.read_csv(UNCERTAINTY_MEMBER_FILE)
    records = []
    for basin in BASINS:
        subset = members[(members["region"] == basin) & (members["depth_band"] == "0-6000m")]
        table = subset.pivot(index="target_year", columns="member", values="global_contribution_mm")
        available_years = np.asarray(sorted(table.index.unique()), dtype=int)
        expected_years = np.asarray([year for year in years if year <= available_years.max()], dtype=int)
        if not np.array_equal(available_years, expected_years):
            raise ValueError("Incomplete ensemble years for {}.".format(basin))
        table = table.reindex(expected_years)
        if table.isna().any().any():
            raise ValueError("Incomplete ensemble members for {}.".format(basin))
        sigma = table.cumsum(axis=0).std(axis=1, ddof=1)
        records.extend(
            {
                "year": int(year),
                "basin": basin,
                "cumulative_1sigma_mm": float(value),
                "member_count": int(table.shape[1]),
            }
            for year, value in sigma.items()
        )
    return pd.DataFrame(records).sort_values(["basin", "year"]).reset_index(drop=True)


def add_panel_label(axis, label):
    axis.text(-0.15, 1.04, label, transform=axis.transAxes, fontweight="bold", fontsize=10)


def draw_figure(annual, closure, basin_sigma):
    total = annual.groupby(["year", "basin"], as_index=False)["contribution_to_gmsl_mm"].sum()
    total["cumulative_mm"] = total.groupby("basin")["contribution_to_gmsl_mm"].cumsum()
    recent = annual[annual.year.between(2005, 2025)].groupby(
        ["basin", "depth_band"], as_index=False
    )["contribution_to_gmsl_mm"].mean()
    cumulative = annual.groupby(["basin", "depth_band"], as_index=False)["contribution_to_gmsl_mm"].sum()

    fig = plt.figure(figsize=(7.2, 5.15))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.42, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.095,
        right=0.985,
        bottom=0.105,
        top=0.93,
        wspace=0.45,
        hspace=0.53,
    )
    axis_a = fig.add_subplot(grid[:, 0])
    axis_b = fig.add_subplot(grid[0, 1])
    axis_c = fig.add_subplot(grid[1, 1])

    # a: Main visual argument. Direct end labels avoid a detached legend.
    label_y = {
        "Atlantic": 28.15,
        "Pacific": 26.45,
        "Indian": 14.15,
        "Southern": 2.25,
        "Arctic": 0.62,
    }
    for basin in BASINS:
        subset = total[total.basin == basin]
        sigma_subset = basin_sigma[basin_sigma.basin == basin]
        uncertainty_years = sigma_subset["year"].to_numpy(dtype=int)
        central_for_uncertainty = subset.set_index("year").reindex(uncertainty_years)["cumulative_mm"].to_numpy(dtype=float)
        uncertainty = sigma_subset["cumulative_1sigma_mm"].to_numpy(dtype=float)
        axis_a.fill_between(
            uncertainty_years,
            central_for_uncertainty - uncertainty,
            central_for_uncertainty + uncertainty,
            color=BASIN_COLORS[basin],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        axis_a.plot(
            subset.year,
            subset.cumulative_mm,
            color=BASIN_COLORS[basin],
            linewidth=1.8,
            solid_capstyle="round",
            zorder=3,
        )
        end_value = float(subset.cumulative_mm.iloc[-1])
        target_y = label_y[basin]
        axis_a.plot([2025.0, 2026.4], [end_value, target_y], color=BASIN_COLORS[basin], linewidth=0.85, clip_on=False)
        axis_a.text(2026.9, target_y, basin, color=BASIN_COLORS[basin], va="center", ha="left", fontsize=6.8, fontweight="bold", clip_on=False)
    axis_a.axhline(0.0, color="0.30", linewidth=0.65, zorder=0)
    axis_a.grid(axis="y", color="0.90", linewidth=0.5, zorder=0)
    axis_a.set_xlim(1940, 2037.5)
    axis_a.set_xticks(np.arange(1940, 2030, 20))
    axis_a.set_xlabel("Year")
    axis_a.set_ylabel("Cumulative contribution to global\nmean sea level (mm)")
    add_panel_label(axis_a, "a")

    # b: Recent depth-resolved contribution fingerprint.
    matrix = recent.pivot(index="basin", columns="depth_band", values="contribution_to_gmsl_mm")
    matrix = matrix.reindex(index=BASINS, columns=DEPTH_BANDS)
    maximum = float(np.nanmax(np.abs(matrix.to_numpy())))
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    image = axis_b.imshow(matrix.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
    axis_b.set_xticks(np.arange(len(DEPTH_BANDS)))
    axis_b.set_xticklabels(["0-700", "700-2000", "2000-6000"], rotation=25, ha="right", fontsize=5.8)
    axis_b.set_yticks(np.arange(len(BASINS)))
    axis_b.set_yticklabels(BASINS)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.iloc[row, column])
            text_color = "white" if abs(value) > maximum * 0.54 else "0.15"
            label = "{:.3f}".format(value) if abs(value) < 0.01 else "{:.2f}".format(value)
            axis_b.text(column, row, label, ha="center", va="center", fontsize=6.0, color=text_color)
    colorbar = fig.colorbar(image, ax=axis_b, fraction=0.052, pad=0.035)
    colorbar.set_label("Annual contribution\n(mm yr$^{-1}$)", fontsize=6.1)
    colorbar.ax.tick_params(labelsize=5.8, width=0.55, length=2)
    axis_b.tick_params(length=0)
    axis_b.set_xlabel("Depth layer (m)", labelpad=1)
    add_panel_label(axis_b, "b")

    # c: Additive long-term depth composition. Labels report totals only so
    # narrow segments remain legible; the source table carries exact values.
    cumulative_matrix = cumulative.pivot(index="basin", columns="depth_band", values="contribution_to_gmsl_mm")
    cumulative_matrix = cumulative_matrix.reindex(index=BASINS, columns=DEPTH_BANDS)
    left = np.zeros(len(BASINS), dtype=float)
    for band in DEPTH_BANDS:
        values = cumulative_matrix[band].to_numpy(dtype=float)
        axis_c.barh(BASINS, values, left=left, height=0.60, color=DEPTH_COLORS[band], label=band, linewidth=0)
        left += values
    for index, value in enumerate(left):
        axis_c.text(value + 0.42, index, "{:.1f}".format(value), va="center", ha="left", fontsize=6.2, color="0.18", fontweight="bold")
    axis_c.invert_yaxis()
    axis_c.set_xlabel("Cumulative contribution (mm)")
    axis_c.set_xlim(0.0, max(left) * 1.18)
    axis_c.set_yticks(np.arange(len(BASINS)))
    axis_c.set_yticklabels(BASINS)
    axis_c.grid(axis="x", color="0.90", linewidth=0.5, zorder=0)
    axis_c.set_axisbelow(True)
    axis_c.legend(
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.18),
        ncol=3,
        frameon=False,
        fontsize=5.8,
        handlelength=1.25,
        columnspacing=0.65,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    add_panel_label(axis_c, "c")

    return fig, recent, cumulative


def save_figure(fig, stem):
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in ("png", "tiff"):
            options["dpi"] = 600
        fig.savefig(FIGURE_DIR / (stem + "." + suffix), **options)
    plt.close(fig)


def main():
    annual, closure = load_basin_history()
    years = np.asarray(sorted(annual["year"].unique()), dtype=int)
    basin_sigma = load_cumulative_basin_sigma(years)
    fig, recent, cumulative = draw_figure(annual, closure, basin_sigma)
    stem = "Fig03_ocean_basin_historical_partition"
    save_figure(fig, stem)
    annual.to_csv(FIGURE_DIR / (stem + "_source_annual.csv"), index=False)
    recent.to_csv(FIGURE_DIR / (stem + "_source_recent_mean.csv"), index=False)
    cumulative.to_csv(FIGURE_DIR / (stem + "_source_cumulative.csv"), index=False)
    closure.to_csv(FIGURE_DIR / (stem + "_source_closure.csv"), index=False)
    basin_sigma.to_csv(FIGURE_DIR / (stem + "_source_cumulative_1sigma.csv"), index=False)
    print("Saved {} in {}".format(stem, FIGURE_DIR))


if __name__ == "__main__":
    main()
