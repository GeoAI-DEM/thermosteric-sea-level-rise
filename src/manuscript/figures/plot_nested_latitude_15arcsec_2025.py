#!/usr/bin/env python3
"""Plot 1941-2025 native-grid nested-latitude thermosteric contributions."""

from manuscript_paths import workspace_root, input_path
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


FIGURES = (workspace_root() / 'figures')
INPUT = FIGURES / "nested_latitude_15arcsec/Fig04_nested_latitude_15arcsec_annual_main.csv"
OUTPUT = FIGURES / "revised_lon_aligned_20260925/nested_latitude_15arcsec_2025"
RANGES = (
    ("15S-15N", -15, 15, r"15$^\circ$S-15$^\circ$N"),
    ("30S-30N", -30, 30, r"30$^\circ$S-30$^\circ$N"),
    ("45S-45N", -45, 45, r"45$^\circ$S-45$^\circ$N"),
    ("60S-60N", -60, 60, r"60$^\circ$S-60$^\circ$N"),
    ("75S-75N", -75, 75, r"75$^\circ$S-75$^\circ$N"),
    ("86S-90N", -86, 90, r"86$^\circ$S-90$^\circ$N"),
)
PERIODS = (
    ("1941-1960", 1941, 1960, "#275d8a"),
    ("1961-1980", 1961, 1980, "#2c9588"),
    ("1981-2000", 1981, 2000, "#82949b"),
    ("2001-2025", 2001, 2025, "#c95d4d"),
)


def load_contributions(path):
    data = pd.read_csv(path)
    data = data[data.depth_band == "0-6000m"].copy()
    expected = {(year, name) for year in range(1941, 2026)
                for name, _, _, _ in RANGES}
    observed = list(zip(data.target_year, data.latitude_band))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Expected one 0-6000m central value per range and year, 1941-2025")
    if not (data.base_year == data.target_year - 1).all():
        raise ValueError("Annual comparison years are inconsistent")
    if not np.isfinite(data.global_contribution_mm).all():
        raise ValueError("Non-finite annual contributions")
    data["period"] = ""
    for label, first, last, _ in PERIODS:
        data.loc[data.target_year.between(first, last), "period"] = label
    if (data.period == "").any():
        raise ValueError("Period definition does not cover all years")
    by_period = (data.groupby(["latitude_band", "period"], as_index=False)
                 .global_contribution_mm.sum())
    totals = (data.groupby("latitude_band", as_index=False)
              .global_contribution_mm.sum()
              .rename(columns={"global_contribution_mm": "cumulative_contribution_mm"}))
    closure = (by_period.groupby("latitude_band").global_contribution_mm.sum()
               - totals.set_index("latitude_band").cumulative_contribution_mm)
    if not np.allclose(closure, 0, atol=1e-9):
        raise ValueError("Period contributions do not sum to cumulative totals")
    return by_period, totals


def draw(by_period, totals):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.2, 3.45), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=0.16, right=0.97, bottom=0.23, top=0.78,
                            width_ratios=[1.65, 3.4], wspace=0.18)
    coverage = fig.add_subplot(grid[0, 0])
    contribution = fig.add_subplot(grid[0, 1], sharey=coverage)
    y = np.arange(len(RANGES))

    coverage.axvline(0, color="#b8c0c5", linewidth=0.7, zorder=0)
    for index, (_, south, north, _) in enumerate(RANGES):
        coverage.barh(index, north - south, left=south, height=0.47,
                      color="#a8d2d0", edgecolor="#528886", linewidth=0.7)
        coverage.plot([south, north], [index, index], "|", color="#2a6e6d",
                      markersize=9, markeredgewidth=1.2)
    coverage.set_xlim(-94, 94)
    coverage.set_xticks([-90, -45, 0, 45, 90],
                        [r"90$^\circ$S", r"45$^\circ$S", "0",
                         r"45$^\circ$N", r"90$^\circ$N"])
    coverage.set_yticks(y, [label for _, _, _, label in RANGES])
    coverage.set_ylim(5.6, -0.6)
    coverage.tick_params(axis="y", length=0, pad=6)
    coverage.tick_params(axis="x", length=3)
    coverage.set_title("a  Nested latitude coverage", loc="left",
                       fontweight="bold", fontsize=8, pad=11)
    coverage.spines[["top", "right", "left"]].set_visible(False)

    values = by_period.set_index(["latitude_band", "period"])
    totals = totals.set_index("latitude_band")
    positive_ends = []
    for index, (name, _, _, _) in enumerate(RANGES):
        positive = 0.0
        negative = 0.0
        for period, _, _, color in PERIODS:
            value = float(values.loc[(name, period), "global_contribution_mm"])
            if value >= 0:
                contribution.barh(index, value, left=positive, height=0.51,
                                  color=color, edgecolor="white", linewidth=0.35)
                positive += value
            else:
                contribution.barh(index, -value, left=negative + value,
                                  height=0.51, color=color, edgecolor="white",
                                  linewidth=0.35)
                negative += value
        total = float(totals.loc[name, "cumulative_contribution_mm"])
        positive_ends.append(max(positive, total))
        contribution.text(max(positive, total) + 1.1, index, "{:.1f}".format(total),
                          ha="left", va="center", fontsize=7.5,
                          fontweight="bold", color="#263238")
    contribution.axvline(0, color="#829198", linewidth=0.7)
    contribution.set_xlim(-2.5, max(positive_ends) + 9)
    contribution.set_xticks([0, 20, 40, 60, 80])
    contribution.grid(axis="x", color="#e5e9eb", linewidth=0.7)
    contribution.set_axisbelow(True)
    contribution.set_xlabel("Cumulative contribution to GMSL, 1941-2025 (mm)")
    contribution.set_title("b  Period contributions | 15-arcsec estimate",
                           loc="left", fontweight="bold", fontsize=8, pad=11)
    contribution.tick_params(axis="y", left=False, labelleft=False)
    contribution.spines[["top", "right", "left"]].set_visible(False)
    handles = [Patch(facecolor=color, edgecolor="none", label=period)
               for period, _, _, color in PERIODS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, 0.94),
               ncol=4, frameon=False, fontsize=7.3, handlelength=1.3,
               columnspacing=1.0)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    periods, totals = load_contributions(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = draw(periods, totals)
    stem = args.output_dir / "Fig04_nested_latitude_period_contributions_15arcsec_2025"
    for suffix in ("png", "pdf", "svg", "tiff"):
        fig.savefig(str(stem) + "." + suffix, bbox_inches="tight",
                    facecolor="white", dpi=600 if suffix in ("png", "tiff") else None)
    plt.close(fig)
    periods.to_csv(str(stem) + "_source_periods.csv", index=False)
    totals.to_csv(str(stem) + "_source_totals.csv", index=False)
    metadata = {
        "input": str(args.input.resolve()),
        "years": "1941-2025",
        "depth_band": "0-6000m",
        "statistic": "native 15-arcsec central estimate",
        "quantity": "contribution to global mean sea level, not regional mean",
        "warning": "Nested ranges overlap; do not sum across rows",
        "uncertainty": "not displayed",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved " + str(stem))


if __name__ == "__main__":
    main()
