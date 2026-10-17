#!/usr/bin/env python3
"""Create a publication-style latitude/time and depth-partition figure."""

from __future__ import print_function
from manuscript_paths import workspace_root, input_path

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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


DEPTH_BANDS = ("0-700m", "700-2000m", "2000-6000m")
PROFILE_BANDS = DEPTH_BANDS + ("0-6000m",)
DEPTH_LABELS = {
    "0-700m": "0-700 m",
    "700-2000m": "700-2000 m",
    "2000-6000m": "2000-6000 m",
    "0-6000m": "0-6000 m",
}
DEPTH_COLORS = {
    "0-700m": "#2764a5",
    "700-2000m": "#23998e",
    "2000-6000m": "#8054a6",
    "0-6000m": "#343a40",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central-file", type=Path, required=True)
    parser.add_argument("--member-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "figures")
    return parser.parse_args()


def load_inputs(central_file, member_file):
    central = pd.read_csv(central_file)
    required_central = {
        "target_year", "latitude_band", "lat_lower_deg", "lat_upper_deg",
        "depth_band", "global_contribution_mm",
    }
    missing = required_central.difference(central.columns)
    if missing:
        raise ValueError("Central CSV lacks columns: {}".format(sorted(missing)))
    central = central.rename(columns={"latitude_band": "region"})
    central = central[central.depth_band.isin(PROFILE_BANDS)].copy()
    central["target_year"] = central.target_year.astype(int)
    years = np.arange(central.target_year.min(), central.target_year.max() + 1)
    if not np.array_equal(np.sort(central.target_year.unique()), years):
        raise ValueError("Central latitude history is not contiguous.")

    members = pd.read_csv(member_file)
    required_members = {"member", "target_year", "region_type", "region", "depth_band", "global_contribution_mm"}
    missing = required_members.difference(members.columns)
    if missing:
        raise ValueError("Member CSV lacks columns: {}".format(sorted(missing)))
    members = members[
        (members.region_type == "latitude_10deg")
        & members.depth_band.isin(PROFILE_BANDS)
    ].copy()
    members["target_year"] = members.target_year.astype(int)

    order = (
        central[["region", "lat_lower_deg", "lat_upper_deg"]]
        .drop_duplicates()
        .sort_values("lat_lower_deg")
        .reset_index(drop=True)
    )
    if len(order) != 18:
        raise ValueError("Expected 18 ten-degree latitude bands; found {}.".format(len(order)))
    return central, members, order, years


def cumulative_profiles(central, members, order):
    end_year = min(2023, int(central.target_year.max()))
    expected_years = np.arange(int(central.target_year.min()), end_year + 1)
    records = []
    member_source = []

    for band in PROFILE_BANDS:
        central_band = central[
            (central.depth_band == band) & (central.target_year <= end_year)
        ]
        member_band = members[
            (members.depth_band == band) & (members.target_year <= end_year)
        ]
        for _, latitude in order.iterrows():
            region = latitude.region
            central_value = float(
                central_band.loc[central_band.region == region, "global_contribution_mm"].sum()
            )
            table = member_band[member_band.region == region].pivot(
                index="target_year", columns="member", values="global_contribution_mm"
            ).reindex(expected_years)
            totals = table.sum(axis=0, min_count=len(expected_years)).dropna()
            if len(totals) < 2:
                raise ValueError("Incomplete member paths for {} {}.".format(region, band))
            records.append({
                "region": region,
                "depth_band": band,
                "central_cumulative_mm": central_value,
                "member_p05_mm": float(np.percentile(totals.to_numpy(dtype=float), 5)),
                "member_p95_mm": float(np.percentile(totals.to_numpy(dtype=float), 95)),
                "member_count": int(len(totals)),
            })
            member_source.extend(
                {
                    "region": region,
                    "depth_band": band,
                    "member": member,
                    "cumulative_member_mm": float(value),
                }
                for member, value in totals.items()
            )
    return pd.DataFrame(records), pd.DataFrame(member_source), end_year


def latitude_tick(region, lower, upper):
    if upper <= 0:
        return "{}-{} S".format(abs(int(upper)), abs(int(lower)))
    if lower >= 0:
        return "{}-{} N".format(int(lower), int(upper))
    return "0-10 {}".format("S" if region.endswith("0S") else "N")


def draw_figure(central, cumulative, order, years, end_year):
    lat_labels = order.region.tolist()
    annual = (
        central[central.depth_band == "0-6000m"]
        .pivot(index="region", columns="target_year", values="global_contribution_mm")
        .reindex(index=lat_labels, columns=years)
    )
    if annual.isna().any().any():
        raise ValueError("Annual latitude contribution matrix contains missing values.")

    values = annual.to_numpy(dtype=float)
    scale = max(float(np.nanpercentile(np.abs(values), 98)), 0.05)
    lat_edges = np.r_[order.lat_lower_deg.to_numpy(dtype=float), float(order.lat_upper_deg.iloc[-1])]
    time_edges = np.r_[years.astype(float) - 0.5, float(years[-1]) + 0.5]

    fig = plt.figure(figsize=(7.25, 6.0))
    grid = fig.add_gridspec(
        2, 4, left=0.15, right=0.985, bottom=0.105, top=0.955,
        height_ratios=(0.88, 1.12), hspace=0.2, wspace=0.25,
    )
    axis_a = fig.add_subplot(grid[0, :])
    axes_b = [fig.add_subplot(grid[1, i], sharey=None if i == 0 else None) for i in range(4)]

    image = axis_a.pcolormesh(
        time_edges, lat_edges, values, shading="flat", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale), rasterized=True,
    )
    axis_a.set_xlim(time_edges[0], time_edges[-1])
    axis_a.set_ylim(-90, 90)
    axis_a.set_xticks([1940, 1960, 1980, 2000, 2020])
    axis_a.set_yticks(np.arange(-80, 100, 20))
    axis_a.set_xlabel("Year")
    axis_a.set_ylabel("Latitude")
    axis_a.tick_params(labelsize=6.2, length=2.5, pad=2)
    axis_a.grid(axis="y", color="white", linewidth=0.45, alpha=0.65)
    colorbar = fig.colorbar(image, ax=axis_a, fraction=0.025, pad=0.012, aspect=25)
    colorbar.set_label("Annual contribution to GMSL (mm yr$^{-1}$)", fontsize=6.2, labelpad=5)
    colorbar.ax.tick_params(labelsize=5.8, width=0.55, length=2)
    axis_a.text(-0.09, 1.03, "a", transform=axis_a.transAxes, fontweight="bold", fontsize=10)

    profile_order = order.sort_values("lat_lower_deg", ascending=False).reset_index(drop=True)
    regions = profile_order.region.tolist()
    y = np.arange(len(regions))
    for index, (axis, band) in enumerate(zip(axes_b, PROFILE_BANDS)):
        subset = cumulative[cumulative.depth_band == band].set_index("region").reindex(regions)
        central_values = subset.central_cumulative_mm.to_numpy(dtype=float)
        lower = subset.member_p05_mm.to_numpy(dtype=float)
        upper = subset.member_p95_mm.to_numpy(dtype=float)
        color = DEPTH_COLORS[band]

        axis.hlines(y - 0.13, lower, upper, color="#818a91", linewidth=1.35, zorder=2)
        axis.scatter(central_values, y + 0.13, color=color, edgecolor="white", linewidth=0.35,
                     s=14, zorder=3)
        axis.axvline(0, color="#70777d", linewidth=0.55, zorder=0)
        axis.grid(axis="y", color="#e6e8e9", linewidth=0.45, zorder=0)
        axis.set_axisbelow(True)
        axis.set_title(DEPTH_LABELS[band], fontsize=6.8, fontweight="bold", pad=4)
        axis.set_ylim(len(regions) - 0.5, -0.5)
        axis.xaxis.set_major_locator(MaxNLocator(3))
        axis.tick_params(axis="x", labelsize=5.5, length=2, pad=2)
        axis.tick_params(axis="y", length=0, pad=2, labelsize=5.2)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#687078")
        axis.grid(axis="x", color="#eceeef", linewidth=0.5, zorder=0)
        if index == 0:
            ticks = [latitude_tick(row.region, row.lat_lower_deg, row.lat_upper_deg)
                     for row in profile_order.itertuples(index=False)]
            axis.set_yticks(y)
            axis.set_yticklabels(ticks)
        else:
            axis.tick_params(axis="y", labelleft=False)
        xmax = max(float(np.nanmax(central_values)), float(np.nanmax(upper)), 0.1)
        xmin = min(float(np.nanmin(lower)), 0.0)
        pad = max((xmax - xmin) * 0.08, 0.04)
        axis.set_xlim(xmin - pad, xmax + pad)

    axes_b[0].set_ylabel("Latitude band", labelpad=5)
    axes_b[0].text(-0.48, 1.12, "b", transform=axes_b[0].transAxes,
                   fontweight="bold", fontsize=10)
    fig.text(0.58, 0.045, "Cumulative contribution to global mean sea level, 1941-{} (mm)".format(end_year),
             ha="center", fontsize=6.5)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#343a40",
               markeredgecolor="white", markersize=4.0, label="15-arcsec central estimate"),
        Line2D([0, 1], [0, 0], color="#737b82", linewidth=1.5,
               label="1-degree member range, 5th-95th (n=7)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.58, 0.002),
               ncol=2, frameon=False, fontsize=5.7, handlelength=1.8,
               columnspacing=1.1, handletextpad=0.45)
    return fig, annual


def save_figure(fig, output_dir):
    stem = "Fig04_latitudinal_thermosteric_partition_redesign"
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix in ("png", "tiff"):
            options["dpi"] = 600
        fig.savefig(output_dir / (stem + "." + suffix), **options)
    plt.close(fig)
    return stem


def main():
    args = parse_args()
    central_file = args.central_file.resolve()
    member_file = args.member_file.resolve()
    output_dir = args.output_dir.resolve()
    if not central_file.is_file() or not member_file.is_file():
        raise FileNotFoundError("Central or member latitude input file is missing.")
    output_dir.mkdir(parents=True, exist_ok=True)
    central, members, order, years = load_inputs(central_file, member_file)
    cumulative, member_source, end_year = cumulative_profiles(central, members, order)
    fig, annual = draw_figure(central, cumulative, order, years, end_year)
    stem = save_figure(fig, output_dir)
    cumulative.to_csv(output_dir / (stem + "_source_cumulative.csv"), index=False)
    member_source.to_csv(output_dir / (stem + "_source_member_cumulative.csv"), index=False)
    print("Saved {} in {}".format(stem, output_dir))


if __name__ == "__main__":
    main()
