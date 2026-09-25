#!/usr/bin/env python3
"""Create the latitude-resolved historical thermosteric contribution figure.

Panel a uses the native 15-arc-second central maps aggregated to 10-degree
latitude bands and accumulated from 1941. Panel b decomposes each 1941-2023
latitude-band contribution by depth, with temporally coherent 5th-95th
percentile member intervals shown for the total.
"""

from __future__ import print_function

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
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
DEPTH_LABELS = {
    "0-700m": "0-700 m",
    "700-2000m": "700-2000 m",
    "2000-6000m": "2000-6000 m",
}
DEPTH_COLORS = {
    "0-700m": "#1f5a9d",
    "700-2000m": "#2b9c94",
    "2000-6000m": "#7956a7",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central-file", type=Path, required=True)
    parser.add_argument("--member-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "figures")
    return parser.parse_args()


def latitude_order(frame):
    return (
        frame[["region", "lat_lower_deg", "lat_upper_deg"]]
        .drop_duplicates()
        .sort_values("lat_lower_deg")
        .reset_index(drop=True)
    )


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
    central = central[central["depth_band"].isin(DEPTH_BANDS + ("0-6000m",))].copy()
    central["target_year"] = central["target_year"].astype(int)
    years = np.arange(int(central.target_year.min()), int(central.target_year.max()) + 1)
    if not np.array_equal(np.sort(central.target_year.unique()), years):
        raise ValueError("Central latitude history is not contiguous.")

    members = pd.read_csv(member_file)
    required_members = {"member", "target_year", "region", "depth_band", "global_contribution_mm"}
    missing = required_members.difference(members.columns)
    if missing:
        raise ValueError("Member CSV lacks columns: {}".format(sorted(missing)))
    members = members[(members["region_type"] == "latitude_10deg") & members["depth_band"].isin(DEPTH_BANDS + ("0-6000m",))].copy()
    members["target_year"] = members["target_year"].astype(int)
    order = latitude_order(central)
    return central, members, order, years


def annual_interval(members, order, years):
    subset = members[members.depth_band == "0-6000m"]
    records = []
    for region in order.region:
        regional = subset[subset.region == region]
        table = regional.pivot(index="target_year", columns="member", values="global_contribution_mm")
        for year in years:
            if year not in table.index:
                continue
            values = table.loc[year].dropna().to_numpy(dtype=float)
            if len(values) < 2:
                continue
            records.append({
                "target_year": int(year),
                "region": region,
                "p05_mm": float(np.percentile(values, 5)),
                "p95_mm": float(np.percentile(values, 95)),
                "member_count": int(len(values)),
            })
    return pd.DataFrame(records)


def cumulative_intervals(central, members, order):
    records = []
    source = []
    for band in DEPTH_BANDS + ("0-6000m",):
        central_band = central[(central.depth_band == band) & (central.target_year <= 2023)]
        member_band = members[(members.depth_band == band) & (members.target_year <= 2023)]
        for _, latitude in order.iterrows():
            region = latitude.region
            central_value = float(central_band[central_band.region == region].global_contribution_mm.sum())
            table = member_band[member_band.region == region].pivot(
                index="target_year", columns="member", values="global_contribution_mm"
            )
            values = table.sum(axis=0, min_count=len(table)).dropna().to_numpy(dtype=float)
            if len(values) < 2:
                raise ValueError("Incomplete member path for {} {}".format(region, band))
            records.append({
                "region": region,
                "depth_band": band,
                "central_cumulative_mm": central_value,
                "member_p05_mm": float(np.percentile(values, 5)),
                "member_p95_mm": float(np.percentile(values, 95)),
                "member_count": int(len(values)),
            })
            source.extend(
                {
                    "region": region,
                    "depth_band": band,
                    "member": member,
                    "cumulative_member_mm": float(value),
                }
                for member, value in table.sum(axis=0, min_count=len(table)).dropna().items()
            )
    return pd.DataFrame(records), pd.DataFrame(source)


def add_panel_label(axis, label):
    axis.text(-0.11, 1.035, label, transform=axis.transAxes, fontweight="bold", fontsize=10)


def draw_figure(central, cumulative, order, years):
    lat_labels = order.region.tolist()
    edges = np.arange(-90.0, 91.0, 10.0)
    annual = (
        central[central.depth_band == "0-6000m"]
        .pivot(index="region", columns="target_year", values="global_contribution_mm")
        .reindex(index=lat_labels, columns=years)
    )
    cumulative_time = annual.cumsum(axis=1)
    heat_values = cumulative_time.to_numpy(dtype=float)
    scale = max(float(np.nanpercentile(np.abs(heat_values), 99)), 0.5)

    fig = plt.figure(figsize=(7.2, 4.7))
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.42, 1.0),
        left=0.09, right=0.985, bottom=0.14, top=0.90, wspace=0.38,
    )
    axis_a = fig.add_subplot(grid[0, 0])
    axis_b = fig.add_subplot(grid[0, 1])

    year_edges = np.arange(years.min() - 0.5, years.max() + 1.5, 1.0)
    image = axis_a.pcolormesh(
        year_edges, edges, heat_values, shading="flat", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale), rasterized=True,
    )
    axis_a.set_xlim(years.min() - 0.5, years.max() + 0.5)
    axis_a.set_ylim(-90, 90)
    axis_a.set_xticks(np.arange(1940, 2030, 20))
    axis_a.set_yticks(np.arange(-80, 100, 20))
    axis_a.set_xlabel("Year")
    axis_a.set_ylabel("Latitude (degrees)")
    colorbar = fig.colorbar(image, ax=axis_a, orientation="horizontal", fraction=0.050, pad=0.12, aspect=30)
    colorbar.set_label("Cumulative contribution to global mean sea level (mm)", fontsize=6.1)
    colorbar.ax.tick_params(labelsize=5.8, width=0.55, length=2)
    add_panel_label(axis_a, "a")

    y = np.arange(len(lat_labels))
    left = np.zeros(len(lat_labels), dtype=float)
    for band in DEPTH_BANDS:
        subset = cumulative[cumulative.depth_band == band].set_index("region").reindex(lat_labels)
        values = subset.central_cumulative_mm.to_numpy(dtype=float)
        axis_b.barh(y, values, left=left, height=0.64, color=DEPTH_COLORS[band], linewidth=0, zorder=2)
        left += values
    total = cumulative[cumulative.depth_band == "0-6000m"].set_index("region").reindex(lat_labels)
    central_total = total.central_cumulative_mm.to_numpy(dtype=float)
    lower = total.member_p05_mm.to_numpy(dtype=float)
    upper = total.member_p95_mm.to_numpy(dtype=float)
    axis_b.errorbar(
        central_total, y, xerr=np.vstack((central_total - lower, upper - central_total)),
        fmt="o", markersize=2.8, color="#202020", ecolor="#202020",
        elinewidth=0.65, capsize=1.4, capthick=0.65, zorder=4,
    )
    axis_b.axvline(0.0, color="0.25", linewidth=0.65, zorder=1)
    axis_b.grid(axis="x", color="0.90", linewidth=0.5, zorder=0)
    axis_b.set_axisbelow(True)
    axis_b.set_yticks(y)
    axis_b.set_yticklabels(lat_labels, fontsize=5.6)
    axis_b.set_ylim(-0.75, len(lat_labels) - 0.25)
    axis_b.set_xlabel("1941-2023 cumulative contribution to GMSL (mm)")
    axis_b.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=DEPTH_COLORS["0-700m"], edgecolor="none", label="0-700 m"),
            plt.Rectangle((0, 0), 1, 1, facecolor=DEPTH_COLORS["700-2000m"], edgecolor="none", label="700-2000 m"),
            plt.Rectangle((0, 0), 1, 1, facecolor=DEPTH_COLORS["2000-6000m"], edgecolor="none", label="2000-6000 m"),
            Line2D([0], [0], color="#202020", marker="o", markersize=3, linewidth=0.65, label="5-95% interval"),
        ],
        loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2,
        frameon=False, fontsize=5.6, handlelength=1.1, columnspacing=0.75,
        handletextpad=0.35, borderaxespad=0.0,
    )
    add_panel_label(axis_b, "b")
    return fig, cumulative_time


def save_figure(fig, output_dir):
    stem = "Fig04_latitudinal_thermosteric_partition"
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
    cumulative, member_cumulative = cumulative_intervals(central, members, order)
    fig, cumulative_time = draw_figure(central, cumulative, order, years)
    stem = save_figure(fig, output_dir)
    cumulative_time.reset_index().to_csv(output_dir / (stem + "_source_cumulative_time.csv"), index=False)
    cumulative.to_csv(output_dir / (stem + "_source_cumulative.csv"), index=False)
    member_cumulative.to_csv(output_dir / (stem + "_source_member_cumulative.csv"), index=False)
    print("Saved {} in {}".format(stem, output_dir))


if __name__ == "__main__":
    main()
