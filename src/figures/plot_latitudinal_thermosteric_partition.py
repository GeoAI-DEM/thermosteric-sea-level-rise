#!/usr/bin/env python3
"""Create the latitude-resolved historical thermosteric contribution figure.

Panel a uses the native 15-arc-second central annual maps aggregated to 10-degree
latitude bands. Hatching identifies annual IAPv4-LE member intervals spanning
zero; it is intentionally absent after 2023 because no member product exists.
Panels b-d show 1941-2023 cumulative 15-arc-second central contributions and
the temporally coherent 5th-95th percentile member intervals for each depth band.
"""

from __future__ import print_function

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
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
    for band in DEPTH_BANDS:
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


def draw_figure(central, annual_uncertainty, cumulative, order, years):
    lat_labels = order.region.tolist()
    latitude_centres = ((order.lat_lower_deg + order.lat_upper_deg) / 2.0).to_numpy(dtype=float)
    edges = np.arange(-90.0, 91.0, 10.0)
    band_centres = (edges[:-1] + edges[1:]) / 2.0
    heat = (
        central[central.depth_band == "0-6000m"]
        .pivot(index="region", columns="target_year", values="global_contribution_mm")
        .reindex(index=lat_labels, columns=years)
    )
    smoothed = heat.T.rolling(window=5, center=True, min_periods=3).mean().T
    heat_values = smoothed.to_numpy(dtype=float)
    scale = max(float(np.nanpercentile(np.abs(heat_values), 95)), 0.05)
    uncertainty_flag = np.zeros_like(heat_values, dtype=bool)
    if not annual_uncertainty.empty:
        for _, row in annual_uncertainty.iterrows():
            year_index = int(row.target_year) - int(years.min())
            region_index = lat_labels.index(row.region)
            uncertainty_flag[region_index, year_index] = row.p05_mm <= 0.0 <= row.p95_mm

    fig = plt.figure(figsize=(7.2, 5.0))
    grid = fig.add_gridspec(
        3, 2, width_ratios=(1.55, 1.0),
        left=0.09, right=0.985, bottom=0.11, top=0.94,
        wspace=0.43, hspace=0.23,
    )
    axis_a = fig.add_subplot(grid[:, 0])
    axes_depth = [fig.add_subplot(grid[index, 1]) for index in range(3)]

    year_edges = np.arange(years.min() - 0.5, years.max() + 1.5, 1.0)
    image = axis_a.pcolormesh(
        year_edges, edges, heat_values, shading="flat", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale), rasterized=True,
    )
    if uncertainty_flag.any():
        overlay = np.ma.masked_where(~uncertainty_flag, np.ones_like(heat_values))
        axis_a.pcolormesh(
            year_edges, edges, overlay, shading="flat",
            cmap=ListedColormap(["#7d7d7d"]), vmin=0.0, vmax=1.0,
            alpha=0.16, rasterized=True,
        )
    axis_a.axvline(2023.5, color="0.2", linewidth=0.75, linestyle=(0, (2, 2)))
    axis_a.set_xlim(years.min() - 0.5, years.max() + 0.5)
    axis_a.set_ylim(-90, 90)
    axis_a.set_xticks(np.arange(1940, 2030, 20))
    axis_a.set_yticks(np.arange(-80, 100, 20))
    axis_a.set_xlabel("Year")
    axis_a.set_ylabel("Latitude (degrees)")
    axis_a.text(2023.0, 84.0, "IAPv4-LE ends", ha="right", va="center", fontsize=5.8, color="0.20")
    axis_a.legend(
        handles=[Patch(facecolor="#7d7d7d", alpha=0.30, edgecolor="none", label="5-95% member interval spans zero")],
        loc="upper left", bbox_to_anchor=(0.01, 0.985), frameon=False,
        fontsize=5.5, handlelength=1.0, handletextpad=0.35, borderaxespad=0.0,
    )
    colorbar = fig.colorbar(image, ax=axis_a, orientation="horizontal", fraction=0.050, pad=0.11, aspect=30)
    colorbar.set_label("Five-year mean annual contribution to global mean sea level (mm yr$^{-1}$)", fontsize=6.1)
    colorbar.ax.tick_params(labelsize=5.8, width=0.55, length=2)
    add_panel_label(axis_a, "a")

    for axis, band, label in zip(axes_depth, DEPTH_BANDS, ("b", "c", "d")):
        subset = cumulative[cumulative.depth_band == band].set_index("region").reindex(lat_labels).reset_index()
        values = subset.central_cumulative_mm.to_numpy(dtype=float)
        lower = subset.member_p05_mm.to_numpy(dtype=float)
        upper = subset.member_p95_mm.to_numpy(dtype=float)
        y = np.arange(len(lat_labels))
        axis.hlines(y, lower, upper, color=DEPTH_COLORS[band], linewidth=1.1, alpha=0.62, zorder=1)
        axis.scatter(values, y, s=13, color=DEPTH_COLORS[band], edgecolor="white", linewidth=0.35, zorder=2)
        axis.axvline(0.0, color="0.35", linewidth=0.6, zorder=0)
        axis.grid(axis="x", color="0.90", linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)
        axis.set_yticks(y)
        axis.set_yticklabels(lat_labels, fontsize=5.4)
        axis.invert_yaxis()
        axis.set_title(DEPTH_LABELS[band], loc="left", fontsize=6.8, fontweight="bold", pad=2)
        if axis is axes_depth[-1]:
            axis.set_xlabel("Cumulative contribution to GMSL (mm)")
        else:
            axis.tick_params(labelbottom=False)
        add_panel_label(axis, label)

    return fig, smoothed, uncertainty_flag


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
    annual_uncertainty = annual_interval(members, order, years)
    cumulative, member_cumulative = cumulative_intervals(central, members, order)
    fig, smoothed, uncertainty_flag = draw_figure(central, annual_uncertainty, cumulative, order, years)
    stem = save_figure(fig, output_dir)
    smoothed.reset_index().to_csv(output_dir / (stem + "_source_5yr_mean.csv"), index=False)
    annual_uncertainty.to_csv(output_dir / (stem + "_source_annual_member_interval.csv"), index=False)
    cumulative.to_csv(output_dir / (stem + "_source_cumulative.csv"), index=False)
    member_cumulative.to_csv(output_dir / (stem + "_source_member_cumulative.csv"), index=False)
    pd.DataFrame({
        "region": np.repeat(order.region.to_numpy(), len(years)),
        "target_year": np.tile(years, len(order)),
        "member_interval_spans_zero": uncertainty_flag.ravel(),
    }).to_csv(output_dir / (stem + "_source_uncertainty_hatching.csv"), index=False)
    print("Saved {} in {}".format(stem, output_dir))


if __name__ == "__main__":
    main()
