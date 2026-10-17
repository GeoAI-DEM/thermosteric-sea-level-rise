#!/usr/bin/env python3
"""Editable ocean thermosteric research schematic for the Introduction."""

from manuscript_paths import workspace_root, input_path
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.shapereader as shp


ROOT = (workspace_root() / 'figures')
OUTPUT = ROOT / "introduction_schematic_20261003"
INK = "#283b45"
MUTED = "#61737c"
BLUE = "#356e9b"
TEAL = "#3f9c97"
PURPLE = "#8874ad"
WARM = "#cf826a"
GREY = "#dadfdf"
COLORS = (BLUE, TEAL, PURPLE)


def label(ax, x, y, value, size=7.4, color=INK, weight="normal", **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   va="center", **kwargs)


def arrow(ax, start, end, color=MUTED, width=1.0, style="-|>", **kwargs):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=9,
                            linewidth=width, color=color, **kwargs)
    ax.add_patch(patch)
    return patch


def globe(fig):
    ax = fig.add_axes([0.055, 0.670, 0.52, 0.23], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("#e8f1f4")
    land_path = shp.natural_earth("110m", "physical", "land")
    ax.add_geometries(shp.Reader(land_path).geometries(), ccrs.PlateCarree(),
                      facecolor="#f0f1ef", edgecolor="#829398", linewidth=0.32)
    ax.gridlines(xlocs=np.arange(-180, 181, 60), ylocs=[-60, -30, 0, 30, 60],
                 linewidth=0.35, color="#a9bfc5", alpha=0.7)
    ax.spines["geo"].set_edgecolor("#b8c6cb")
    ax.spines["geo"].set_linewidth(0.6)
    return str(land_path)


def ocean_section(ax):
    x = np.linspace(0.8, 8.0, 400)
    # Illustrative continental shelf, slope, abyssal floor and seamount.
    floor = np.interp(x, [0.8, 1.12, 1.65, 2.02, 2.7, 3.6, 4.4, 5.35,
                         6.0, 6.35, 6.8, 7.2, 7.65, 8.0],
                      [4.62, 4.56, 4.38, 3.77, 1.26, 0.94, 1.09, 1.0,
                       1.36, 2.1, 1.25, 1.02, 1.14, 1.46])
    surface = 5.08
    boundaries = (4.38, 3.39, 0.83)
    tops = (surface, boundaries[0], boundaries[1])
    tints = ("#c7dce8", "#c6e1de", "#ded6e8")
    for top, bottom, color in zip(tops, boundaries, tints):
        low = np.maximum(floor, bottom)
        ax.fill_between(x, low, top, where=floor < top, color=color,
                        interpolate=True, linewidth=0)
    ax.fill_between(x, 0.72, floor, color=GREY, linewidth=0)
    ax.plot(x, floor, color="#7b9096", linewidth=1.0)
    ax.plot([0.8, 8.0], [surface, surface], color=BLUE, linewidth=1.1)
    for y in boundaries[:2]:
        ax.hlines(y, 0.8, 8.0, color="#ffffff", linewidth=0.8)

    ax.plot([0.60, 0.60], [0.83, surface], color="#a8b6bc", linewidth=0.7)
    for y, text in zip((surface, 4.38, 3.39, 0.83), ("0", "700", "2000", "6000")):
        ax.plot([0.54, 0.66], [y, y], color="#a8b6bc", linewidth=0.7)
        label(ax, 0.45, y, text, size=6.5, ha="right", color=MUTED)
    label(ax, -0.55, 2.9, "Depth (m)", size=7, rotation=90)

    for y, text, color in zip((4.73, 3.86, 2.58),
                              ("0-700 m", "700-2000 m", "2000-6000 m"), COLORS):
        label(ax, 7.76, y, text, size=7.5, weight="bold", color=color, ha="right")

    # A sample wet column shows partial-cell volume at the seabed.
    cx0, cx1 = 3.28, 3.84
    column_floor = np.interp(np.array([cx0, cx1]), x, floor)
    polygon = Polygon([(cx0, surface), (cx1, surface), (cx1, column_floor[1]),
                       (cx0, column_floor[0])], facecolor="white", alpha=0.44,
                       edgecolor=BLUE, linewidth=0.9)
    ax.add_patch(polygon)
    for y in np.linspace(1.17, 4.88, 13):
        ax.plot([cx0, cx1], [y, y], color=BLUE, alpha=0.28, linewidth=0.35)
    label(ax, 3.55, 0.40, "15\N{DOUBLE PRIME} wet-cell volumes", size=7, ha="center")
    arrow(ax, (3.55, 0.62), (3.55, 1.14), color=BLUE, width=0.7)
    label(ax, 6.20, 0.42, "GEBCO bathymetry", size=7, color=MUTED, ha="center")

    label(ax, 3.9, 5.85, "Ocean heat uptake", color=WARM, size=8, weight="bold", ha="center")
    for xh in (2.1, 3.5, 4.9, 6.3):
        arrow(ax, (xh, 5.6), (xh, 5.14), color=WARM, width=1.4)
    label(ax, 4.4, -0.10, "Ocean profile and depths are schematic", size=6,
          ha="center", color=MUTED)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(-0.3, 6.8)
    ax.axis("off")


def response(ax):
    label(ax, 0.03, 0.95, "b", size=10, weight="bold")
    label(ax, 0.10, 0.95, "Thermosteric response", size=9, weight="bold")
    # Same-mass water parcels illustrate warming and volume increase only.
    for x, height, tint, edge in ((0.14, 0.16, "#d7e6ed", BLUE),
                                 (0.57, 0.205, "#edd6cc", WARM)):
        ax.add_patch(Rectangle((x, 0.68), 0.19, height, facecolor=tint,
                                edgecolor=edge, linewidth=0.9))
    arrow(ax, (0.37, 0.77), (0.52, 0.77), color=WARM, width=1.2)
    label(ax, 0.44, 0.855, "Warming", color=WARM, size=7, ha="center")
    label(ax, 0.23, 0.635, "Reference", size=6.5, ha="center")
    label(ax, 0.66, 0.635, "Expanded", size=6.5, ha="center")
    label(ax, 0.47, 0.583, "Conceptual response at fixed mass", size=6, color=MUTED, ha="center")
    label(ax, 0.47, 0.503, "TEOS-10", size=8.5, weight="bold", ha="center")
    label(ax, 0.47, 0.422,
          r"$\Delta V_{i,k}=V_{i,k}\,\overline{\alpha}_{i,k}\,\Delta CT_{i,k}$",
          size=10.4, ha="center")
    label(ax, 0.47, 0.354, "Wet-cell volume  \N{MULTIPLICATION SIGN}  expansion coefficient  \N{MULTIPLICATION SIGN}  temperature change",
          size=6.1, ha="center", color=MUTED)
    arrow(ax, (0.47, 0.314), (0.47, 0.238), color=BLUE, width=1.1)
    label(ax, 0.47, 0.179, "Global mean sea-level contribution", size=8.4,
          weight="bold", ha="center", color=BLUE)
    label(ax, 0.47, 0.083, r"$\Delta\eta_{\rm global}=\frac{\sum_{i,k}\Delta V_{i,k}}{A_{\rm ocean}}$",
          size=12, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def bottom(fig):
    ax = fig.add_axes([0.055, 0.042, 0.89, 0.23])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.98, 0, 1, color="#ccd6d9", linewidth=0.7)
    label(ax, 0.00, 0.85, "c", size=10, weight="bold")
    label(ax, 0.035, 0.85, "Where, how deep, and how certain?", size=9, weight="bold")

    # Three depth swatches tie regional partitions to the physical section.
    for k, color in enumerate(COLORS):
        ax.add_patch(Rectangle((0.006, 0.53-k*0.075), 0.045, 0.055,
                              facecolor=color, edgecolor="none", alpha=0.85))
    label(ax, 0.067, 0.56, "Depth, ocean basin", size=7.6, weight="bold")
    label(ax, 0.067, 0.41, "and latitude partitions", size=7.6, weight="bold")
    label(ax, 0.006, 0.16, "15\N{DOUBLE PRIME} central reconstruction", size=6.5, color=MUTED)

    ax.vlines(0.31, 0.08, 0.69, color="#e0e6e8", linewidth=0.7)
    label(ax, 0.35, 0.59, "1\N{DEGREE SIGN} ensemble uncertainty", size=7.6, weight="bold", color=TEAL)
    label(ax, 0.35, 0.43, "Aggregated DEM ocean volumes", size=6.8)
    label(ax, 0.35, 0.29, "+ IAPv4-LE members, 1940-2023", size=6.8)
    label(ax, 0.35, 0.15, "Central annual salinity held fixed", size=6.3, color=MUTED)

    ax.vlines(0.70, 0.08, 0.69, color="#e0e6e8", linewidth=0.7)
    label(ax, 0.74, 0.59, "Near-term outlook", size=7.6, weight="bold", color=WARM)
    label(ax, 0.74, 0.41, "Annual series", size=6.8)
    arrow(ax, (0.745, 0.265), (0.98, 0.265), color=WARM, width=0.9)
    label(ax, 0.98, 0.41, "2030", size=6.8, ha="right")
    label(ax, 0.74, 0.10, "2026-2030 statistical forecast", size=6.3, color=MUTED)
    return ax


def draw():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.5, 6.4), facecolor="white")
    header = fig.add_axes([0, 0, 1, 1], zorder=0)
    header.axis("off")
    label(header, 0.055, 0.957, "a", size=10, weight="bold")
    label(header, 0.080, 0.957, "Global ocean reconstruction", size=9, weight="bold")
    label(header, 0.575, 0.920, "1940-2025  |  0-6000 m", size=7, color=MUTED, ha="right")
    land_path = globe(fig)
    label(header, 0.315, 0.642, "IAP temperature + salinity: 1\N{DEGREE SIGN} fields", size=7.4, ha="center")
    label(header, 0.315, 0.620, "Interpolated to the 15\N{DOUBLE PRIME} calculation grid", size=6.7,
          ha="center", color=MUTED)
    section = fig.add_axes([0.055, 0.285, 0.52, 0.35])
    ocean_section(section)
    right = fig.add_axes([0.645, 0.31, 0.31, 0.665])
    response(right)
    # The broad arrow represents integration across cells and depth layers.
    arrow(header, (0.584, 0.454), (0.632, 0.454), color=BLUE, width=1.5)
    bottom(fig)
    return fig, land_path


CAPTION = """Figure. Conceptual framework for reconstructing ocean thermal expansion.
(a) Global annual temperature and salinity fields are interpolated to a 15-arcsec
calculation grid. GEBCO bathymetry defines the water volume in each depth layer,
including partially wet cells near the seabed. The three illustrated depth
classes are 0-700 m, 700-2000 m, and 2000-6000 m. (b) The schematic water parcels
illustrate expansion during warming at fixed mass. The implemented estimate is
Delta V = V * mean(alpha) * Delta CT, where alpha is the two-year mean TEOS-10
thermal expansion coefficient and CT is Conservative Temperature. The global
mean thermosteric sea-level contribution is total volume change divided by the
fixed ocean area used in the calculation (361.9 million km2). (c) The central
reconstruction is partitioned by depth, ocean basin, and latitude. Temperature
ensemble spread is propagated separately on 1-degree, bathymetry-aggregated
ocean volumes using the central salinity field for each year. Annual series
provide the input for the statistical forecast for 2026-2030.

Image-integrity note: The map uses Natural Earth land geometry. All other
geometry is a conceptual vector illustration; parcel expansion, seabed relief,
depth spacing and arrows are not quantitative or to scale. No geographical
temperature pattern, local sea-level map, ensemble interval, or future field is
fabricated. Fine bathymetric resolution does not imply independent temperature
and salinity observations at 15 arcsec. Annual ensemble data cover 1940-2023;
uncertainty extensions for 2024-2025, where used in other figures, are proxies.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, land_path = draw()
    stem = args.output_dir / "Fig_Introduction_ocean_thermal_expansion_framework"
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(str(stem)+"."+suffix, facecolor="white", dpi=600,
                    bbox_inches="tight", pad_inches=0.12, **options)
    plt.close(fig)
    (args.output_dir / "figure_caption.txt").write_text(CAPTION, encoding="utf-8")
    metadata = {
        "figure_role": "Introduction: conceptual research framework",
        "claim": "Bathymetry-resolved thermal volume change links ocean temperature to global sea-level contribution and regional partitions.",
        "backend": "Python matplotlib + cartopy",
        "geometry_source": land_path,
        "drawing_type": "schematic-led composite; conceptual geometry; no numerical result series",
        "main_formula": "V * 0.5 * (alpha_base + alpha_target) * (CT_target - CT_base)",
        "ocean_area_m2": 361.9e12,
        "main_calculation_grid": "15 arcsec; T/S interpolated from 1-degree fields",
        "uncertainty_grid": "1 degree; DEM-aggregated ocean volume",
        "uncertainty_scope": "IAPv4-LE temperature ensemble; central annual salinity held fixed",
        "illustration_limit": "Seabed, depth spacing, heat arrows, and parcel expansion are not to scale.",
        "palette": {"upper": BLUE, "intermediate": TEAL, "deep": PURPLE, "warming": WARM},
        "export": "600-dpi PNG/TIFF; editable-text PDF/SVG",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved " + str(stem))


if __name__ == "__main__":
    main()
