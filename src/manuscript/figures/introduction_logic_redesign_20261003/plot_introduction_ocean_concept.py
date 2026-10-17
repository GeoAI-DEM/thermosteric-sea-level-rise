#!/usr/bin/env python3
"""A single conceptual ocean-warming illustration for the Introduction."""

from manuscript_paths import workspace_root, input_path
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from netCDF4 import Dataset
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.shapereader as shp


OUTPUT = (workspace_root() / 'figures/introduction_logic_redesign_20261003')
ANNUAL_INPUTS = input_path('THERMOSTERIC_ANNUAL_DIR', 'inputs/annual_averages')
INK = "#2e414b"
MUTED = "#6a7e85"
BLUE = "#477c98"
TEAL = "#4b9691"
PURPLE = "#8a78a6"
WARM = "#c77f66"


def text(ax, x, y, value, size=8, color=INK, weight="normal", **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   va="center", **kwargs)


def arrow(ax, start, end, color=WARM, width=1.0, style="-|>", scale=9, **kwargs):
    item = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale,
                           linewidth=width, color=color, **kwargs)
    ax.add_patch(item)
    return item


def flow_arrow(ax, start, end, color=BLUE, scale=17):
    return arrow(ax, start, end, color=color, width=0,
                 style="simple,tail_width=0.22,head_width=0.65,head_length=0.55",
                 scale=scale, shrinkA=0, shrinkB=0)


def calculation_grid_map(fig):
    ax = fig.add_axes([0.68, 0.724, 0.25, 0.176], projection=ccrs.Robinson())
    ax.set_global()
    ax.set_facecolor("#e8f1f4")
    source = shp.natural_earth("110m", "physical", "land")
    ax.add_geometries(shp.Reader(source).geometries(), ccrs.PlateCarree(),
                      facecolor="#f1f2ef", edgecolor="#8a9ba0", linewidth=0.2)
    ax.gridlines(xlocs=np.arange(-180, 181, 60), ylocs=[-60, -30, 0, 30, 60],
                 linewidth=0.25, color="#aec3c9", alpha=0.75)
    ax.spines["geo"].set_edgecolor("#a4b6bd")
    ax.spines["geo"].set_linewidth(0.4)


def input_symbols(fig):
    palettes = (
        LinearSegmentedColormap.from_list("ocean_temperature",
            ["#396b89", "#a5c6d1", "#f0e1c4", "#d59a73", "#ac584c"]),
        LinearSegmentedColormap.from_list("ocean_salinity",
            ["#e0eeeb", "#a2ccc5", "#519b99", "#2d6877"]))
    fields = (("Annual_Temp_2025.nc", "temp", 0.075, -2, 32),
              ("Annual_Salinity_2025.nc", "salinity", 0.365, 30, 39))
    source = shp.natural_earth("110m", "physical", "land")
    land = list(shp.Reader(source).geometries())
    records = []
    for (filename, variable, left, low, high), cmap in zip(fields, palettes):
        with Dataset(str(ANNUAL_INPUTS / filename), "r") as ds:
            var = ds.variables[variable]
            index = tuple(0 if d == "depth_std" else slice(None) for d in var.dimensions)
            surface = np.ma.filled(var[index], np.nan).astype(float)
            lat = np.asarray(ds.variables["lat"][:])
            lon = (np.asarray(ds.variables["lon"][:]) + 180) % 360 - 180
            order = np.argsort(lon)
            lon, surface = lon[order], surface[:, order]
            if surface.shape != (len(lat), len(lon)) or not np.isfinite(surface).any():
                raise ValueError("Invalid surface field: " + filename)
            records.append({"file": str(ANNUAL_INPUTS / filename), "variable": variable,
                            "units": var.units, "year": 2025,
                            "depth_m": float(ds.variables["depth_std"][0]),
                            "color_limits": [low, high], "role": "qualitative input thumbnail"})
        ax = fig.add_axes([left, 0.724, 0.25, 0.176], projection=ccrs.Robinson())
        ax.set_global()
        ax.set_facecolor("#edf2f2")
        ax.pcolormesh(lon, lat, np.ma.masked_invalid(surface), cmap=cmap,
                      vmin=low, vmax=high, shading="nearest",
                      transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_geometries(land, ccrs.PlateCarree(), facecolor="#f1f2ef",
                          edgecolor="#819398", linewidth=0.2)
        ax.spines["geo"].set_edgecolor("#a4b6bd")
        ax.spines["geo"].set_linewidth(0.4)
    fig.input_field_metadata = records
    return str(source)


def surface_response(fig):
    ax = fig.add_axes([0.695, 0.310, 0.255, 0.175])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    x = np.linspace(0.06, 0.86, 250)
    wave = 0.012 * np.sin(2*np.pi*(x-0.06)/0.21)
    before = 0.48 + wave
    after = 0.65 + wave
    ax.fill_between(x, 0.25, before, color="#d5e5eb", linewidth=0)
    ax.fill_between(x, before, after, color="#edd2c5", linewidth=0)
    ax.plot(x, before, color=BLUE, linewidth=0.85, linestyle=(0, (3, 3)))
    ax.plot(x, after, color=WARM, linewidth=1.2)
    for xpos in (0.24, 0.47, 0.70):
        arrow(ax, (xpos, 0.35), (xpos, 0.60), color=WARM, width=0.9, scale=7)
    arrow(ax, (0.93, 0.48), (0.93, 0.65), color=INK, style="<->", width=0.8, scale=7)


def ocean(fig):
    ax = fig.add_axes([0.02, 0.205, 0.61, 0.43])
    x = np.linspace(0.8, 8.0, 600)
    bed = np.interp(x, [0.8, 1.12, 1.65, 2.02, 2.7, 3.6, 4.4,
                       5.35, 6.0, 6.35, 6.8, 7.2, 7.65, 8.0],
                    [4.62, 4.56, 4.38, 3.77, 1.26, 0.94, 1.09,
                     1.0, 1.36, 2.1, 1.25, 1.02, 1.14, 1.46])
    surface = 5.08
    boundaries = (4.38, 3.39, 0.83)
    tops = (surface, boundaries[0], boundaries[1])
    for top, bottom, tint in zip(tops, boundaries, ("#c9dde7", "#cbe2de", "#ddd5e7")):
        ax.fill_between(x, np.maximum(bottom, bed), top, where=bed < top,
                        color=tint, interpolate=True, linewidth=0)
    ax.fill_between(x, 0.72, bed, color="#dce1df", linewidth=0)
    ax.plot(x, bed, color="#819797", linewidth=1.0)
    ax.plot(x, np.full_like(x, surface), color=BLUE, linewidth=1.15)
    for boundary in boundaries[:2]:
        # Mask each boundary below the seabed; no artificial line crosses land.
        line = np.where(bed < boundary, boundary, np.nan)
        ax.plot(x, line, color="#7e959e", linewidth=0.6,
                linestyle=(0, (4, 4)), alpha=0.70)

    column_x = np.linspace(3.28, 3.84, 100)
    column_bed = np.interp(column_x, x, bed)
    ax.fill_between(column_x, column_bed, surface, color="white", alpha=0.43,
                    linewidth=0, zorder=4)
    ax.plot(column_x, column_bed, color=BLUE, linewidth=0.85, alpha=0.8, zorder=5)
    for xpos in (column_x[0], column_x[-1]):
        ax.plot([xpos, xpos], [np.interp(xpos, x, bed), surface],
                color=BLUE, linewidth=0.85, alpha=0.8, zorder=5)
    for y in np.linspace(0.95, surface, 15)[1:-1]:
        line = np.where(column_bed < y, y, np.nan)
        ax.plot(column_x, line, color=BLUE, linewidth=0.35, alpha=0.48, zorder=5)

    ax.plot([0.60, 0.60], [0.83, surface], color=MUTED, linewidth=0.65)
    for y, label in ((surface, "0"), (4.38, "700"), (3.39, "2000"), (0.83, "6000")):
        ax.plot([0.52, 0.66], [y, y], color=MUTED, linewidth=0.65)
        text(ax, 0.43, y, label, size=6.5, color=MUTED, ha="right")
    text(ax, -0.34, 2.94, "Depth (m)", size=7, color=MUTED,
         rotation=90, ha="center")

    text(ax, 3.9, 6.15, "Ocean heat uptake", size=9, weight="bold", color=WARM, ha="center")
    for xpos in (2.1, 3.5, 4.9, 6.3):
        arrow(ax, (xpos, 5.83), (xpos, 5.17), width=1.4, scale=9)
    for y, description, color in zip((4.73, 3.86, 2.58),
          ("0-700 m", "700-2000 m", "2000-6000 m"), (BLUE, TEAL, PURPLE)):
        text(ax, 7.76, y, description, size=7.5, color=color, weight="bold", ha="right")
    text(ax, 3.56, 0.15, "15\u2033 wet-cell volumes", size=7, ha="center", color=BLUE)
    arrow(ax, (3.56, 0.45), (3.56, np.interp(3.56, x, bed)),
          color=BLUE, width=0.7, scale=6)
    text(ax, 6.8, 0.15, "GEBCO bathymetry", size=7, color=MUTED, ha="center")
    ax.set_xlim(-0.8, 8.5)
    ax.set_ylim(-0.15, 6.85)
    ax.axis("off")


def time_axis(fig):
    ax = fig.add_axes([0.12, 0.035, 0.78, 0.095])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    text(ax, 0.36, 0.90, "Reconstruction", size=7.5, color=BLUE, ha="center")
    text(ax, 0.87, 0.90, "Forecast", size=7.5, color=WARM, ha="center")
    ax.plot([0.02, 0.72], [0.49, 0.49], color=BLUE, linewidth=1.0)
    arrow(ax, (0.72, 0.49), (0.98, 0.49), color=WARM, width=1.0,
          linestyle=(0, (3, 3)), scale=7)
    for x, date, color in ((0.02, "1940", BLUE), (0.72, "2025", BLUE), (0.98, "2030", WARM)):
        ax.plot(x, 0.49, marker="o", markersize=2.5, color=color)
        text(ax, x, 0.10, date, size=7, color=MUTED, ha="center")


def draw():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.5, 5.6), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], zorder=0)
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    source = input_symbols(fig)
    calculation_grid_map(fig)
    text(canvas, 0.20, 0.937, "Temperature", size=9, weight="bold", color=WARM, ha="center")
    text(canvas, 0.49, 0.937, "Salinity", size=9, weight="bold", color=TEAL, ha="center")
    text(canvas, 0.805, 0.937, "15\u2033 calculation grid", size=9, weight="bold", ha="center")
    text(canvas, 0.345, 0.710, "1\u00b0 \u2192 15\u2033 interpolation", size=7.5,
         color=BLUE, ha="center")
    # The restored neutral globe is geographic context, not a rendered 15-arcsec DEM.
    for xpos in (0.20, 0.49):
        canvas.plot([xpos, xpos], [0.72, 0.677], color=BLUE, linewidth=3.7,
                    solid_capstyle="butt")
    canvas.plot([0.20, 0.638, 0.638], [0.677, 0.677, 0.812], color=BLUE,
                linewidth=3.7, solid_joinstyle="round", solid_capstyle="butt")
    flow_arrow(canvas, (0.638, 0.812), (0.676, 0.812), scale=15)
    flow_arrow(canvas, (0.805, 0.705), (0.805, 0.645))
    text(canvas, 0.805, 0.605, "TEOS-10", size=14, weight="bold", ha="center")
    flow_arrow(canvas, (0.805, 0.574), (0.805, 0.548), color=INK, scale=13)
    text(canvas, 0.805, 0.512, r"$\Delta V\approx V\,\overline{\alpha}\,\Delta CT$",
         size=12, ha="center")
    flow_arrow(canvas, (0.638, 0.512), (0.716, 0.512))
    flow_arrow(canvas, (0.805, 0.484), (0.805, 0.465), color=WARM, scale=11)
    text(canvas, 0.805, 0.444, "Volume change", size=8, color=WARM,
         weight="bold", ha="center")
    surface_response(fig)
    flow_arrow(canvas, (0.805, 0.328), (0.805, 0.297), scale=13)
    text(canvas, 0.805, 0.267, "Global thermosteric contribution", size=7.6,
         color=BLUE, weight="bold", ha="center")
    text(canvas, 0.805, 0.203,
         r"$\Delta h=\frac{\sum\Delta V}{A_{\mathrm{ocean}}}$",
         size=12, ha="center")
    ocean(fig)
    time_axis(fig)
    return fig, source


CAPTION = """Conceptual illustration of ocean warming and its thermosteric contribution
to global mean sea level. Ocean heat uptake changes seawater density and volume.
Inputs are annual IAP temperature and salinity fields from two comparison years,
interpolated from their 1-degree grid to the 15-arcsec bathymetry grid. GEBCO
bathymetry defines the seawater volumes. TEOS-10 provides Conservative
Temperature and the expansion coefficient with respect to Conservative
Temperature. The comparison uses Delta CT and the mean of the two years'
expansion coefficients. Combining these with wet-cell volumes gives
Delta V approximately V * mean(alpha) * Delta CT. Summing over horizontal
cells i and depth layers k and dividing by global ocean area gives Delta h.
This is a thermal-expansion estimate, not a total steric density-difference
calculation including a separate halosteric contribution.
The two input thumbnails display actual IAP annual temperature and salinity at
1 m depth in 2025, not thermal-expansion results. They qualitatively identify
the input fields, rather than compare regional magnitudes. Temperature is in
degrees Celsius and salinity in g/kg, following the input file metadata.
Thumbnail color limits are -2 to 32 degrees Celsius and 30 to 39 g/kg; values
outside these limits are clipped for display only. No input values are changed.
The main reconstruction uses the comparison years and full depth profiles;
the segmented column illustrates 15-arcsec wet-cell volumes, including seabed-
truncated cells. Its width and vertical segmentation are schematic.
The two input maps and restored neutral global basemap have identical printed
size and projection. The neutral globe represents the geographic extent of the
15-arcsec calculation grid, not an actual rendered DEM or temperature field.
Its coarse graticules are geographic reference lines, not 15-arcsec cell borders.
The map sizes establish the geographical scope; their
printed size does not represent grid spacing. Refining the calculation grid
does not create independent 15-arcsec temperature or salinity observations.
The ocean section distinguishes upper (0-700 m), intermediate
(700-2000 m), and deep (2000-6000 m) contributions and illustrates how bathymetry
constrains the water volume. The raised water surface represents a conceptual
thermosteric contribution, not a local sea-level prediction. The timeline shows
the historical reconstruction from 1940 to 2025 and statistical forecasts for
2026-2030; it does not encode the magnitude or trend of either series.

All heat arrows, water-surface displacements, seabed geometry, and layer spacing
are schematic and not to scale. The map uses Natural Earth land geometry.
No calculated warming-change pattern, regional sea-level field, or uncertainty range
is displayed in this illustration. Ocean heat uptake represents the physical
background, not an additional measured heat-flux input. Computational detail belongs in the Methods:
the main calculation uses 15-arcsec DEM-resolved water volumes and interpolated
1-degree annual temperature and salinity, with Delta V = V * mean(alpha) *
Delta CT. Temperature ensemble spread is evaluated separately on bathymetry-
aggregated 1-degree volumes with the central annual salinity held fixed.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, coast = draw()
    stem = args.output_dir / "Fig_Introduction_ocean_thermal_expansion_framework"
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(str(stem)+"."+suffix, dpi=600, bbox_inches="tight", pad_inches=0.14,
                    facecolor="white", **options)
    plt.close(fig)
    (args.output_dir / "figure_caption.txt").write_text(CAPTION, encoding="utf-8")
    metadata = {"role": "Introduction conceptual illustration",
                "claim": "Ocean warming changes seawater volume and contributes to global mean sea level across space, depth and time.",
                "backend": "Python matplotlib + cartopy",
                "layout": "three equal-size globes, filled block flow arrows, two input branches, minimal text; no a/b/c labels",
                "figure_contract": {"archetype": "schematic-led composite",
                                    "evidence": "annual T/S to CT and alpha; GEBCO to V; V, alpha and Delta CT to Delta V; global integration to Delta h",
                                    "review_risks": "input maps qualitative only; 15-arcsec interpolation not independent observations; profile and expansion schematic; no local sea-level prediction"},
                "geography": coast,
                "neutral_globe": "restored Natural Earth basemap, schematic extent of the 15-arcsec grid, not a DEM raster",
                "numerical_results": "real 2025 surface input thumbnails only; displacement and section geometry schematic",
                "export": "600-dpi PNG/TIFF; editable text PDF/SVG",
                "methods_detail": "TEOS-10 CT/alpha, interpolation, wet-cell V, volume change and global integration explicitly linked",
                "depth_boundaries": "700 m and 2000 m: subtle dashed lines clipped above seabed",
                "input_symbols": "real ocean temperature and salinity map thumbnails",
                "input_thumbnail_sources": fig.input_field_metadata,
                "volume_column": "schematic segmented 15-arcsec wet-cell column clipped above seabed",
                "palette": {"upper": BLUE, "intermediate": TEAL, "deep": PURPLE, "heat": WARM}}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved " + str(stem))


if __name__ == "__main__":
    main()
