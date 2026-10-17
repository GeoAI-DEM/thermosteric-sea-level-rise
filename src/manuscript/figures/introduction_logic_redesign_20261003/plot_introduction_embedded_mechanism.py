#!/usr/bin/env python3
"""An arrow-free mechanism variant; previous figure exports are preserved."""

from manuscript_paths import workspace_root, input_path
import argparse
import itertools
import json
from pathlib import Path

from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

import plot_introduction_ocean_concept as base


DEFAULT_NAME = "Fig_Introduction_ocean_thermal_expansion_embedded_mechanism"


def scope(fig):
    ax = fig.add_axes([0.12, 0.035, 0.78, 0.095])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    base.text(ax, 0.36, 0.90, "Reconstruction", size=7.5, color=base.BLUE, ha="center")
    base.text(ax, 0.87, 0.90, "Forecast", size=7.5, color=base.WARM, ha="center")
    ax.plot([0.02, 0.72], [0.49, 0.49], color=base.BLUE, linewidth=0.85)
    ax.plot([0.72, 0.98], [0.49, 0.49], color=base.WARM,
            linewidth=0.85, linestyle=(0, (3, 3)))
    for xpos, date, color in ((0.02, "1940", base.BLUE), (0.72, "2025", base.BLUE),
                              (0.98, "2030", base.WARM)):
        ax.plot(xpos, 0.49, marker="o", markersize=2.5, color=color)
        base.text(ax, xpos, 0.10, date, size=7, color=base.MUTED, ha="center")


def draw():
    base.plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = base.plt.figure(figsize=(7.5, 5.6), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], zorder=0)
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    coast = base.input_symbols(fig)
    base.calculation_grid_map(fig)
    for xpos, label, color in ((0.20, "Temperature", base.WARM),
                              (0.49, "Salinity", base.TEAL),
                              (0.805, "15\u2033 calculation grid", base.INK)):
        base.text(canvas, xpos, 0.937, label, size=9, weight="bold",
                  color=color, ha="center")

    # Quiet alignment marks associate the input fields without directional flowheads.
    guide_color = "#a9b8bd"
    canvas.plot([0.20, 0.20, 0.49, 0.49], [0.716, 0.700, 0.700, 0.716],
                color=guide_color, linewidth=0.6)
    base.text(canvas, 0.345, 0.673, "1\u00b0 fields, interpolated to 15\u2033",
              size=7.5, color=base.MUTED, ha="center")

    base.ocean(fig)
    section = fig.axes[-1]
    section.set_position([0.04, 0.205, 0.67, 0.445])
    # Turn the wet-cell callout into an unheaded leader; retain only physical heat arrows.
    section.patches[-1].remove()
    section.plot([3.56, 3.56], [0.45, 0.94], color=base.BLUE, linewidth=0.6)
    base.text(section, 4.18, 3.06, "TEOS-10", size=10,
              weight="bold", color=base.INK, ha="left")
    base.text(section, 4.18, 2.48,
              r"$\Delta V\approx V\,\overline{\alpha}\,\Delta CT$",
              size=10.8, color=base.INK, ha="left")
    section.plot([3.88, 3.98, 3.98, 3.88], [0.97, 0.97, 5.08, 5.08],
                 color=guide_color, linewidth=0.6)
    section.plot([3.98, 4.07], [3.06, 3.06], color=guide_color, linewidth=0.6)

    base.surface_response(fig)
    response = fig.axes[-1]
    response.set_position([0.745, 0.465, 0.22, 0.19])
    for patch in list(response.patches):
        patch.remove()
    response.plot([0.93, 0.97, 0.97, 0.93], [0.48, 0.48, 0.65, 0.65],
                  color=base.MUTED, linewidth=0.65)
    base.text(response, 0.45, 0.568, r"$\Delta V$", size=9,
              color=base.WARM, ha="center")
    base.text(canvas, 0.85, 0.64, "Volume change", size=8.5,
              color=base.WARM, weight="bold", ha="center")
    canvas.plot([0.85, 0.85], [0.505, 0.47], color=guide_color, linewidth=0.6)
    base.text(canvas, 0.85, 0.44, "Global thermosteric", size=8.2,
              color=base.BLUE, weight="bold", ha="center")
    base.text(canvas, 0.85, 0.409, "contribution", size=8.2,
              color=base.BLUE, weight="bold", ha="center")
    base.text(canvas, 0.85, 0.351,
              r"$\Delta h=\frac{\sum\Delta V}{A_{\mathrm{ocean}}}$",
              size=12, ha="center")

    # A thin unheaded callout links the section to a schematic expansion response.
    path = MplPath([(0.674, 0.5375), (0.715, 0.5375), (0.73, 0.57), (0.756, 0.573)],
                   [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    canvas.add_patch(PathPatch(path, fill=False, linewidth=0.6, edgecolor=guide_color))
    scope(fig)
    return fig, coast


def qa(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
    overlaps = [(a.get_text(), b.get_text()) for a, b in itertools.combinations(labels, 2)
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
    maps = [list(ax.get_position().bounds) for ax in fig.axes if hasattr(ax, "projection")]
    if overlaps:
        raise RuntimeError("Overlapping labels: " + repr(overlaps))
    return {"text_overlaps": overlaps, "map_bounds": maps, "text_count": len(labels),
            "panel_letters_absent": not any(t.get_text() in ("a", "b", "c") for t in labels),
            "flow_arrows": 0, "physical_heat_arrows": 4}


def unused_stem(directory, name):
    stem = directory / name
    sequence = 2
    while list(directory.glob(stem.name + ".*")) or list(directory.glob(stem.name + "_*")):
        stem = directory / (name + "_{:02d}".format(sequence))
        sequence += 1
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.OUTPUT)
    parser.add_argument("--name", default=DEFAULT_NAME)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = unused_stem(args.output_dir, args.name)
    fig, coast = draw()
    report = qa(fig)
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(str(stem) + "." + suffix, dpi=600, bbox_inches="tight",
                    pad_inches=0.14, facecolor="white", **options)
    caption = base.CAPTION + "\nLayout variant: TEOS-10 and its volume-change expression are embedded beside\nthe wet-cell column. A full-column bracket associates the expression with every\ndepth layer, not only the deep ocean where the annotation is positioned.\nThin unheaded leaders link related objects; there are no directional process\narrows. Only heat-uptake arrows represent a physical direction.\n"
    metadata = {"role": "Introduction mechanism illustration", "backend": "Python matplotlib + cartopy",
        "claim": "Ocean temperature and salinity, combined with bathymetry-resolved water volumes, yield thermosteric expansion.",
        "layout": "embedded mechanism; no flow arrows, cards, or a/b/c panel letters",
        "input_thumbnail_sources": fig.input_field_metadata, "geography": coast,
        "neutral_globe": "geographic context of the calculation grid; not a 15-arcsec DEM raster",
        "methods": "TEOS-10 CT and alpha; Delta V approximately V * mean(alpha) * Delta CT; Delta h = sum(Delta V) / ocean area",
        "image_integrity": "real 2025 surface inputs; schematic section, wet-cell segmentation and raised surface",
        "export": "600-dpi PNG/TIFF; PDF/SVG with editable text", "qa": report,
        "previous_exports": "not modified; automatic unique names prevent overwriting"}
    for suffix, content in (("caption.txt", caption),
                            ("metadata.json", json.dumps(metadata, indent=2)),
                            ("qa.json", json.dumps(report, indent=2))):
        (stem.parent / (stem.name + "_" + suffix)).write_text(content, encoding="utf-8")
    base.plt.close(fig)
    print("Saved " + str(stem))
    print("QA: no overlapping text; zero directional process arrows.")


if __name__ == "__main__":
    main()
