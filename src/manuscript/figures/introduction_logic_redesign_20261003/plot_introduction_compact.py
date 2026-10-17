#!/usr/bin/env python3
"""A compact framework variant with explicit, unobscured forecast arrowhead."""

from manuscript_paths import workspace_root, input_path
import argparse
import itertools
import json
from pathlib import Path

from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Polygon

import plot_introduction_refined_connectors as refined


base = refined.base
NAME = "Fig_Introduction_ocean_thermal_expansion_framework_compact"


def draw():
    fig, coast = refined.draw()
    if len(fig.axes) != 7:
        raise RuntimeError("Unexpected source framework axes.")
    fig.set_size_inches(7.5, 4.9, forward=True)
    canvas = fig.axes[0]
    for ax in fig.axes[1:4]:
        left = ax.get_position().x0
        ax.set_position([left, 0.746, 0.25, 0.198])
    response, section, timeline = fig.axes[4:7]
    response.set_position([0.695, 0.284, 0.255, 0.177])
    response.patches[-1].remove()
    section.set_position([0.02, 0.225, 0.61, 0.455])
    timeline.set_position([0.12, 0.024, 0.78, 0.094])

    label_y = {
        "Temperature": 0.964,
        "Salinity": 0.964,
        "15\u2033 calculation grid": 0.964,
        "1\u00b0 \u2192 15\u2033 interpolation": 0.728,
        "TEOS-10": 0.631,
        r"$\Delta V\approx V\,\overline{\alpha}\,\Delta CT$": 0.542,
        "Volume change": 0.426,
        "Global thermosteric contribution": 0.247,
        r"$\Delta h=\frac{\sum\Delta V}{A_{\mathrm{ocean}}}$": 0.185,
    }
    for item in canvas.texts:
        item.set_y(label_y[item.get_text()])
    for item in list(canvas.patches):
        item.remove()

    trunk = MplPath([(0.20, 0.742),
        (0.20, 0.713), (0.207, 0.700), (0.246, 0.700),
        (0.365, 0.700), (0.55, 0.700), (0.588, 0.700),
        (0.628, 0.700), (0.638, 0.706), (0.638, 0.747),
        (0.638, 0.808), (0.638, 0.845), (0.676, 0.845)],
        [MplPath.MOVETO] + [MplPath.CURVE4] * 12)
    branch = MplPath([(0.49, 0.742), (0.49, 0.713), (0.501, 0.700), (0.534, 0.700)],
        [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
    canvas.add_patch(PathPatch(branch, fill=False, edgecolor=refined.CONNECTOR,
        linewidth=1.7, capstyle="round", joinstyle="round"))
    refined.connector(canvas, path=trunk)
    refined.connector(canvas, (0.805, 0.729), (0.805, 0.667))
    refined.connector(canvas, (0.805, 0.603), (0.805, 0.578), color=base.INK, scale=8)
    refined.connector(canvas, (0.638, 0.542), (0.716, 0.542))
    refined.connector(canvas, (0.805, 0.515), (0.805, 0.451), color=base.WARM, scale=8)
    refined.connector(canvas, (0.805, 0.305), (0.805, 0.275), scale=8)

    # Draw the dashed forecast shaft and solid arrowhead separately from the endpoint.
    for item in list(timeline.patches):
        item.remove()
    timeline.plot([0.72, 0.953], [0.49, 0.49], color=base.WARM,
                  linewidth=1.0, linestyle=(0, (3, 3)))
    head = Polygon([(0.967, 0.49), (0.950, 0.565), (0.950, 0.415)],
                   closed=True, facecolor=base.WARM, edgecolor="none",
                   clip_on=False, zorder=5)
    timeline.add_patch(head)
    fig.forecast_head = head
    fig.forecast_axis = timeline
    return fig, coast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = refined.unique_stem(args.output_dir, NAME)
    fig, coast = draw()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
    overlaps = [(a.get_text(), b.get_text()) for a, b in itertools.combinations(texts, 2)
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
    if overlaps:
        raise RuntimeError("Overlapping text: " + repr(overlaps))
    head_box = fig.forecast_head.get_window_extent(renderer)
    axis_box = fig.forecast_axis.get_window_extent(renderer)
    head_inside = bool(axis_box.x0 < head_box.x0 < head_box.x1 < axis_box.x1
                       and axis_box.y0 < head_box.y0 < head_box.y1 < axis_box.y1)
    if not head_inside:
        raise RuntimeError("Forecast arrowhead exceeds the visible timeline extent.")
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(str(stem) + "." + suffix, dpi=600, bbox_inches="tight",
                    pad_inches=0.12, facecolor="white", **options)
    caption = base.CAPTION + "\nCompact layout variant: printed height is reduced from 5.6 to 4.9 inches.\nThe double-headed displacement symbol beside the expansion illustration is\nomitted. The connector below the volume-change equation has a longer shaft.\nThe forecast timeline uses a dashed shaft with a separate solid arrowhead,\nspaced from the 2030 marker and inside the visible axes extent.\n"
    metadata = {"role": "Introduction compact framework", "backend": "Python matplotlib + cartopy",
        "changes": ["reduced vertical whitespace", "displacement symbol removed",
                    "longer arrow shaft beneath Delta V", "complete solid forecast arrowhead"],
        "figure_size_inches": [7.5, 4.9], "previous_size_inches": [7.5, 5.6],
        "height_reduction_percent": 12.5, "text_overlaps": overlaps,
        "forecast_head_inside_axes": head_inside,
        "forecast_arrowhead": "solid triangle separate from dashed shaft, not covered by 2030 marker",
        "equation_arrow_length_points": (0.515 - 0.451) * 4.9 * 72,
        "expansion_symbol_removed": True, "geography": coast,
        "input_thumbnail_sources": fig.input_field_metadata,
        "image_integrity": "real 2025 surface T/S examples; section and expansion schematic",
        "export": "600-dpi PNG/TIFF; editable PDF/SVG", "previous_exports": "preserved"}
    for suffix, value in (("caption.txt", caption),
                          ("metadata.json", json.dumps(metadata, indent=2))):
        (stem.parent / (stem.name + "_" + suffix)).write_text(value, encoding="utf-8")
    base.plt.close(fig)
    print("Saved " + str(stem))
    print("QA: no overlapping text; forecast head fully inside axes.")


if __name__ == "__main__":
    main()
