#!/usr/bin/env python3
"""Preserve the framework layout and refine only its process connectors."""

from manuscript_paths import workspace_root, input_path
import argparse
import itertools
import json
from pathlib import Path

from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, PathPatch

import plot_introduction_ocean_concept as base


NAME = "Fig_Introduction_ocean_thermal_expansion_framework_refined_connectors"
CONNECTOR = "#63899b"


def connector(ax, start=None, end=None, path=None, color=CONNECTOR, scale=10):
    options = {"path": path} if path is not None else {"posA": start, "posB": end}
    item = FancyArrowPatch(arrowstyle="-|>,head_length=0.55,head_width=0.25",
        mutation_scale=scale, linewidth=1.7, color=color,
        capstyle="round", joinstyle="round", shrinkA=0.5, shrinkB=0.5, **options)
    ax.add_patch(item)
    return item


def layout_signature(fig):
    return [(t.get_text(), list(t.get_position()), t.get_fontsize(), t.get_color())
            for ax in fig.axes for t in ax.texts]


def draw():
    fig, coast = base.draw()
    before = layout_signature(fig)
    positions = [list(ax.get_position().bounds) for ax in fig.axes]
    canvas = fig.axes[0]
    process_arrows = [p for p in canvas.patches if p.get_linewidth() == 0]
    if len(process_arrows) != 6 or len(canvas.lines) != 3:
        raise RuntimeError("The source framework no longer matches the expected layout.")
    for item in process_arrows + list(canvas.lines):
        item.remove()

    # Merge the two inputs through a rounded route, keeping the existing text and maps.
    trunk = MplPath([(0.20, 0.720),
        (0.20, 0.689), (0.207, 0.682), (0.246, 0.682),
        (0.365, 0.682), (0.55, 0.682), (0.588, 0.682),
        (0.628, 0.682), (0.638, 0.688), (0.638, 0.727),
        (0.638, 0.786), (0.638, 0.812), (0.676, 0.812)],
        [MplPath.MOVETO] + [MplPath.CURVE4] * 12)
    branch = MplPath([(0.49, 0.720), (0.49, 0.691), (0.501, 0.682), (0.534, 0.682)],
        [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
    canvas.add_patch(PathPatch(branch, fill=False, edgecolor=CONNECTOR,
                              linewidth=1.7, capstyle="round", joinstyle="round"))
    connector(canvas, path=trunk)
    connector(canvas, (0.805, 0.705), (0.805, 0.645))
    connector(canvas, (0.805, 0.574), (0.805, 0.548), color=base.INK, scale=8)
    connector(canvas, (0.638, 0.512), (0.716, 0.512))
    connector(canvas, (0.805, 0.484), (0.805, 0.465), color=base.WARM, scale=7)
    connector(canvas, (0.805, 0.328), (0.805, 0.297), scale=8)
    if layout_signature(fig) != before or [list(ax.get_position().bounds) for ax in fig.axes] != positions:
        raise RuntimeError("A non-connector layout element was changed.")
    fig.layout_unchanged = True
    return fig, coast


def unique_stem(directory, name):
    index = 1
    while True:
        stem = directory / (name if index == 1 else name + "_{:02d}".format(index))
        if not list(directory.glob(stem.name + ".*")) and not list(directory.glob(stem.name + "_*")):
            return stem
        index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = unique_stem(args.output_dir, NAME)
    fig, coast = draw()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
    overlaps = [(a.get_text(), b.get_text()) for a, b in itertools.combinations(texts, 2)
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
    if overlaps:
        raise RuntimeError("Overlapping text: " + repr(overlaps))
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(str(stem) + "." + suffix, dpi=600, bbox_inches="tight",
                    pad_inches=0.14, facecolor="white", **options)
    caption = base.CAPTION + "\nConnector refinement: the original framework layout, input maps, section,\nTEOS-10 label, equations, physical heat arrows and timeline are unchanged.\nOnly process connectors use rounded curves, moderate stroke widths and\nsmaller filled arrowheads.\n"
    metadata = {"role": "Introduction framework with refined connectors",
        "backend": "Python matplotlib + cartopy", "source_layout": "original framework",
        "changes": "process connectors only", "layout_unchanged": fig.layout_unchanged,
        "text_overlaps": overlaps, "process_arrows": 6,
        "connector_style": "rounded curves, 1.7-point strokes, compact filled arrowheads",
        "input_thumbnail_sources": fig.input_field_metadata, "geography": coast,
        "image_integrity": "actual 2025 surface temperature/salinity examples; schematic section and expansion",
        "previous_exports": "not modified; automatic unique naming",
        "export": "600-dpi PNG/TIFF; editable PDF/SVG"}
    for suffix, value in (("caption.txt", caption),
                          ("metadata.json", json.dumps(metadata, indent=2))):
        (stem.parent / (stem.name + "_" + suffix)).write_text(value, encoding="utf-8")
    base.plt.close(fig)
    print("Saved " + str(stem))
    print("QA: original text and axes preserved; no text overlap.")


if __name__ == "__main__":
    main()
