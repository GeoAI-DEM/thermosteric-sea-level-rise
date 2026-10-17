#!/usr/bin/env python3
"""Shift the complete right-hand framework column without breaking alignment."""

from manuscript_paths import workspace_root, input_path
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import plot_introduction_equal_arrows as equal


base = equal.base
NAME = "Fig_Introduction_ocean_thermal_expansion_framework_right_column"
SHIFT = 0.03


def draw():
    fig, coast = equal.draw()
    canvas = fig.axes[0]
    for axis in (fig.axes[3], fig.axes[4]):
        x, y, width, height = axis.get_position().bounds
        axis.set_position([x + SHIFT, y, width, height])
    for label in canvas.texts:
        if label.get_position()[0] > 0.7:
            label.set_x(label.get_position()[0] + SHIFT)
    grid_position = fig.axes[3].get_position()
    center = grid_position.x0 + grid_position.width / 2
    for index, start in zip(equal.ARROW_INDICES, equal.ARROW_STARTS):
        canvas.patches[index].set_positions(
            (center, start), (center, start - equal.ARROW_LENGTH))
    canvas.patches[4].set_positions((0.638, 0.542), (0.716 + SHIFT, 0.542))
    # Extend the incoming curved connector only near the moved map.
    trunk = canvas.patches[1]
    path = trunk._path_original
    vertices = path.vertices.copy()
    vertices[-2:, 0] += SHIFT
    trunk._path_original = type(path)(vertices, path.codes)
    trunk.stale = True
    return fig, coast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = equal.aligned.compact.refined.unique_stem(args.output_dir, NAME)
    fig, coast = draw()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.axes[0]
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
    overlaps = [(a.get_text(), b.get_text()) for a, b in itertools.combinations(texts, 2)
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
    if overlaps:
        raise RuntimeError("Overlapping labels: " + repr(overlaps))
    map_bounds = fig.axes[3].get_window_extent(renderer)
    map_center = (map_bounds.x0 + map_bounds.x1) / 2
    heights, offsets = [], []
    for index in equal.ARROW_INDICES:
        arrow = canvas.patches[index]
        bounds = arrow.get_window_extent(renderer)
        heights.append(float(bounds.height))
        offsets.append(float((bounds.x0 + bounds.x1) / 2 - map_center))
        if any(bounds.overlaps(t.get_window_extent(renderer)) for t in texts):
            raise RuntimeError("Flow arrow overlaps a label.")
    if max(abs(value) for value in offsets) > 0.1 or np.ptp(heights) > 0.1:
        raise RuntimeError("Arrow centering or equal lengths failed.")
    title = next(t for t in canvas.texts if t.get_text() == "15\u2033 calculation grid")
    bounds = title.get_window_extent(renderer)
    if abs((bounds.x0 + bounds.x1) / 2 - map_center) > 0.1:
        raise RuntimeError("Grid title is not centered.")
    crop = equal.aligned.content_bbox(fig, renderer)
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(str(stem) + "." + suffix, dpi=600, bbox_inches=crop,
                    pad_inches=0, facecolor="white", **options)
    metadata = {
        "backend": "Python matplotlib + cartopy",
        "right_column_shift_inches": SHIFT * fig.get_figwidth(),
        "arrow_center_offsets_pixels": offsets,
        "arrow_rendered_heights_pixels": heights,
        "text_overlaps": overlaps,
        "input_thumbnail_sources": fig.input_field_metadata,
        "geography": coast,
        "calculation_changes": "none",
        "previous_exports": "preserved",
        "export": "600-dpi PNG/TIFF; editable PDF/SVG",
    }
    caption = base.CAPTION + (
        "\nThe complete right-hand column, including its geographic input map, is\n"
        "shifted right by 0.225 inches. The map, its title and four equal-length\n"
        "vertical connectors share the same center line. Other layouts are preserved.\n")
    for suffix, value in (("caption.txt", caption),
                          ("metadata.json", json.dumps(metadata, indent=2))):
        (stem.parent / (stem.name + "_" + suffix)).write_text(value, encoding="utf-8")
    base.plt.close(fig)
    print("Saved " + str(stem))
    print("QA: arrow-center offsets {}; equal heights {}; no overlaps.".format(offsets, heights))


if __name__ == "__main__":
    main()
