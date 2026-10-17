#!/usr/bin/env python3
"""Compact Introduction framework with four equal-length vertical connectors."""

from manuscript_paths import workspace_root, input_path
import argparse
import itertools
import json
from pathlib import Path

import plot_introduction_compact_aligned as aligned


base = aligned.base
NAME = "Fig_Introduction_ocean_thermal_expansion_framework_equal_arrows"
ARROW_INDICES = (2, 3, 5, 6)
ARROW_STARTS = (0.746, 0.634, 0.513, 0.323)
ARROW_LENGTH = 0.055


def draw():
    fig, coast = aligned.draw()
    canvas = fig.axes[0]
    for index, start in zip(ARROW_INDICES, ARROW_STARTS):
        arrow = canvas.patches[index]
        arrow.set_positions((0.805, start), (0.805, start - ARROW_LENGTH))
        arrow.set_mutation_scale(8)
    return fig, coast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = aligned.compact.refined.unique_stem(args.output_dir, NAME)
    fig, coast = draw()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
    overlaps = [(a.get_text(), b.get_text())
                for a, b in itertools.combinations(texts, 2)
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
    if overlaps:
        raise RuntimeError("Overlapping text: " + repr(overlaps))
    canvas = fig.axes[0]
    heights = []
    for index in ARROW_INDICES:
        arrow = canvas.patches[index]
        bounds = arrow.get_window_extent(renderer)
        heights.append(float(bounds.height))
        for label in texts:
            if bounds.overlaps(label.get_window_extent(renderer)):
                raise RuntimeError("Flow arrow overlaps label: " + label.get_text())
    if max(heights) - min(heights) > 0.1:
        raise RuntimeError("Vertical flow arrows differ in rendered length.")
    crop = aligned.content_bbox(fig, renderer)
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(str(stem) + "." + suffix, dpi=600, bbox_inches=crop,
                    pad_inches=0, facecolor="white", **options)
    caption = base.CAPTION + (
        "\nEqual-connector compact variant: all four right-hand vertical flow arrows\n"
        "have equal rendered lengths and identical arrowhead size and line width.\n"
        "The input titles remain centered on their respective maps; the heat-uptake\n"
        "title remains centered on the ocean profile. Previous exports are preserved.\n")
    metadata = {
        "role": "Introduction schematic-led framework",
        "backend": "Python matplotlib + cartopy",
        "figure_size_inches": list(fig.get_size_inches()),
        "export_bounds_inches": list(crop.extents),
        "vertical_arrow_endpoint_length_points": ARROW_LENGTH * 4.9 * 72,
        "vertical_arrow_rendered_heights_pixels": heights,
        "vertical_arrow_head_scale": 8,
        "text_overlaps": overlaps,
        "arrow_text_overlaps": [],
        "input_thumbnail_sources": fig.input_field_metadata,
        "geography": coast,
        "image_integrity": "real surface inputs; section and displacement schematic",
        "calculation_changes": "none",
        "previous_exports": "preserved; unique filename",
        "export": "600-dpi PNG/TIFF; editable PDF/SVG",
    }
    for suffix, value in (("caption.txt", caption),
                          ("metadata.json", json.dumps(metadata, indent=2))):
        (stem.parent / (stem.name + "_" + suffix)).write_text(value, encoding="utf-8")
    base.plt.close(fig)
    print("Saved " + str(stem))
    print("QA: equal arrow heights {}; no text/arrow overlaps.".format(heights))


if __name__ == "__main__":
    main()
