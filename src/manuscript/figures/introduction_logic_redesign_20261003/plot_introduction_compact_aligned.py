#!/usr/bin/env python3
"""Centered labels, longer TEOS connector and tightly cropped framework."""

from manuscript_paths import workspace_root, input_path
import argparse
import itertools
import json
from pathlib import Path

from matplotlib.transforms import Bbox

import plot_introduction_compact as compact


base = compact.base
NAME = "Fig_Introduction_ocean_thermal_expansion_framework_compact_aligned"


def draw():
    fig, coast = compact.draw()
    canvas = fig.axes[0]
    labels = {t.get_text(): t for t in canvas.texts}
    for ax, title in zip(fig.axes[1:4],
                         ("Temperature", "Salinity", "15\u2033 calculation grid")):
        position = ax.get_position()
        labels[title].set_x(position.x0 + position.width / 2)
        labels[title].set_ha("center")
    section = fig.axes[5]
    heat_label = next(t for t in section.texts if t.get_text() == "Ocean heat uptake")
    heat_label.set_x((0.8 + 8.0) / 2)
    heat_label.set_ha("center")
    for patch, xpos in zip(section.patches[:4], (2.3, 3.7, 5.1, 6.5)):
        patch.set_positions((xpos, 5.83), (xpos, 5.17))
    labels["TEOS-10"].set_y(0.665)
    if len(canvas.patches) != 7:
        raise RuntimeError("Unexpected compact connector layout.")
    canvas.patches[2].set_positions((0.805, 0.729), (0.805, 0.697))
    canvas.patches[3].set_positions((0.805, 0.637), (0.805, 0.574))
    return fig, coast


def content_bbox(fig, renderer, margin=0.055):
    boxes = []
    for ax in fig.axes:
        if hasattr(ax, "projection"):
            boxes.append(ax.get_window_extent(renderer))
        for artist in list(ax.texts) + list(ax.lines) + list(ax.patches):
            if artist.get_visible():
                boxes.append(artist.get_window_extent(renderer))
    bounds = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(bounds.x0 - margin, bounds.y0 - margin,
                             bounds.x1 + margin, bounds.y1 + margin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=base.OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = compact.refined.unique_stem(args.output_dir, NAME)
    fig, coast = draw()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for ax in fig.axes for t in ax.texts if t.get_text().strip()]
    overlaps = [(a.get_text(), b.get_text()) for a, b in itertools.combinations(texts, 2)
                if a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))]
    if overlaps:
        raise RuntimeError("Overlapping text: " + repr(overlaps))
    canvas = fig.axes[0]
    map_alignment = []
    for ax, title in zip(fig.axes[1:4],
                         ("Temperature", "Salinity", "15\u2033 calculation grid")):
        label = next(t for t in canvas.texts if t.get_text() == title)
        rendered = label.get_window_extent(renderer)
        bounds = ax.get_window_extent(renderer)
        offset = float((rendered.x0 + rendered.x1 - bounds.x0 - bounds.x1) / 2)
        map_alignment.append({"label": title, "offset_pixels": offset})
        if abs(offset) > 0.1:
            raise RuntimeError("Map title is not centered: " + title)
    section = fig.axes[5]
    label = next(t for t in section.texts if t.get_text() == "Ocean heat uptake")
    label_box = label.get_window_extent(renderer)
    middle = section.transData.transform((4.4, 5.08))[0]
    heat_offset = float((label_box.x0 + label_box.x1) / 2 - middle)
    if abs(heat_offset) > 0.1:
        raise RuntimeError("Heat-uptake label is not centered above the ocean section.")
    crop = content_bbox(fig, renderer)
    for suffix in ("png", "pdf", "svg", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}
        fig.savefig(str(stem) + "." + suffix, dpi=600, bbox_inches=crop,
                    pad_inches=0, facecolor="white", **options)
    caption = base.CAPTION + "\nAligned compact variant: each map title is centered on its own map frame;\nOcean heat uptake is centered on the seawater profile extent. The connector\nbelow TEOS-10 has a longer shaft. Export bounds are calculated from visible\ncontent with a 0.055-inch safety margin. Earlier exports are preserved.\n"
    metadata = {"role": "Introduction aligned compact framework", "backend": "Python matplotlib + cartopy",
        "figure_size_inches": [7.5, 4.9], "export_bounds_inches": list(crop.extents),
        "safety_margin_inches": 0.055, "map_label_alignment": map_alignment,
        "heat_label_offset_pixels": heat_offset, "text_overlaps": overlaps,
        "TEOS_connector_length_points": (0.637 - 0.574) * 4.9 * 72,
        "expansion_symbol_removed": True, "forecast_head": "separate solid head with endpoint clearance",
        "input_thumbnail_sources": fig.input_field_metadata, "geography": coast,
        "image_integrity": "real surface inputs; section and displacement schematic",
        "previous_exports": "preserved; unique filename", "export": "600-dpi PNG/TIFF; editable PDF/SVG"}
    for suffix, value in (("caption.txt", caption),
                          ("metadata.json", json.dumps(metadata, indent=2))):
        (stem.parent / (stem.name + "_" + suffix)).write_text(value, encoding="utf-8")
    base.plt.close(fig)
    print("Saved " + str(stem))
    print("QA: labels geometrically centered; no overlapping text; content-aware crop.")


if __name__ == "__main__":
    main()
