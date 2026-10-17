#!/usr/bin/env python3
"""Signed annual ridgelines for disjoint 15-arcsec latitude shells."""

from manuscript_paths import workspace_root, input_path
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


FIGURES = (workspace_root() / 'figures')
INPUT = FIGURES / "nested_latitude_15arcsec/Fig04_nested_latitude_15arcsec_annual_main.csv"
OUTPUT = FIGURES / "revised_lon_aligned_20260925/latitude_shell_ridgelines_2025"
RANGES = (
    ("15S-15N", "15\N{DEGREE SIGN}S\N{EN DASH}15\N{DEGREE SIGN}N"),
    ("30S-30N", "15\N{DEGREE SIGN}\N{EN DASH}30\N{DEGREE SIGN} S/N"),
    ("45S-45N", "30\N{DEGREE SIGN}\N{EN DASH}45\N{DEGREE SIGN} S/N"),
    ("60S-60N", "45\N{DEGREE SIGN}\N{EN DASH}60\N{DEGREE SIGN} S/N"),
    ("75S-75N", "60\N{DEGREE SIGN}\N{EN DASH}75\N{DEGREE SIGN} S/N"),
    ("86S-90N", "86\N{DEGREE SIGN}S\N{EN DASH}75\N{DEGREE SIGN}S; 75\N{DEGREE SIGN}N\N{EN DASH}90\N{DEGREE SIGN}N"),
)
POSITIVE = "#447c8e"
NEGATIVE = "#ba7669"
BASELINE = "#abb8ba"
HEIGHT_PER_MM = 0.19


def load_shells(path):
    data = pd.read_csv(path)
    data = data[data.depth_band == "0-6000m"].copy()
    expected = {(year, name) for year in range(1941, 2026)
                for name, _ in RANGES}
    observed = list(zip(data.target_year, data.latitude_band))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Expected one annual value per range, 1941-2025")
    if not (data.base_year == data.target_year - 1).all():
        raise ValueError("Nonconsecutive annual comparisons")
    if not np.isfinite(data.global_contribution_mm).all():
        raise ValueError("Non-finite annual contributions")
    order = [name for name, _ in RANGES]
    nested = (data.pivot(index="target_year", columns="latitude_band",
                         values="global_contribution_mm")
              .reindex(columns=order).sort_index())
    shells = nested.copy()
    shells.iloc[:, 1:] = nested.iloc[:, 1:].to_numpy() - nested.iloc[:, :-1].to_numpy()
    if not np.allclose(shells.sum(axis=1), nested.iloc[:, -1], atol=1e-10):
        raise ValueError("Annual shell closure failed")
    source = (shells.reset_index().melt(id_vars="target_year", var_name="nested_range",
                                        value_name="annual_shell_contribution_mm"))
    return shells, source


def draw(shells):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.5, 4.4), facecolor="white")
    ax = fig.add_axes([0.30, 0.16, 0.66, 0.70])
    years = shells.index.to_numpy(dtype=float)
    levels = np.arange(len(RANGES))[::-1] * 1.42 + 0.65

    for i, (name, _) in enumerate(RANGES):
        base = float(levels[i])
        values = shells[name].to_numpy(dtype=float)
        profile = base + values * HEIGHT_PER_MM
        ax.hlines(base, 1940, 2026, color=BASELINE, linewidth=0.65, zorder=1)
        ax.fill_between(years, base, profile, where=values >= 0, interpolate=True,
                        color=POSITIVE, alpha=0.78, linewidth=0, zorder=2)
        ax.fill_between(years, base, profile, where=values < 0, interpolate=True,
                        color=NEGATIVE, alpha=0.78, linewidth=0, zorder=2)
        ax.plot(years, profile, color="#34434a", linewidth=0.58, alpha=0.85, zorder=3)

    ax.set_xlim(1940, 2026)
    ax.set_ylim(-0.15, 9.0)
    ax.set_yticks(levels, [label for _, label in RANGES])
    ax.set_xticks([1940, 1960, 1980, 2000, 2025])
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", length=3)
    ax.grid(axis="x", color="#e5eaea", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Year", labelpad=7)
    ax.plot([1945, 1945], [8.60, 8.60 + HEIGHT_PER_MM],
            color="#263238", linewidth=1.0, clip_on=False)
    ax.plot([1944.6, 1945.4], [8.60, 8.60], color="#263238", linewidth=0.8)
    ax.plot([1944.6, 1945.4], [8.60 + HEIGHT_PER_MM, 8.60 + HEIGHT_PER_MM],
            color="#263238", linewidth=0.8)
    ax.text(1947, 8.60 + HEIGHT_PER_MM / 2, "Vertical scale: 1 mm yr$^{-1}$", va="center",
            fontsize=7.2, color="#45565c")
    handles = [Patch(facecolor=POSITIVE, label="Expansion (+)"),
               Patch(facecolor=NEGATIVE, label="Contraction (-)")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.96, 0.88),
               ncol=2, frameon=False, fontsize=7.5, handlelength=1.2)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    shells, source = load_shells(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = draw(shells)
    stem = args.output_dir / "Fig04_latitude_shell_ridgelines_15arcsec_2025"
    for suffix in ("png", "pdf", "svg", "tiff"):
        fig.savefig(str(stem) + "." + suffix, bbox_inches="tight",
                    facecolor="white", dpi=600 if suffix in ("png", "tiff") else None)
    plt.close(fig)
    source.to_csv(str(stem) + "_source_data.csv", index=False)
    metadata = {
        "input": str(args.input.resolve()),
        "years": "1941-2025",
        "depth_band": "0-6000m",
        "derivation": "successive difference of nested annual ranges",
        "ridge_scale": "common linear scale across all shells; unsmoothed annual values",
        "quantity": "annual contribution to global mean sea level (mm yr-1)",
        "cumulative_numbers": "not displayed",
        "palette": {"expansion": POSITIVE, "contraction": NEGATIVE},
        "total_coverage": "86S-90N; not strictly 90S-90N",
        "uncertainty": "not displayed",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved " + str(stem))


if __name__ == "__main__":
    main()
