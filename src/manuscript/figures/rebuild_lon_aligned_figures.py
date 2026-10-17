#!/usr/bin/env python3
"""Rebuild uncertainty-dependent forecasts and figures in a separate directory."""

from manuscript_paths import workspace_root, input_path
import argparse
import importlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = workspace_root()
UNCERTAINTY = ROOT / "uncertainty_analysis/full_spatiotemporal_uncertainty/results_1940_2023_lon_aligned"
DEFAULT_OUTPUT = ROOT / "figures/revised_lon_aligned_20260925"


def validate():
    metadata = json.loads((UNCERTAINTY / "metadata.json").read_text(encoding="utf-8"))
    if (metadata["start_year"], metadata["end_year"]) != (1940, 2023):
        raise ValueError("Corrected uncertainty years are incomplete")
    if "reordered" not in metadata.get("member_longitude_alignment", ""):
        raise ValueError("Longitude alignment is not documented")
    summary = pd.read_csv(UNCERTAINTY / "uncertainty_summary_global.csv")
    for band, member_count in (("0-700m", 11), ("700-2000m", 7),
                               ("2000-6000m", 7), ("0-6000m", 7)):
        rows = summary.loc[summary.depth_band == band]
        if set(rows.target_year) != set(range(1941, 2024)):
            raise ValueError("Missing years for " + band)
        if not (rows.member_count == member_count).all():
            raise ValueError("Wrong member count for " + band)


def configure(output):
    for directory in (ROOT / "figures", ROOT / "short_term_projection",
                      ROOT / "uncertainty_analysis/tail_uncertainty_proxy"):
        pass  # Source imports are configured by the launcher.
    names = (
        "run_global_projection", "run_explainable_dynamic_projection",
        "run_sparse_bsts_projection", "run_performance_weighted_ensemble",
        "run_depth_band_ensemble_projection", "build_tail_uncertainty_proxy",
        "plot_thermal_expansion_with_uncertainty",
        "plot_ocean_basin_historical_partition",
        "plot_latitudinal_thermosteric_partition_redesign",
    )
    m = {name: importlib.import_module(name) for name in names}
    global_model = m["run_global_projection"]
    bsts = m["run_sparse_bsts_projection"]
    ensemble = m["run_performance_weighted_ensemble"]
    depth = m["run_depth_band_ensemble_projection"]
    tail = m["build_tail_uncertainty_proxy"]
    global_model.UNCERTAINTY_RESULTS = UNCERTAINTY
    global_model.OUTPUT_DIR = output / "projection/results"
    bsts.OUTPUT_DIR = output / "projection/sparse_bsts_results"
    ensemble.BASELINE_DIR = global_model.OUTPUT_DIR
    ensemble.BSTS_DIR = bsts.OUTPUT_DIR
    ensemble.OUTPUT_DIR = output / "projection/performance_weighted_ensemble_results"
    depth.UNCERTAINTY_FILE = UNCERTAINTY / "uncertainty_summary_global.csv"
    depth.TOTAL_SAMPLES_FILE = ensemble.OUTPUT_DIR / "ensemble_projection_samples.npz"
    depth.OUTPUT_DIR = output / "projection/depth_band_ensemble_results"
    tail.ROOT = output / "tail_uncertainty_proxy"
    tail.INPUT_FILE = UNCERTAINTY / "uncertainty_summary_global.csv"
    tail.OUTPUT_FILE = tail.ROOT / "tail_uncertainty_proxy_2024_2025.csv"
    tail.VALIDATION_FILE = tail.ROOT / "tail_uncertainty_proxy_rolling_validation.csv"
    tail.METADATA_FILE = tail.ROOT / "tail_uncertainty_proxy_metadata.json"

    global_plot = m["plot_thermal_expansion_with_uncertainty"]
    global_plot.FIGURE_DIR = output
    global_plot.UNCERTAINTY_DIR = UNCERTAINTY
    global_plot.TOTAL_PROJECTION_FILE = ensemble.OUTPUT_DIR / "ensemble_projection_2026_2030.csv"
    global_plot.TOTAL_PROJECTION_SAMPLES = ensemble.OUTPUT_DIR / "ensemble_projection_samples.npz"
    global_plot.DEPTH_PROJECTION_FILE = depth.OUTPUT_DIR / "depth_band_projection_reconciled_2026_2030.csv"
    global_plot.DEPTH_PROJECTION_SAMPLES = depth.OUTPUT_DIR / "depth_band_projection_samples_reconciled.npz"
    global_plot.TAIL_UNCERTAINTY_FILE = tail.OUTPUT_FILE
    basin = m["plot_ocean_basin_historical_partition"]
    basin.FIGURE_DIR = output
    basin.UNCERTAINTY_MEMBER_FILE = UNCERTAINTY / "member_results_basin.csv"
    latitude = m["plot_latitudinal_thermosteric_partition_redesign"]
    latitude.parse_args = lambda: argparse.Namespace(
        central_file=ROOT / "figures/Fig04_latitudinal_15arcsec_annual_main.csv",
        member_file=UNCERTAINTY / "member_results_latitude_10deg.csv",
        output_dir=output,
    )
    return m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "tail", "projection", "figures"), default="all")
    args = parser.parse_args()
    validate()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    modules = configure(output)
    if args.stage in ("all", "tail"):
        print("Updating 2024-2025 uncertainty proxy", flush=True)
        modules["build_tail_uncertainty_proxy"].main()
    if args.stage in ("all", "projection"):
        for name in ("run_global_projection", "run_sparse_bsts_projection",
                     "run_performance_weighted_ensemble", "run_depth_band_ensemble_projection"):
            print("Running " + name, flush=True)
            modules[name].main()
    if args.stage in ("all", "figures"):
        for name in ("plot_thermal_expansion_with_uncertainty",
                     "plot_ocean_basin_historical_partition",
                     "plot_latitudinal_thermosteric_partition_redesign"):
            print("Drawing " + name, flush=True)
            modules[name].main()
    manifest = {
        "uncertainty": str(UNCERTAINTY),
        "central_15arcsec": str(ROOT / "Results/yearly_1940_2025"),
        "stage": args.stage,
        "upper_ocean_members": 11,
        "full_depth_members": 7,
        "salinity": "fixed",
        "tail_2024_2025": "extrapolated proxy, not observed ensemble uncertainty",
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Finished: " + str(output), flush=True)


if __name__ == "__main__":
    main()
