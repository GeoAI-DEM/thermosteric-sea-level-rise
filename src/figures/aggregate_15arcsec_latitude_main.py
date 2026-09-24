#!/usr/bin/env python3
"""Aggregate native 15-arc-second thermosteric maps to latitude bands.

The calculation itself is not repeated.  Each annual output already contains
cumulative 0-700 m, 0-2000 m and 0-6000 m volume-change maps.  This script
streams those maps by latitude blocks and derives non-overlapping depth bands:
0-700 m, 700-2000 m, and 2000-6000 m.  It never substitutes the 1-degree
ensemble mean for the 15-arc-second central estimate.
"""

from __future__ import print_function

import argparse
import logging
import re
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd


RESULTS_ROOT = None
OUTPUT_DEFAULT = Path("outputs") / "Fig04_latitudinal_15arcsec_annual_main.csv"
PAIR_PATTERN = re.compile(r"^(?P<target>\d{4})_vs_(?P<base>\d{4})$")
VARIABLE = "thermal_expansion_visual_filled_km3"
DEPTH_OUTPUTS = (
    ("0-700m", "700"),
    ("0-2000m", "2000"),
    ("0-6000m", "6000"),
)


def latitude_specs():
    specs = []
    for lower in range(-90, 90, 10):
        upper = lower + 10
        if upper <= 0:
            label = "{}S-{}S".format(abs(lower), abs(upper))
        elif lower >= 0:
            label = "{}N-{}N".format(lower, upper)
        else:
            raise ValueError("Latitude band crosses the equator.")
        specs.append((label, float(lower), float(upper)))
    return tuple(specs)


LATITUDE_SPECS = latitude_specs()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="Root containing YYYY_vs_YYYY-1 annual result directories")
    parser.add_argument("--start-year", type=int, default=1941)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--block-rows", type=int, default=128, help="NetCDF rows read at once")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--overwrite", action="store_true", help="Recalculate years already present in output")
    return parser.parse_args()


def logger():
    log = logging.getLogger("latitude_15arcsec_aggregate")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(handler)
    return log


def find_pair_directories(start_year, end_year):
    pairs = []
    for path in RESULTS_ROOT.iterdir():
        match = PAIR_PATTERN.match(path.name)
        if match is None:
            continue
        target = int(match.group("target"))
        if start_year <= target <= end_year:
            pairs.append((target, int(match.group("base")), path))
    return sorted(pairs)


def volume_to_mm_conversion(pair_dir):
    summary_file = pair_dir / "summary_basin.csv"
    summary = pd.read_csv(summary_file)
    if len(summary) != 1:
        raise ValueError("Expected one row in {}".format(summary_file))
    row = summary.iloc[0]
    volume = float(row["total_volume_change_km3"])
    sea_level = float(row["global_sea_level_rise_mm"])
    if not np.isfinite(volume) or not np.isfinite(sea_level) or abs(sea_level) < 1.0e-12:
        raise ValueError("Invalid global conversion in {}".format(summary_file))
    return volume / sea_level


def read_block(variable, start, stop):
    """Return a numeric array with land and missing cells set to zero."""
    block = variable[start:stop, :]
    if np.ma.isMaskedArray(block):
        return np.asarray(block.filled(0.0), dtype=np.float32)
    return np.nan_to_num(np.asarray(block, dtype=np.float32), nan=0.0)


def latitude_index(latitudes):
    index = np.floor((np.asarray(latitudes, dtype=float) + 90.0) / 10.0).astype(int)
    return np.clip(index, 0, len(LATITUDE_SPECS) - 1)


def aggregate_year(target_year, base_year, pair_dir, block_rows, log):
    paths = {name: pair_dir / "{}_{}.nc".format(target_year, suffix) for name, suffix in DEPTH_OUTPUTS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing NetCDF output(s): {}".format("; ".join(missing)))

    datasets = {name: nc.Dataset(path) for name, path in paths.items()}
    try:
        reference = datasets["0-6000m"]
        latitudes = np.asarray(reference.variables["lat"][:], dtype=float)
        expected_shape = (len(latitudes), len(reference.dimensions["lon"]))
        for name, dataset in datasets.items():
            variable = dataset.variables.get(VARIABLE)
            if variable is None or variable.shape != expected_shape:
                raise ValueError("Unexpected map layout in {}".format(paths[name]))

        sums = {band: np.zeros(len(LATITUDE_SPECS), dtype=np.float64) for band in ("0-700m", "700-2000m", "2000-6000m", "0-6000m")}
        for start in range(0, expected_shape[0], block_rows):
            stop = min(start + block_rows, expected_shape[0])
            depth_700 = read_block(datasets["0-700m"].variables[VARIABLE], start, stop)
            depth_2000 = read_block(datasets["0-2000m"].variables[VARIABLE], start, stop)
            depth_6000 = read_block(datasets["0-6000m"].variables[VARIABLE], start, stop)
            group = latitude_index(latitudes[start:stop])
            # Summing each cumulative field first avoids allocating two extra
            # 15-arc-second difference arrays for every latitude block.
            sum_700 = depth_700.sum(axis=1, dtype=np.float64)
            sum_2000 = depth_2000.sum(axis=1, dtype=np.float64)
            sum_6000 = depth_6000.sum(axis=1, dtype=np.float64)
            block_sums = {
                "0-700m": sum_700,
                "700-2000m": sum_2000 - sum_700,
                "2000-6000m": sum_6000 - sum_2000,
                "0-6000m": sum_6000,
            }
            for band, values in block_sums.items():
                sums[band] += np.bincount(group, weights=values, minlength=len(LATITUDE_SPECS))
        total_from_bands = sums["0-700m"] + sums["700-2000m"] + sums["2000-6000m"]
        mismatch = float(np.abs(total_from_bands.sum() - sums["0-6000m"].sum()))
        tolerance = max(1.0e-5, abs(float(sums["0-6000m"].sum())) * 2.0e-6)
        if mismatch > tolerance:
            raise ValueError("Depth-band closure failed for {}: {:.6g} km3".format(target_year, mismatch))
        declared_total = float(getattr(reference, "total_volume_change_km3"))
        declared_mismatch = float(abs(sums["0-6000m"].sum() - declared_total))
        if declared_mismatch > tolerance:
            raise ValueError("NetCDF total mismatch for {}: {:.6g} km3".format(target_year, declared_mismatch))
    finally:
        for dataset in datasets.values():
            dataset.close()

    conversion = volume_to_mm_conversion(pair_dir)
    records = []
    for band, values in sums.items():
        for index, (label, lower, upper) in enumerate(LATITUDE_SPECS):
            records.append({
                "base_year": base_year,
                "target_year": target_year,
                "latitude_band": label,
                "lat_lower_deg": lower,
                "lat_upper_deg": upper,
                "depth_band": band,
                "volume_change_km3": float(values[index]),
                "global_contribution_mm": float(values[index] / conversion),
            })
    log.info(
        "%s 相对于 %s 完成；0-6000 m = %.6f km3，闭合误差 %.3g km3",
        target_year,
        base_year,
        sums["0-6000m"].sum(),
        mismatch,
    )
    return records


def main():
    global RESULTS_ROOT
    args = parse_args()
    if args.block_rows <= 0:
        raise ValueError("--block-rows must be positive")
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    RESULTS_ROOT = args.results_root.resolve()
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError("Annual results directory not found: {}".format(RESULTS_ROOT))
    log = logger()
    pairs = find_pair_directories(args.start_year, args.end_year)
    if not pairs:
        raise FileNotFoundError("No annual result directories found in requested range.")

    existing = pd.DataFrame()
    if args.output.is_file() and not args.overwrite:
        existing = pd.read_csv(args.output)
    completed = set(existing["target_year"].unique()) if not existing.empty else set()
    pending = [pair for pair in pairs if pair[0] not in completed or args.overwrite]
    log.info("请求年份 %s-%s；待聚合 %s 个年度", args.start_year, args.end_year, len(pending))
    fresh = []
    for target, base, pair_dir in pending:
        fresh.extend(aggregate_year(target, base, pair_dir, args.block_rows, log))

    if args.overwrite and not existing.empty:
        requested_years = {target for target, _, _ in pairs}
        existing = existing[~existing["target_year"].isin(requested_years)]
    output = pd.concat([existing, pd.DataFrame(fresh)], ignore_index=True)
    output = output.sort_values(["target_year", "latitude_band", "depth_band"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    log.info("已写入 %s 条纬度带主值记录: %s", len(output), args.output)


if __name__ == "__main__":
    main()
