#!/usr/bin/env python3
"""Sum existing 15-arcsec annual maps over six overlapping latitude ranges.

This reads existing thermal-expansion maps; it does not rerun TEOS-10 or
change the original NetCDF files. Results are contributions to global mean
sea level, not mean sea-level change within each latitude range.
"""

from manuscript_paths import workspace_root, input_path
import argparse
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd


ROOT = workspace_root()
DEFAULT_INPUT = ROOT / "Results/yearly_1940_2025"
DEFAULT_OUTPUT = ROOT / "figures/nested_latitude_15arcsec/Fig04_nested_latitude_15arcsec_annual_main.csv"
PAIR = re.compile(r"^(\d{4})_vs_(\d{4})$")
VARIABLE = "thermal_expansion_visual_filled_km3"
RANGES = (
    ("15S-15N", -15.0, 15.0),
    ("30S-30N", -30.0, 30.0),
    ("45S-45N", -45.0, 45.0),
    ("60S-60N", -60.0, 60.0),
    ("75S-75N", -75.0, 75.0),
    ("86S-90N", -86.0, 90.0),
)
DEPTH_BANDS = ("0-700m", "700-2000m", "2000-6000m", "0-6000m")
OUTPUT_COLUMNS = (
    "base_year", "target_year", "latitude_band", "lat_lower_deg",
    "lat_upper_deg", "depth_band", "volume_change_km3",
    "global_contribution_mm",
)


def read_block(variable, start, stop):
    block = variable[start:stop, :]
    if np.ma.isMaskedArray(block):
        return np.asarray(block.filled(0.0), dtype=np.float32)
    return np.nan_to_num(np.asarray(block, dtype=np.float32), nan=0.0)


def sum_rows_by_range(latitudes, row_sums):
    """Sum complete native-resolution rows; lower bound is inclusive."""
    return np.array([
        row_sums[(latitudes >= lower) & (latitudes < upper)].sum(dtype=np.float64)
        for _, lower, upper in RANGES
    ], dtype=np.float64)


def mm_divisor(pair_dir):
    summary = pd.read_csv(pair_dir / "summary_basin.csv")
    if len(summary) != 1:
        raise ValueError("Expected one row in summary_basin.csv: {}".format(pair_dir))
    volume = float(summary.iloc[0]["total_volume_change_km3"])
    rise = float(summary.iloc[0]["global_sea_level_rise_mm"])
    if not np.isfinite(volume) or not np.isfinite(rise) or abs(rise) < 1e-12:
        raise ValueError("Invalid global conversion in {}".format(pair_dir))
    divisor = volume / rise
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Invalid km3/mm divisor in {}".format(pair_dir))
    return divisor


def aggregate_year(target, base, pair_dir, block_rows):
    paths = {suffix: pair_dir / "{}_{}.nc".format(target, suffix)
             for suffix in ("700", "2000", "6000")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))
    datasets = {suffix: nc.Dataset(str(path)) for suffix, path in paths.items()}
    try:
        reference = datasets["6000"]
        latitudes = np.asarray(reference.variables["lat"][:], dtype=np.float64)
        shape = (len(latitudes), len(reference.dimensions["lon"]))
        for suffix, dataset in datasets.items():
            if dataset.variables[VARIABLE].shape != shape:
                raise ValueError("Map shape mismatch: {}".format(paths[suffix]))
            if not np.array_equal(dataset.variables["lat"][:], latitudes):
                raise ValueError("Latitude grid mismatch: {}".format(paths[suffix]))
            if not np.array_equal(dataset.variables["lon"][:], reference.variables["lon"][:]):
                raise ValueError("Longitude grid mismatch: {}".format(paths[suffix]))

        sums = {band: np.zeros(len(RANGES), dtype=np.float64)
                for band in DEPTH_BANDS}
        total_all_latitudes = 0.0
        for start in range(0, shape[0], block_rows):
            stop = min(start + block_rows, shape[0])
            rows = {suffix: read_block(datasets[suffix].variables[VARIABLE], start, stop)
                    .sum(axis=1, dtype=np.float64) for suffix in paths}
            block_lat = latitudes[start:stop]
            block_bands = {
                "0-700m": rows["700"],
                "700-2000m": rows["2000"] - rows["700"],
                "2000-6000m": rows["6000"] - rows["2000"],
                "0-6000m": rows["6000"],
            }
            for band, row_sums in block_bands.items():
                sums[band] += sum_rows_by_range(block_lat, row_sums)
            total_all_latitudes += rows["6000"].sum(dtype=np.float64)

        tolerance = max(1e-5, abs(total_all_latitudes) * 2e-6)
        declared_total = float(getattr(reference, "total_volume_change_km3"))
        if abs(total_all_latitudes - declared_total) > tolerance:
            raise ValueError("NC total mismatch for {}: {:.6g} km3".format(
                target, total_all_latitudes - declared_total))
        depth_total = sums["0-700m"] + sums["700-2000m"] + sums["2000-6000m"]
        if not np.allclose(depth_total, sums["0-6000m"], rtol=2e-6, atol=1e-5):
            raise ValueError("Depth-band closure failed for {}".format(target))
    finally:
        for dataset in datasets.values():
            dataset.close()

    divisor = mm_divisor(pair_dir)
    records = []
    for band in DEPTH_BANDS:
        for index, (label, lower, upper) in enumerate(RANGES):
            volume = float(sums[band][index])
            records.append({
                "base_year": base,
                "target_year": target,
                "latitude_band": label,
                "lat_lower_deg": lower,
                "lat_upper_deg": upper,
                "depth_band": band,
                "volume_change_km3": volume,
                "global_contribution_mm": volume / divisor,
            })
    return records


def completed_years(existing):
    completed = set()
    for year, rows in existing.groupby("target_year"):
        keys = set(zip(rows.latitude_band, rows.depth_band))
        expected = {(region, depth) for region, _, _ in RANGES for depth in DEPTH_BANDS}
        if len(rows) != len(expected) or keys != expected or rows.base_year.nunique() != 1:
            raise ValueError("Incomplete or duplicate records for year {}".format(year))
        completed.add(int(year))
    return completed


def save_atomic(output, data):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    data.to_csv(temporary, index=False, columns=OUTPUT_COLUMNS)
    os.replace(str(temporary), str(output))


def append_year(output, existing, target, base, records):
    updated = pd.concat([existing, pd.DataFrame.from_records(records)], ignore_index=True)
    updated = updated.sort_values(["target_year", "latitude_band", "depth_band"])
    save_atomic(output, updated)
    logging.info("Saved %d vs %d (%d records): %s", target, base, len(records), output)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1941)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent annual processes; try 2 before 4")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.block_rows <= 0 or args.workers <= 0 or args.start_year > args.end_year:
        parser.error("Require positive --block-rows/--workers and start-year <= end-year")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    pairs = []
    for directory in args.input_root.iterdir():
        match = PAIR.match(directory.name)
        if match and directory.is_dir():
            target, base = map(int, match.groups())
            if args.start_year <= target <= args.end_year:
                if base != target - 1:
                    raise ValueError("Not consecutive years: {}".format(directory))
                pairs.append((target, base, directory))
    pairs.sort()
    expected_years = set(range(args.start_year, args.end_year + 1))
    if {target for target, _, _ in pairs} != expected_years:
        raise FileNotFoundError("Missing annual result directories for {}".format(
            sorted(expected_years - {target for target, _, _ in pairs})))

    existing = pd.read_csv(args.output) if args.output.is_file() else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not existing.empty and set(existing.columns) != set(OUTPUT_COLUMNS):
        raise ValueError("Unexpected existing output schema: {}".format(args.output))
    completed = completed_years(existing)
    logging.info("Requested %d years; %d already complete", len(pairs),
                 len(expected_years & completed))
    pending = [(target, base, directory) for target, base, directory in pairs
               if target not in completed]
    if args.workers == 1:
        for target, base, directory in pending:
            records = aggregate_year(target, base, directory, args.block_rows)
            existing = append_year(args.output, existing, target, base, records)
    else:
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(aggregate_year, target, base, directory, args.block_rows):
                       (target, base) for target, base, directory in pending}
            for future in as_completed(futures):
                target, base = futures[future]
                try:
                    records = future.result()
                except Exception:
                    logging.exception("Aggregation failed for %d vs %d", target, base)
                    failures.append(target)
                    continue
                existing = append_year(args.output, existing, target, base, records)
        if failures:
            raise RuntimeError("Failed years (rerun to retry): {}".format(sorted(failures)))
    logging.info("Completed %d requested years", len(pairs))


if __name__ == "__main__":
    main()
