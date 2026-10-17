#!/usr/bin/env python3
"""Pilot temperature-ensemble uncertainty propagation on 1-degree GEBCO volumes.

This script uses IAPv4-LE temperature anomalies plus the matching monthly
climatology to reconstruct absolute in-situ temperature. Annual salinity is
held fixed to the central IAP salinity product. It intentionally writes only
global thermal-expansion totals, not maps or member NetCDF files.
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gsw
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, label
from scipy.spatial.distance import cdist


ANNUAL_DIR = None
VOLUME_FILE = None
TEMP_BASE_URL = "http://www.ocean.iap.ac.cn/ftp/ensemble/Ensemble_data_temp_all"
CLIM_BASE_URL = "http://www.ocean.iap.ac.cn/ftp/ensemble/Climatology_2000m_20052020Clim_2023new2"

# Fixed preliminary sample: four available members from each climatology class.
PILOT_MEMBERS = (
    "11111114", "11222114", "12123116", "13322139",
    "21111114", "21323135", "22222329", "23323139",
    "31111113", "31313337", "32223138", "33323335",
)

GROUPS = (
    ("0-700m", 0.0, 700.0),
    ("700-2000m", 700.0, 2000.0),
    ("2000-6000m", 2000.0, 6000.1),
    ("0-6000m", 0.0, 6000.1),
)


def configure_logger(log_file):
    logger = logging.getLogger("iapv4_le_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def normalize_longitude(lon):
    return np.mod(np.asarray(lon, dtype=np.float64) + 180.0, 360.0) - 180.0


def align_member_longitude(values):
    """Align the archive's 0.5-359.5 E columns with the -180-to-180 volume grid."""
    source_lon = np.arange(0.5, 360.0, 1.0)
    if values.shape[-1] != source_lon.size:
        raise ValueError(f"Unexpected IAP member longitude count: {values.shape[-1]}")
    order = np.argsort(normalize_longitude(source_lon))
    return np.take(values, order, axis=-1)


def fill_island_region(data, island_mask):
    filled = data.copy()
    expanded = np.zeros_like(island_mask)
    rows, cols = np.where(island_mask)
    for row, col in zip(rows, cols):
        expanded[max(0, row - 3):row + 4, max(0, col - 3):col + 4] = True

    search = expanded & ~island_mask & ~np.isnan(data)
    if not np.any(search):
        filled[island_mask] = np.nanmean(data)
        return filled

    valid_coords = np.column_stack(np.where(search))
    valid_values = data[search]
    fill_coords = np.column_stack(np.where(island_mask))
    distances = cdist(fill_coords, valid_coords)
    for index, (row, col) in enumerate(fill_coords):
        nearest = np.argsort(distances[index])[:4]
        weights = 1.0 / (distances[index][nearest] + 1e-8)
        weights /= weights.sum()
        filled[row, col] = np.sum(weights * valid_values[nearest])
    return filled


def fill_small_enclosed_islands(data, max_island_size=5, min_ocean_ratio=0.8):
    filled = np.ascontiguousarray(data).copy()
    nan_mask = np.isnan(data)
    labels, count = label(nan_mask, structure=np.ones((3, 3)))
    for island_id in range(1, count + 1):
        island = labels == island_id
        if island.sum() > max_island_size:
            continue
        boundary = binary_dilation(island, iterations=1) & ~island
        if boundary.sum() == 0:
            continue
        if np.sum(~np.isnan(data[boundary])) / boundary.sum() >= min_ocean_ratio:
            filled = fill_island_region(filled, island)
    return filled


def fix_missing_layers(values):
    fixed = values.copy()
    valid_percent = np.mean(np.isfinite(values), axis=(1, 2)) * 100.0
    valid_layers = np.where(valid_percent >= 1.0)[0]
    for depth_index in np.where(valid_percent < 1.0)[0]:
        nearest = valid_layers[np.argmin(np.abs(valid_layers - depth_index))]
        fixed[depth_index] = fixed[nearest]
    return fixed


def regrid_to_target(values, source_lat, source_lon, target_lat, target_lon):
    if np.array_equal(source_lat, target_lat) and np.array_equal(source_lon, target_lon):
        return values

    clipped_lat = np.clip(target_lat, source_lat[0], source_lat[-1])
    clipped_lon = np.clip(target_lon, source_lon[0], source_lon[-1])
    lon_2d, lat_2d = np.meshgrid(clipped_lon, clipped_lat, indexing="xy")
    points = np.column_stack((lat_2d.ravel(), lon_2d.ravel()))
    output = np.empty((values.shape[0], target_lat.size, target_lon.size), dtype=np.float64)
    for depth_index, layer_values in enumerate(values):
        interpolator = RegularGridInterpolator(
            (source_lat, source_lon), layer_values, method="linear", bounds_error=False, fill_value=None
        )
        output[depth_index] = interpolator(points).reshape(target_lat.size, target_lon.size)
    return output


def read_annual_variable(year, variable_name):
    filename = ANNUAL_DIR / f"Annual_{'Temp' if variable_name == 'temp' else 'Salinity'}_{year}.nc"
    with nc.Dataset(filename) as dataset:
        variable = dataset.variables[variable_name]
        values = np.ma.filled(variable[:].astype(np.float64), np.nan)
        values[np.isclose(values, 999.0)] = np.nan
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = normalize_longitude(dataset.variables["lon"][:])
        depth = np.asarray(dataset.variables["depth_std"][:], dtype=np.float64)

    if lat[0] > lat[-1]:
        lat = lat[::-1].copy()
        values = values[::-1, :, :].copy()
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    values = values[:, lon_order, :].copy()
    return np.transpose(values, (2, 0, 1)), depth, lat, lon


def read_volume_data():
    with nc.Dataset(VOLUME_FILE) as dataset:
        volumes = np.asarray(dataset.variables["layer_ocean_volume_m3"][:], dtype=np.float64)
        area = float(np.sum(dataset.variables["ocean_area_m2"][:], dtype=np.float64))
        depth = np.asarray(dataset.variables["depth_std"][:], dtype=np.float64)
        lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
    return volumes, area, depth, lat, lon


def prepare_fixed_salinity(years, target_depth, target_lat, target_lon):
    salinity = {}
    for year in years:
        values, depth, lat, lon = read_annual_variable(year, "salinity")
        if not np.array_equal(depth, target_depth):
            raise ValueError(f"Salinity depth coordinates differ for {year}")
        values = fix_missing_layers(values)
        for depth_index in range(values.shape[0]):
            values[depth_index] = fill_small_enclosed_islands(values[depth_index])
        salinity[year] = regrid_to_target(values, lat, lon, target_lat, target_lon)
    return salinity


def month_index(year, month):
    # IAPv4-LE index 1 is January 1940 and index 1008 is December 2023.
    return 1 + (year - 1940) * 12 + (month - 1)


def climatology_spec(member, month):
    climatology_index = member[0]
    if climatology_index == "1":
        filename = f"Climatology_analysis_ENOI2024_final_v1_{month}_len_25_Mean_3_t_1131111.nc"
        return filename, "Tclim_filling"
    if climatology_index == "2":
        return f"Climatology_analysis_standarlise_WAGHC_{month}.nc", "ohc_analysis"
    if climatology_index == "3":
        return f"Climatology_analysis_standarlise_WOA_{month}.nc", "ohc_analysis"
    raise ValueError(f"Unknown climatology index in member {member}")


def download_file(url, destination, timeout=300):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "IAPv4-LE-1deg-pilot/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(partial, "wb") as output:
                shutil.copyfileobj(response, output)
        if partial.stat().st_size == 0:
            raise RuntimeError("response is empty")
        partial.replace(destination)
    except Exception as error:
        if partial.exists():
            partial.unlink()
        raise RuntimeError(f"Download failed for {url}: {error}") from None
    return destination


def prepare_climatology_cache(cache_dir, logger):
    output = {}
    for index, member in (("1", "11111114"), ("2", "21111114"), ("3", "31111113")):
        output[index] = []
        for month in range(1, 13):
            filename, variable_name = climatology_spec(member, month)
            destination = cache_dir / "climatology" / filename
            if not destination.exists():
                logger.info("Downloading climatology %s", filename)
                download_file(f"{CLIM_BASE_URL}/{filename}", destination)
            output[index].append((destination, variable_name))
    return output


def load_climatology_month(path, variable_name):
    with nc.Dataset(path) as dataset:
        values = np.ma.filled(dataset.variables[variable_name][:].astype(np.float64), np.nan)
    return values


def member_monthly_file_map(member, cache_dir):
    """Map IAP month indices to the member's actual filenames.

    Different IAPv4-LE members use different middle filename identifiers, so
    the numeric identifier cannot be hard-coded as zero.
    """
    mapping_file = cache_dir / "member_file_maps" / f"{member}.json"
    if mapping_file.exists():
        with open(mapping_file, encoding="utf-8") as source:
            return {int(index): filename for index, filename in json.load(source).items()}

    url = f"{TEMP_BASE_URL}/{member}/"
    request = urllib.request.Request(url, headers={"User-Agent": "IAPv4-LE-1deg-pilot/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            page = response.read().decode("utf-8", errors="replace")
    except Exception as error:
        raise RuntimeError(f"Cannot list IAPv4-LE member {member}: {error}") from None

    pattern = re.compile(
        r'href="(assigrid_temperature_[^_]+_tm_([0-9]+)_matlab_data_container\.nc)"'
    )
    mapping = {int(index): filename for filename, index in pattern.findall(page)}
    expected = set(range(1, 1009))
    missing = sorted(expected - set(mapping))
    if missing:
        raise RuntimeError(
            f"IAPv4-LE member {member} lacks {len(missing)} expected monthly files; "
            f"first missing index: {missing[0]}"
        )
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_file, "w", encoding="utf-8") as output:
        json.dump({str(index): mapping[index] for index in sorted(mapping)}, output, ensure_ascii=True)
    return mapping


def read_member_annual_temperature(member, year, climatology_files, member_file_map, raw_cache_dir, keep_member_files):
    annual_sum = None
    for month in range(1, 13):
        index = month_index(year, month)
        filename = member_file_map[index]
        local_file = raw_cache_dir / member / filename
        url = f"{TEMP_BASE_URL}/{member}/{filename}"
        download_file(url, local_file)
        try:
            with nc.Dataset(local_file) as dataset:
                anomaly = np.ma.filled(dataset.variables["temp"][:].astype(np.float64), np.nan)
                anomaly[np.isclose(anomaly, -9999.0)] = np.nan
                anomaly /= 10000.0
            climate_path, climate_variable = climatology_files[month - 1]
            absolute_temperature = anomaly + load_climatology_month(climate_path, climate_variable)
            if annual_sum is None:
                annual_sum = np.zeros_like(absolute_temperature, dtype=np.float64)
            annual_sum += absolute_temperature
        finally:
            if not keep_member_files and local_file.exists():
                local_file.unlink()
    member_dir = raw_cache_dir / member
    if not keep_member_files and member_dir.exists():
        try:
            member_dir.rmdir()
        except OSError:
            pass
    return align_member_longitude(annual_sum / 12.0)


def calculate_layer_changes(temp_base, temp_target, salt_base, salt_target, depth, volumes):
    changes = np.zeros(depth.size, dtype=np.float64)
    for depth_index, depth_m in enumerate(depth):
        t0 = temp_base[depth_index]
        t1 = temp_target[depth_index]
        sa0 = salt_base[depth_index]
        sa1 = salt_target[depth_index]
        valid = (
            np.isfinite(t0) & np.isfinite(t1) & np.isfinite(sa0) & np.isfinite(sa1)
            & (volumes[depth_index] > 0.0)
        )
        if not np.any(valid):
            continue
        ct0 = gsw.CT_from_t(sa0[valid], t0[valid], depth_m)
        ct1 = gsw.CT_from_t(sa1[valid], t1[valid], depth_m)
        alpha0 = gsw.alpha(sa0[valid], ct0, depth_m)
        alpha1 = gsw.alpha(sa1[valid], ct1, depth_m)
        changes[depth_index] = np.sum(
            volumes[depth_index][valid] * ((alpha0 + alpha1) / 2.0) * (ct1 - ct0),
            dtype=np.float64,
        )
    return changes


def grouped_results(changes, depth, ocean_area):
    results = {}
    for name, lower, upper in GROUPS:
        amount_m3 = float(np.sum(changes[(depth >= lower) & (depth < upper)], dtype=np.float64))
        results[name] = {
            "volume_change_km3": amount_m3 / 1.0e9,
            "sea_level_mm": amount_m3 / ocean_area * 1000.0,
        }
    return results


def member_worker(member, years, cache_dir_str, keep_member_files):
    cache_dir = Path(cache_dir_str)
    volumes, ocean_area, depth, target_lat, target_lon = read_volume_data()
    salinity = prepare_fixed_salinity(years, depth, target_lat, target_lon)
    climatology_files = []
    for month in range(1, 13):
        filename, variable_name = climatology_spec(member, month)
        climatology_files.append((cache_dir / "climatology" / filename, variable_name))
    member_file_map = member_monthly_file_map(member, cache_dir)

    annual_temperature = {
        year: read_member_annual_temperature(
            member, year, climatology_files, member_file_map, cache_dir / "member_monthly", keep_member_files
        )
        for year in years
    }

    records = []
    for base_year, target_year in zip(years[:-1], years[1:]):
        changes = calculate_layer_changes(
            annual_temperature[base_year], annual_temperature[target_year],
            salinity[base_year], salinity[target_year], depth, volumes,
        )
        for band, result in grouped_results(changes, depth, ocean_area).items():
            records.append({
                "member": member,
                "climatology_index": member[0],
                "base_year": base_year,
                "target_year": target_year,
                "depth_band": band,
                **result,
            })
    return records


def central_results(years, volumes, ocean_area, depth, target_lat, target_lon):
    temperature = {}
    for year in years:
        values, temp_depth, temp_lat, temp_lon = read_annual_variable(year, "temp")
        if not np.array_equal(temp_depth, depth):
            raise ValueError(f"Temperature depth coordinates differ for {year}")
        values = fix_missing_layers(values)
        for depth_index in range(values.shape[0]):
            values[depth_index] = fill_small_enclosed_islands(values[depth_index])
        temperature[year] = regrid_to_target(values, temp_lat, temp_lon, target_lat, target_lon)
    salinity = prepare_fixed_salinity(years, depth, target_lat, target_lon)

    records = []
    for base_year, target_year in zip(years[:-1], years[1:]):
        changes = calculate_layer_changes(
            temperature[base_year], temperature[target_year],
            salinity[base_year], salinity[target_year], depth, volumes,
        )
        for band, result in grouped_results(changes, depth, ocean_area).items():
            records.append({
                "base_year": base_year,
                "target_year": target_year,
                "depth_band": band,
                **result,
            })
    return records


def percentile(values, level):
    return float(np.percentile(np.asarray(values, dtype=np.float64), level))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(member_rows, central_rows):
    central_by_key = {
        (row["base_year"], row["target_year"], row["depth_band"]): row
        for row in central_rows
    }
    grouped = {}
    for row in member_rows:
        key = (row["base_year"], row["target_year"], row["depth_band"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for key in sorted(grouped):
        rows = grouped[key]
        member_values = [row["sea_level_mm"] for row in rows]
        member_median = float(np.median(member_values))
        deviations = np.asarray(member_values, dtype=np.float64) - member_median
        central = central_by_key[key]["sea_level_mm"]
        summary_rows.append({
            "base_year": key[0],
            "target_year": key[1],
            "depth_band": key[2],
            "member_count": len(member_values),
            "central_1deg_mm": central,
            "ensemble_mean_mm": float(np.mean(member_values)),
            "ensemble_median_mm": member_median,
            "ensemble_sd_mm": float(np.std(member_values, ddof=1)),
            "ensemble_p05_mm": percentile(member_values, 5.0),
            "ensemble_p95_mm": percentile(member_values, 95.0),
            "temperature_uncertainty_p05_mm": percentile(deviations, 5.0),
            "temperature_uncertainty_p95_mm": percentile(deviations, 95.0),
            "central_plus_p05_mm": central + percentile(deviations, 5.0),
            "central_plus_p95_mm": central + percentile(deviations, 95.0),
        })
    return summary_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-dir", type=Path, required=True, help="Directory containing annual IAP T/S NetCDF files")
    parser.add_argument("--volume-file", type=Path, required=True, help="1-degree GEBCO layer-volume NetCDF")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "iapv4_le_1deg_pilot")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent ensemble members; default: 3")
    parser.add_argument("--keep-member-files", action="store_true", help="Retain downloaded monthly member files for debugging")
    return parser.parse_args()


def main():
    args = parse_args()
    global ANNUAL_DIR, VOLUME_FILE
    ANNUAL_DIR = args.annual_dir.resolve()
    VOLUME_FILE = args.volume_file.resolve()
    if not ANNUAL_DIR.is_dir():
        raise FileNotFoundError(f"Annual IAP directory not found: {ANNUAL_DIR}")
    if not VOLUME_FILE.is_file():
        raise FileNotFoundError(f"1-degree volume file not found: {VOLUME_FILE}")
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(output_dir / "pilot.log")
    years = (2020, 2021, 2022, 2023)

    logger.info("Pilot periods: %s", ", ".join(f"{year}_vs_{year - 1}" for year in years[1:]))
    logger.info("Pilot members (%d): %s", len(PILOT_MEMBERS), ", ".join(PILOT_MEMBERS))
    logger.info("Preparing climatology cache")
    prepare_climatology_cache(cache_dir, logger)

    volumes, ocean_area, depth, target_lat, target_lon = read_volume_data()
    logger.info("Computing central 1-degree results with fixed annual salinity")
    central_rows = central_results(years, volumes, ocean_area, depth, target_lat, target_lon)
    write_csv(
        output_dir / "central_1deg_results.csv",
        central_rows,
        ["base_year", "target_year", "depth_band", "volume_change_km3", "sea_level_mm"],
    )

    start_time = time.monotonic()
    member_rows = []
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(member_worker, member, years, str(cache_dir), args.keep_member_files): member
            for member in PILOT_MEMBERS
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            member = futures[future]
            try:
                rows = future.result()
                member_rows.extend(rows)
                logger.info("Completed member %s (%d/%d)", member, completed, len(PILOT_MEMBERS))
            except Exception as error:
                failures.append({"member": member, "error": repr(error)})
                logger.exception("Member %s failed", member)

    write_csv(
        output_dir / "member_results.csv",
        member_rows,
        ["member", "climatology_index", "base_year", "target_year", "depth_band", "volume_change_km3", "sea_level_mm"],
    )
    summary_rows = summarize(member_rows, central_rows)
    write_csv(
        output_dir / "uncertainty_summary.csv",
        summary_rows,
        [
            "base_year", "target_year", "depth_band", "member_count", "central_1deg_mm",
            "ensemble_mean_mm", "ensemble_median_mm", "ensemble_sd_mm", "ensemble_p05_mm",
            "ensemble_p95_mm", "temperature_uncertainty_p05_mm", "temperature_uncertainty_p95_mm",
            "central_plus_p05_mm", "central_plus_p95_mm",
        ],
    )
    metadata = {
        "purpose": "preliminary pilot only; not a final publication uncertainty interval",
        "temperature_source": "IAPv4-LE temperature anomaly members plus matching monthly climatology",
        "salinity_treatment": "fixed central annual IAP salinity for each year",
        "thermodynamic_method": "direct TEOS-10 CT_from_t and alpha propagation",
        "geometry": "15-arcsecond GEBCO seawater volume aggregated to one-degree cells",
        "periods": [f"{year}_vs_{year - 1}" for year in years[1:]],
        "members": list(PILOT_MEMBERS),
        "member_count_requested": len(PILOT_MEMBERS),
        "member_count_completed": len(PILOT_MEMBERS) - len(failures),
        "failures": failures,
        "elapsed_minutes": (time.monotonic() - start_time) / 60.0,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as output:
        json.dump(metadata, output, ensure_ascii=True, indent=2)
    logger.info("Pilot completed in %.2f minutes; successful members: %d", metadata["elapsed_minutes"], metadata["member_count_completed"])
    if failures:
        logger.warning("Some members failed; inspect metadata.json before interpreting the summary")


if __name__ == "__main__":
    main()
