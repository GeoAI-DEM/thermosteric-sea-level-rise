#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate a 15-arcsecond GEBCO DEM into 1-degree layer-volume weights.

Only bathymetry geometry is calculated here: no temperature, salinity, TEOS-10,
or thermal expansion is evaluated.  The layer formula deliberately matches
``TEOS-10V4_beifen_Wutian.py``:

    V_pixel = A_pixel * min(thickness, max(seafloor_depth - layer_depth, 0))

for pixels with negative GEBCO elevation only.  It uses the same spherical
Earth radius (6,371,000 m), pixel-centre latitude correction, and standard
depth/thickness definition as that script.

The DEM is read in one-degree latitude strips.  Within every strip, a
depth-by-longitude weighted histogram preserves the 15-arcsecond depth and
area information while avoiding a 15-arcsecond 3-D volume array.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
from osgeo import gdal
from tqdm import tqdm


EARTH_RADIUS_M = 6_371_000.0


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gebco_1deg_volume")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(output_dir / "gebco_1deg_volume.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def normalise_longitude(lon: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(lon, dtype=np.float64) + 180.0, 360.0) - 180.0


def get_coordinate(dataset: nc.Dataset, candidates: tuple[str, ...]) -> np.ndarray:
    for name in candidates:
        if name in dataset.variables:
            return np.asarray(dataset.variables[name][:], dtype=np.float64)
    raise KeyError(f"Missing coordinate; tried {candidates}, available={list(dataset.variables)}")


def read_target_grid(annual_temp_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the annual main T/S grid as the definitive 1-degree grid definition."""
    with nc.Dataset(annual_temp_file, "r") as dataset:
        lat = get_coordinate(dataset, ("lat", "latitude", "y"))
        lon = get_coordinate(dataset, ("lon", "longitude", "x"))
        depth = get_coordinate(dataset, ("depth_std", "depth", "z", "level", "lev"))
    return np.sort(lat), np.sort(normalise_longitude(lon)), np.sort(depth)


def layer_thicknesses(depths: np.ndarray) -> np.ndarray:
    """Match the current TEOS-10V4 layer-thickness implementation exactly."""
    if depths.size < 2:
        raise ValueError("At least two standard depths are required")
    thickness = np.empty(depths.size, dtype=np.float64)
    thickness[0] = depths[1] - depths[0]
    thickness[-1] = depths[-1] - depths[-2]
    thickness[1:-1] = (depths[2:] - depths[:-2]) / 2.0
    if np.any(thickness <= 0):
        raise ValueError("Depth levels must be strictly increasing")
    return thickness


def validate_raster(
    dataset: gdal.Dataset,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> tuple[int, int, int, int, tuple[float, float, float, float, float, float]]:
    """Validate that the source is a global, regularly gridded DEM nested in 1-degree cells."""
    geotransform = dataset.GetGeoTransform()
    width, height = dataset.RasterXSize, dataset.RasterYSize
    lon_res = abs(geotransform[1])
    lat_res = abs(geotransform[5])
    pixels_per_lon_degree = int(round(1.0 / lon_res))
    pixels_per_lat_degree = int(round(1.0 / lat_res))

    if not np.isclose(lon_res * pixels_per_lon_degree, 1.0, atol=1.0e-10):
        raise ValueError(f"Longitude resolution {lon_res} is not an integer subdivision of 1 degree")
    if not np.isclose(lat_res * pixels_per_lat_degree, 1.0, atol=1.0e-10):
        raise ValueError(f"Latitude resolution {lat_res} is not an integer subdivision of 1 degree")
    if pixels_per_lon_degree != pixels_per_lat_degree:
        raise ValueError("Longitude and latitude pixel counts per degree differ")
    if width != target_lon.size * pixels_per_lon_degree:
        raise ValueError(f"DEM width {width} does not nest in target longitude grid {target_lon.size}")
    if height != target_lat.size * pixels_per_lat_degree:
        raise ValueError(f"DEM height {height} does not nest in target latitude grid {target_lat.size}")
    if not (np.isclose(geotransform[0], -180.0) and np.isclose(geotransform[3], 90.0)):
        raise ValueError(f"Expected global DEM origin (-180, 90), got ({geotransform[0]}, {geotransform[3]})")
    if geotransform[5] >= 0:
        raise ValueError("Expected north-to-south raster rows with a negative latitude resolution")
    return width, height, pixels_per_lon_degree, pixels_per_lat_degree, geotransform


def weighted_depth_histogram(
    bathymetry: np.ndarray,
    row_areas_m2: np.ndarray,
    pixels_per_degree: int,
    n_lon: int,
    depth_bin_count: int,
    nodata: float | None,
) -> tuple[np.ndarray, int]:
    """Return area weighted by exact water depth for each target longitude cell.

    The returned array has shape ``[n_lon, depth_bin_count]`` and units m2.
    Bin ``h`` contains the sum of 15-arcsecond pixel areas whose water depth is
    exactly ``h`` metres.  The source GEBCO band is Int16, so preserving integer
    metre depths is lossless for this input.
    """
    ocean = bathymetry < 0
    if nodata is not None and np.isfinite(nodata):
        ocean &= bathymetry != nodata

    water_depth = np.where(ocean, -bathymetry, 0).astype(np.int32, copy=False)
    if water_depth.max(initial=0) >= depth_bin_count:
        raise ValueError("DEM water depth exceeds the allocated histogram range")

    # Every consecutive ``pixels_per_degree`` columns belongs to one 1-degree cell.
    lon_offsets = np.repeat(
        np.arange(n_lon, dtype=np.int32) * depth_bin_count,
        pixels_per_degree,
    )
    histogram_index = water_depth + lon_offsets[np.newaxis, :]
    row_weights = np.broadcast_to(
        np.asarray(row_areas_m2, dtype=np.float64)[:, np.newaxis],
        bathymetry.shape,
    )
    valid = water_depth > 0

    histogram = np.bincount(
        histogram_index[valid],
        weights=row_weights[valid],
        minlength=n_lon * depth_bin_count,
    ).reshape(n_lon, depth_bin_count)
    return histogram, int(np.count_nonzero(valid))


def layer_volumes_from_histogram(
    histogram_m2: np.ndarray,
    depths_m: np.ndarray,
    thicknesses_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact 15-arcsecond-derived volume for every standard depth layer."""
    n_lon, depth_bin_count = histogram_m2.shape
    depth_axis = np.arange(depth_bin_count, dtype=np.float64)
    depth_area_hist = histogram_m2 * depth_axis[np.newaxis, :]

    # ``area_ge[:, h]`` is the 15-arcsecond ocean area with H >= h metres.
    area_ge = np.empty((n_lon, depth_bin_count + 1), dtype=np.float64)
    depth_area_ge = np.empty((n_lon, depth_bin_count + 1), dtype=np.float64)
    area_ge[:, :-1] = np.cumsum(histogram_m2[:, ::-1], axis=1)[:, ::-1]
    depth_area_ge[:, :-1] = np.cumsum(depth_area_hist[:, ::-1], axis=1)[:, ::-1]
    area_ge[:, -1] = 0.0
    depth_area_ge[:, -1] = 0.0

    volumes = np.zeros((depths_m.size, n_lon), dtype=np.float64)
    for index, (depth, thickness) in enumerate(zip(depths_m, thicknesses_m)):
        # GEBCO depths are integer metres.  These bounds implement H > depth and
        # H >= depth + thickness without approximating individual pixel thickness.
        lower = int(math.floor(depth)) + 1
        high = int(math.ceil(depth + thickness))
        lower = max(0, min(lower, depth_bin_count))
        high = max(lower, min(high, depth_bin_count))

        partial_area = area_ge[:, lower] - area_ge[:, high]
        partial_depth_area = depth_area_ge[:, lower] - depth_area_ge[:, high]
        volumes[index, :] = (
            partial_depth_area - depth * partial_area + thickness * area_ge[:, high]
        )

    ocean_area = histogram_m2.sum(axis=1)
    full_water_column_volume = depth_area_hist.sum(axis=1)
    return volumes, ocean_area, full_water_column_volume


def calculate_group_volume(layer_volume_m3: np.ndarray, depths_m: np.ndarray, lower: float, upper: float) -> float:
    """Summarise existing model layers by depth-centre membership for reporting."""
    if lower == 0.0:
        mask = depths_m <= upper
    else:
        mask = (depths_m > lower) & (depths_m <= upper)
    return float(np.sum(layer_volume_m3[mask], dtype=np.float64))


def write_netcdf(
    path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    depths: np.ndarray,
    thicknesses: np.ndarray,
    layer_volume_m3: np.ndarray,
    ocean_area_m2: np.ndarray,
    full_water_column_volume_m3: np.ndarray,
    dem_path: Path,
    geotransform: tuple[float, float, float, float, float, float],
    ocean_pixels: int,
) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")

    with nc.Dataset(temporary, "w", format="NETCDF4") as dataset:
        dataset.createDimension("depth_std", depths.size)
        dataset.createDimension("lat", lat.size)
        dataset.createDimension("lon", lon.size)

        depth_var = dataset.createVariable("depth_std", "f4", ("depth_std",))
        lat_var = dataset.createVariable("lat", "f4", ("lat",))
        lon_var = dataset.createVariable("lon", "f4", ("lon",))
        thickness_var = dataset.createVariable("layer_thickness_m", "f4", ("depth_std",))
        depth_var[:] = depths.astype(np.float32)
        lat_var[:] = lat.astype(np.float32)
        lon_var[:] = lon.astype(np.float32)
        thickness_var[:] = thicknesses.astype(np.float32)
        depth_var.units = "m"
        depth_var.positive = "down"
        depth_var.standard_name = "depth"
        lat_var.units = "degrees_north"
        lon_var.units = "degrees_east"
        lat_var.standard_name = "latitude"
        lon_var.standard_name = "longitude"
        thickness_var.units = "m"
        thickness_var.long_name = "Layer thickness matching TEOS-10V4 volume calculation"

        layer_var = dataset.createVariable(
            "layer_ocean_volume_m3",
            "f8",
            ("depth_std", "lat", "lon"),
            zlib=True,
            complevel=4,
            shuffle=True,
            chunksizes=(1, min(30, lat.size), min(90, lon.size)),
        )
        layer_var[:] = layer_volume_m3
        layer_var.units = "m3"
        layer_var.long_name = "15-arcsecond GEBCO-derived seawater volume for each thermal-expansion layer"
        layer_var.layer_formula = "pixel_area * min(layer_thickness, max(seafloor_depth - layer_depth, 0))"

        area_var = dataset.createVariable(
            "ocean_area_m2", "f8", ("lat", "lon"), zlib=True, complevel=4, shuffle=True
        )
        area_var[:] = ocean_area_m2
        area_var.units = "m2"
        area_var.long_name = "15-arcsecond GEBCO ocean area aggregated to 1-degree cell"

        full_volume_var = dataset.createVariable(
            "full_water_column_volume_m3", "f8", ("lat", "lon"), zlib=True, complevel=4, shuffle=True
        )
        full_volume_var[:] = full_water_column_volume_m3
        full_volume_var.units = "m3"
        full_volume_var.long_name = "Physical 15-arcsecond GEBCO water-column volume aggregated to 1-degree cell"

        dataset.title = "One-degree layer volume weights derived from 15-arcsecond GEBCO"
        dataset.source_dem = str(dem_path)
        dataset.source_dem_resolution = "15 arcseconds"
        dataset.target_grid_resolution = "1 degree"
        dataset.ocean_mask = "GEBCO bathymetry < 0 and not NoData"
        dataset.earth_radius_m = EARTH_RADIUS_M
        dataset.area_formula = "R^2 * radians(dlon) * radians(dlat) * cos(pixel_center_latitude)"
        dataset.volume_calculation_compatibility = "TEOS-10V4_beifen_Wutian.py enhanced_gpu_volume_calculation"
        dataset.longitude_convention = "-180 to <180 degrees, sorted west to east"
        dataset.geotransform = ", ".join(f"{value:.16g}" for value in geotransform)
        dataset.ocean_pixel_count = ocean_pixels
        dataset.created_utc = datetime.now(timezone.utc).isoformat()
        dataset.created_by = "build_1deg_layer_volumes.py"

    os.replace(temporary, path)


def write_summary(
    output_dir: Path,
    depths: np.ndarray,
    thicknesses: np.ndarray,
    layer_volume_m3: np.ndarray,
    ocean_area_m2: np.ndarray,
    full_water_column_volume_m3: np.ndarray,
    ocean_pixels: int,
    dem_path: Path,
) -> dict[str, float | int | str]:
    layer_totals = np.sum(layer_volume_m3, axis=(1, 2), dtype=np.float64)
    physical_volume = float(np.sum(full_water_column_volume_m3, dtype=np.float64))
    model_volume = float(np.sum(layer_totals, dtype=np.float64))
    ocean_area = float(np.sum(ocean_area_m2, dtype=np.float64))

    summary = {
        "source_dem": str(dem_path),
        "ocean_pixel_count": ocean_pixels,
        "ocean_area_m2": ocean_area,
        "ocean_area_km2": ocean_area / 1.0e6,
        "physical_full_water_column_volume_m3": physical_volume,
        "physical_full_water_column_volume_km3": physical_volume / 1.0e9,
        "teos_model_layer_volume_0_6000m_m3": model_volume,
        "teos_model_layer_volume_0_6000m_km3": model_volume / 1.0e9,
        "layer_center_group_0_700m_km3": calculate_group_volume(layer_volume_m3, depths, 0.0, 700.0) / 1.0e9,
        "layer_center_group_700_2000m_km3": calculate_group_volume(layer_volume_m3, depths, 700.0, 2000.0) / 1.0e9,
        "layer_center_group_2000_6000m_km3": calculate_group_volume(layer_volume_m3, depths, 2000.0, 6000.0) / 1.0e9,
    }

    csv_path = output_dir / "gebco_1deg_layer_volume_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "record_type", "depth_m", "layer_thickness_m", "volume_m3", "volume_km3", "cumulative_volume_km3"
        ])
        cumulative = 0.0
        for depth, thickness, volume in zip(depths, thicknesses, layer_totals):
            cumulative += float(volume)
            writer.writerow(["layer", f"{depth:.6f}", f"{thickness:.6f}", f"{volume:.6f}", f"{volume/1e9:.9f}", f"{cumulative/1e9:.9f}"])
        writer.writerow(["physical_full_water_column", "", "", f"{physical_volume:.6f}", f"{physical_volume/1e9:.9f}", ""])
        for lower, upper, label in ((0.0, 700.0, "depth_center_0_700"), (700.0, 2000.0, "depth_center_700_2000"), (2000.0, 6000.0, "depth_center_2000_6000")):
            volume = calculate_group_volume(layer_volume_m3, depths, lower, upper)
            writer.writerow([label, "", "", f"{volume:.6f}", f"{volume/1e9:.9f}", ""])

    json_path = output_dir / "gebco_1deg_layer_volume_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate 15-arcsecond GEBCO to 1-degree thermal-expansion layer volumes."
    )
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--annual-temp", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs") / "gebco_1deg_layer_volumes"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not args.dem.is_file():
        raise FileNotFoundError(f"DEM not found: {args.dem}")
    if not args.annual_temp.is_file():
        raise FileNotFoundError(f"Annual temperature file not found: {args.annual_temp}")

    output_nc = args.output_dir / "gebco_1deg_layer_volume_0_6000m.nc"
    if output_nc.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_nc}; use --overwrite to replace it")

    logger = configure_logging(args.output_dir)
    logger.info("Starting 15-arcsecond GEBCO -> 1-degree layer-volume aggregation")
    logger.info("DEM: %s", args.dem)
    logger.info("Annual T/S grid: %s", args.annual_temp)
    logger.info("Output: %s", output_nc)

    gdal.UseExceptions()
    dataset = gdal.Open(str(args.dem), gdal.GA_ReadOnly)
    if dataset is None:
        raise RuntimeError(f"Unable to open DEM: {args.dem}")

    try:
        target_lat, target_lon, depths = read_target_grid(args.annual_temp)
        width, height, pixels_x, pixels_y, geotransform = validate_raster(dataset, target_lat, target_lon)
        if pixels_x != pixels_y:
            raise ValueError("Only square arcsecond DEM pixels are supported")
        pixels_per_degree = pixels_x
        thicknesses = layer_thicknesses(depths)
        band = dataset.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        min_value = band.GetMinimum()
        if min_value is None:
            min_value, _ = band.ComputeRasterMinMax(approx_ok=False)
        deepest_required = max(float(-min_value), float(np.max(depths + thicknesses)))
        depth_bin_count = int(math.ceil(deepest_required)) + 2

        logger.info("DEM grid: %d x %d, %d pixels per degree", width, height, pixels_per_degree)
        logger.info("Target grid: %d depth layers x %d latitude x %d longitude", depths.size, target_lat.size, target_lon.size)
        logger.info("Depth range: %.1f to %.1f m", depths[0], depths[-1])
        logger.info("NoData: %s; histogram depth bins: %d", nodata, depth_bin_count)

        layer_volume = np.zeros((depths.size, target_lat.size, target_lon.size), dtype=np.float64)
        ocean_area = np.zeros((target_lat.size, target_lon.size), dtype=np.float64)
        full_water_column_volume = np.zeros((target_lat.size, target_lon.size), dtype=np.float64)
        ocean_pixel_count = 0

        base_pixel_area = (
            EARTH_RADIUS_M * np.radians(abs(geotransform[1]))
        ) * (
            EARTH_RADIUS_M * np.radians(abs(geotransform[5]))
        )

        for north_band in tqdm(range(target_lat.size), desc="Aggregating 1-degree latitude bands", unit="band"):
            row_start = north_band * pixels_per_degree
            bathymetry = band.ReadAsArray(0, row_start, width, pixels_per_degree)
            if bathymetry is None or bathymetry.shape != (pixels_per_degree, width):
                raise RuntimeError(f"Failed to read DEM rows {row_start}:{row_start + pixels_per_degree}")

            # This is the pixel-centre latitude convention in TEOS-10V4.
            row_numbers = row_start + np.arange(pixels_per_degree, dtype=np.float64)
            pixel_lats = geotransform[3] + (row_numbers + 0.5) * geotransform[5]
            row_areas = base_pixel_area * np.cos(np.radians(pixel_lats))

            histogram, block_ocean_pixels = weighted_depth_histogram(
                bathymetry,
                row_areas,
                pixels_per_degree,
                target_lon.size,
                depth_bin_count,
                nodata,
            )
            block_layer_volumes, block_area, block_full_volume = layer_volumes_from_histogram(
                histogram, depths, thicknesses
            )

            # The raster is north-to-south; output coordinates are south-to-north.
            output_lat_index = target_lat.size - 1 - north_band
            layer_volume[:, output_lat_index, :] = block_layer_volumes
            ocean_area[output_lat_index, :] = block_area
            full_water_column_volume[output_lat_index, :] = block_full_volume
            ocean_pixel_count += block_ocean_pixels

        write_netcdf(
            output_nc,
            target_lat,
            target_lon,
            depths,
            thicknesses,
            layer_volume,
            ocean_area,
            full_water_column_volume,
            args.dem,
            geotransform,
            ocean_pixel_count,
        )
        summary = write_summary(
            args.output_dir,
            depths,
            thicknesses,
            layer_volume,
            ocean_area,
            full_water_column_volume,
            ocean_pixel_count,
            args.dem,
        )

        logger.info("Aggregation complete")
        logger.info("Ocean pixels: %s", f"{summary['ocean_pixel_count']:,}")
        logger.info("Ocean area: %.6f million km2", float(summary["ocean_area_km2"]) / 1.0e6)
        logger.info("Physical full water-column volume: %.6f million km3", float(summary["physical_full_water_column_volume_km3"]) / 1.0e6)
        logger.info("TEOS-compatible model-layer volume through 6000 m: %.6f million km3", float(summary["teos_model_layer_volume_0_6000m_km3"]) / 1.0e6)
        logger.info("0-700 m (depth-centre grouping): %.6f million km3", float(summary["layer_center_group_0_700m_km3"]) / 1.0e6)
        logger.info("700-2000 m (depth-centre grouping): %.6f million km3", float(summary["layer_center_group_700_2000m_km3"]) / 1.0e6)
        logger.info("2000-6000 m (depth-centre grouping): %.6f million km3", float(summary["layer_center_group_2000_6000m_km3"]) / 1.0e6)
        return 0
    finally:
        dataset = None


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.exception("Volume aggregation failed: %s", exc)
        sys.exit(1)
