#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare GO-SHIP versus IAP deep-ocean T/S residuals for uncertainty analysis.

This script is deliberately an audit and preprocessing step.  It does not
calculate thermal expansion and it does not manufacture a global deep-ocean
one-sigma field.  It compares the independent, repeat-hydrography GO-SHIP
observations with the annual IAP 1-degree fields at matched locations and
standard depths.  Its output keeps the repeat section occupation as the
fundamental sampling block, so later bootstrap calculations can preserve the
strong spatial and vertical dependence within a hydrographic section.

The output is limited by default to 1993--2021 and 2000--6000 m.  This is the
period in which the IAPv4 publication regards the abyssal estimate as most
robust and it overlaps the existing annual upper-ocean one-sigma inputs.

Inputs
------
* GO-SHIP Easy Ocean gridded NetCDF section time series.  These files contain
  quality-controlled, horizontally gridded CTD observations and their actual
  station dates.
* Annual IAP temperature and Absolute Salinity fields on the main 1-degree
  calculation grid.

Outputs
-------
``goship_iap_deep_residuals.csv``
    One row per [section occupation, standard depth].  Temperature and
    Absolute Salinity residuals are GO-SHIP minus IAP, aggregated only along
    that individual section occupation.
``goship_iap_deep_occupations.csv``
    One row per section occupation with date, geographic extent, and sample
    counts for audit and bootstrap blocking.
``metadata.json`` and ``prepare_goship_iap_deep_residuals.log``
    Run configuration and reproducibility information.
"""

from __future__ import annotations

import argparse
import csv
import grp
import json
import logging
import os
import pwd
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import gsw
import netCDF4 as nc
import numpy as np


COORDINATE_CANDIDATES = {
    "depth": ("depth_std", "depth", "z", "level", "lev"),
    "lat": ("lat", "latitude", "y"),
    "lon": ("lon", "longitude", "x"),
}

RESIDUAL_FIELDS = [
    "line_id",
    "basin",
    "occupation_id",
    "section_repeat_number",
    "observation_year",
    "observation_date_median",
    "depth_m",
    "n_locations",
    "temperature_residual_mean_c",
    "temperature_residual_median_c",
    "temperature_residual_std_c",
    "absolute_salinity_residual_mean_gkg",
    "absolute_salinity_residual_median_gkg",
    "absolute_salinity_residual_std_gkg",
    "temp_salinity_residual_covariance",
    "longitude_min",
    "longitude_max",
    "latitude_min",
    "latitude_max",
]

OCCUPATION_FIELDS = [
    "line_id",
    "basin",
    "occupation_id",
    "section_repeat_number",
    "observation_year",
    "observation_date_median",
    "source_file",
    "n_horizontal_locations",
    "n_deep_depth_levels",
    "longitude_min",
    "longitude_max",
    "latitude_min",
    "latitude_max",
]


@dataclass(frozen=True)
class IapGrid:
    """IAP data in canonical [depth, latitude, longitude] order.

    Temperature and salinity source files have the same depth and latitude
    axes, but their longitude centres can differ by half a degree.  This
    mirrors the main thermal-expansion loader, which interpolates both fields
    independently rather than coercing one field onto the other grid.
    """

    depth: np.ndarray
    lat: np.ndarray
    temperature_lon: np.ndarray
    salinity_lon: np.ndarray
    temperature: np.ndarray
    absolute_salinity: np.ndarray


@dataclass(frozen=True)
class HorizontalWeights:
    """Bilinear weights on a periodic longitude / bounded latitude grid."""

    lat0: np.ndarray
    lat1: np.ndarray
    lon0: np.ndarray
    lon1: np.ndarray
    weight00: np.ndarray
    weight01: np.ndarray
    weight10: np.ndarray
    weight11: np.ndarray


def configure_logging(output_dir: Path, verbose: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("goship_iap_deep_residuals")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(
        output_dir / "prepare_goship_iap_deep_residuals.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def resolve_owner(owner: str) -> tuple[int, int]:
    user = pwd.getpwnam(owner)
    try:
        group = grp.getgrgid(user.pw_gid)
    except KeyError:
        group = grp.getgrnam(owner)
    return user.pw_uid, group.gr_gid


def set_owner_recursively(path: Path, owner: str, logger: logging.Logger) -> None:
    """Set requested ownership when permissions allow it."""
    uid, gid = resolve_owner(owner)
    paths = [path]
    if path.is_dir():
        paths.extend(sorted(path.rglob("*")))
    for item in paths:
        try:
            stat = item.stat()
            if stat.st_uid != uid or stat.st_gid != gid:
                os.chown(item, uid, gid)
        except OSError as exc:
            logger.warning("Cannot change owner of %s to %s: %s", item, owner, exc)
            return


def normalise_longitude(lon: np.ndarray) -> np.ndarray:
    """Normalise longitude to [-180, 180), matching the IAP calculation grid."""
    return np.mod(np.asarray(lon, dtype=np.float64) + 180.0, 360.0) - 180.0


def fill_to_nan(
    variable: nc.Variable, invalid_magnitude: float | None = 900.0
) -> np.ndarray:
    """Read a NetCDF variable and replace masked / declared fill values by NaN."""
    raw = variable[:]
    if np.ma.isMaskedArray(raw):
        data = np.ma.filled(raw, np.nan)
    else:
        data = np.asarray(raw)
    data = np.asarray(data, dtype=np.float64)
    for attr in ("_FillValue", "missing_value"):
        if hasattr(variable, attr):
            values = np.atleast_1d(getattr(variable, attr))
            for value in values:
                if np.isfinite(value):
                    data[np.isclose(data, float(value), rtol=0.0, atol=1.0e-12)] = np.nan
    if invalid_magnitude is not None:
        data[np.abs(data) >= invalid_magnitude] = np.nan
    return data


def find_coordinate_name(dataset: nc.Dataset, kind: str) -> str:
    for candidate in COORDINATE_CANDIDATES[kind]:
        if candidate in dataset.variables:
            return candidate
    raise KeyError(f"Cannot identify {kind} coordinate in {dataset.filepath()}")


def identify_axis(dimensions: tuple[str, ...], kind: str) -> int:
    candidates = COORDINATE_CANDIDATES[kind]
    for index, dimension in enumerate(dimensions):
        lowered = dimension.lower()
        if any(candidate.lower() == lowered for candidate in candidates):
            return index
    raise ValueError(f"Cannot identify {kind} axis in dimensions {dimensions}")


def read_iap_variable(path: Path, variable_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one annual IAP variable in [depth, lat, lon] coordinate order."""
    with nc.Dataset(path, "r") as dataset:
        if variable_name not in dataset.variables:
            raise KeyError(f"{path}: missing variable {variable_name}")
        variable = dataset.variables[variable_name]
        data = fill_to_nan(variable)
        depth_name = find_coordinate_name(dataset, "depth")
        lat_name = find_coordinate_name(dataset, "lat")
        lon_name = find_coordinate_name(dataset, "lon")
        depth = np.asarray(dataset.variables[depth_name][:], dtype=np.float64)
        lat = np.asarray(dataset.variables[lat_name][:], dtype=np.float64)
        lon = normalise_longitude(dataset.variables[lon_name][:])

        axes = [
            identify_axis(variable.dimensions, "depth"),
            identify_axis(variable.dimensions, "lat"),
            identify_axis(variable.dimensions, "lon"),
        ]
        data = np.transpose(data, axes)

    depth_order = np.argsort(depth)
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    return (
        depth[depth_order],
        lat[lat_order],
        lon[lon_order],
        data[np.ix_(depth_order, lat_order, lon_order)],
    )


class IapYearCache:
    """Small LRU cache that prevents repeated annual IAP NetCDF reads."""

    def __init__(self, annual_dir: Path, max_items: int, logger: logging.Logger):
        self.annual_dir = annual_dir
        self.max_items = max_items
        self.logger = logger
        self.cache: OrderedDict[int, IapGrid] = OrderedDict()

    def get(self, year: int) -> IapGrid:
        cached = self.cache.pop(year, None)
        if cached is not None:
            self.cache[year] = cached
            return cached

        temperature_path = self.annual_dir / f"Annual_Temp_{year}.nc"
        salinity_path = self.annual_dir / f"Annual_Salinity_{year}.nc"
        if not temperature_path.exists() or not salinity_path.exists():
            raise FileNotFoundError(
                f"Missing annual IAP files for {year}: {temperature_path}, {salinity_path}"
            )

        depth, lat, lon, temperature = read_iap_variable(temperature_path, "temp")
        salt_depth, salt_lat, salt_lon, salinity = read_iap_variable(salinity_path, "salinity")
        if not (np.array_equal(depth, salt_depth) and np.array_equal(lat, salt_lat)):
            raise ValueError(f"IAP temperature and salinity depth/latitude mismatch for {year}")

        result = IapGrid(
            depth=depth,
            lat=lat,
            temperature_lon=lon,
            salinity_lon=salt_lon,
            temperature=temperature,
            absolute_salinity=salinity,
        )
        self.cache[year] = result
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        self.logger.info("Loaded IAP annual fields for %d", year)
        return result


def horizontal_weights(
    lat: np.ndarray,
    lon: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> HorizontalWeights:
    """Build reusable bilinear weights for GO-SHIP station locations."""
    target_lat = np.clip(np.asarray(lat, dtype=np.float64), grid_lat[0], grid_lat[-1])
    target_lon = normalise_longitude(lon)

    lat1 = np.searchsorted(grid_lat, target_lat, side="right")
    lat1 = np.clip(lat1, 1, grid_lat.size - 1)
    lat0 = lat1 - 1
    lat_fraction = (target_lat - grid_lat[lat0]) / (grid_lat[lat1] - grid_lat[lat0])

    lon_extended = np.concatenate((grid_lon, [grid_lon[0] + 360.0]))
    lon_unwrapped = np.where(target_lon < grid_lon[0], target_lon + 360.0, target_lon)
    lon1_unwrapped = np.searchsorted(lon_extended, lon_unwrapped, side="right")
    lon1_unwrapped = np.clip(lon1_unwrapped, 1, lon_extended.size - 1)
    lon0 = lon1_unwrapped - 1
    lon1 = lon1_unwrapped % grid_lon.size
    lon_fraction = (
        (lon_unwrapped - lon_extended[lon0])
        / (lon_extended[lon1_unwrapped] - lon_extended[lon0])
    )

    return HorizontalWeights(
        lat0=lat0,
        lat1=lat1,
        lon0=lon0,
        lon1=lon1,
        weight00=(1.0 - lat_fraction) * (1.0 - lon_fraction),
        weight01=(1.0 - lat_fraction) * lon_fraction,
        weight10=lat_fraction * (1.0 - lon_fraction),
        weight11=lat_fraction * lon_fraction,
    )


def bilinear_sample(values: np.ndarray, weights: HorizontalWeights) -> np.ndarray:
    """Sample [depth, lat, lon] values at locations with valid-neighbour renormalisation."""
    corners = (
        (values[:, weights.lat0, weights.lon0], weights.weight00),
        (values[:, weights.lat0, weights.lon1], weights.weight01),
        (values[:, weights.lat1, weights.lon0], weights.weight10),
        (values[:, weights.lat1, weights.lon1], weights.weight11),
    )
    numerator = np.zeros_like(corners[0][0], dtype=np.float64)
    denominator = np.zeros_like(corners[0][0], dtype=np.float64)
    for corner, weight in corners:
        valid = np.isfinite(corner)
        numerator += np.where(valid, corner * weight[np.newaxis, :], 0.0)
        denominator += np.where(valid, weight[np.newaxis, :], 0.0)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )


def basin_for_line(line_id: str) -> str:
    """Assign GO-SHIP line identifiers to broad ocean basins for block bootstrap."""
    line = line_id.upper()
    if line == "75N":
        return "Arctic"
    # AR07E/W are North Atlantic repeat lines near Greenland, not Arctic-basin
    # transects.  They therefore belong with the Atlantic section blocks.
    if line.startswith("AR"):
        return "Atlantic"
    if line.startswith("A"):
        return "Atlantic"
    if line.startswith("I"):
        return "Indian"
    if line.startswith("P"):
        return "Pacific"
    if line.startswith("S"):
        return "Southern"
    return "Unknown"


def line_id_from_dataset(dataset: nc.Dataset, path: Path) -> str:
    value = getattr(dataset, "goship_woce_line_id", "")
    if value:
        return str(value).strip().upper()
    return path.stem.upper()


def to_datetime_array(time_variable: nc.Variable, section_index: int) -> np.ndarray:
    """Read a section's station dates as an array matching its horizontal coordinate."""
    dimensions = time_variable.dimensions
    # NetCDF time is measured in days since 1950 and is normally far larger
    # than the ocean-variable missing-value threshold used elsewhere.
    values = fill_to_nan(time_variable, invalid_magnitude=None)
    section_axis = None
    for index, dimension in enumerate(dimensions):
        if dimension.lower() == "gridded_section":
            section_axis = index
            break
    if section_axis is None:
        raise ValueError(f"Cannot find gridded_section axis in time dimensions {dimensions}")
    section_time = np.take(values, section_index, axis=section_axis)
    dates = np.empty(section_time.shape, dtype=object)
    dates[:] = None
    valid = np.isfinite(section_time)
    if np.any(valid):
        converted = nc.num2date(
            section_time[valid],
            units=time_variable.units,
            calendar=getattr(time_variable, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        dates[valid] = list(np.atleast_1d(converted))
    return dates


def section_coordinates(dataset: nc.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened [lat, lon] coordinates in the NetCDF data storage order."""
    lat = np.asarray(dataset.variables[find_coordinate_name(dataset, "lat")][:], dtype=np.float64)
    lon = np.asarray(dataset.variables[find_coordinate_name(dataset, "lon")][:], dtype=np.float64)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return lat_grid.ravel(), lon_grid.ravel()


def section_times_flat(
    dataset: nc.Dataset,
    section_index: int,
    n_lat: int,
    n_lon: int,
) -> np.ndarray:
    """Broadcast time coordinates to the flattened [lat, lon] section shape."""
    time_variable = dataset.variables["time"]
    time_values = to_datetime_array(time_variable, section_index)
    remaining_dimensions = tuple(
        dimension
        for dimension in time_variable.dimensions
        if dimension.lower() != "gridded_section"
    )
    if not remaining_dimensions:
        return np.full(n_lat * n_lon, time_values.item(), dtype=object)
    if remaining_dimensions == ("longitude",):
        return np.broadcast_to(time_values[np.newaxis, :], (n_lat, n_lon)).ravel()
    if remaining_dimensions == ("latitude",):
        return np.broadcast_to(time_values[:, np.newaxis], (n_lat, n_lon)).ravel()
    if set(remaining_dimensions) == {"latitude", "longitude"}:
        if remaining_dimensions != ("latitude", "longitude"):
            axes = [remaining_dimensions.index("latitude"), remaining_dimensions.index("longitude")]
            time_values = np.transpose(time_values, axes)
        return time_values.ravel()
    raise ValueError(f"Unsupported GO-SHIP time dimensions {time_variable.dimensions}")


def read_section_variable(dataset: nc.Dataset, variable_name: str, section_index: int) -> np.ndarray:
    """Read a GO-SHIP variable as [pressure, latitude, longitude] for one repeat."""
    variable = dataset.variables[variable_name]
    dimensions = variable.dimensions
    section_axis = dimensions.index("gridded_section")
    values = np.take(fill_to_nan(variable), section_index, axis=section_axis)
    remaining = tuple(dimension for dimension in dimensions if dimension != "gridded_section")
    target = ("pressure", "latitude", "longitude")
    if set(remaining) != set(target):
        raise ValueError(f"Unexpected dimensions for {variable_name}: {dimensions}")
    axes = [remaining.index(name) for name in target]
    return np.transpose(values, axes)


def median_date(dates: np.ndarray) -> tuple[int | None, str]:
    """Return representative observation year/date for an occupation."""
    valid = [date for date in dates.ravel() if date is not None]
    if not valid:
        return None, ""
    years = [int(date.year) for date in valid]
    year = Counter(years).most_common(1)[0][0]
    ordered = sorted(valid)
    middle = ordered[len(ordered) // 2]
    return year, f"{middle.year:04d}-{middle.month:02d}-{middle.day:02d}"


def interpolate_vertical(
    pressure_dbar: np.ndarray,
    values: np.ndarray,
    target_pressure_dbar: np.ndarray,
) -> np.ndarray:
    """Linearly sample [pressure, location] section values at target pressure."""
    if values.ndim != 2:
        raise ValueError("values must have dimensions [pressure, location]")
    if target_pressure_dbar.ndim != 2:
        raise ValueError("target pressure must have dimensions [depth, location]")
    if values.shape[1] != target_pressure_dbar.shape[1]:
        raise ValueError("target pressure location dimension mismatch")

    lower = np.searchsorted(pressure_dbar, target_pressure_dbar, side="right") - 1
    lower = np.clip(lower, 0, pressure_dbar.size - 2)
    upper = lower + 1
    x0 = pressure_dbar[lower]
    x1 = pressure_dbar[upper]
    columns = np.broadcast_to(
        np.arange(values.shape[1], dtype=np.int64), target_pressure_dbar.shape
    )
    y0 = values[lower, columns]
    y1 = values[upper, columns]
    fraction = (target_pressure_dbar - x0) / (x1 - x0)
    valid = (
        (target_pressure_dbar >= pressure_dbar[0])
        & (target_pressure_dbar <= pressure_dbar[-1])
        & np.isfinite(y0)
        & np.isfinite(y1)
    )
    interpolated = y0 + fraction * (y1 - y0)
    return np.where(valid, interpolated, np.nan)


def scalar_summary(values: np.ndarray) -> tuple[int, float, float, float]:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return 0, np.nan, np.nan, np.nan
    standard_deviation = float(np.std(valid, ddof=1)) if valid.size > 1 else np.nan
    return int(valid.size), float(np.mean(valid)), float(np.median(valid)), standard_deviation


def residual_row(
    *,
    line_id: str,
    basin: str,
    occupation_id: str,
    repeat_number: int,
    observation_year: int,
    observation_date: str,
    depth_m: float,
    temperature_residual: np.ndarray,
    salinity_residual: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
) -> dict[str, object]:
    n_temp, temp_mean, temp_median, temp_std = scalar_summary(temperature_residual)
    n_salt, salt_mean, salt_median, salt_std = scalar_summary(salinity_residual)
    paired = np.isfinite(temperature_residual) & np.isfinite(salinity_residual)
    covariance = (
        float(np.cov(temperature_residual[paired], salinity_residual[paired], ddof=1)[0, 1])
        if np.count_nonzero(paired) > 1
        else np.nan
    )
    n_locations = max(n_temp, n_salt)
    return {
        "line_id": line_id,
        "basin": basin,
        "occupation_id": occupation_id,
        "section_repeat_number": repeat_number,
        "observation_year": observation_year,
        "observation_date_median": observation_date,
        "depth_m": f"{depth_m:.1f}",
        "n_locations": n_locations,
        "temperature_residual_mean_c": temp_mean,
        "temperature_residual_median_c": temp_median,
        "temperature_residual_std_c": temp_std,
        "absolute_salinity_residual_mean_gkg": salt_mean,
        "absolute_salinity_residual_median_gkg": salt_median,
        "absolute_salinity_residual_std_gkg": salt_std,
        "temp_salinity_residual_covariance": covariance,
        "longitude_min": float(np.nanmin(lon)),
        "longitude_max": float(np.nanmax(lon)),
        "latitude_min": float(np.nanmin(lat)),
        "latitude_max": float(np.nanmax(lat)),
    }


def process_file(
    path: Path,
    cache: IapYearCache,
    min_year: int,
    max_year: int,
    min_depth: float,
    max_depth: float,
    residual_writer: csv.DictWriter,
    occupation_writer: csv.DictWriter,
    logger: logging.Logger,
) -> tuple[int, int]:
    """Process all repeat occupations in one GO-SHIP line NetCDF file."""
    residual_count = 0
    occupation_count = 0
    with nc.Dataset(path, "r") as dataset:
        line_id = line_id_from_dataset(dataset, path)
        basin = basin_for_line(line_id)
        pressure = np.asarray(dataset.variables["pressure"][:], dtype=np.float64)
        lat_values = np.asarray(dataset.variables["latitude"][:], dtype=np.float64)
        lon_values = np.asarray(dataset.variables["longitude"][:], dtype=np.float64)
        n_lat, n_lon = lat_values.size, lon_values.size
        lat, lon = section_coordinates(dataset)
        repeat_values = np.asarray(dataset.variables["gridded_section"][:], dtype=np.float64)

        for section_index, repeat_value in enumerate(repeat_values):
            dates = section_times_flat(dataset, section_index, n_lat, n_lon)
            observation_year, observation_date = median_date(dates)
            if observation_year is None:
                logger.warning("Skip %s repeat %s: no station dates", line_id, repeat_value)
                continue
            if not min_year <= observation_year <= max_year:
                continue

            grid = cache.get(observation_year)
            depth_mask = (grid.depth > min_depth) & (grid.depth <= max_depth)
            deep_depths = grid.depth[depth_mask]
            if deep_depths.size == 0:
                raise ValueError(f"No IAP levels in ({min_depth}, {max_depth}] m")

            temperature = read_section_variable(dataset, "temperature", section_index)
            practical_salinity = read_section_variable(
                dataset, "practical_salinity", section_index
            )
            temperature = temperature.reshape(pressure.size, -1)
            practical_salinity = practical_salinity.reshape(pressure.size, -1)

            target_pressure = gsw.p_from_z(
                -deep_depths[:, np.newaxis], lat[np.newaxis, :]
            )
            observed_temperature = interpolate_vertical(
                pressure, temperature, target_pressure
            )
            observed_sp = interpolate_vertical(pressure, practical_salinity, target_pressure)
            observed_salinity = np.full_like(observed_sp, np.nan)
            valid_sp = np.isfinite(observed_sp)
            if np.any(valid_sp):
                observed_salinity[valid_sp] = gsw.SA_from_SP(
                    observed_sp[valid_sp],
                    target_pressure[valid_sp],
                    np.broadcast_to(lon[np.newaxis, :], observed_sp.shape)[valid_sp],
                    np.broadcast_to(lat[np.newaxis, :], observed_sp.shape)[valid_sp],
                )

            # The Easy Ocean gridded collection can retain placeholder profiles
            # without a conventional fill value (for example, SP near 3 g kg-1
            # and constant temperature).  Treat T/S as one paired observation:
            # a physically invalid member invalidates both variables so that a
            # residual profile cannot mix real temperature with fake salinity.
            valid_observation = (
                (observed_temperature >= -3.0)
                & (observed_temperature <= 40.0)
                & (observed_salinity >= 20.0)
                & (observed_salinity <= 45.0)
            )
            observed_temperature = np.where(valid_observation, observed_temperature, np.nan)
            observed_salinity = np.where(valid_observation, observed_salinity, np.nan)

            temperature_weights = horizontal_weights(
                lat, lon, grid.lat, grid.temperature_lon
            )
            salinity_weights = horizontal_weights(lat, lon, grid.lat, grid.salinity_lon)
            iap_temperature_values = grid.temperature[depth_mask].copy()
            iap_salinity_values = grid.absolute_salinity[depth_mask].copy()
            # Retain only physically plausible deep values before bilinear
            # interpolation.  A few annual source cells contain non-fill,
            # nonphysical salinity values that would otherwise contaminate a
            # section residual despite the surrounding field being valid.
            iap_temperature_values[
                (iap_temperature_values < -3.0) | (iap_temperature_values > 40.0)
            ] = np.nan
            iap_salinity_values[
                (iap_salinity_values < 20.0) | (iap_salinity_values > 45.0)
            ] = np.nan
            iap_temperature = bilinear_sample(iap_temperature_values, temperature_weights)
            iap_salinity = bilinear_sample(iap_salinity_values, salinity_weights)
            temperature_residual = observed_temperature - iap_temperature
            salinity_residual = observed_salinity - iap_salinity

            repeat_number = int(round(repeat_value))
            occupation_id = f"{line_id}_{observation_year}_{repeat_number:02d}"
            occupation_writer.writerow(
                {
                    "line_id": line_id,
                    "basin": basin,
                    "occupation_id": occupation_id,
                    "section_repeat_number": repeat_number,
                    "observation_year": observation_year,
                    "observation_date_median": observation_date,
                    "source_file": str(path),
                    "n_horizontal_locations": int(lat.size),
                    "n_deep_depth_levels": int(deep_depths.size),
                    "longitude_min": float(np.nanmin(lon)),
                    "longitude_max": float(np.nanmax(lon)),
                    "latitude_min": float(np.nanmin(lat)),
                    "latitude_max": float(np.nanmax(lat)),
                }
            )
            occupation_count += 1

            for depth_index, depth in enumerate(deep_depths):
                residual_writer.writerow(
                    residual_row(
                        line_id=line_id,
                        basin=basin,
                        occupation_id=occupation_id,
                        repeat_number=repeat_number,
                        observation_year=observation_year,
                        observation_date=observation_date,
                        depth_m=float(depth),
                        temperature_residual=temperature_residual[depth_index],
                        salinity_residual=salinity_residual[depth_index],
                        lon=lon,
                        lat=lat,
                    )
                )
                residual_count += 1
            logger.info(
                "Processed %s repeat=%d year=%d: locations=%d deep_levels=%d",
                line_id,
                repeat_number,
                observation_year,
                lat.size,
                deep_depths.size,
            )
    return residual_count, occupation_count


def iter_goship_files(root: Path, requested_lines: list[str] | None = None) -> Iterable[Path]:
    files = sorted(root.glob("*/*.nc"))
    if not requested_lines:
        return files
    requested = {line.strip().upper() for line in requested_lines}
    selected = [
        path
        for path in files
        if path.parent.name.upper() in requested or path.stem.upper() in requested
    ]
    found = {path.parent.name.upper() for path in selected}
    unknown = requested - found
    if unknown:
        raise ValueError(f"Requested GO-SHIP line(s) not found: {sorted(unknown)}")
    return selected


def output_counts(residual_path: Path, occupation_path: Path) -> tuple[int, int, int]:
    """Return final residual rows, occupations, and distinct source line files."""
    with residual_path.open("r", newline="", encoding="utf-8") as handle:
        residual_rows = max(0, sum(1 for _ in handle) - 1)
    sources: set[str] = set()
    occupation_rows = 0
    with occupation_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            occupation_rows += 1
            sources.add(row["source_file"])
    return residual_rows, occupation_rows, len(sources)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare section-blocked GO-SHIP/IAP deep T/S residuals."
    )
    parser.add_argument("--goship-dir", type=Path, required=True)
    parser.add_argument("--annual-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "goship_iap_deep_residuals")
    parser.add_argument("--start-year", type=int, default=1993)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument(
        "--min-depth",
        type=float,
        default=2000.0,
        help="Exclusive lower standard-depth bound in metres (default: 2000).",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=6000.0,
        help="Inclusive upper standard-depth bound in metres (default: 6000).",
    )
    parser.add_argument(
        "--year-cache-size",
        type=int,
        default=2,
        help="Number of annual IAP field pairs retained in RAM (default: 2).",
    )
    parser.add_argument(
        "--line",
        dest="lines",
        action="append",
        default=None,
        help="Process one GO-SHIP line only; repeat for a batch of lines.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append a line batch to existing residual CSV outputs.",
    )
    parser.add_argument("--owner", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    if args.min_depth < 0 or args.max_depth <= args.min_depth:
        raise ValueError("Depth bounds must satisfy 0 <= min-depth < max-depth")
    if args.year_cache_size < 1:
        raise ValueError("--year-cache-size must be at least 1")
    if args.overwrite and args.append:
        raise ValueError("--overwrite and --append cannot be used together")
    if not args.goship_dir.is_dir():
        raise FileNotFoundError(f"GO-SHIP NetCDF directory not found: {args.goship_dir}")
    if not args.annual_dir.is_dir():
        raise FileNotFoundError(f"Annual IAP directory not found: {args.annual_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir, args.verbose)
    residual_path = args.output_dir / "goship_iap_deep_residuals.csv"
    occupation_path = args.output_dir / "goship_iap_deep_occupations.csv"
    metadata_path = args.output_dir / "metadata.json"

    existing = [path for path in (residual_path, occupation_path, metadata_path) if path.exists()]
    if existing and not args.overwrite and not args.append:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists; rerun with --overwrite or --append: {names}")
    if args.append and (not residual_path.exists() or not occupation_path.exists()):
        raise FileNotFoundError("--append requires existing residual and occupation CSV outputs")

    files = list(iter_goship_files(args.goship_dir, args.lines))
    if not files:
        raise FileNotFoundError(f"No GO-SHIP NetCDF files found under {args.goship_dir}")

    logger.info(
        "GO-SHIP/IAP deep residual audit: files=%d, years=%d-%d, depths=(%.1f, %.1f] m",
        len(files),
        args.start_year,
        args.end_year,
        args.min_depth,
        args.max_depth,
    )
    cache = IapYearCache(args.annual_dir, args.year_cache_size, logger)
    total_residual_rows = 0
    total_occupations = 0

    mode = "a" if args.append else "w"
    write_header = not args.append or residual_path.stat().st_size == 0
    with residual_path.open(mode, newline="", encoding="utf-8") as residual_handle, occupation_path.open(
        mode, newline="", encoding="utf-8"
    ) as occupation_handle:
        residual_writer = csv.DictWriter(residual_handle, fieldnames=RESIDUAL_FIELDS)
        occupation_writer = csv.DictWriter(occupation_handle, fieldnames=OCCUPATION_FIELDS)
        if write_header:
            residual_writer.writeheader()
            occupation_writer.writeheader()
        for path in files:
            rows, occupations = process_file(
                path,
                cache,
                args.start_year,
                args.end_year,
                args.min_depth,
                args.max_depth,
                residual_writer,
                occupation_writer,
                logger,
            )
            total_residual_rows += rows
            total_occupations += occupations

    final_residual_rows, final_occupations, final_source_files = output_counts(
        residual_path, occupation_path
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "GO-SHIP versus IAP deep T/S residual preprocessing for section-block bootstrap",
        "goship_source_directory": str(args.goship_dir),
        "annual_iap_directory": str(args.annual_dir),
        "year_range": [args.start_year, args.end_year],
        "depth_selection_m": {"exclusive_lower": args.min_depth, "inclusive_upper": args.max_depth},
        "current_run_line_files": len(files),
        "current_run_section_occupations": total_occupations,
        "current_run_residual_rows": total_residual_rows,
        "n_goship_line_files": final_source_files,
        "n_section_occupations": final_occupations,
        "n_residual_rows": final_residual_rows,
        "selected_line_filter": args.lines or "all",
        "temperature_residual_definition": "GO-SHIP in-situ temperature (ITS-90) minus IAP annual temperature",
        "salinity_residual_definition": "GO-SHIP Absolute Salinity converted from CTD practical salinity minus IAP Absolute Salinity",
        "section_blocking": "Each repeat hydrographic section occupation remains one bootstrap block; CTD samples are not independent bootstrap members.",
        "vertical_coordinate": "GO-SHIP pressure converted from IAP target depth using gsw.p_from_z at station latitude.",
        "horizontal_matching": "Bilinear interpolation of IAP 1-degree values, with periodic longitude and valid-neighbour renormalisation.",
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    set_owner_recursively(args.output_dir, args.owner, logger)
    logger.info(
        "Complete: occupations=%d, residual rows=%d, output=%s",
        final_occupations,
        final_residual_rows,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
