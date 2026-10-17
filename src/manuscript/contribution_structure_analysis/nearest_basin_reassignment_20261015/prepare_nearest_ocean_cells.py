#!/usr/bin/env python3
"""Locate unassigned wet cells and assign the nearest existing basin on a sphere."""
from manuscript_paths import workspace_root, input_path
import hashlib
import json
import logging
import os
from pathlib import Path

import netCDF4 as nc
import numpy as np
from osgeo import gdal, ogr
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

HERE = (workspace_root() / 'contribution_structure_analysis/nearest_basin_reassignment_20261015')
ROOT = HERE.parent.parent
DEM = input_path('THERMOSTERIC_GEBCO', 'inputs/GEBCO_ocean.tif')
SHP = input_path('THERMOSTERIC_BASINS', 'inputs/IHO_ocean_basins')
YEARS = ROOT / 'Results/yearly_1940_2025'
BASINS = ('Arctic', 'Atlantic', 'Indian', 'Pacific', 'Southern')
SHAPES = ('Arctic_Ocean/Arctic_Ocean.shp', 'Atlantic_Ocean_Shp/Atlantic_Ocean.shp',
          'Indian_Ocean/Indian_Ocean.shp', 'Pacific_Ocean_Shp/Pacific_Ocean.shp',
          'Southern Ocean/Antarctic_Ocean.shp')


def signature():
    source = {}
    for relative in SHAPES:
        for path in sorted((SHP / relative).parent.glob(Path(relative).stem + '.*')):
            source[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = DEM.stat()
    return {'dem': str(DEM), 'dem_bytes': stat.st_size, 'dem_mtime_ns': stat.st_mtime_ns,
            'shapes_sha256': source, 'method': 'production_float32_coordinate_rasterization; nearest_assigned_wet_center_on_sphere_v1'}


def logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(str(HERE / 'preparation.log'))])
    return logging.getLogger('nearest_basin')


def xyz(lat, lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def prepare(log):
    final = HERE / 'nearest_assignments.npz'
    fingerprint = signature()
    if final.exists():
        metadata = json.loads((HERE / 'geometry.json').read_text())
        if metadata['signature'] != fingerprint:
            raise ValueError('Existing geometry has different inputs')
        log.info('Reusing nearest-cell assignments')
        return final
    with nc.Dataset(str(YEARS / '2025_vs_2024/2025_6000.nc')) as ds:
        lat = np.asarray(ds.variables['lat'][:], dtype=np.float64)
        lon = np.asarray(ds.variables['lon'][:], dtype=np.float64)
    if len(lat) != 43200 or len(lon) != 86400 or np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError('Unexpected main-grid coordinates')
    # Reproduce the production label geometry, including its coordinate precision.
    dx, dy = float(lon[1] - lon[0]), float(lat[1] - lat[0])
    gt = (float(lon[0] - dx / 2), dx, 0., float(lat[0] - dy / 2), 0., dy)
    labels_path = HERE / 'production_basin_labels_uint8.npy'
    gap_rows, gap_cols = [], []
    dem = gdal.Open(str(DEM), gdal.GA_ReadOnly)
    dem_band = dem.GetRasterBand(1)
    nodata = dem_band.GetNoDataValue()
    log.info('Rasterizing existing five-basin classification at native resolution')
    raster = gdal.GetDriverByName('MEM').Create('', len(lon), len(lat), 1, gdal.GDT_Byte)
    raster.SetGeoTransform(gt)
    raster.GetRasterBand(1).Fill(0)
    for index, relative in enumerate(SHAPES, 1):
        source = ogr.Open(str(SHP / relative), 0)
        if source is None or gdal.RasterizeLayer(raster, [1], source.GetLayer(0), burn_values=[index]):
            raise ValueError('Cannot reproduce basin geometry')
        log.info('Rasterized %s', BASINS[index - 1])
    labels = np.lib.format.open_memmap(str(labels_path), mode='w+', dtype=np.uint8,
                                      shape=(len(lat), len(lon)))
    for start in range(0, len(lat), 512):
        stop = min(start + 512, len(lat))
        raw = dem_band.ReadAsArray(0, len(lat) - stop, len(lon), stop - start)[::-1]
        wet = np.isfinite(raw) & (raw < 0)
        if nodata is not None:
            wet &= raw != nodata
        block = raster.GetRasterBand(1).ReadAsArray(0, start, len(lon), stop - start)
        block[~wet] = 0
        labels[start:stop] = block
        rr, cc = np.nonzero(wet & (block == 0))
        gap_rows.append((rr + start).astype(np.int32))
        gap_cols.append(cc.astype(np.int32))
        if start % (512 * 10) == 0:
            log.info('Wet-cell scan %d/%d rows; unassigned cells so far %d',
                     stop, len(lat), sum(len(a) for a in gap_rows))
    labels.flush()
    raster, dem, dem_band = None, None, None
    rows, cols = np.concatenate(gap_rows), np.concatenate(gap_cols)
    log.info('Located %d unassigned ocean cells (NoData/land excluded)', len(rows))
    np.savez_compressed(str(HERE / 'unassigned_cells.npz'), rows=rows, cols=cols)
    if not len(rows):
        raise ValueError('No spatially unassigned cells: arithmetic residual cannot be spatially allocated')
    boundary_rows, boundary_cols, boundary_ids = [], [], []
    for start in range(0, len(lat), 256):
        stop = min(start + 256, len(lat))
        lo, hi = max(0, start - 1), min(len(lat), stop + 1)
        occupied = np.asarray(labels[lo:hi]) > 0
        occupied = np.pad(occupied, ((0, 0), (1, 1)), mode='wrap')
        eroded = binary_erosion(occupied, structure=np.ones((3, 3), dtype=bool), border_value=0)
        edge = occupied[start - lo:stop - lo, 1:-1] & ~eroded[start - lo:stop - lo, 1:-1]
        rr, cc = np.nonzero(edge)
        boundary_rows.append((rr + start).astype(np.int32))
        boundary_cols.append(cc.astype(np.int32))
        boundary_ids.append(np.asarray(labels[rr + start, cc], dtype=np.uint8))
    rr, cc, ids = np.concatenate(boundary_rows), np.concatenate(boundary_cols), np.concatenate(boundary_ids)
    log.info('Building spherical nearest-neighbor tree from %d assigned wet boundary cells', len(rr))
    tree = cKDTree(xyz(lat[rr], lon[cc]))
    assigned, distances = np.empty(len(rows), dtype=np.uint8), np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), 250000):
        stop = min(start + 250000, len(rows))
        distance, index = tree.query(xyz(lat[rows[start:stop]], lon[cols[start:stop]]), k=1)
        assigned[start:stop] = ids[index]
        distances[start:stop] = (2. * np.arcsin(np.minimum(distance / 2., 1.)) * 6371.).astype(np.float32)
    if not np.isin(assigned, np.arange(1, 6)).all():
        raise ValueError('Invalid nearest-basin assignment')
    # Chunk ordering allows sparse sampling from original compressed NetCDF fields.
    chunk_key = (rows // 128).astype(np.int64) * 22 + cols // 4096
    order = np.argsort(chunk_key, kind='stable')
    np.savez_compressed(str(final), rows=rows[order], cols=cols[order], basin_id=assigned[order],
                        distance_km=distances[order], chunk_key=chunk_key[order], lat=lat, lon=lon)
    metadata = {'signature': fingerprint, 'unassigned_wet_pixels': len(rows),
                'source_basin_order': list(BASINS), 'production_label_geotransform': gt,
                'nearest_distance_km_percentiles': {str(q): float(np.percentile(distances, q)) for q in (0, 50, 95, 99, 100)},
                'assigned_pixel_counts': {name: int(np.count_nonzero(assigned == i)) for i, name in enumerate(BASINS, 1)},
                'distance': 'great-circle distance between original grid cell centers; dateline periodicity handled in 3D',
                'existing_classified_pixels_changed': False, 'land_or_nodata_assigned': False}
    (HERE / 'geometry.json').write_text(json.dumps(metadata, indent=2))
    log.info('Saved assignments: %s', metadata['assigned_pixel_counts'])
    log.info('Nearest distance quantiles km: %s', metadata['nearest_distance_km_percentiles'])
    return final


def sample_year(year, assignments, log):
    path = YEARS / ('{}_vs_{}'.format(year, year - 1)) / ('{}_6000.nc'.format(year))
    before = path.stat()
    with np.load(str(assignments)) as data:
        rows, cols, ids, keys = [data[key] for key in ('rows', 'cols', 'basin_id', 'chunk_key')]
        lat, lon = data['lat'], data['lon']
    cuts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
    totals = np.zeros(6, dtype=np.float64)
    finite = 0
    with nc.Dataset(str(path), 'r') as ds:
        np.testing.assert_array_equal(ds.variables['lat'][:], lat)
        np.testing.assert_array_equal(ds.variables['lon'][:], lon)
        variable = ds.variables['thermal_expansion_visual_filled_km3']
        if variable.chunking() != [128, 4096]:
            raise ValueError('Unexpected original NetCDF chunk layout')
        variable.set_auto_mask(False)
        for k, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
            r0, c0 = int(rows[start] // 128 * 128), int(cols[start] // 4096 * 4096)
            block = variable[r0:min(r0 + 128, len(lat)), c0:min(c0 + 4096, len(lon))]
            values = np.asarray(block[rows[start:stop] - r0, cols[start:stop] - c0], dtype=np.float64)
            valid = np.isfinite(values)
            totals += np.bincount(ids[start:stop][valid], weights=values[valid], minlength=6)
            finite += int(valid.sum())
            if k % 500 == 0:
                log.info('%d sparse map chunks %d/%d', year, k, len(cuts) - 1)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Original NetCDF changed while reading')
    record = {'year': year, 'source_nc_bytes': before.st_size, 'source_nc_mtime_ns': before.st_mtime_ns,
              'finite_unassigned_pixels': finite, 'sampled_nc_chunks': len(cuts) - 1,
              'basin_corrections_mm': {name: float(totals[i] / 361.9) for i, name in enumerate(BASINS, 1)},
              'spatial_unassigned_total_mm': float(totals.sum() / 361.9), 'source_nc_read_only': True}
    (HERE / ('correction_{}.json'.format(year))).write_text(json.dumps(record, indent=2))
    log.info('%d spatial correction total %.12g mm: %s', year, record['spatial_unassigned_total_mm'], record['basin_corrections_mm'])
    return record


if __name__ == '__main__':
    HERE.mkdir(exist_ok=True)
    try:
        log = logger()
        path = prepare(log)
        sample_year(2025, path, log)
    finally:
        owner = ROOT.stat()
        for item in [HERE] + list(HERE.rglob('*')):
            os.chown(str(item), owner.st_uid, owner.st_gid)
            item.chmod(0o755 if item.is_dir() else 0o644)
