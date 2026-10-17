"""Input adaptation and shared legacy-formula calculations for the EN4 comparison."""
from manuscript_paths import workspace_root, input_path
import hashlib
import json
import time
import zipfile
from pathlib import Path

import gsw
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import prepare_en4_uniform_reference as download


CACHE_VERSION = 'en4-g10-monthly-pt-SP-to-t-SA-nanmean-periodic-linear-v1'
GROUPS = ('0-700m', '700-2000m', '0-2000m')
FIELDS = ('iap_t', 'iap_sa', 'en4_t', 'en4_sa')


def group_mask(depth, group):
    if group == '0-700m':
        return depth <= 700.
    if group == '700-2000m':
        return (depth > 700.) & (depth <= 2000.)
    if group == '0-2000m':
        return depth <= 2000.
    raise ValueError('Unsupported depth group')


def decoded_temperature_c(variable, values):
    unit = getattr(variable, 'units', '').lower()
    if unit in ('kelvin', 'k'):
        # netCDF4 has already applied scale_factor/add_offset: decoded values are kelvin.
        return values - 273.15
    if unit in ('degree celsius', 'degrees celsius', 'degree_celsius', 'degrees_celsius', 'celsius',
                'degree celsius (in situ)', 'degrees celsius (in situ)'):
        return values
    raise ValueError('Unsupported temperature units: ' + repr(unit))


def horizontal(values, lat, lon, target_lat, target_lon):
    lat, lon = np.asarray(lat), (np.asarray(lon) + 180.) % 360. - 180.
    yi, xi = np.argsort(lat), np.argsort(lon)
    lat, lon = lat[yi], lon[xi]
    values = np.asarray(values, dtype=np.float64)[:, yi, :][:, :, xi]
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError('Source horizontal coordinates must be unique')
    if np.array_equal(lat, target_lat) and np.array_equal(lon, target_lon):
        return values
    cyclic_lon = np.r_[lon[-1] - 360., lon, lon[0] + 360.]
    cyclic = np.concatenate([values[:, :, -1:], values, values[:, :, :1]], axis=2)
    lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing='ij')
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    valid = np.isfinite(cyclic)
    data = np.where(valid, cyclic, 0.).transpose(1, 2, 0)
    support = valid.astype(np.float64).transpose(1, 2, 0)
    interpolate = RegularGridInterpolator((lat, cyclic_lon), data, bounds_error=False, fill_value=np.nan)
    interpolate_support = RegularGridInterpolator((lat, cyclic_lon), support, bounds_error=False, fill_value=0.)
    result = interpolate(points)
    confidence = interpolate_support(points)
    result[confidence < 1. - 1e-12] = np.nan
    return result.reshape(len(target_lat), len(target_lon), len(values)).transpose(2, 0, 1)


def vertical(values, source_depth, target_depth):
    source_depth, target_depth = np.asarray(source_depth), np.asarray(target_depth)
    if np.any(np.diff(source_depth) <= 0):
        raise ValueError('Source depths must increase')
    if np.array_equal(source_depth, target_depth):
        return values
    if np.any(target_depth > source_depth[-1]):
        raise ValueError('Deep vertical extrapolation is forbidden')
    right = np.searchsorted(source_depth, target_depth, side='left')
    right = np.clip(right, 0, len(source_depth) - 1)
    left = np.maximum(right - 1, 0)
    denominator = source_depth[right] - source_depth[left]
    weight = np.divide(target_depth - source_depth[left], denominator,
                       out=np.zeros(len(target_depth)), where=denominator > 0)
    weight = np.clip(weight, 0., 1.)[:, None, None]
    a, b = values[left], values[right]
    va, vb = np.isfinite(a), np.isfinite(b)
    result = (1. - weight) * np.where(va, a, 0.) + weight * np.where(vb, b, 0.)
    support = (1. - weight) * va + weight * vb
    result[support < 1. - 1e-12] = np.nan
    return result


def read_iap(year, annual_dir, target_depth, target_lat, target_lon):
    fields, metadata = {}, {}
    for label, name, key in [('Temp', 'temp', 'iap_t'), ('Salinity', 'salinity', 'iap_sa')]:
        path = annual_dir / 'Annual_{}_{}.nc'.format(label, year)
        before = download.sha256(path)
        with nc.Dataset(str(path), 'r') as dataset:
            variable = dataset.variables[name]
            if variable.dimensions != ('lat', 'lon', 'depth_std'):
                raise ValueError('Unexpected IAP variable dimensions: ' + str(path))
            source_depth = np.asarray(dataset.variables['depth_std'][:], dtype=np.float64)
            keep = np.flatnonzero(source_depth <= 2000.)
            if not np.array_equal(source_depth[keep], target_depth):
                raise ValueError('IAP depths must match existing volume reference depths')
            values = np.asarray(np.ma.filled(variable[:, :, :len(keep)], np.nan), dtype=np.float64).transpose(2, 0, 1)
            lat = np.asarray(dataset.variables['lat'][:], dtype=np.float64)
            lon = np.asarray(dataset.variables['lon'][:], dtype=np.float64)
            attrs = {k: str(getattr(variable, k)) for k in variable.ncattrs()}
            if key == 'iap_t':
                values = decoded_temperature_c(variable, values)
            elif getattr(variable, 'units', '').lower() not in ('absolute salinity (g/kg)', 'g/kg', 'g kg-1'):
                raise ValueError('IAP salinity is not declared absolute; inspect before harmonizing')
        if download.sha256(path) != before:
            raise RuntimeError('IAP file changed while reading')
        fields[key] = horizontal(values, lat, lon, target_lat, target_lon)
        metadata[key] = {'path': str(path), 'sha256': before, 'attributes': attrs,
                         'salinity_handling': 'retain_as_declared_SA_no_second_conversion' if key == 'iap_sa' else '',
                         'longitude_first_three': lon[:3].tolist()}
    return fields, metadata


def accumulate(total, count, values):
    valid = np.isfinite(values)
    total += np.where(valid, values, 0.)
    count += valid


def divide_mean(total, count):
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def en4_monthly_to_annual(path, year, target_depth, target_lat, target_lon, log):
    members = download.monthly_members(path, year)
    totals, counts, month_metadata = {}, {}, []
    reference_coords = None
    with zipfile.ZipFile(str(path)) as archive:
        for month in range(1, 13):
            name = members[month]
            # Read original monthly bytes in memory; no extracted monthly files are modified or discarded.
            with nc.Dataset(name, memory=archive.read(name)) as dataset:
                time_var = dataset.variables['time']
                date = nc.num2date(time_var[:], time_var.units, calendar=getattr(time_var, 'calendar', 'standard'))[0]
                if (date.year, date.month) != (year, month):
                    raise ValueError('Native EN4 CF date differs from requested month')
                depth = np.asarray(dataset.variables['depth'][:], dtype=np.float64)
                stop = min(len(depth), int(np.searchsorted(depth, target_depth[-1], side='left')) + 1)
                depth = depth[:stop]
                lat = np.asarray(dataset.variables['lat'][:], dtype=np.float64)
                lon = np.asarray(dataset.variables['lon'][:], dtype=np.float64)
                coords = (depth, lat, lon)
                if reference_coords is None:
                    reference_coords = coords
                elif not all(np.array_equal(a, b) for a, b in zip(coords, reference_coords)):
                    raise ValueError('Monthly EN4 native coordinates changed')
                temperature_var = dataset.variables['temperature']
                salinity_var = dataset.variables['salinity']
                if temperature_var.standard_name != 'sea_water_potential_temperature':
                    raise ValueError('Expected EN4 potential temperature')
                if salinity_var.units != '1' or salinity_var.standard_name != 'sea_water_salinity':
                    raise ValueError('Expected EN4 practical salinity as documented in official FAQ')
                pt = np.asarray(np.ma.filled(temperature_var[0, :stop], np.nan), dtype=np.float64)
                pt = decoded_temperature_c(temperature_var, pt)
                sp = np.asarray(np.ma.filled(salinity_var[0, :stop], np.nan), dtype=np.float64)
                pressure = depth[:, None, None]
                sa = gsw.SA_from_SP(sp, pressure, lon[None, None, :] % 360., lat[None, :, None])
                ct = gsw.CT_from_pt(sa, pt)
                in_situ = gsw.t_from_CT(sa, ct, pressure)
                fields = {'en4_t': in_situ, 'en4_sa': sa}
                for key, variable in [('en4_t_weight', 'temperature_observation_weights'),
                                      ('en4_s_weight', 'salinity_observation_weights')]:
                    fields[key] = np.asarray(np.ma.filled(dataset.variables[variable][0, :stop], np.nan), dtype=np.float64)
                for key, values in fields.items():
                    if key not in totals:
                        totals[key], counts[key] = np.zeros_like(values), np.zeros(values.shape, dtype=np.uint8)
                    accumulate(totals[key], counts[key], values)
                month_metadata.append({'member': name, 'native_cf_time': date.isoformat(),
                                       'temperature_units_decoded': temperature_var.units,
                                       'temperature_add_offset': float(getattr(temperature_var, 'add_offset', 0.)),
                                       'temperature_type': temperature_var.standard_name,
                                       'salinity_units': salinity_var.units,
                                       'global_start_date_not_used': str(getattr(dataset, 'start_date', ''))})
            log.info('EN4 %d month %02d converted PT/SP to in-situ T/SA', year, month)
    mapped = {}
    for key in totals:
        annual = divide_mean(totals[key], counts[key])
        mapped[key] = vertical(horizontal(annual, lat, lon, target_lat, target_lon), depth, target_depth)
    complete = ((counts['en4_t'] == 12) & (counts['en4_sa'] == 12)).astype(np.float64)
    complete = vertical(horizontal(complete, lat, lon, target_lat, target_lon), depth, target_depth)
    mapped['en4_complete_year'] = np.isfinite(complete) & (complete >= 1. - 1e-12)
    return mapped, {'months': month_metadata, 'source_depth': depth.tolist(),
                    'surface_extension_to_m': float(depth[0]),
                    'annual_mean': 'equal_month_nanmean_after_monthly_native_T_SA_conversion',
                    'native_latitude_extent': [float(lat[0]), float(lat[-1])],
                    'native_longitude_extent': [float(lon[0]), float(lon[-1])],
                    'temperature_valid_month_range': [int(counts['en4_t'].min()), int(counts['en4_t'].max())]}


def grid_fingerprint(depth, lat, lon):
    digest = hashlib.sha256(CACHE_VERSION.encode('ascii'))
    for array in (depth, lat, lon):
        digest.update(np.asarray(array, dtype='<f8').tobytes())
    return digest.hexdigest()


def year_worker(year, correction, url, out_string, annual_dir_string, depth, lat, lon):
    out, annual_dir = Path(out_string), Path(annual_dir_string)
    log = download.log_setup(out / 'annual_processing' / '{}.log'.format(year))
    archive, record = download.download_year(year, correction, url, out, log)
    cache_path = out / 'annual_cache' / '{}.npz'.format(year)
    meta_path = cache_path.with_suffix('.json')
    fingerprint = grid_fingerprint(depth, lat, lon)
    if cache_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata['grid_fingerprint'] != fingerprint or metadata['archive_sha256'] != record['sha256']:
            raise ValueError('Annual cache configuration changed')
        for value in metadata['iap_inputs'].values():
            if download.sha256(Path(value['path'])) != value['sha256']:
                raise ValueError('IAP input changed since annual cache creation')
        if download.sha256(cache_path) != metadata['annual_cache_sha256']:
            raise ValueError('Annual cache contents changed')
        log.info('Reuse annual cache %d', year)
        return str(cache_path)
    started = time.monotonic()
    if download.sha256(archive) != record['sha256']:
        raise ValueError('Downloaded EN4 archive no longer matches its hash')
    fields, iap_metadata = read_iap(year, annual_dir, depth, lat, lon)
    en4_fields, en4_metadata = en4_monthly_to_annual(archive, year, depth, lat, lon, log)
    fields.update(en4_fields)
    temporary = cache_path.with_suffix('.npz.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **fields)
    temporary.replace(cache_path)
    metadata = {'year': year, 'correction': correction, 'grid_fingerprint': fingerprint,
                'archive_sha256': record['sha256'], 'iap_inputs': iap_metadata,
                'en4_adaptation': en4_metadata, 'processing_seconds': time.monotonic() - started,
                'annual_cache_sha256': download.sha256(cache_path), 'cache_version': CACHE_VERSION,
                'new_common_mask': 'annual_fields_only_final_all_year_intersection_applied_later'}
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info('Annual state %d ready in %.1f s', year, metadata['processing_seconds'])
    return str(cache_path)


def common_mask(states, volumes):
    mask = np.isfinite(volumes) & (volumes > 0.)
    for state in states:
        for key in FIELDS:
            if state[key].shape != mask.shape:
                raise ValueError('All annual states must share the volume grid')
            mask &= np.isfinite(state[key])
    return mask


def layer_changes(t0, sa0, t1, sa1, depth, volumes, mask):
    output = np.zeros_like(volumes, dtype=np.float64)
    for k, z in enumerate(depth):
        valid = mask[k]
        if not valid.any():
            continue
        ct0 = gsw.CT_from_t(sa0[k][valid], t0[k][valid], z)
        ct1 = gsw.CT_from_t(sa1[k][valid], t1[k][valid], z)
        alpha0 = gsw.alpha(sa0[k][valid], ct0, z)
        alpha1 = gsw.alpha(sa1[k][valid], ct1, z)
        changes = volumes[k][valid] * ((alpha0 + alpha1) / 2.) * (ct1 - ct0)
        if not np.isfinite(changes).all():
            raise ValueError('Nonfinite TEOS-10 result; refusing silent zero replacement')
        output[k][valid] = changes
    return output
