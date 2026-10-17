"""Download small official snapshots and check annual text against monthly grids."""
from manuscript_paths import workspace_root, input_path
import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from osgeo import gdal

import compare_noaa_thermosteric as common

BASE = 'https://climate.mri-jma.go.jp/pub/ocean/ts/v7.3.1/'


def first_complete_grib_messages(payload):
    end = 0
    while end < len(payload):
        if payload[end:end + 4] != b'GRIB':
            raise ValueError('Unexpected GRIB framing at byte {}'.format(end))
        if end + 8 > len(payload):
            break
        if payload[end + 7] != 1:
            raise ValueError('Expected GRIB edition 1')
        size = int.from_bytes(payload[end + 4:end + 7], 'big')
        if size < 12:
            raise ValueError('Invalid GRIB message size')
        if end + size > len(payload):
            break
        if payload[end + size - 4:end + size] != b'7777':
            raise ValueError('Missing GRIB terminator')
        end += size
    if end == 0:
        raise ValueError('No complete GRIB messages')
    return payload[:end]


def annual_grid_check(path, text_path):
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    try:
        dataset = gdal.Open(str(path))
    finally:
        gdal.PopErrorHandler()
    if dataset is None:
        raise ValueError('Cannot open official GRIB prefix')
    transform = dataset.GetGeoTransform()
    latitude = transform[3] + (np.arange(dataset.RasterYSize) + .5) * transform[5]
    weights = np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None],
                              (dataset.RasterYSize, dataset.RasterXSize))
    samples = []
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        meta = band.GetMetadata()
        if i <= 2:
            print('GRID_METADATA', path.name, i, meta, flush=True)
        # The official ctl identifies parameter 131 as anomaly, 132 as error.
        if meta.get('GRIB_ELEMENT') != 'var131 of table 2 of center 34':
            continue
        if len(samples) == 24:
            break
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        try:
            field = band.ReadAsArray().astype(np.float64)
        finally:
            gdal.PopErrorHandler()
        valid = np.isfinite(field) & (np.abs(field) < 100)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            valid &= field != nodata
        samples.append({'reference_time': meta.get('GRIB_REF_TIME'),
                        'valid_time': meta.get('GRIB_VALID_TIME'),
                        'weighted_mean_m': float(np.sum(field[valid] * weights[valid]) / weights[valid].sum()),
                        'valid_cell_count': int(valid.sum())})
    if len(samples) != 24:
        raise ValueError('Need 24 monthly fields, found {}'.format(len(samples)))
    table = np.loadtxt(str(text_path))
    dates = [datetime.fromtimestamp(int(row['valid_time']), timezone.utc) for row in samples]
    if [(date.year, date.month) for date in dates] != [(int(table[0, 0]) + y, m) for y in range(2) for m in range(1, 13)]:
        raise ValueError('Grid prefix must cover January-December of first two text years')
    means_m = [np.mean([row['weighted_mean_m'] for row in samples[start:start + 12]]) for start in (0, 12)]
    delta_m = means_m[1] - means_m[0]
    delta_text = table[1, 1] - table[0, 1]
    result = {'first_text_year': int(table[0, 0]), 'text_first_anomaly': float(table[0, 1]),
              'first_two_annual_means_m': means_m,
              'annual_grid_change_m': float(delta_m), 'annual_text_change': float(delta_text),
              'text_change_to_grid_change_ratio': float(delta_text / delta_m),
              'interpretation': 'Annual changes remove an arbitrary grid/text reference offset; ctl gives grid units in m.',
              'months': samples, 'geotransform': transform}
    print('UNIT_CHECK', path.name, json.dumps(result), flush=True)
    return result


def prepare(out):
    import logging
    log = logging.getLogger('ishii_download')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    raw = out / 'raw_ishii'
    raw.mkdir(parents=True, exist_ok=True)
    provenance_path = out / 'downloads.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else []
    files = ['Thermosteric_0-700.txt', 'Thermosteric_700-2000.txt', 'Thermosteric_0-2000.txt',
             'doc/00README_v7.3.1', 'Thermosteric_700m.ctl', 'Thermosteric_2000m.ctl']
    for name in files:
        destination = raw / Path(name).name
        if not destination.exists():
            provenance.append(common.fetch(BASE + name, destination, log))
        elif not any(p['local_file'] == str(destination) for p in provenance):
            provenance.append({'url': BASE + name, 'local_file': str(destination),
                               'sha256': common.step1.sha256(destination),
                               'bytes': destination.stat().st_size,
                               'retrieval_note': 'Reused official snapshot from initial unit-check attempt'})
        provenance_path.write_text(json.dumps(provenance, indent=2))
    checks = {}
    for depth in ('700', '2000'):
        url = BASE + 'Thermosteric_{}m.grb'.format(depth)
        destination = raw / 'Thermosteric_{}m_first_two_years.grb'.format(depth)
        if not destination.exists():
            request = urllib.request.Request(url, headers={'Range': 'bytes=0-4999999'})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read(5000000)
                headers = dict(response.headers.items())
                status = response.status
            destination.write_bytes(first_complete_grib_messages(payload))
            provenance.append({'url': url, 'local_file': str(destination),
                               'http_status': status, 'response_headers': headers,
                               'requested_prefix_bytes': 5000000,
                               'saved_complete_message_bytes': destination.stat().st_size,
                               'sha256': common.step1.sha256(destination),
                               'retrieved_utc': datetime.now(timezone.utc).isoformat()})
        elif not any(p['local_file'] == str(destination) for p in provenance):
            provenance.append({'url': url, 'local_file': str(destination),
                               'sha256': common.step1.sha256(destination),
                               'saved_complete_message_bytes': destination.stat().st_size,
                               'retrieval_note': 'Reused official prefix from initial unit-check attempt'})
        provenance_path.write_text(json.dumps(provenance, indent=2))
        checks['0-' + depth + 'm'] = annual_grid_check(destination, raw / ('Thermosteric_0-{}.txt'.format(depth)))
    (out / 'unit_checks.json').write_text(json.dumps({'checks': checks, 'downloads': provenance}, indent=2))
    return checks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    try:
        prepare(args.output_dir)
    finally:
        owner = args.output_dir.parent.parent.stat()
        if args.output_dir.exists():
            for path in [args.output_dir] + list(args.output_dir.rglob('*')):
                os.chown(str(path), owner.st_uid, owner.st_gid)
