"""Download native thermosteric OMI files from their public official STAC assets."""
from manuscript_paths import workspace_root, input_path
import argparse
import json
import logging
import os
from pathlib import Path
from urllib.parse import urljoin

import compare_noaa_thermosteric as common


STAC_BASE = 'https://stac.marine.copernicus.eu/metadata/'
PRODUCTS = {
    '0-700m': 'OMI_CLIMATE_THSL_GLOBAL_area_averaged_anomalies_0_700',
    '0-2000m': 'OMI_CLIMATE_THSL_GLOBAL_area_averaged_anomalies_0_2000',
}


def prepare(out):
    raw = out / 'raw_copernicus'
    raw.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('copernicus_download')
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    manifest_path = out / 'downloads.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []

    def fetch(url, path):
        previous = next((row for row in manifest if row['local_file'] == str(path)), None)
        if previous:
            if common.step1.sha256(path) != previous['sha256']:
                raise ValueError('Reference snapshot changed: ' + str(path))
        elif path.exists():
            raise ValueError('Reference exists without provenance: ' + str(path))
        else:
            log.info('Downloading %s', url)
            manifest.append(common.fetch(url, path, log))
            manifest_path.write_text(json.dumps(manifest, indent=2))

    files = {}
    versions = {}
    for band, product in PRODUCTS.items():
        product_url = STAC_BASE + product + '/product.stac.json'
        product_path = raw / (product + '.stac.json')
        fetch(product_url, product_path)
        collection = json.loads(product_path.read_text())
        links = [link for link in collection['links'] if link['rel'] == 'item']
        if len(links) != 1:
            raise ValueError('Expected one active dataset version; inspect changed STAC metadata')
        dataset_url = urljoin(product_url, links[0]['href'])
        dataset_path = raw / (product + '.dataset.stac.json')
        fetch(dataset_url, dataset_path)
        dataset = json.loads(dataset_path.read_text())
        versions[band] = dataset['id']
        native = dataset['assets']['native']
        if native['type'] != 'application/x-netcdf':
            raise ValueError('Expected native NetCDF series, not ARCO conversion')
        path = raw / (product + '.nc')
        fetch(native['href'], path)
        files[band] = path
        for link in collection['links']:
            if link['rel'] == 'describedby' and link.get('type') == 'application/pdf':
                fetch(link['href'], raw / link['href'].rsplit('/', 1)[-1])
    return files, versions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    try:
        files, versions = prepare(args.output_dir)
        print(json.dumps({'files': {key: str(p) for key, p in files.items()}, 'versions': versions}, indent=2))
    finally:
        if args.output_dir.exists():
            owner = args.output_dir.parent.parent.stat()
            for path in [args.output_dir] + list(args.output_dir.rglob('*')):
                os.chown(str(path), owner.st_uid, owner.st_gid)
                path.chmod(0o755 if path.is_dir() else 0o644)
