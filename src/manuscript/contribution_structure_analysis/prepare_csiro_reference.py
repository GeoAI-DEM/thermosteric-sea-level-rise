"""Retrieve the public BODC Domingues 2026 v3 archive and its metadata."""
from manuscript_paths import workspace_root, input_path
import argparse
import json
import logging
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import compare_noaa_thermosteric as common


UUID = '4e025925-dc1f-3753-e063-7086abc01cce'
DOI = '10.5285/' + UUID
API = 'https://www.bodc.ac.uk/data/published_data_library/api'


def post_json(url):
    request = urllib.request.Request(url, data=b'', method='POST')
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def prepare(out):
    raw = out / 'raw_csiro'
    raw.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('csiro_reference')
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    provenance_file = out / 'downloads.json'
    provenance = json.loads(provenance_file.read_text()) if provenance_file.exists() else []
    meta_path = raw / 'bodc_dataset_metadata.json'
    if not meta_path.exists():
        provenance.append(common.fetch(API + '/doi/' + UUID, meta_path, log))
        provenance_file.write_text(json.dumps(provenance, indent=2))
    metadata = json.loads(meta_path.read_text())
    if metadata.get('endEmbargo') or metadata.get('replacedBy'):
        raise ValueError('Dataset is embargoed or superseded; inspect metadata first')
    zip_path = raw / 'domingues_1960_2025_v3.zip'
    if not zip_path.exists():
        result = post_json(API + '/download/prepare/' + UUID)
        (raw / 'download_prepare_response.json').write_text(json.dumps(result, indent=2))
        if result.get('submitstatus') != 'success':
            raise RuntimeError('Public download preparation refused: ' + repr(result))
        for attempt in range(30):
            result = post_json(API + '/download/check/' + UUID)
            if result.get('status') == 'ready':
                filename = result.get('filename')
                if not filename or '/' in filename or '\\' in filename:
                    raise ValueError('Unexpected prepared archive filename')
                url = 'https://www.bodc.ac.uk/data/published_data_library/catalogue/10.5285/123/' + filename
                provenance.append(common.fetch(url, zip_path, log))
                provenance_file.write_text(json.dumps(provenance, indent=2))
                break
            if result.get('status') != 'notready':
                raise RuntimeError('Public download not available: ' + repr(result))
            time.sleep(3)
        else:
            raise RuntimeError('Public archive preparation did not finish in 90 seconds')
        (raw / 'download_check_response.json').write_text(json.dumps(result, indent=2))
    with zipfile.ZipFile(str(zip_path)) as archive:
        manifest = []
        for member in archive.infolist():
            manifest.append({'member': member.filename, 'size_bytes': member.file_size})
            print('ARCHIVE_MEMBER', member.filename, member.file_size, flush=True)
            if member.is_dir():
                continue
            if member.file_size > 20 * 1024 * 1024:
                raise ValueError('Unexpected large reference file; inspect before extraction')
            destination = raw / Path(member.filename).name
            if destination.exists():
                if destination.read_bytes() != archive.read(member):
                    raise ValueError('Existing extracted reference differs: ' + str(destination))
                continue
            destination.write_bytes(archive.read(member))
        (raw / 'archive_manifest.json').write_text(json.dumps(manifest, indent=2))
    return metadata


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
