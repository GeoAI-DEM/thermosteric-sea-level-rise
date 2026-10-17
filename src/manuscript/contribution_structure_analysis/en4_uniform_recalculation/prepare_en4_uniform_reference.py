"""Download a prespecified EN4 objective-analysis version from its official index."""
from manuscript_paths import workspace_root, input_path
import argparse
import hashlib
import json
import logging
import os
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin


INDEX = 'https://www.metoffice.gov.uk/hadobs/en4/download-en4-2-2.html'
FAQ = 'https://www.metoffice.gov.uk/hadobs/en4/FAQ.html'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = {}

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == 'a' and 'href' in attributes:
            url = urljoin(INDEX, attributes['href'])
            self.links[url.rsplit('/', 1)[-1]] = url


def log_setup(path):
    log = logging.getLogger('en4_uniform')
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(str(path), encoding='utf-8')):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_index(out):
    raw = out / 'raw_en4'
    raw.mkdir(parents=True, exist_ok=True)
    index_path = raw / 'official_download_index.html'
    if not index_path.exists():
        with urllib.request.urlopen(INDEX, timeout=90) as response:
            index_path.write_bytes(response.read())
    faq_path = raw / 'official_faq.html'
    if not faq_path.exists():
        with urllib.request.urlopen(FAQ, timeout=90) as response:
            faq_path.write_bytes(response.read())
    parser = Links()
    parser.feed(index_path.read_text(encoding='utf-8'))
    return parser.links


def monthly_members(path, year):
    with zipfile.ZipFile(str(path)) as archive:
        names = [name for name in archive.namelist() if name.endswith('.nc')]
    months = {}
    for month in range(1, 13):
        candidates = [name for name in names if name.endswith('{}{:02d}.nc'.format(year, month))]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one monthly analysis for {}-{:02d}'.format(year, month))
        months[month] = candidates[0]
    if len(names) != 12:
        raise ValueError('Unexpected extra NetCDF records in annual archive')
    return months


def download_year(year, correction, url, out, log):
    raw = out / 'raw_en4'
    filename = 'EN.4.2.2.analyses.{}.{}.zip'.format(correction, year)
    path = raw / filename
    record_path = raw / (filename + '.json')
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if record['url'] != url or path.stat().st_size != record['bytes']:
            raise ValueError('Cached download identity changed: ' + str(path))
        monthly_members(path, year)
        log.info('Reuse EN4 %d archive: %.1f MB', year, record['bytes'] / 1e6)
        return path, record
    if path.exists():
        raise ValueError('Existing reference archive has no provenance record: ' + str(path))
    partial = path.with_suffix('.zip.part')
    error = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'EN4UniformThermostericComparison/1.0'})
            log.info('Download EN4 %d attempt %d/5: %s', year, attempt + 1, url)
            started = last_print = time.monotonic()
            digest = hashlib.sha256()
            count = 0
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
                headers = dict(response.headers.items())
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    stream.write(block)
                    digest.update(block)
                    count += len(block)
                    now = time.monotonic()
                    if now - last_print >= 20:
                        log.info('EN4 %d downloaded %.1f MB, mean %.2f MB/s', year, count / 1e6,
                                 count / 1e6 / (now - started))
                        last_print = now
            expected = headers.get('Content-Length')
            if not count or (expected is not None and count != int(expected)):
                raise ValueError('Incomplete reference download')
            monthly_members(partial, year)
            partial.replace(path)
            record = {'year': year, 'correction': correction, 'url': url, 'bytes': count,
                      'sha256': digest.hexdigest(), 'http_headers': headers,
                      'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                      'download_seconds': time.monotonic() - started}
            record_path.write_text(json.dumps(record, indent=2))
            log.info('EN4 %d archive complete: %.1f MB in %.1f s', year, count / 1e6,
                     record['download_seconds'])
            return path, record
        except Exception as exc:
            error = exc
            log.warning('EN4 %d download attempt %d failed: %s', year, attempt + 1, exc)
            if attempt < 4:
                time.sleep(min(30, 2**(attempt + 1)))
    raise RuntimeError('EN4 {} download failed after five attempts: {}'.format(year, error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--year', type=int, default=2005)
    parser.add_argument('--correction', choices=['g10', 'l09', 'c13', 'c14'], default='g10')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        log = log_setup(args.output_dir / 'download_preparation.log')
        urls = prepare_index(args.output_dir)
        filename = 'EN.4.2.2.analyses.{}.{}.zip'.format(args.correction, args.year)
        path, record = download_year(args.year, args.correction, urls[filename], args.output_dir, log)
        print('ARCHIVE=' + str(path))
        print(json.dumps(monthly_members(path, args.year), indent=2))
    finally:
        owner = args.output_dir.parent.parent.parent.stat()
        for p in [args.output_dir] + list(args.output_dir.rglob('*')):
            os.chown(str(p), owner.st_uid, owner.st_gid)
