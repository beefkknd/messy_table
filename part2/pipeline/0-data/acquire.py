#!/usr/bin/env python3
"""Download the publisher archive in resumable, verified HTTP byte ranges.
Extract with ZIP CRC checks; never changes existing experiment data.
"""
import concurrent.futures, hashlib, json, os, pathlib, time, urllib.request, uuid, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
SIZE = 22625661806
CHUNK = 32 * 1024 * 1024
URL = 'https://drive.usercontent.google.com/download?id=1i4mJz9xsDwhzWes7sVLXuhLKP9eNtbBG&export=download&confirm=t'
ARCHIVE = ROOT / 'sources' / 'MessyTable.zip'
PARTIAL = ARCHIVE.with_suffix('.zip.partial')
LEDGER = ROOT / 'sources' / 'download-progress.json'

def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)

def fetch(index):
    start = index * CHUNK
    end = min(SIZE, start + CHUNK) - 1
    for attempt in range(8):
        try:
            req = urllib.request.Request(URL + '&slice=' + uuid.uuid4().hex,
                                         headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as response:
                expected = f'bytes {start}-{end}/{SIZE}'
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise ValueError(f'Unexpected range response: {response.status}, {response.headers.get("Content-Range")}')
                data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError(f'Incomplete chunk {index}: {len(data)} bytes')
            return index, data, hashlib.sha256(data).hexdigest()
        except Exception as exc:
            print(f'Retry chunk {index} attempt {attempt+1}: {type(exc).__name__}: {exc}', flush=True)
            if attempt == 7:
                raise
            time.sleep(min(2 ** attempt, 30))

def main():
    ROOT.joinpath('sources').mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        state = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'size': SIZE, 'chunk_size': CHUNK, 'chunks': {}}
        if state['size'] != SIZE or state['chunk_size'] != CHUNK:
            raise ValueError('Resume configuration mismatch')
        fd = os.open(PARTIAL, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            # Recheck saved ranges before trusting a resumed download.
            for key, digest in list(state['chunks'].items()):
                index = int(key)
                data = os.pread(fd, min(CHUNK, SIZE-index*CHUNK), index*CHUNK)
                if hashlib.sha256(data).hexdigest() != digest:
                    del state['chunks'][key]
            count = (SIZE + CHUNK - 1) // CHUNK
            pending = [i for i in range(count) if str(i) not in state['chunks']]
            print(f'Archive {SIZE/1e9:.2f} GB; {len(state["chunks"])}/{count} ranges already verified', flush=True)
            started = time.monotonic()
            downloaded = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = {pool.submit(fetch, i) for i in pending}
                for future in concurrent.futures.as_completed(futures):
                    index, data, digest = future.result()
                    futures.remove(future)
                    offset = index * CHUNK
                    written = 0
                    while written < len(data):
                        written += os.pwrite(fd, data[written:], offset + written)
                    state['chunks'][str(index)] = digest
                    downloaded += len(data)
                    save(LEDGER, state)
                    if len(state['chunks']) % 8 == 0 or len(state['chunks']) == count:
                        elapsed = time.monotonic()-started
                        print(f'Download {len(state["chunks"])}/{count} ranges; {downloaded/1e9:.2f} GB this run; {downloaded/max(elapsed,1)/1e6:.1f} MB/s', flush=True)
            os.fsync(fd)
        finally:
            os.close(fd)
        if PARTIAL.stat().st_size != SIZE:
            raise ValueError('Archive size mismatch')
        PARTIAL.replace(ARCHIVE)
    archive_hash = hashlib.sha256()
    with ARCHIVE.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            archive_hash.update(block)
    digest = archive_hash.hexdigest()
    print(f'Archive SHA256 {digest}', flush=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = archive.infolist()
        print(f'ZIP contains {len(members)} entries, {sum(m.file_size for m in members)/1e9:.2f} GB uncompressed', flush=True)
        for i, member in enumerate(members):
            path = ROOT / member.filename
            if not path.resolve().is_relative_to(ROOT):
                raise ValueError(f'Unsafe ZIP path: {member.filename}')
            # ZIP extraction checks each member CRC as it is decompressed.
            archive.extract(member, ROOT)
            if (i+1) % 5000 == 0:
                print(f'Extracted and CRC checked {i+1}/{len(members)} entries', flush=True)
        inventory = {'source_repository': 'https://github.com/caizhongang/MessyTable',
                     'source_url': URL, 'archive_bytes': ARCHIVE.stat().st_size,
                     'archive_sha256': digest, 'zip_entries': len(members),
                     'image_count': sum(m.filename.lower().endswith(('.jpg','.jpeg','.png')) for m in members),
                     'uncompressed_bytes': sum(m.file_size for m in members),
                     'label_files': [m.filename for m in members if m.filename.endswith('.json')],
                     'verification': 'Exact HTTP Content-Range and length per chunk; local SHA256 per chunk and archive; ZIP member CRC checked during extraction. Publisher checksum not provided.'}
    save(ROOT / 'sources' / 'inventory.json', inventory)
    print(json.dumps(inventory, indent=2), flush=True)

if __name__ == '__main__':
    main()
