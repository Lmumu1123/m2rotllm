#!/usr/bin/env python3
"""Check the packaged migration file list and SHA-256 hashes without ML dependencies."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    m = json.loads((ROOT / 'migration/manifests/repository_files.json').read_text())
    errors, total = [], 0
    for row in m['files']:
        path = ROOT / row['path']
        if not path.is_file():
            errors.append(dict(path=row['path'], problem='missing'))
            continue
        if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            errors.append(dict(path=row['path'], problem='changed'))
        total += path.stat().st_size
    result = dict(status='passed' if not errors else 'failed', files=len(m['files']),
                  verified_bytes=total, errors=errors,
                  note='Manifest excludes itself and post-verification receipt; checks packaged files, not externally omitted data.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
