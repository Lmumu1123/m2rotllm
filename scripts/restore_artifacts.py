#!/usr/bin/env python3
"""Restore byte-identical Qwen weights / large result files; standard library only."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def valid_target(path):
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError('Manifest target outside this checkout')
    return target

def restore(target, parts, expected, size):
    dest = valid_target(target)
    if dest.exists():
        if dest.stat().st_size == size and digest(dest) == expected:
            print('Already verified:', target)
            return
        raise RuntimeError(f'Existing file differs; inspect it before replacing: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + '.restoring')
    h, length = hashlib.sha256(), 0
    try:
        with temporary.open('xb') as out:
            for item in parts:
                archive = valid_target(item['path'])
                if digest(archive) != item['archive_sha256']:
                    raise RuntimeError(f'Archive checksum mismatch: {archive}')
                piece, count = hashlib.sha256(), 0
                with gzip.open(archive, 'rb') as f:
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                        out.write(block)
                        piece.update(block)
                        h.update(block)
                        count += len(block)
                if piece.hexdigest() != item['sha256'] or count != item['bytes']:
                    raise RuntimeError(f'Restored chunk checksum mismatch: {archive}')
                length += count
        if h.hexdigest() != expected or length != size:
            raise RuntimeError(f'Restored file checksum mismatch: {dest}')
        os.replace(temporary, dest)
    except BaseException:
        # Do not delete a pre-existing .restoring file owned by another process.
        raise
    print('Restored and verified:', target, length)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qwen', action='store_true', help='Restore the 3.09 GB base language model')
    parser.add_argument('--results', action='store_true', help='Restore complete historical evaluation JSONL files')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if not (args.qwen or args.results or args.all):
        parser.error('Choose --qwen, --results, or --all')
    if args.qwen or args.all:
        m = json.loads((ROOT / 'migration/manifests/qwen_parts.json').read_text())
        restore(m['target'], m['parts'], m['sha256'], m['bytes'])
    if args.results or args.all:
        for m in json.loads((ROOT / 'migration/manifests/compressed_results.json').read_text()):
            part = dict(path=m['archive'], archive_sha256=m['archive_sha256'],
                        sha256=m['sha256'], bytes=m['bytes'])
            restore(m['target'], [part], m['sha256'], m['bytes'])

if __name__ == '__main__':
    main()
