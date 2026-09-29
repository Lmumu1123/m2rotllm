#!/usr/bin/env python3
"""Reproduce the archived Qwen gzip parts without running any Git command.

All missing/incorrect parts are staged and SHA-verified before installing any.
The manifest and source file are never modified. Python standard library only.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib

REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
MODEL_SHA256 = 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
MODEL_BYTES = 3087467144
URL = f'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/{REVISION}/model.safetensors?download=true'
PART_PREFIX = PurePosixPath('migration/model_parts/qwen_model')
BLOCK = 8 * 1024 * 1024


def progress(event, **kwargs):
    print(json.dumps({'event': event, **kwargs}, ensure_ascii=False), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(BLOCK), b''):
            h.update(data)
    return h.hexdigest()


def part_path(repo, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.parent != PART_PREFIX or p.suffix != '.gz':
        raise ValueError('unexpected_part_destination')
    result = repo / p.as_posix()
    if result.resolve().is_relative_to(repo) is False:
        raise ValueError('part_destination_escapes_repository')
    if result.is_symlink():
        raise ValueError('part_destination_is_symlink')
    return result


def validate_manifest(info, repo):
    if info.get('sha256') != MODEL_SHA256 or info.get('bytes') != MODEL_BYTES:
        raise ValueError('manifest_base_weight_identity_mismatch')
    if info.get('revision') != REVISION:
        raise ValueError('manifest_revision_mismatch')
    parts = info['parts']
    if sum(p['bytes'] for p in parts) != MODEL_BYTES:
        raise ValueError('manifest_part_sizes_do_not_sum_to_model_size')
    names = [p['path'] for p in parts]
    if len(set(names)) != len(names):
        raise ValueError('manifest_contains_duplicate_part_paths')
    for part in parts:
        part_path(repo, part['path'])
        if not (0 < part['bytes'] <= 64 * 1024 * 1024):
            raise ValueError('manifest_part_size_out_of_bounds')
        for field in ('sha256', 'archive_sha256'):
            if len(part[field]) != 64 or any(c not in '0123456789abcdef' for c in part[field]):
                raise ValueError('invalid_manifest_hash')
    return parts


def archive_matches(path, part):
    return (path.is_file() and not path.is_symlink()
            and path.stat().st_size == part['archive_bytes']
            and sha256(path) == part['archive_sha256'])


def verify_archives_full(repo, parts):
    h = hashlib.sha256()
    total = 0
    for i, part in enumerate(parts):
        p = part_path(repo, part['path'])
        if not archive_matches(p, part):
            raise ValueError('existing_archive_verification_failed')
        ph = hashlib.sha256()
        n = 0
        with gzip.open(p, 'rb') as stream:
            for block in iter(lambda: stream.read(BLOCK), b''):
                h.update(block); ph.update(block)
                n += len(block); total += len(block)
        if n != part['bytes'] or ph.hexdigest() != part['sha256']:
            raise ValueError('existing_part_raw_hash_mismatch')
        if (i + 1) % 10 == 0 or i + 1 == len(parts):
            progress('verified_existing_parts', completed=i + 1, total=len(parts))
    if h.hexdigest() != MODEL_SHA256 or total != MODEL_BYTES:
        raise ValueError('existing_parts_full_model_hash_mismatch')
    progress('full_raw_verified', sha256=h.hexdigest(), bytes=total)


def download(destination, attempts):
    for attempt in range(1, attempts + 1):
        h = hashlib.sha256()
        count = 0
        last_report = 0
        progress('download_attempt', attempt=attempt, total_attempts=attempts)
        try:
            req = urllib.request.Request(URL, headers={'User-Agent':'C2R-reproducible-model-migration/1.0'})
            with urllib.request.urlopen(req, timeout=120) as response, destination.open('wb') as out:
                for block in iter(lambda: response.read(BLOCK), b''):
                    out.write(block); h.update(block); count += len(block)
                    if count > MODEL_BYTES:
                        raise ValueError('download_exceeds_expected_model_size')
                    if count - last_report >= 256 * 1024 * 1024:
                        progress('download_progress', received_bytes=count, total_bytes=MODEL_BYTES)
                        last_report = count
            if count != MODEL_BYTES or h.hexdigest() != MODEL_SHA256:
                progress('download_hash_mismatch', actual_sha256=h.hexdigest(),
                         expected_sha256=MODEL_SHA256, bytes=count)
                raise ValueError('download_weight_identity_mismatch')
            progress('download_verified', sha256=h.hexdigest(), bytes=count)
            return destination
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # Do not log redirect URLs, headers, tokens, or arbitrary HTTP response bodies.
            progress('download_attempt_failed', attempt=attempt, error_type=type(exc).__name__,
                     http_status=getattr(exc, 'code', None))
            if attempt >= attempts:
                raise RuntimeError('download_failed_after_finite_retries') from None
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError('unreachable_download_state')


def reproduce(repo, info, parts, source, stage, keep):
    h = hashlib.sha256()
    total = 0
    staged = []
    with source.open('rb') as stream:
        for index, part in enumerate(parts):
            raw = stream.read(part['bytes'])
            raw_sha = hashlib.sha256(raw).hexdigest()
            if len(raw) != part['bytes'] or raw_sha != part['sha256']:
                progress('raw_part_hash_mismatch', part=index, actual_sha256=raw_sha,
                         expected_sha256=part['sha256'])
                raise ValueError('source_part_identity_mismatch')
            h.update(raw); total += len(raw)
            if index not in keep:
                candidate = stage / f'part-{index:03d}.gz'
                # These settings exactly match the original migration script.
                with candidate.open('wb') as out:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=3) as archive:
                        archive.write(raw)
                archive_sha = sha256(candidate)
                if (archive_sha != part['archive_sha256'] or
                        candidate.stat().st_size != part['archive_bytes']):
                    progress('archive_hash_mismatch', part=index, actual_sha256=archive_sha,
                             expected_sha256=part['archive_sha256'],
                             zlib_build=zlib.ZLIB_VERSION,zlib_runtime=zlib.ZLIB_RUNTIME_VERSION,
                             python=sys.version.split()[0])
                    raise ValueError('gzip_bytes_mismatch_no_parts_installed')
                staged.append((part_path(repo,part['path']),candidate,part))
            progress('part_verified', part=index, total_parts=len(parts), raw_sha256=raw_sha,
                     archive_sha256=part['archive_sha256'], retained_existing=index in keep)
        if stream.read(1):
            raise ValueError('source_has_trailing_bytes')
    if total != info['bytes'] or h.hexdigest() != info['sha256']:
        progress('full_raw_hash_mismatch', actual_sha256=h.hexdigest(),expected_sha256=info['sha256'])
        raise ValueError('full_raw_model_identity_mismatch_no_parts_installed')
    progress('all_staged_verified', new_parts=len(staged), existing_parts=len(keep),
             full_raw_sha256=h.hexdigest(), bytes=total)
    # No repository part has been changed before this point.
    for destination,candidate,part in staged:
        destination.parent.mkdir(parents=True,exist_ok=True)
        # A concurrent process may have supplied the exact archive meanwhile.
        if not archive_matches(destination,part):
            os.replace(candidate,destination)
        if not archive_matches(destination,part):
            raise ValueError('installed_archive_verification_failed')
    for index in keep:
        if not archive_matches(part_path(repo,parts[index]['path']),parts[index]):
            raise ValueError('previously_correct_archive_changed_concurrently')
    progress('complete', materialized_parts=len(staged), retained_parts=len(keep),
             total_parts=len(parts), full_raw_sha256=h.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--manifest', type=Path, help='Defaults to repo/migration/manifests/qwen_parts.json')
    parser.add_argument('--source', type=Path, help='Existing complete local model; omitted: fixed official HF download')
    parser.add_argument('--attempts', type=int, default=3, choices=range(1,6))
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        raise ValueError('repository_directory_missing')
    manifest = args.manifest.resolve() if args.manifest else repo/'migration/manifests/qwen_parts.json'
    manifest_before = sha256(manifest)
    info = json.loads(manifest.read_text(encoding='utf-8'))
    parts = validate_manifest(info,repo)
    progress('compression_runtime',python=sys.version.split()[0],
             zlib_build=zlib.ZLIB_VERSION,zlib_runtime=zlib.ZLIB_RUNTIME_VERSION)
    keep = {i for i,part in enumerate(parts) if archive_matches(part_path(repo,part['path']),part)}
    progress('existing_archive_scan', correct_parts=len(keep), total_parts=len(parts))
    if len(keep) == len(parts):
        verify_archives_full(repo,parts)
        progress('complete', materialized_parts=0, retained_parts=len(keep),total_parts=len(parts))
    else:
        with tempfile.TemporaryDirectory(prefix='.qwen-materialize-',dir=repo) as temp:
            stage = Path(temp)
            if args.source is not None:
                source=args.source.expanduser().resolve()
                if not source.is_file() or source.stat().st_size != MODEL_BYTES:
                    raise ValueError('local_source_missing_or_wrong_size')
            else:
                source=download(stage/'source.safetensors',args.attempts)
            reproduce(repo,info,parts,source,stage,keep)
    if sha256(manifest) != manifest_before:
        raise RuntimeError('manifest_changed_concurrently')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        progress('failed',error_type=type(exc).__name__,reason=str(exc) if isinstance(exc,(ValueError,RuntimeError)) else 'io_error')
        sys.exit(1)
