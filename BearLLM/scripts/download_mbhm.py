"""Download and cryptographically verify the complete pinned official MBHM release.

Example: python scripts/download_mbhm.py --data-dir /path/to/mbhm_dataset \
    --endpoint https://hf-mirror.com --no-proxy --connections 16
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import requests

import download_demo_assets as download_helpers


REPO = "SIA-IDE/MBHM"
REVISION = "78cd9b8b6b65cd43eebf4879b060d783d5f7dcfe"


def download_large(url, path, info):
    """Keep compatible 32 MiB part files, but rotate slow HTTP connections.

    Smaller 4 MiB requests avoid long-lived mirror transfers that sometimes
    decay to tens of KiB/s. Every request checks its exact Content-Range and
    partial requests resume at the last byte written.
    """
    parts = path.with_suffix(path.suffix + ".parts")
    parts.mkdir(exist_ok=True)
    part_size = 32 * 1024 * 1024
    request_size = 4 * 1024 * 1024
    ranges = [(start, min(start + part_size, info["size"]) - 1)
              for start in range(0, info["size"], part_size)]

    def fetch(bounds):
        start, end = bounds
        target = parts / str(start)
        failures = 0
        while True:
            offset = target.stat().st_size if target.exists() else 0
            if offset == end - start + 1:
                return target
            if offset > end - start + 1:
                target.unlink()
                offset = 0
            first = start + offset
            last = min(first + request_size - 1, end)
            before = time.monotonic()
            written = 0
            try:
                with download_helpers.get(
                    url, headers={"Range": f"bytes={first}-{last}"},
                    stream=True, timeout=(30, 30),
                ) as response:
                    response.raise_for_status()
                    expected = f"bytes {first}-{last}/{info['size']}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor byte range")
                    with target.open("ab") as output:
                        for chunk in response.iter_content(128 * 1024):
                            output.write(chunk)
                            written += len(chunk)
                            elapsed = time.monotonic() - before
                            if elapsed > 25 and written / elapsed < 256 * 1024:
                                raise TimeoutError("Rotating slow range connection")
                if written != last - first + 1:
                    raise ValueError("Incomplete byte range")
                failures = 0
            except (requests.RequestException, TimeoutError, ValueError) as exc:
                failures += 1
                print(f"Range {first}-{last}: retry {failures}/24 ({type(exc).__name__})", flush=True)
                if failures >= 24:
                    raise RuntimeError(f"Failed byte range {first}-{last}") from None
                time.sleep(min(failures, 5))
            if target.exists() and target.stat().st_size == end - start + 1:
                print(f"{path.name}: chunk {start // part_size + 1}/{len(ranges)} verified size", flush=True)
                return target

    with ThreadPoolExecutor(max_workers=download_helpers.CONNECTIONS) as executor:
        complete = list(executor.map(fetch, ranges))
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Assembling {len(complete)} verified-size ranges into {partial.name}", flush=True)
    with partial.open("wb") as output:
        for part in complete:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    print(f"Verifying full-file SHA-256: {path.name}", flush=True)
    if not download_helpers.verify(partial, info):
        raise ValueError(f"Hash verification failed: {path.name}; clear {parts} and retry")
    partial.replace(path)
    shutil.rmtree(parts)
    print(f"Downloaded and verified {path.name} ({info['size']:,} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--connections", type=int, choices=range(1, 33), default=16)
    parser.add_argument("--no-proxy", action="store_true")
    args = parser.parse_args()
    download_helpers.CONNECTIONS = args.connections
    download_helpers.TRUST_ENV = not args.no_proxy
    download_helpers.download_large = download_large
    endpoint = args.endpoint.rstrip("/")
    response = download_helpers.get(
        f"{endpoint}/api/datasets/{REPO}/revision/{REVISION}",
        params={"blobs": "true"}, timeout=30,
    )
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != REVISION:
        raise ValueError("Dataset revision mismatch")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    files = metadata["siblings"]
    manifest = {
        "repo": REPO,
        "revision": REVISION,
        "endpoint": endpoint,
        "directory": str(args.data_dir.resolve()),
        "verified": False,
        "files": files,
    }
    manifest_path = args.data_dir / "mbhm_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [
        (f"{endpoint}/datasets/{REPO}/resolve/{REVISION}/{item['rfilename']}",
         args.data_dir / item["rfilename"], item)
        for item in files
    ]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download_helpers.download, jobs))
    manifest["verified"] = True
    manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete MBHM release verified: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
