"""Download only the official demo assets, pinned to immutable HF revisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import requests


ASSETS = [
    ("models", "Qwen/Qwen2.5-1.5B-Instruct",
     "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "qwen_weights",
     ["config.json", "generation_config.json", "merges.txt", "model.safetensors",
      "tokenizer.json", "tokenizer_config.json", "vocab.json", "LICENSE"]),
    ("models", "SIA-IDE/BearLLM",
     "fd2859d9ea8fbebe7f815ca8e02ca48bb1025191", "bearllm_weights",
     ["adapter_config.json", "adapter_model.safetensors", "vibration_adapter.pth"]),
    ("datasets", "SIA-IDE/MBHM",
     "78cd9b8b6b65cd43eebf4879b060d783d5f7dcfe", "mbhm_dataset",
     ["demo_data.json"]),
]
CONNECTIONS = 1
TRUST_ENV = True


def get(url, **kwargs):
    session = requests.Session()
    session.trust_env = TRUST_ENV
    return session.get(url, **kwargs)


def verify(path, info):
    if not path.is_file() or path.stat().st_size != info["size"]:
        return False
    if "lfs" in info:
        with path.open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest() == info["lfs"]["sha256"]
    digest = hashlib.sha1(f"blob {info['size']}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == info["blobId"]


def download_large(url, path, info):
    """Use resumable, strictly checked byte ranges on slow single connections."""
    parts = path.with_suffix(path.suffix + ".parts")
    parts.mkdir(exist_ok=True)
    chunk_size = 32 * 1024 * 1024
    ranges = [(start, min(start + chunk_size, info["size"]) - 1)
              for start in range(0, info["size"], chunk_size)]

    def fetch(byte_range):
        start, end = byte_range
        target = parts / str(start)
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        for attempt in range(4):
            try:
                offset = target.stat().st_size if target.exists() else 0
                if offset == end - start + 1:
                    return target
                if offset > end - start + 1:
                    offset = 0
                resume_start = start + offset
                with get(url, headers={"Range": f"bytes={resume_start}-{end}"},
                         stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    expected = f"bytes {resume_start}-{end}/{info['size']}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor byte range")
                    with target.open("ab" if offset else "wb") as f:
                        for chunk in response.iter_content(1024 * 1024):
                            f.write(chunk)
                if target.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete byte range")
                print(f"{path.name}: chunk {start // chunk_size + 1}/{len(ranges)} verified size", flush=True)
                return target
            except (requests.RequestException, ValueError) as exc:
                print(f"Chunk retry {attempt + 1}/4 ({type(exc).__name__})", flush=True)
                if attempt == 3:
                    raise RuntimeError(f"Failed byte range {start}-{end}") from None
                time.sleep(2 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=CONNECTIONS) as executor:
        completed = list(executor.map(fetch, ranges))
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("wb") as output:
        for part in completed:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    if not verify(partial, info):
        raise ValueError(f"Hash verification failed: {path.name}; clear {parts} and retry")
    partial.replace(path)
    shutil.rmtree(parts)
    print(f"Downloaded and verified {path.name} ({info['size']:,} bytes)", flush=True)


def download(job):
    url, path, info = job
    path.parent.mkdir(parents=True, exist_ok=True)
    if verify(path, info):
        print(f"Verified existing {path.name}", flush=True)
        return
    if CONNECTIONS > 1 and info["size"] > 128 * 1024 * 1024:
        return download_large(url, path, info)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == info["size"] and verify(partial, info):
                partial.replace(path)
                return
            if offset >= info["size"]:
                partial.unlink()
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected resume range")
                written = offset if append else 0
                last_report = time.monotonic()
                with partial.open("ab" if append else "wb") as f:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        f.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_report > 20:
                            print(f"{path.name}: {written / info['size']:.1%}", flush=True)
                            last_report = time.monotonic()
            if not verify(partial, info):
                raise ValueError(f"Size/hash verification failed for {path.name}")
            partial.replace(path)
            print(f"Downloaded and verified {path.name} ({info['size']:,} bytes)", flush=True)
            return
        except (requests.RequestException, ValueError) as exc:
            print(f"Retry {attempt + 1}/4: {path.name}: {type(exc).__name__}", flush=True)
            if attempt == 3:
                raise RuntimeError(f"Could not download {path.name}") from None
            time.sleep(2 * (attempt + 1))


def main():
    global CONNECTIONS, TRUST_ENV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--connections", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--no-proxy", action="store_true", help="Ignore proxy environment variables")
    args = parser.parse_args()
    CONNECTIONS, TRUST_ENV = args.connections, not args.no_proxy
    endpoint = args.endpoint.rstrip("/")
    jobs, manifest = [], []
    for kind, repo, revision, folder, names in ASSETS:
        response = get(f"{endpoint}/api/{kind}/{repo}/revision/{revision}",
                                params={"blobs": "true"}, timeout=30)
        response.raise_for_status()
        metadata = response.json()
        if metadata["sha"] != revision:
            raise ValueError(f"Revision mismatch for {repo}")
        files = {item["rfilename"]: item for item in metadata["siblings"]}
        prefix = "datasets/" if kind == "datasets" else ""
        for name in names:
            jobs.append((f"{endpoint}/{prefix}{repo}/resolve/{revision}/{name}",
                         args.data_dir / folder / name, files[name]))
        manifest.append({"repo": repo, "revision": revision, "directory": folder,
                         "files": [files[name] for name in names]})
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download, jobs))
    (args.data_dir / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("All demo assets verified.", flush=True)


if __name__ == "__main__":
    main()
