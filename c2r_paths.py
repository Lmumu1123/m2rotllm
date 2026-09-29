"""Explicit relocation of historical C2R paths; never modifies result metadata."""
from pathlib import Path
import os


def project_root():
    return Path(os.environ.get('C2R_ROOT', Path(__file__).resolve().parent)).resolve()


def resolve_path(value):
    """Return a string with known historical prefixes relocated, otherwise unchanged."""
    if not isinstance(value, (str, os.PathLike)):
        return value
    text = os.fspath(value)
    if not isinstance(text, str):
        return value
    root = project_root()
    mappings = [
        ('/media/nas_users/huangyating/data', Path(os.environ.get('C2R_DATA_ROOT', root / 'datasets/r2'))),
        ('/media/nas_users/huangyating/bearllm-assets/mbhm_dataset', Path(os.environ.get('C2R_MBHM_ROOT', root / 'datasets/mbhm_dataset'))),
        ('/media/nas_users/huangyating/bearllm-assets', Path(os.environ.get('C2R_ASSETS_ROOT', root / 'external/bearllm-assets'))),
        ('/media/nas_users/huangyating/bearllm-runs', Path(os.environ.get('C2R_BEAR_RUNS_ROOT', root / 'external/bearllm-runs'))),
        ('/media/nas_users/huangyating', root / 'external'),
        ('/home/huangyating', root),
    ]
    # Resolution is idempotent even if a staging copy itself lives underneath
    # the old user's home or NAS directory.
    known_targets = [target for _, target in mappings]
    known_targets.append(Path(__file__).resolve().parent)
    if os.environ.get('C2R_ARCHIVE_ROOT'):
        known_targets.append(Path(os.environ['C2R_ARCHIVE_ROOT']))
    for target in known_targets:
        target_text = str(target)
        if text == target_text or text.startswith(target_text + '/'):
            return text
    for old, new in mappings:
        if text == old or text.startswith(old + '/'):
            return str(new) + text[len(old):]
    return text
