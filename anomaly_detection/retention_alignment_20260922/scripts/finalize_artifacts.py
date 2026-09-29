"""Verify immutable inputs, copy the label-free demo, and inventory artifacts.

Does not train or overwrite experiment/model results. The generated inventory
excludes its own hash. Existing demo copies must match the archived originals.
"""
from pathlib import Path
import ast
import hashlib
import importlib.metadata as im
import json
import platform
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'four_class_chain_20260922'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    verified_inputs = []
    manifests = [ROOT/'preprocessing/input_manifest.json', OLD/'radar/input_manifest.json']
    for manifest in manifests:
        for rec in json.loads(manifest.read_text()):
            p = Path(rec.get('file', rec.get('path')))
            actual = sha(p)
            assert actual == rec['sha256'], f'Original input changed: {p}'
            verified_inputs.append(dict(path=str(p), sha256=actual))
    weights = {}
    for p in [ROOT/'source/provenance.json', ROOT/'results/llm_native_p10_chain/provenance.json']:
        for name, expected in json.loads(p.read_text())['weights'].items():
            actual = sha(name)
            assert actual == expected, f'Original weight changed: {name}'
            weights[name] = actual
    old_files = json.loads((OLD/'artifact_manifest.json').read_text())['files']
    for rec in old_files:
        assert sha(OLD/rec['path']) == rec['sha256'], f'Prior artifact changed: {rec["path"]}'

    demo = ROOT/'inputs/radar_only'
    demo.mkdir(parents=True, exist_ok=True)
    demo_inputs = []
    for name in ['features.npz', 'metadata.csv', 'README.txt']:
        src = OLD/'results/radar_only_demo_inputs'/name
        dest = demo/name
        if not dest.exists():
            shutil.copy2(src, dest)
        assert sha(src) == sha(dest)
        demo_inputs.append(dict(source=str(src), copy=str(dest), sha256=sha(dest)))
    write_json(demo/'copy_manifest.json', demo_inputs)

    py_files = list(ROOT.rglob('*.py'))
    for p in py_files:
        ast.parse(p.read_text(), filename=str(p))
    missing = []
    links_checked = 0
    for name in ['README.md', '原能力保持与跨模态对齐实测报告.md']:
        for target in re.findall(r'\]\(([^\n)]+)\)', (ROOT/name).read_text()):
            if target.startswith(('http://', 'https://', '#')):
                continue
            dest = Path(target) if target.startswith('/') else ROOT/target
            links_checked += 1
            if not dest.exists():
                missing.append(str(dest))
    assert not missing, missing
    versions = dict(python=sys.version, executable=sys.executable, platform=platform.platform())
    for name in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'transformers', 'peft', 'h5py', 'matplotlib']:
        versions[name] = im.version(name)
    assert Path(sys.executable).parent.parent.name == 'm2vllm', sys.executable
    write_json(ROOT/'environment_versions.json', versions)
    write_json(ROOT/'final_integrity.json', dict(
        status='passed', original_input_npz_count=len(verified_inputs), original_inputs=verified_inputs,
        original_weights_unchanged=weights, previous_artifact_files_unchanged=len(old_files),
        copied_radar_demo_exact=True, python_files_syntax_checked=len(py_files),
        main_document_local_links_checked=links_checked, main_document_missing_links=missing,
        experiment_verification_files=[
            'preprocessing/verification.json', 'results/retention_verification/verification.json',
            'results/retention_verification/shared_model_verification.json',
            'results/full_old_train_regression/verification.json',
            'radar/radar_only_conservative_v0/verification.json',
            'results/llm_native_p10_combined/verification.json'],
        note='Integrity verification does not upgrade exploratory scores into independent generalization evidence.'))

    entries = []
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name == 'artifact_manifest.json':
            continue
        entries.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)))
    write_json(ROOT/'artifact_manifest.json', dict(
        root_name=ROOT.name, exclusions=['__pycache__', 'artifact_manifest.json self-hash'], files=entries))
    print(json.dumps(dict(status='passed', original_inputs=len(verified_inputs), old_files=len(old_files),
        files=len(entries), total_bytes=sum(r['bytes'] for r in entries), syntax=len(py_files), links=links_checked)))


if __name__ == '__main__':
    main()
