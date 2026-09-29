"""Verify frozen inputs, local report links and script syntax; package this run."""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import ast, hashlib, importlib.metadata, json, platform, re, sys, zipfile
from datetime import datetime, timezone
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]
SOURCE = ROOT.parent / 'four_class_preprocessing_20260921/results'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def keep(p):
    r = p.relative_to(ROOT)
    return p.is_file() and '__pycache__' not in r.parts and ('smoke_5steps' not in r.parts) and (p.name != 'smoke_5steps.log')

def main():
    if '/envs/m2vllm/' not in sys.executable:
        raise ValueError('Use the authorized m2vllm environment')
    versions = {}
    for name in ['torch', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'transformers', 'peft', 'safetensors', 'accelerate']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write(ROOT / 'environment_versions.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=versions, environment_modified=False))
    refs = json.loads((SOURCE / 'verification.json').read_text())['files']
    checked = []
    for ref in refs:
        p = SOURCE / ref['file']
        actual = sha(p)
        assert actual == ref['sha256'], f'Input changed: {p}'
        checked.append(dict(file=str(p), sha256=actual, unchanged=True))
    assert len(checked) == 24
    radar_manifest = json.loads((ROOT / 'radar/input_manifest.json').read_text())
    for record in radar_manifest:
        assert sha(Path(_c2r_resolve_path(record['path'])).with_suffix('.json')) == record['sidecar_sha256']
    runtime = json.loads((ROOT / 'results/chain_v1/runtime.json').read_text())
    assert sha(ROOT / 'scripts/run_chain.py') == runtime['script_sha256']
    assert sha(ROOT / 'protocol.json') == runtime['protocol_sha256']
    planned = {ROOT / 'artifact_verification.json', ROOT / 'artifact_manifest.json'}
    link_count = 0
    bad = []
    for doc in ROOT.rglob('*.md'):
        for target in re.findall('!?\\[[^\\]]*\\]\\(([^\\n)]+)\\)', doc.read_text()):
            if re.match('^(https?://|mailto:|#)', target):
                continue
            target = target.split('#', 1)[0].strip('<>')
            if not target:
                continue
            path = (doc.parent / target).resolve() if not target.startswith('/') else Path(_c2r_resolve_path(target))
            link_count += 1
            if not path.exists() and path not in planned:
                bad.append(dict(doc=str(doc), target=target))
    assert not bad, bad
    parsed = []
    for script in ROOT.rglob('*.py'):
        if not keep(script):
            continue
        ast.parse(script.read_text(), filename=str(script))
        parsed.append(str(script.relative_to(ROOT)))
    core = json.loads((ROOT / 'results/chain_v1/verification.json').read_text())
    demo = json.loads((ROOT / 'results/radar_only_chain_demo/verification.json').read_text())
    assert core['status'] == 'passed' and demo['no_label_or_state_or_contact_inputs']
    write(ROOT / 'artifact_verification.json', dict(status='passed', time_utc=datetime.now(timezone.utc).isoformat(), source_npz_count=len(checked), source_npz_unchanged=checked, radar_sidecars_unchanged=12, formal_training_script_and_protocol_hash_match=True, local_markdown_links_checked=link_count, broken_local_links=bad, scripts_syntax_checked=parsed, primary_model_reload_checks=core['model_reload_count'], standalone_radar_demo_verified=True, experimental_limitations=['wrong outer/keep ROI unresolved without original ADC bins', 'no precise window synchronization', 'no demonstrated distillation classification advantage', 'primary chain fails external healthy motor cases', 'small dependent recording splits'], validation_scope='artifact consistency; does not remove experimental limitations'))
    files = sorted((p for p in ROOT.rglob('*') if keep(p) and p.name != 'artifact_manifest.json'))
    write(ROOT / 'artifact_manifest.json', dict(root_name=ROOT.name, exclusions=['__pycache__', 'results/smoke_5steps', 'results/smoke_5steps.log', 'artifact_manifest.json self-hash'], files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in files]))
    files.append(ROOT / 'artifact_manifest.json')
    archive = ROOT.parent / (ROOT.name + '_bundle.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, arcname=str(Path(_c2r_resolve_path(ROOT.name)) / p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)
    digest = sha(archive)
    archive.with_suffix('.zip.sha256').write_text(digest + '  ' + archive.name + '\n')
    print(json.dumps(dict(status='passed', source_files=len(checked), links=link_count, scripts=len(parsed), package=str(archive), archive_files=len(files), archive_bytes=archive.stat().st_size, archive_sha256=digest), ensure_ascii=False))
if __name__ == '__main__':
    main()
