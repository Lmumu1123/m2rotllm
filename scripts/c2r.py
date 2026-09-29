#!/usr/bin/env python3
"""Portable C2R environment checks, cached replay, and isolated R2 experiments."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
R2 = Path('anomaly_detection/r2_validation_20260928')
sys.path.insert(0, str(REPO))
os.environ.setdefault('C2R_ROOT', str(REPO))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from c2r_paths import resolve_path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(args, cwd=None, root=None):
    env = os.environ.copy()
    env['C2R_ROOT'] = str(root or REPO)
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    print('RUN:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd or REPO, env=env, check=True)


def doctor():
    packages = {}
    for name in ('numpy', 'scipy', 'h5py', 'pandas', 'scikit-learn', 'torch', 'safetensors', 'matplotlib', 'transformers', 'peft'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    required = [
        'anomaly_detection/r2_validation_20260928/contact/contact_features.npz',
        'anomaly_detection/r2_validation_20260928/radar/features.npz',
        'anomaly_detection/four_class_chain_20260922/contact/external_references.npz',
        'anomaly_detection/retention_alignment_20260922/results/retained_head_v1/models/v0/all_known/retained_head.npz',
        'external/bearllm-runs/released-code-seed42/pretrain/fcn/feature_encoder.pth',
        'external/bearllm-runs/released-code-seed42/pretrain/fcn/classifier.pth',
    ]
    result = dict(project=str(REPO), python=sys.version, packages=packages,
                  required_files={p: (REPO / p).is_file() for p in required},
                  optional_r2_export_root=resolve_path('/media/nas_users/huangyating/data'),
                  optional_r2_exports_present=Path(resolve_path('/media/nas_users/huangyating/data/upload')).exists(),
                  optional_qwen_restored=(REPO / 'external/bearllm-assets/qwen_weights/model.safetensors').is_file(),
                  note='Qwen language generation is optional for contact/radar classification. Restore large artifacts before running LLM tasks.')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(any(v is None for k, v in packages.items() if k not in ('transformers', 'peft')) or not all(result['required_files'].values()))


def configure():
    """Generate only documented local BearLLM paths and public model constants."""
    target = REPO / 'BearLLM/.env'
    if target.exists():
        raise FileExistsError(f'Existing config is preserved; edit manually: {target}')
    values = dict(MBHM_DATASET=resolve_path('/media/nas_users/huangyating/bearllm-assets/mbhm_dataset'),
                  QWEN_WEIGHTS=resolve_path('/media/nas_users/huangyating/bearllm-assets/qwen_weights'),
                  BEARLLM_WEIGHTS=resolve_path('/media/nas_users/huangyating/bearllm-assets/bearllm_weights'),
                  DESCRIPTION_LEN=5, LLM_HIDDEN_SIZE=1536, SIGNAL_TOKEN_ID=151925)
    target.write_text('\n'.join(f'{k}={json.dumps(v)}' for k, v in values.items()) + '\n')
    print(str(target))
    return 0


def verify(args):
    """Run numerical frontend tests and independently replay a saved student."""
    forbidden = ('/home/huangyating/anomaly_detection', '/home/huangyating/BearLLM',
                 '/home/huangyating/research_proposals', '/media/nas_users/huangyating/bearllm-assets',
                 '/media/nas_users/huangyating/bearllm-runs', '/media/nas_users/huangyating/data')
    def disallow_original_dependencies(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            path = os.fsdecode(values[0])
            if any(path == p or path.startswith(p + '/') for p in forbidden):
                raise RuntimeError('Relocation check tried to read the original server dependency: ' + path)
    sys.addaudithook(disallow_original_dependencies)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    command([sys.executable, '-m', 'unittest', 'discover', '-p', 'test_*.py', '-v'],
            cwd=REPO / 'anomaly_detection/local_preprocessing_r2_20260928')
    command([sys.executable, '-m', 'unittest', 'discover', '-p', 'test_features.py', '-v'],
            cwd=REPO / R2 / 'radar')
    exp = REPO / R2 / 'experiments/archived__complex_shape'
    key = 'holdout_1000__full__seed17'
    checkpoint = exp / 'models' / (key + '.pt')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = np.load(exp / 'predictions' / (key + '.npz'), allow_pickle=False)
    target = np.load(exp / 'targets/holdout_1000.npz', allow_pickle=False)
    features = np.load(REPO / R2 / 'radar/features.npz', allow_pickle=False)
    mod = imported(REPO / R2 / 'experiments/run_alignment.py', 'c2r_alignment_verify')
    model = mod.Encoder(target['contact_mean'], target['contact_scale']).eval()
    model.load_state_dict(state['state_dict'])
    row = saved['row_index'][:32]
    x = (features[state['feature']][row].astype(float) - state['radar_mean']) / state['radar_scale']
    with torch.inference_mode():
        hidden = model(torch.from_numpy(x).float())
        w = torch.from_numpy(target['head_weight']).float()
        b = torch.from_numpy(target['head_bias']).float()
        probs = mod.group(hidden @ w.T + b).softmax(-1).numpy()
    error = float(np.max(np.abs(probs - saved['probabilities4'][:len(row)])))
    assert error < 2e-5, error
    # Load actual Bear FCN and run one cached query against all nine fixed refs.
    api = imported(REPO / 'anomaly_detection/four_class_chain_20260922/contact/evaluate_contact.py', 'c2r_bear_api_verify')
    contact_api = imported(REPO / R2 / 'contact/evaluate_new_contact.py', 'c2r_contact_verify')
    refs, metadata = api.external_references()
    bear, prov = api.load_model('retrained_fcn')
    raw = np.load(REPO / R2 / 'contact/raw_contact_windows.npz', allow_pickle=False)['raw_xyz'][:1]
    q = contact_api.query_transform(raw, demean=True)
    hh, pp = contact_api.infer(bear, q, refs, 'cpu', 3)
    contact_saved = np.load(REPO / R2 / 'contact/contact_features.npz', allow_pickle=False)
    reference = contact_saved['hidden_axes'][:1]
    contact_error = float(np.max(np.abs(hh - reference)))
    contact_relative_error = float(np.linalg.norm(hh - reference) / max(np.linalg.norm(reference), 1e-12))
    contact_probability = contact_api.probability_from_hidden(hh.mean(1), contact_saved['retained_weight10'], contact_saved['retained_bias10'])
    contact_probability_error = float(np.max(np.abs(contact_probability - contact_saved['retained_probabilities4'][:1])))
    # Raw cache is float32; original HDF5 inference used float64, so bitwise
    # equality is neither expected nor claimed.
    assert np.isfinite(hh).all() and np.isfinite(contact_probability).all()
    original_cpu = np.load(REPO / 'migration/verification/contact_original_cpu_reference.npz', allow_pickle=False)
    original_cpu_error = float(np.max(np.abs(hh - original_cpu['hidden_axes'])))
    np.testing.assert_allclose(hh, original_cpu['hidden_axes'], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pp, original_cpu['probabilities10_axes'], rtol=1e-5, atol=1e-6)
    result = dict(ok=True, root=str(REPO), checkpoint=str(checkpoint.relative_to(REPO)), checkpoint_sha256=sha(checkpoint),
                  replay_windows=len(row), radar_probability_max_absolute_error=error,
                  contact_cached_packet_count=1, contact_views=27,
                  contact_hidden_max_absolute_error=contact_error,
                  contact_hidden_relative_l2_error=contact_relative_error,
                  contact_probability_max_absolute_error=contact_probability_error,
                  contact_migrated_vs_original_same_CPU_same_input_max_absolute_error=original_cpu_error,
                  contact_raw_cache_precision='float32 cache versus original float64 HDF5',
                  contact_source_weights=prov['weights'], fixed_references=len(metadata))
    result['original_project_and_NAS_dependencies_blocked_during_model_replay'] = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def workspace(path, from_exports=False):
    """Create a fresh directory: source/model dependencies are read-only links.

    Only the active R2 working tree is copied. There are no archived experiment
    outputs in this tree, so old checkpoints and CSVs are never overwritten.
    """
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f'Use a NEW workspace path: {path}')
    path.mkdir(parents=True)
    for name in ('BearLLM', 'research_proposals', 'external', 'datasets'):
        if (REPO / name).exists():
            (path / name).symlink_to(REPO / name, target_is_directory=True)
    (path / 'anomaly_detection').mkdir()
    for item in (REPO / 'anomaly_detection').iterdir():
        if item.name != R2.name:
            (path / 'anomaly_detection' / item.name).symlink_to(item, target_is_directory=item.is_dir())
    active = path / R2
    original = REPO / R2
    for source in original.rglob('*.py'):
        if '__pycache__' in source.parts:
            continue
        target = active / source.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(original / 'protocol_before_results.json', active / 'protocol_before_results.json')
    if not from_exports:
        inputs = ['contact/contact_features.npz', 'contact/raw_contact_windows.npz', 'contact/metadata.csv',
                  'radar/features.npz', 'audit/radar_quality.csv']
        for relative in inputs:
            src = original / relative
            if src.exists():
                dst = active / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for source in (original / 'audit').glob('*.csv'):
            dst = active / 'audit' / source.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dst)
    (path / 'run_manifest.json').write_text(json.dumps(dict(source_repository=str(REPO),
            source_protocol_sha256=sha(original / 'protocol_before_results.json'),
            initial_inputs='regenerate from external HDF5 exports' if from_exports else 'archived derived features and raw contact cache',
            note='A rerun is computational reproduction, not a new physical test set.'), indent=2) + '\n')
    return path


def run(args):
    if args.from_exports and args.task not in ('full', 'contact', 'radar'):
        raise ValueError('--from-exports applies to full/contact/radar only')
    if args.task in ('contact', 'radar') and not args.from_exports:
        raise ValueError('Raw contact/radar frontend requires --from-exports and C2R_DATA_ROOT; cached-feature diagnostics use contact-heads/alignment.')
    root = workspace(args.workspace, args.from_exports)
    active = root / R2
    device = args.device
    def script(relative, *extra):
        command([sys.executable, active / relative, *extra], root=root)
    if args.from_exports:
        data = Path(resolve_path('/media/nas_users/huangyating/data'))
        if not (data / 'upload').is_dir():
            raise FileNotFoundError('Set C2R_DATA_ROOT to the directory containing upload/ and wide_cache_local/.')
        if args.task in ('full', 'radar'):
            script('audit/run_audit.py')
            script('radar/extract_r2_features.py', '--data-root', str(data), '--output', str(active / 'radar'))
        if args.task in ('full', 'contact'):
            script('contact/evaluate_new_contact.py', '--data', str(data / 'upload/contacts'), '--device', device, '--batch', '48')
    if args.task in ('full', 'contact-heads'):
        script('contact/diagnose_loso.py')
    if args.task in ('full', 'baselines'):
        script('baselines/run_baselines.py')
    if args.task in ('full', 'alignment'):
        argv = ['--head', args.head, '--feature', args.feature, '--device', device, '--seeds', *map(str, args.seeds)]
        if args.methods:
            argv += ['--methods', *args.methods]
        if args.head == 'diagnostic' and args.task == 'alignment':
            script('contact/diagnose_loso.py')
        script('experiments/run_alignment.py', *argv)
    print(json.dumps(dict(completed=True, workspace=str(root), outputs=str(active), archived_results_unchanged=True), indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Show dependencies and distinguish optional large datasets/Qwen')
    sub.add_parser('configure', help='Create a fresh local BearLLM .env from public model constants and current paths')
    p = sub.add_parser('verify', help='16 parser tests + 7 radar tests + saved contact/radar model replay')
    p.add_argument('--output', type=Path)
    p = sub.add_parser('run', help='Run experiments in a NEW workspace; archives stay immutable')
    p.add_argument('task', choices=['full', 'contact', 'radar', 'contact-heads', 'baselines', 'alignment'])
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--from-exports', action='store_true')
    p.add_argument('--head', choices=['archived', 'diagnostic'], default='archived')
    p.add_argument('--feature', default='complex_shape')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 42, 73])
    p.add_argument('--methods', nargs='+')
    args = parser.parse_args()
    if args.command == 'doctor':
        return doctor()
    if args.command == 'configure':
        return configure()
    if args.command == 'verify':
        return verify(args)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
