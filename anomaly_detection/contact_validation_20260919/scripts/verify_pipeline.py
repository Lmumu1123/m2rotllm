"""Focused checks for frequency coordinates, corrupt exports and frozen features."""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import importlib.util, json, hashlib, sys, ast
import numpy as np
import torch
from contact_pipeline import ROOT, FS, N_FREQ, dcn_1s, prepare_pair, read_recording, write_json
from teacher_bridge import REPO, load_teacher, embed

def fails(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Expected rejection')

def main():
    torch.set_num_threads(2)
    n = np.arange(FS)
    x = np.cos(2 * np.pi * 125 * (n + 0.5) / FS) + 0.2 * np.cos(2 * np.pi * 320 * (n + 0.5) / FS) - 0.91
    normalized = dcn_1s(x)
    assert np.argmax(abs(normalized)) == 250
    assert np.all(normalized[4000:] == 0)
    assert abs(float(np.sqrt(np.mean(normalized ** 2))) - 0.01) < 1e-08
    assert np.allclose(normalized, dcn_1s(x * 9.80665 + 3), atol=1e-07)
    spec = importlib.util.spec_from_file_location('released_dataset_dcn', _c2r_resolve_path('/media/nas_users/huangyating/bearllm-assets/mbhm_dataset/src/dcn.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert np.allclose(normalized, mod.dcn(x), atol=1e-07)
    fails(lambda: dcn_1s(x[:1000]))
    fails(lambda: dcn_1s(np.ones(FS)))
    fails(lambda: dcn_1s(np.full(FS, np.nan)))
    query = np.c_[x, np.roll(x, 10), np.roll(x, 20)]
    ref = np.c_[np.roll(x, 100), np.roll(x, 110), np.roll(x, 120)]
    fails(lambda: prepare_pair(query, ref, 'same', 'same'))
    pair = prepare_pair(query, ref, 'synthetic_query', 'synthetic_reference')
    assert pair.shape == (3, 2, N_FREQ)
    m = load_teacher('retrained_fcn', 'cpu')
    buffers = {k: v.clone() for k, v in m.named_buffers()}
    fmap, h, logits = embed(m, torch.from_numpy(pair))
    assert tuple(fmap.shape) == (3, 128, 47) and tuple(h.shape) == (3, 128) and (tuple(logits.shape) == (3, 10))
    assert all((torch.equal(v, buffers[k]) for k, v in m.named_buffers()))
    np.savez_compressed(ROOT / 'results/synthetic_interface_test.npz', feature_map=fmap.numpy(), hidden=h.numpy(), logits=logits.numpy())
    recordings = []
    original = {r['file']: r for r in json.loads((ROOT / 'results/quality.json').read_text())}
    for path in sorted((ROOT.parent / '接触式').glob('*.DAT')):
        record = read_recording(path)
        assert record['txt_payload_identical']
        assert record['dat_sha256'] == original[path.name]['dat_sha256']
        assert record['txt_sha256'] == original[path.name]['txt_sha256']
        assert not any((len(s) >= FS for s in record['segments']))
        for s in record['segments']:
            counters = np.array([r['counter'] for r in s])
            assert np.all(np.diff(counters) == 1)
            assert len(set((r['packet_index'] for r in s))) == 1
        recordings.append(dict(file=path.name, sha256=record['dat_sha256']))
    script_hashes = {}
    for p in sorted((ROOT / 'scripts').glob('*.py')):
        ast.parse(p.read_text())
        script_hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    stats = np.genfromtxt(ROOT / 'results/axis_statistics.csv', delimiter=',', names=True, dtype=None, encoding='utf8')
    dc = []
    for row in stats:
        mu = float(row['mean'])
        sd = float(row['std'])
        dc.append(dict(rpm=int(row['rpm']), axis=str(row['axis']), uncentered_dct_dc_energy_fraction=2 * mu * mu / (2 * mu * mu + sd * sd)))
    write_json(ROOT / 'results/uncentered_dc_audit.json', dc)
    write_json(ROOT / 'results/verification.json', dict(status='passed', python=sys.executable, frequency_mapping_125Hz='DCT index 250 at 0.5 Hz spacing', matches_released_dataset_preprocessing=True, unit_scale_and_offset_invariance=True, rejects_short_constant_nonfinite_and_self_reference=True, synthetic_shapes=dict(feature_map=list(fmap.shape), hidden=list(h.shape), logits=list(logits.shape)), synthetic_is_not_real_contact_diagnosis=True, actual_recording_strict_windows=0, frozen_batchnorm_verified=True, dat_txt_payload_match=True, input_hashes_unchanged=True, recordings=recordings, script_hashes=script_hashes))
    print((ROOT / 'results/verification.json').read_text())
if __name__ == '__main__':
    main()
