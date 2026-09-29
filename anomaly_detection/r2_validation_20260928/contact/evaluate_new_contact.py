"""Frozen contact-teacher transfer audit on the 2026-09-27 R2 collection.

No labels choose preprocessing, references, weights, or filtering. Existing
models are only read, and all exported complete packets are evaluated.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time
import h5py
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
HERE = Path(_c2r_resolve_path(__file__)).resolve().parent
BASE = HERE.parents[1]
DATA = Path(_c2r_resolve_path('/media/nas_users/huangyating/data/upload/contacts'))
API_PATH = BASE / 'four_class_chain_20260922/contact/evaluate_contact.py'
PHYS_PATH = BASE / 'encoder_validation_20260923_v2/scripts/physical_readouts.py'
HEAD_ROOT = BASE / 'retention_alignment_20260922/results/retained_head_v1/models'
LABELS = {'normal': 0, 'inBroken': 1, 'outBroken': 2, 'roll': 3}
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def sha(path):
    h = hashlib.sha256()
    with Path(_c2r_resolve_path(path)).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def writej(path, obj):
    Path(_c2r_resolve_path(path)).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def metrics(y, pred):
    return dict(n=len(y), correct=int(np.sum(np.asarray(y) == np.asarray(pred))), accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, labels=range(4), average='macro', zero_division=0)), balanced_accuracy=float(balanced_accuracy_score(y, pred)), confusion_matrix=confusion_matrix(y, pred, labels=range(4)).tolist())

def query_transform(raw, demean):
    x = raw.astype(np.float64)
    if demean:
        x = x - x.mean(1, keepdims=True)
    c = dct(x, type=2, norm='backward', axis=1).transpose(0, 2, 1)
    norms = np.linalg.norm(c, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError('Zero-energy DCN input; retain packet and report, do not silently drop')
    q = np.zeros((len(raw), 3, 24000), np.float32)
    q[:, :, :4000] = c * (0.01 * np.sqrt(24000) / norms)
    assert np.allclose(np.sqrt(np.mean(q.astype(float) ** 2, -1)), 0.01, atol=1e-08)
    return q

def infer(model, q, refs, device, batch):
    model = model.to(device).eval()
    before = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    flat = q.reshape(-1, 24000)
    hidden_sum = np.zeros((len(flat), 128), np.float64)
    probability_sum = np.zeros((len(flat), 10), np.float64)
    with torch.inference_mode():
        for reference_idx, ref in enumerate(refs):
            for start in range(0, len(flat), batch):
                query = flat[start:start + batch]
                inp = torch.from_numpy(np.stack([query, np.broadcast_to(ref, query.shape)], axis=1)).to(device)
                fm = model.encoder(inp)
                h = torch.relu(model.classifier.linear1(fm.flatten(1)))
                p = model.classifier.linear2(h).softmax(-1)
                if not torch.isfinite(h).all() or not torch.isfinite(p).all():
                    raise ValueError('Nonfinite model output')
                hidden_sum[start:start + len(query)] += h.cpu().numpy()
                probability_sum[start:start + len(query)] += p.cpu().numpy()
            print(f'completed reference {reference_idx + 1}/{len(refs)}', flush=True)
    for name, value in model.named_buffers():
        assert torch.equal(before[name], value.cpu()), name
    return ((hidden_sum / len(refs)).reshape(len(q), 3, 128).astype(np.float32), (probability_sum / len(refs)).reshape(len(q), 3, 10).astype(np.float32))

def probability_from_hidden(h, w, b):
    logits = h @ w.T + b
    logits = logits - logits.max(-1, keepdims=True)
    p10 = np.exp(logits)
    p10 /= p10.sum(-1, keepdims=True)
    return np.stack([p10[:, g].sum(-1) for g in GROUPS], -1)

def summarize_predictions(md, p, variant, model_name, rows, packet_rows, session_rows):
    label = md.label.to_numpy()
    pred = p.argmax(-1)
    base = dict(variant=variant, model=model_name)
    rows.append({**base, 'level': 'packet', 'scope': 'all', **metrics(label, pred)})
    for rpm in sorted(md.rpm.unique()):
        ids = md.rpm.eq(rpm).to_numpy()
        rows.append({**base, 'level': 'packet', 'scope': f'rpm_{rpm}', **metrics(label[ids], pred[ids])})
    dd = md.copy()
    dd['variant'], dd['model'], dd['prediction'] = (variant, model_name, pred)
    for k in range(4):
        dd[f'p{k}'] = p[:, k]
    packet_rows.append(dd)
    ss = dd.groupby(['session_id', 'label', 'rpm'], as_index=False)[[f'p{k}' for k in range(4)]].mean()
    ss['prediction'] = ss[[f'p{k}' for k in range(4)]].to_numpy().argmax(-1)
    ss['variant'], ss['model'] = (variant, model_name)
    session_rows.append(ss)
    rows.append({**base, 'level': 'session', 'scope': 'all', **metrics(ss.label, ss.prediction)})
    for row in ss.itertuples():
        ids = md.session_id.eq(row.session_id).to_numpy()
        rows.append({**base, 'level': 'packet', 'scope': row.session_id, **metrics(label[ids], pred[ids])})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--sensitivity', action='store_true')
    args = ap.parse_args()
    assert sys.version_info >= (3, 10)
    if (HERE / 'contact_features.npz').exists():
        raise FileExistsError('Completed primary output exists; avoid replacing new-data evaluation')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    api, phys = (module(API_PATH, 'old_contact_api'), module(PHYS_PATH, 'old_physical_targets'))
    records, raw, quality, inputs = ([], [], [], [])
    for session in sorted(args.data.iterdir()):
        if not session.is_dir():
            continue
        state, rpm_s, _ = session.name.rsplit('_', 2)
        rpm, label = (int(rpm_s.removesuffix('r')), LABELS[state])
        path = session / 'contact.h5'
        qc = json.loads((session / 'contact_qc.json').read_text())
        with h5py.File(path, 'r') as f:
            assert f.attrs['sample_rate_hz'] == 4000
            packets = f['raw_xyz'][:]
            assert packets.shape[1:] == (4096, 3)
            assert np.isfinite(packets).all()
            selected = packets[:, 48:4048]
            ordinals = f['packet_ordinal_observed'][:]
            byte_start, byte_end = (f['byte_start'][:], f['byte_end_exclusive'][:])
            for j, ordinal in enumerate(ordinals):
                records.append(dict(session_id=session.name, label=label, class_name=state, rpm=rpm, packet_index=j, packet_ordinal_observed=int(ordinal), byte_start=int(byte_start[j]), byte_end_exclusive=int(byte_end[j]), absolute_packet_time_known=False))
            raw.append(selected)
        complete, incomplete = (qc['complete_packets'], qc['incomplete_candidates'])
        quality.append(dict(session_id=session.name, label=label, rpm=rpm, complete_packets=complete, incomplete_candidates=incomplete, packet_start_count=qc['packet_starts_observed'], complete_fraction_of_candidates=complete / (complete + incomplete), valid_syntax_rows=qc['valid_syntax_rows'], complete_samples=complete * 4096, complete_duration_s=complete * 4096 / 4000, one_second_windows=complete, real_wall_duration_unknown=True, issues=qc['issue_reason_counts']))
        inputs.append(dict(path=str(path), sha256=sha(path), source_sha256=qc['source_sha256']))
    md, raw = (pd.DataFrame(records), np.concatenate(raw))
    assert raw.shape == (343, 4000, 3)
    md.to_csv(HERE / 'metadata.csv', index=False)
    pd.DataFrame(quality).to_csv(HERE / 'session_quality.csv', index=False)
    refs, refmeta = api.external_references()
    writej(HERE / 'protocol_before_results.json', dict(preprocessing_primary='v0', query='4000 Hz; [48:4048] from each complete 4096 XYZ packet; demean each axis; signed DCT-II backward; zero-pad 4000 coefficients to 24000; RMS=0.01', reference='same fixed nine public-source training healthy references selected by old pipeline seed42; not same-rig reference', references=refmeta, primary_teacher='retrained_fcn', embedding='ReLU(classifier.linear1(encoder(query,reference))); mean over 9 references then XYZ; raw unstandardized nonnegative 128D', head_primary=str(HEAD_ROOT / 'v0/all_known/retained_head.npz'), head_training='only old 2026-09-20 data and public replay; no new R2 labels used', evaluation='all 343 complete packets; 12 sessions; 4 labels; 3 RPM; pooled packets, per-RPM, per-session and session mean probabilities', adapted_head_pooling='apply head after pooling hidden; not mean of per-view probabilities', original_native_pooling='report both source head applied to pooled hidden and original mean-of-probabilities', sensitivity_predeclared=args.sensitivity, axis_sensitivity='XYZ primary; XY-only averaging applies uniformly to every class, motivated before inference by independent audit of possible Z-axis clipping; never choose best axis by new-data accuracy', no_new_training=True, no_model_selection=True, no_wrong_prediction_filtering=True, four_class_groups=GROUPS, targets=phys.TARGETS, physical_target_definition='legacy six raw-signal descriptors; 20-800 Hz zero-phase Butterworth; trim 0.1s each end; same frozen descriptor code; not severity labels', inputs=inputs, source_sha256=sha(__file__), reference_api_sha256=sha(API_PATH), physical_script_sha256=sha(PHYS_PATH)))
    targets = np.stack([phys.descriptor(x) for x in raw])
    desc = md.copy()
    for j, name in enumerate(phys.TARGETS):
        desc[name] = targets[:, j]
    for j, axis in enumerate('xyz'):
        desc[f'raw_mean_{axis}'] = raw.mean(1)[:, j]
        desc[f'raw_ac_rms_{axis}'] = raw.std(1)[:, j]
        desc[f'raw_abs_max_{axis}'] = np.abs(raw[:, :, j]).max(1)
        desc[f'raw_abs_ge_1p99_fraction_{axis}'] = (np.abs(raw[:, :, j]) >= 1.99).mean(1)
    desc.to_csv(HERE / 'physical_targets.csv', index=False)
    np.savez_compressed(HERE / 'raw_contact_windows.npz', raw_xyz=raw.astype(np.float32), session_id=md.session_id.to_numpy(str), packet_index=md.packet_index.to_numpy())
    variants = [('v0', True, False, 'retrained_fcn')]
    if args.sensitivity:
        variants += [('query_no_demean_ref2k', False, True, 'retrained_fcn'), ('v0', True, False, 'official_final')]
    rows, packet_rows, session_rows, checks, provenance = ([], [], [], [], [])
    started = time.monotonic()
    for variant, demean, bandlimit, kind in variants:
        model, prov = api.load_model(kind)
        q = query_transform(raw, demean)
        reference = refs.copy()
        if bandlimit:
            reference[:, 4000:] = 0
            reference *= 0.01 * np.sqrt(24000) / np.linalg.norm(reference.astype(float), axis=-1, keepdims=True)
        hidden_axes, probs_axes = infer(model, q, reference, args.device, args.batch)
        hidden = hidden_axes.mean(1)
        hidden_xy = hidden_axes[:, :2].mean(1)
        w = model.classifier.linear2.weight.detach().cpu().numpy()
        b = model.classifier.linear2.bias.detach().cpu().numpy()
        p10 = probs_axes.mean(1)
        prob_view = np.stack([p10[:, g].sum(-1) for g in GROUPS], -1)
        prob_hidden = probability_from_hidden(hidden, w, b)
        summarize_predictions(md, prob_view, variant, f'{kind}_original_probability_pool', rows, packet_rows, session_rows)
        summarize_predictions(md, prob_hidden, variant, f'{kind}_original_hidden_pool', rows, packet_rows, session_rows)
        summarize_predictions(md, probability_from_hidden(hidden_xy, w, b), variant, f'{kind}_original_hidden_pool_xy_sensitivity', rows, packet_rows, session_rows)
        result = dict(embedding=hidden, embedding_xy_sensitivity=hidden_xy, hidden_axes=hidden_axes, labels=md.label.to_numpy(), rpm=md.rpm.to_numpy(), session_id=md.session_id.to_numpy(str), packet_index=md.packet_index.to_numpy(), packet_ordinal_observed=md.packet_ordinal_observed.to_numpy(), physical_targets=targets, physical_target_names=np.array(phys.TARGETS), original_weight10=w, original_bias10=b, original_probability_pool=prob_view, original_hidden_pool=prob_hidden)
        if kind == 'retrained_fcn':
            hp = HEAD_ROOT / variant / 'all_known/retained_head.npz'
            with np.load(hp, allow_pickle=False) as z:
                rw, rb = (z['weight10'], z['bias10'])
                old_bags = z['local_training_bags'].tolist()
            assert not set(old_bags) & set(md.session_id)
            retained_p = probability_from_hidden(hidden, rw, rb)
            summarize_predictions(md, retained_p, variant, 'existing_retained_head_hidden_pool', rows, packet_rows, session_rows)
            summarize_predictions(md, probability_from_hidden(hidden_xy, rw, rb), variant, 'existing_retained_head_hidden_pool_xy_sensitivity', rows, packet_rows, session_rows)
            result.update(retained_weight10=rw, retained_bias10=rb, retained_probabilities4=retained_p)
            provenance.append(dict(head_path=str(hp), sha256=sha(hp), trained_old_bags=old_bags, source_acceptance=json.loads((hp.parent / 'acceptance.json').read_text())))
        stem = 'contact_features' if variant == 'v0' and kind == 'retrained_fcn' else f'contact_features_{kind}_{variant}'
        np.savez_compressed(HERE / f'{stem}.npz', **result)
        for path, expected in prov['weights'].items():
            assert sha(path) == expected
        checks.append(dict(variant=variant, kind=kind, finite_hidden=bool(np.isfinite(hidden).all()), hidden_min=float(hidden.min()), n_packets=len(hidden), model_inputs_require_grad=False, model_batchnorm_buffers_unchanged=True, pretrained_files_unchanged=True, model_provenance=prov))
        pd.DataFrame(rows).to_csv(HERE / 'metrics.csv', index=False)
        pd.concat(packet_rows).to_csv(HERE / 'packet_predictions.csv', index=False)
        pd.concat(session_rows).to_csv(HERE / 'session_predictions.csv', index=False)
        print(f'{kind} {variant}: ', [r for r in rows if r['scope'] == 'all'][-6:], flush=True)
        del model, q
        torch.cuda.empty_cache()
    for item in inputs:
        assert sha(item['path']) == item['sha256']
    for item in provenance:
        assert sha(item['head_path']) == item['sha256']
    writej(HERE / 'metrics.json', rows)
    writej(HERE / 'verification.json', dict(checks=checks, heads=provenance, no_new_label_training=True, input_files_unchanged=True, source_checkpoints_unchanged=True, n_packets=len(md), sessions=md.session_id.nunique(), elapsed_seconds=time.monotonic() - started, python=sys.executable, torch=torch.__version__, raw_output_is_unstandardized=True, one_contact_session_shared_across_radar_distances_and_repeats=True, independent_cross_modal_pairs_are_sessions_not_144_recordings=True, contact_preprocessing_source_sha256=sha(__file__)))
if __name__ == '__main__':
    main()
