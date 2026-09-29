"""Frozen BearLLM FCN/adapter diagnostics on local contact data.

No local query is ever used as a reference. External references are selected
before predictions from original MBHM training queries (one per source, seed 42).
The reference condition deliberately differs from the local rig: NOT a strict
reproduction of the paper's same-condition reference protocol.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import argparse, hashlib, importlib.util, json, sqlite3, sys, time
import numpy as np
import pandas as pd
import h5py
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from safetensors.torch import load_file
ROOT = Path(_c2r_resolve_path('/home/huangyating/BearLLM'))
ASSETS = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-assets'))
RUNS = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42'))
LABELS = ['normal', 'inner', 'outer', 'ball']
MAP = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]

def sha(path):
    h = hashlib.sha256()
    with open(_c2r_resolve_path(path), 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def savej(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def external_references():
    cached = Path(__file__).resolve().parent / 'external_references.npz'
    protocol = cached.with_name('external_reference_protocol.json')
    if cached.exists() and protocol.exists():
        with np.load(cached, allow_pickle=False) as saved:
            refs = saved['dcn'].astype(np.float32)
            ids = saved['file_id'].tolist()
        chosen = json.loads(protocol.read_text())['references']
        assert refs.shape == (9, 24000) and np.isfinite(refs).all()
        assert ids == [row['file_id'] for row in chosen]
        return (refs, chosen)
    dataset = ASSETS / 'mbhm_dataset'
    with sqlite3.connect(f'file:{dataset}/metadata.sqlite?mode=ro', uri=True) as c:
        rows = c.execute('SELECT f.file_id,f.condition_id,f.label,c.dataset FROM file_info f JOIN "condition" c ON f.condition_id=c.condition_id ORDER BY f.file_id').fetchall()
    healthy_conditions = {r[1] for r in rows if r[2] == 0}
    eligible = [r for r in rows if r[1] in healthy_conditions]
    train_healthy = [r for i, r in enumerate(eligible) if i % 10 < 7 and r[2] == 0]
    rng = np.random.default_rng(42)
    chosen = []
    for source in sorted({r[3] for r in train_healthy}):
        pool = [r for r in train_healthy if r[3] == source]
        r = pool[int(rng.integers(len(pool)))]
        chosen.append(dict(file_id=r[0], condition_id=r[1], label=r[2], source=r[3], source_training_query=True))
    with h5py.File(dataset / 'data.hdf5', 'r') as f:
        refs = np.stack([f['vibration'][r['file_id']] for r in chosen]).astype(np.float32)
    assert refs.shape == (9, 24000) and np.isfinite(refs).all()
    return (refs, chosen)

def load_model(kind):
    fcnmod = module(ROOT / 'models/FCN.py', 'standalone_fcn')
    model = fcnmod.FaultClassificationNetwork()
    provenance = {'kind': kind, 'weights': {}, 'lora_merged': False}
    if kind == 'retrained_fcn':
        wd = RUNS / 'pretrain/fcn'
        for attr, fn in [('encoder', 'feature_encoder.pth'), ('classifier', 'classifier.pth')]:
            path = wd / fn
            getattr(model, attr).load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            provenance['weights'][str(path)] = sha(path)
    else:
        wd = RUNS / 'finetune/weights' if kind == 'retrained_final' else ASSETS / 'bearllm_weights'
        base = torch.load(wd / 'vibration_adapter.pth', map_location='cpu', weights_only=True)
        enc = {k.removeprefix('feature_encoder.'): v for k, v in base.items() if k.startswith('feature_encoder.')}
        head = {k.removeprefix('alignment_layer.'): v.clone() for k, v in base.items() if k.startswith(('alignment_layer.linear1.', 'alignment_layer.linear2.'))}
        cfg = json.loads((wd / 'adapter_config.json').read_text())
        assert not any((cfg.get(k) for k in ['use_rslora', 'use_dora', 'rank_pattern', 'alpha_pattern', 'fan_in_fan_out']))
        lora = load_file(wd / 'adapter_model.safetensors')
        prefix = 'base_model.model.model.embed_tokens.adapter.alignment_layer.'
        for layer in ['linear1', 'linear2']:
            a = lora[prefix + layer + '.lora_A.weight'].float()
            b = lora[prefix + layer + '.lora_B.weight'].float()
            head[layer + '.weight'] = head[layer + '.weight'].float() + b @ a * (cfg['lora_alpha'] / cfg['r'])
        model.encoder.load_state_dict(enc)
        model.classifier.load_state_dict(head)
        for fn in ['vibration_adapter.pth', 'adapter_model.safetensors', 'adapter_config.json']:
            provenance['weights'][str(wd / fn)] = sha(wd / fn)
        provenance.update(lora_merged=True, lora_scale=cfg['lora_alpha'] / cfg['r'])
    for p in model.parameters():
        p.requires_grad_(False)
    return (model.eval(), provenance)

def read_inputs(data):
    summary = json.loads((data / 'summary.json').read_text())
    bags = {r['contact']: r for r in json.loads((data / 'recording_bags.json').read_text())}
    contacts = sorted([r for r in summary if r['modality'] == 'contact'], key=lambda x: x['output'])
    q, metadata, manifest = ([], [], [])
    for item in contacts:
        p = data / item['output']
        meta = json.loads(p.with_suffix('.json').read_text())
        with np.load(p, allow_pickle=False) as z:
            query = z['dcn']
            raw = z['raw_xyz']
            assert query.shape[1:] == (3, 24000) and np.all(query[:, :, 4000:] == 0)
            assert np.allclose(np.sqrt(np.mean(query ** 2, -1)), 0.01, atol=1e-06)
            from scipy.fft import dct
            ac = raw.astype(np.float64) - raw.mean(1, keepdims=True, dtype=np.float64)
            coef = dct(ac, axis=1, type=2, norm='backward').transpose(0, 2, 1)
            coef *= 0.01 * np.sqrt(24000) / np.linalg.norm(coef, axis=-1, keepdims=True)
            assert np.allclose(query[:, :, :4000], coef, atol=2e-05)
            q.append(query)
        for i, row in enumerate(meta['windows']):
            metadata.append(dict(file=p.name, window_row=i, label=item['label'], state=item['state'], baud=item['baud_candidate'], bag_id=bags.get(p.name, {}).get('candidate_bag_id', ''), bag_pairing='simultaneous_recording_user_confirmed_not_window_time_aligned', packet_ordinal_observed=row['packet_ordinal_observed'], independent_run_id=None, condition_id=None, sample_rate_hz=row['sample_rate_hz']))
        manifest.append(dict(file=str(p), sha256=sha(p), windows=len(query)))
    return (np.concatenate(q), pd.DataFrame(metadata), manifest)

def metrics(df):
    y = df['label']
    pred = df['pred4']
    return dict(n=len(df), accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, labels=range(4), average='macro', zero_division=0)), balanced_accuracy=float(balanced_accuracy_score(y, pred)), confusion_matrix=confusion_matrix(y, pred, labels=range(4)).tolist())

def evaluate(model, q, refs, md, output, device, batch):
    output.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    before = {n: t.detach().cpu().clone() for n, t in model.named_buffers()}
    features, hidden, logits, probs = ([], [], [], [])
    flat = q.reshape(-1, 24000)
    hsum = np.zeros((len(flat), 128), np.float64)
    fsum = np.zeros((len(flat), 128, 47), np.float64)
    lsum = np.zeros((len(flat), 10), np.float64)
    psum = np.zeros_like(lsum)
    with torch.inference_mode():
        for ref in refs:
            for start in range(0, len(flat), batch):
                query = flat[start:start + batch]
                x = torch.from_numpy(np.stack([query, np.broadcast_to(ref, query.shape)], axis=1)).to(device)
                f = model.encoder(x)
                h = torch.relu(model.classifier.linear1(f.flatten(1)))
                l = model.classifier.linear2(h)
                p = l.softmax(-1)
                assert all((torch.isfinite(t).all() for t in [f, h, l, p]))
                for dest, t in [(fsum, f), (hsum, h), (lsum, l), (psum, p)]:
                    dest[start:start + len(query)] += t.cpu().numpy()
    for n, t in model.named_buffers():
        assert torch.equal(before[n], t.cpu()), n
    n = len(q)
    nr = len(refs)
    hidden = (hsum / nr).reshape(n, 3, 128).astype(np.float32)
    fmap = (fsum / nr).reshape(n, 3, 128, 47).astype(np.float32)
    logits = (lsum / nr).reshape(n, 3, 10).astype(np.float32)
    probs = (psum / nr).reshape(n, 3, 10).astype(np.float32)
    p10 = probs.mean(1)
    p4 = np.stack([p10[:, idx].sum(1) for idx in MAP], axis=1)
    np.savez_compressed(output / 'contact_features.npz', hidden_mean=hidden.mean(1), hidden_axes=hidden, feature_map_mean=fmap.mean(1), logits10_axes=logits, logits10_mean=logits.mean(1), probs10_axes=probs, probs10_mean=p10, probs4_mean=p4, labels=md.label.to_numpy(), file_names=md.file.to_numpy(dtype=str), window_rows=md.window_row.to_numpy(), baud=md.baud.to_numpy(), bag_id=md.bag_id.to_numpy(dtype=str), states=md.state.to_numpy(dtype=str))
    md.to_csv(output / 'metadata.csv', index=False)
    pred = md.copy()
    pred['pred4'] = p4.argmax(1)
    pred['pred10'] = p10.argmax(1)
    for i in range(10):
        pred[f'p10_{i}'] = p10[:, i]
    for i in range(4):
        pred[f'p4_{i}'] = p4[:, i]
    pred.to_csv(output / 'window_predictions.csv', index=False)
    file = pred.groupby(['file', 'label', 'state', 'baud', 'bag_id'], as_index=False)[[f'p10_{i}' for i in range(10)] + [f'p4_{i}' for i in range(4)]].mean()
    file['pred4'] = file[[f'p4_{i}' for i in range(4)]].to_numpy().argmax(1)
    file['pred10'] = file[[f'p10_{i}' for i in range(10)]].to_numpy().argmax(1)
    file.to_csv(output / 'file_predictions.csv', index=False)
    result = dict(windows=metrics(pred[pred.label >= 0]), files=metrics(file[file.label >= 0]), buffers_unchanged=True, external_files=file[file.label < 0][['file', 'state', 'pred4', 'pred10']].to_dict(orient='records'), total_exported_windows=len(md), unknown_rows_excluded_from_four_class_metrics=True)
    savej(output / 'metrics.json', result)
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results')))
    p.add_argument('--output', type=Path, default=Path(_c2r_resolve_path(__file__)).resolve().parent)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch', type=int, default=48)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    q, md, manifest = read_inputs(args.data)
    refs, reference_metadata = external_references()
    savej(args.output / 'external_reference_protocol.json', dict(seed=42, selection='one training healthy query per MBHM source, NumPy default_rng; no local data selection', references=reference_metadata, condition='external mismatched condition; diagnostic fallback, not same-condition BearLLM protocol', pooling='mean probabilities and mean hidden across references; then mean across XYZ axes', zero_reference='diagnostic ablation only; outside training support'))
    np.savez_compressed(args.output / 'external_references.npz', dcn=refs, file_id=np.array([r['file_id'] for r in reference_metadata]))
    savej(args.output / 'input_manifest.json', manifest)
    allresults = []
    for kind in ['retrained_fcn', 'retrained_final', 'official_final']:
        model, provenance = load_model(kind)
        for refname, ref in [('fixed_external', refs), ('zero_reference', np.zeros((1, 24000), np.float32))]:
            name = f'{kind}_{refname}'
            started = time.monotonic()
            result = evaluate(model, q, ref, md, args.output / name, args.device, args.batch)
            savej(args.output / name / 'provenance.json', {**provenance, 'reference_protocol': refname, 'elapsed_seconds': time.monotonic() - started})
            allresults.append(dict(model=kind, reference=refname, **{f'file_{k}': v for k, v in result['files'].items() if k != 'confusion_matrix'}, **{f'window_{k}': v for k, v in result['windows'].items() if k != 'confusion_matrix'}))
            print(name, json.dumps(result), flush=True)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(allresults).to_csv(args.output / 'contact_metrics.csv', index=False)
    savej(args.output / 'runtime.json', dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__, device=args.device, source_sha256=sha(__file__), FCN_sha256=sha(ROOT / 'models/FCN.py'), input_files=len(manifest), windows=len(q), label_map=MAP, local_references_used=False, original_weights_modified=False))
if __name__ == '__main__':
    main()
