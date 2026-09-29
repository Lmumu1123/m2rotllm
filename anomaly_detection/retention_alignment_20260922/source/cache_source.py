"""Read-only MBHM regression cache for the independently pretrained frozen FCN.

Replay selection is label/source-stratified with a fixed seed, before inference.
All original validation/test queries are retained. Never fits anything to test.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import argparse, collections, hashlib, importlib.util, json, random, sqlite3, sys, time
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
REPO = Path(_c2r_resolve_path('/home/huangyating/BearLLM'))
DATA = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-assets/mbhm_dataset'))
PRE = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain'))
sys.path.insert(0, str(REPO / 'scripts'))
from evaluate_mbhm import read_metadata, make_protocol
sys.path.insert(0, str(REPO))
from models.FCN import FaultClassificationNetwork
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]
MAP = np.array([0, 1, 1, 1, 3, 3, 3, 2, 2, 2])

def sha(p):
    h = hashlib.sha256()
    with open(_c2r_resolve_path(p), 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def savej(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def metrics(y, prob, classes):
    pred = prob.argmax(-1)
    return dict(n=int(len(y)), accuracy=float(accuracy_score(y, pred)), macro_f1=float(f1_score(y, pred, labels=list(range(classes)), average='macro', zero_division=0)), confusion_matrix=confusion_matrix(y, pred, labels=list(range(classes))).tolist())

def collapse(p):
    return np.stack([p[..., g].sum(-1) for g in GROUPS], -1)

def source_stratified(rows, limit, seed):
    rng = random.Random(seed)
    out = []
    for label in range(10):
        buckets = collections.defaultdict(list)
        for row in rows:
            if row['label'] == label:
                buckets[row['source']].append(row)
        for pool in buckets.values():
            rng.shuffle(pool)
        sources = sorted(buckets)
        chosen = []
        i = 0
        while len(chosen) < limit and any(buckets.values()):
            source = sources[i % len(sources)]
            i += 1
            if buckets[source]:
                chosen.append(buckets[source].pop())
        out.extend(chosen)
    return sorted(out, key=lambda r: r['file_id'])

def protocol(output):
    rows = read_metadata(DATA)
    pairs, excluded = make_protocol(rows, 42, 3)
    old = json.loads((PRE / 'dataset.json').read_text())
    assert all((old[s] == [[r['file_id'], r['ref_id'], r['label']] for r in pairs if r['split'] == s] for s in old))
    queries = []
    for i in range(0, len(pairs), 3):
        r = pairs[i]
        q = {k: r[k] for k in ['file_id', 'condition_id', 'label', 'source', 'split']}
        q['references'] = [p['ref_id'] for p in pairs[i:i + 3]]
        queries.append(q)
    splits = {r['file_id']: r['split'] for r in queries}
    train = source_stratified([r for r in queries if r['split'] == 'train'], 128, 20260922)
    selected = train + [r for r in queries if r['split'] in ['val', 'test']]
    rng = random.Random(20260923)
    buckets = collections.defaultdict(list)
    for r in queries:
        if r['split'] == 'val':
            buckets[r['source'], r['label']].append(r)
    sensitivity = []
    for key, pool in sorted(buckets.items()):
        rng.shuffle(pool)
        sensitivity.extend(pool[:8])
    sensitivity = sorted(sensitivity, key=lambda r: r['file_id'])
    stats = {s: {'query_count': sum((r['split'] == s for r in queries)), 'selected_count': sum((r['split'] == s for r in selected)), 'same_query_ref_pairs': sum((p['split'] == s and p['file_id'] == p['ref_id'] for p in pairs)), 'reference_query_split_counts': dict(collections.Counter((splits[p['ref_id']] for p in pairs if p['split'] == s)))} for s in ['train', 'val', 'test']}
    selected_refs = {x for r in train for x in r['references']}
    test_condition = {r['condition_id'] for r in queries if r['split'] == 'test'}
    train_condition = {r['condition_id'] for r in queries if r['split'] == 'train'}
    info = dict(seed=42, reference_draws=3, metadata_rows=len(rows), eligible_queries=len(queries), excluded_queries=len(excluded), saved_dataset_json_matches_reconstructed_seed42=True, splits=stats, selection='All original val/test; train max128 per 10-class, round-robin available sources, seed20260922. No score filtering.', replay_train_selected_by_10class=dict(collections.Counter((r['label'] for r in train))), replay_train_selected_by_source=dict(collections.Counter((r['source'] for r in train))), sensitivity_selection='Validation only; each source x 10-class cell <=8, seed20260923; fixed before inference.', sensitivity_queries=len(sensitivity), sensitivity_ids=[r['file_id'] for r in sensitivity], replay_reference_query_split_counts=dict(collections.Counter((splits[x] for x in selected_refs))), heldout_test_conditions_overlapping_training=len(test_condition & train_condition), total_test_conditions=len(test_condition), labels4=['normal', 'inner', 'outer', 'ball'], label10_to4=MAP.tolist(), limitations=['Upstream split is query ordinal modulo10, NOT acquisition/motor/bearing entity split.', 'Original reference pool includes train/val/test healthy queries and can include the query itself; retained for comparable regression, not independent reference-clean evaluation.', 'Train replay subset is source-balanced and is NOT the original train prevalence.', 'Cache stores mean hidden, mean logits and mean probabilities separately: softmax(mean logits) differs from mean softmax(logits).'])
    savej(output / 'protocol.json', info)
    return (selected, sensitivity, info)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path(_c2r_resolve_path(__file__)).resolve().parent)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--query-batch', type=int, default=64)
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'cache.npz').exists():
        raise FileExistsError('Refusing to replace completed cache')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    selected, sensitivity, info = protocol(out)
    model = FaultClassificationNetwork()
    for attr, fn in [('encoder', 'feature_encoder.pth'), ('classifier', 'classifier.pth')]:
        getattr(model, attr).load_state_dict(torch.load(PRE / 'fcn' / fn, map_location='cpu', weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(args.device).eval()
    before = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    stats = json.loads((PRE / 'results.json').read_text())
    provenance = dict(checkpoint='independent FCN exported by released-code-seed42/pretrain; NOT final LLM LoRA adapter', weights={str(PRE / 'fcn' / f): sha(PRE / 'fcn' / f) for f in ['feature_encoder.pth', 'classifier.pth']}, metadata_sha256=sha(DATA / 'metadata.sqlite'), training_dataset_json_sha256=sha(PRE / 'dataset.json'), epochs_completed=stats['epochs_completed'], best_epoch=stats['best_epoch'], source_selected_epoch=stats['source_selected_epoch'], python=sys.executable, torch=torch.__version__, source_sha256=sha(__file__), fcndef_sha256=sha(REPO / 'models/FCN.py'))
    savej(out / 'provenance.json', provenance)
    start = time.monotonic()
    with h5py.File(DATA / 'data.hdf5', 'r') as f:
        signal = f['vibration'][:]
    assert signal.shape == (135516, 24000)
    print(json.dumps(dict(stage='loaded', seconds=time.monotonic() - start, queries=len(selected), signal_GB=signal.nbytes / 1000000000.0)), flush=True)

    def infer(items, cutoff=None, renorm=True):
        hidden = np.empty((len(items), 3, 128), np.float32)
        logits = np.empty((len(items), 3, 10), np.float32)
        with torch.inference_mode():
            for start in range(0, len(items), args.query_batch):
                batch = items[start:start + args.query_batch]
                ids = np.asarray([[[r['file_id'], ref] for ref in r['references']] for r in batch])
                x = signal[ids].reshape(-1, 2, 24000).copy()
                if cutoff is not None:
                    x[:, :, cutoff:] = 0
                    if renorm:
                        x *= 0.01 / np.maximum(np.sqrt(np.mean(x.astype(np.float64) ** 2, -1, keepdims=True)), 1e-20)
                t = torch.from_numpy(x).to(args.device)
                f = model.encoder(t)
                h = torch.relu(model.classifier.linear1(f.flatten(1)))
                l = model.classifier.linear2(h)
                assert torch.isfinite(h).all() and torch.isfinite(l).all()
                hidden[start:start + len(batch)] = h.cpu().numpy().reshape(-1, 3, 128)
                logits[start:start + len(batch)] = l.cpu().numpy().reshape(-1, 3, 10)
                if cutoff is None and (start == 0 or start % (args.query_batch * 50) == 0):
                    print(json.dumps(dict(stage='encode', complete_queries=start + len(batch), total_queries=len(items), elapsed_seconds=time.monotonic() - start_all)), flush=True)
        p = torch.from_numpy(logits).softmax(-1).numpy()
        return (hidden, logits, p)
    start_all = time.monotonic()
    h, l, p = infer(selected)
    md = pd.DataFrame([{k: v for k, v in r.items() if k != 'references'} | {'ref_id_0': r['references'][0], 'ref_id_1': r['references'][1], 'ref_id_2': r['references'][2]} for r in selected])
    md['label4'] = MAP[md.label.to_numpy()]
    md['cache_row'] = np.arange(len(md))
    md.to_csv(out / 'metadata.csv', index=False)
    np.savez_compressed(out / 'cache.npz', hidden_mean=h.mean(1), hidden_refs=h, logits10_mean=l.mean(1), logits10_refs=l, probs10_mean=p.mean(1), probs10_refs=p, probs4_mean=collapse(p.mean(1)), label10=md.label.to_numpy(), label4=md.label4.to_numpy(), split=md.split.to_numpy(dtype=str), query_id=md.file_id.to_numpy(), condition_id=md.condition_id.to_numpy(), source=md.source.to_numpy(dtype=str), reference_ids=md[['ref_id_0', 'ref_id_1', 'ref_id_2']].to_numpy())
    np.savez(out / 'original_head.npz', weight10=before['classifier.linear2.weight'].numpy(), bias10=before['classifier.linear2.bias'].numpy())
    print(json.dumps(dict(stage='cache_saved', path=str(out / 'cache.npz'), seconds=time.monotonic() - start_all)), flush=True)
    metrics_rows = []
    detail = {}
    for split in ['train', 'val', 'test']:
        m = md.split.to_numpy() == split
        for source in ['all'] + sorted(md.loc[m, 'source'].unique().tolist()):
            keep = m if source == 'all' else m & (md.source.to_numpy() == source)
            if not keep.any():
                continue
            modes = {'mean_probability_10': (md.label.to_numpy()[keep], p[keep].mean(1), 10), 'mean_probability_4': (md.label4.to_numpy()[keep], collapse(p[keep].mean(1)), 4), 'mean_hidden_original_10': (md.label.to_numpy()[keep], torch.from_numpy(l[keep].mean(1)).softmax(-1).numpy(), 10), 'mean_hidden_original_4': (md.label4.to_numpy()[keep], collapse(torch.from_numpy(l[keep].mean(1)).softmax(-1).numpy()), 4), 'pair_probability_10': (np.repeat(md.label.to_numpy()[keep], 3), p[keep].reshape(-1, 10), 10), 'pair_probability_4': (np.repeat(md.label4.to_numpy()[keep], 3), collapse(p[keep].reshape(-1, 10)), 4)}
            for name, (y, probs, k) in modes.items():
                met = metrics(y, probs, k)
                detail[f'{split}/{source}/{name}'] = met
                metrics_rows.append(dict(split=split, source=source, method=name, **{a: b for a, b in met.items() if a != 'confusion_matrix'}))
    pd.DataFrame(metrics_rows).to_csv(out / 'baseline_metrics.csv', index=False)
    savej(out / 'baseline_metrics_detail.json', detail)
    sens = []
    sens_arrays = {}
    index = {r['file_id']: i for i, r in enumerate(selected)}
    ix = np.asarray([index[r['file_id']] for r in sensitivity])
    y10 = md.label.to_numpy()[ix]
    y4 = md.label4.to_numpy()[ix]
    for name, cut, renorm in [('full', None, False), ('cut_2000Hz_renorm', 4000, True), ('cut_800Hz_renorm', 1600, True), ('cut_2000Hz_no_renorm', 4000, False)]:
        hh, ll, pp = (h[ix], l[ix], p[ix]) if cut is None else infer(sensitivity, cut, renorm)
        sens_arrays[name + '_hidden_mean'] = hh.mean(1)
        sens_arrays[name + '_probs10_mean'] = pp.mean(1)
        for k, yy, prob in [(10, y10, pp.mean(1)), (4, y4, collapse(pp.mean(1)))]:
            met = metrics(yy, prob, k)
            sens.append(dict(variant=name, classes=k, **{a: b for a, b in met.items() if a != 'confusion_matrix'}))
    pd.DataFrame(sens).to_csv(out / 'bandwidth_sensitivity.csv', index=False)
    np.savez_compressed(out / 'bandwidth_sensitivity.npz', **sens_arrays, query_id=md.file_id.to_numpy()[ix], label10=y10, label4=y4, source=md.source.to_numpy(dtype=str)[ix])
    unchanged = all((torch.equal(before[n], t.detach().cpu()) for n, t in model.state_dict().items()))
    assert unchanged
    reconstructed = h.mean(1) @ before['classifier.linear2.weight'].numpy().T + before['classifier.linear2.bias'].numpy()
    error = float(np.max(np.abs(reconstructed - l.mean(1))))
    assert error < 0.001
    savej(out / 'completion.json', dict(complete=True, queries=len(md), cached_reference_pairs=len(md) * 3, weights_and_buffers_unchanged=unchanged, source_head_reconstruction_max_abs_error=error, elapsed_seconds=time.monotonic() - start_all, cache_sha256=sha(out / 'cache.npz'), metadata_sha256=sha(out / 'metadata.csv')))
    print(json.dumps(dict(stage='completed', queries=len(md), seconds=time.monotonic() - start_all)), flush=True)
if __name__ == '__main__':
    main()
