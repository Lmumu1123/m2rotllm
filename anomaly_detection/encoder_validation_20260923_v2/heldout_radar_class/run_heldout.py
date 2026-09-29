"""Fixed-protocol leave-one-radar-class transfer; all 96 models reported."""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import hashlib, importlib.util, json, random, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parent
RADAR = ROOT.parent / 'geometry_corrected/radar'
CONTACT = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922/preprocessing/variants/v0/retrained_fcn'))
ARCHIVE = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922/radar/stage_b_v0'))
SOURCE = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/encoder_validation_20260923/controls/run_controls.py'))
spec = importlib.util.spec_from_file_location('existing_controls_architecture', SOURCE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
METHODS = ['ce_only', 'matched_bag_mse', 'full', 'label_code_mse']
SEEDS = [17, 42, 73]

def sha(p):
    return hashlib.sha256(Path(_c2r_resolve_path(p)).read_bytes()).hexdigest()

def savej(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def make_codes(w, b, mean, scale, labels):
    mean = torch.tensor(mean, dtype=torch.float32)
    scale = torch.tensor(scale, dtype=torch.float32)
    z = nn.Parameter(torch.zeros(len(labels), 128))
    opt = torch.optim.Adam([z], lr=0.05)
    yt = torch.tensor(labels)
    for _ in range(400):
        h = F.relu(mean + scale * z)
        zz = (h - mean) / scale
        loss = F.cross_entropy(api.grouping(h @ w.T + b), yt) + 0.001 * zz.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = F.relu(mean + scale * z)
        zz = (h - mean) / scale
        p = api.grouping(h @ w.T + b).softmax(-1)
    return (h, zz, p)

def metrics(y, p, heldout, common, unit):
    yp = p.argmax(1)
    rows = []
    for scope, mask in [('all_four', np.ones(len(y), bool)), ('seen_three', y != heldout), ('unseen_one', y == heldout)]:
        labels = list(range(4)) if scope == 'all_four' else [heldout] if scope == 'unseen_one' else [k for k in range(4) if k != heldout]
        yy = y[mask]
        pp = yp[mask]
        rows.append(dict(**common, unit=unit, scope=scope, n=int(mask.sum()), correct=int((yy == pp).sum()), accuracy=float(accuracy_score(yy, pp)), macro_f1=float(f1_score(yy, pp, labels=labels, average='macro', zero_division=0)), predicted_heldout=int((pp == heldout).sum()), predicted_heldout_rate=float((pp == heldout).mean())))
    return rows

def main():
    assert 'envs/m2vllm/' in sys.executable
    torch.set_num_threads(2)
    started = time.time()
    out = ROOT / 'results'
    assert not out.exists(), 'Refusing to overwrite completed/running run'
    out.mkdir()
    for d in ['models', 'windows', 'targets']:
        (out / d).mkdir()
    protocol = json.loads((ROOT / 'protocol_before_results.json').read_text())
    cm = pd.read_csv(CONTACT / 'metadata.csv')
    rm = pd.read_csv(RADAR / 'metadata.csv')
    h = np.load(CONTACT / 'contact_features.npz')['hidden_mean'].astype(np.float32)
    rz = np.load(RADAR / 'features.npz')
    x = rz['frame_shape'].astype(np.float32)
    assert np.array_equal(rz['bag_id'], rm.bag_id.to_numpy(str))
    assert not rm.loc[rm.label >= 0, 'geometry_roi_mismatch'].any()
    splits = [s for s in json.loads((ARCHIVE / 'splits.json').read_text()) if s['test_bags']]
    files = [SOURCE, ROOT / 'protocol_before_results.json', CONTACT / 'metadata.csv', CONTACT / 'contact_features.npz', RADAR / 'features.npz', RADAR / 'metadata.csv', ARCHIVE / 'splits.json'] + [Path(_c2r_resolve_path(s['shared_head'])) for s in splits]
    hashes = {str(p): sha(p) for p in files}
    savej(out / 'input_hashes.json', hashes)
    rm.to_csv(out / 'radar_metadata.csv', index=False)
    cm.to_csv(out / 'contact_metadata.csv', index=False)
    rows = []
    bags = []
    checks = []
    split_records = []
    traces = []
    for sp in splits:
        fold = sp['fold']
        original_train = sp['train_bags']
        test_bags = sp['test_bags']
        hd = np.load(sp['shared_head'])
        assert set(hd['local_training_bags'].tolist()) == set(original_train)
        w = torch.tensor(hd['weight10'], dtype=torch.float32)
        b = torch.tensor(hd['bias10'], dtype=torch.float32)
        wbefore = w.clone()
        bbefore = b.clone()
        for heldout in range(4):
            train_bags = [bag for bag in original_train if int(cm.loc[cm.bag_id.eq(bag), 'label'].iloc[0]) != heldout]
            assert len(train_bags) == 3
            ctr = cm.bag_id.isin(train_bags).to_numpy()
            rtr = rm.bag_id.isin(train_bags).to_numpy()
            rte = rm.bag_id.isin(test_bags).to_numpy()
            assert not (rtr & rte).any() and (not (rm.loc[rtr, 'label'] == heldout).any())
            assert not (cm.loc[ctr, 'label'] == heldout).any() and (rm.loc[rte, 'label'] == heldout).any()
            seenlabels = np.array([int(cm.loc[cm.bag_id.eq(bag), 'label'].iloc[0]) for bag in train_bags])
            csc = StandardScaler().fit(h[ctr], sample_weight=api.weights(cm[ctr]))
            rsc = StandardScaler().fit(x[rtr], sample_weight=api.weights(rm[rtr]))
            xt = torch.tensor(rsc.transform(x).astype(np.float32))
            hz = csc.transform(h).astype(np.float32)
            targets = torch.tensor(np.stack([hz[cm.bag_id.eq(bag)].mean(0) for bag in train_bags]))
            yi = torch.tensor(seenlabels)
            positive = torch.tensor(seenlabels[:, None] == seenlabels[None, :])
            with torch.no_grad():
                tl = torch.tensor(h) @ w.T + b
                qt = torch.stack([torch.stack([(tl[cm.bag_id.eq(bag).to_numpy()] / 2).softmax(-1)[:, g].sum(-1) for g in api.GROUPS], -1).mean(0) for bag in train_bags])
            code_h, code_z, code_p = make_codes(w, b, csc.mean_, csc.scale_, seenlabels)
            pools = [np.flatnonzero(rm.bag_id.eq(bag).to_numpy()) for bag in train_bags]
            assert all((not (rm.iloc[p].label == heldout).any() for p in pools))
            ident = f'{fold}__heldout{heldout}'
            np.savez_compressed(out / 'targets' / f'{ident}.npz', seen_labels=seenlabels, train_bags=np.array(train_bags), teacher_targets_standardized=targets.numpy(), artificial_codes_standardized=code_z.numpy(), artificial_codes_hidden=code_h.numpy(), artificial_codes_probabilities4=code_p.numpy(), contact_mean=csc.mean_, contact_scale=csc.scale_, radar_mean=rsc.mean_, radar_scale=rsc.scale_)
            split_records.append(dict(fold=fold, heldout_label=heldout, student_train_bags=train_bags, test_bags=test_bags, excluded_source_bags=sorted(set(original_train) - set(train_bags)), radar_scaler_fit_rows=np.flatnonzero(rtr).tolist(), contact_scaler_fit_rows=np.flatnonzero(ctr).tolist(), head_prior_contact_training_bags=original_train, shared_head=sp['shared_head'], target_radar_class_seen_for_fit=False, whole_system_zero_shot=False))
            for method in METHODS:
                target = code_z if method == 'label_code_mse' else targets
                for seed in SEEDS:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    model = api.Encoder(csc.mean_, csc.scale_)
                    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                    rng = np.random.default_rng(seed)
                    for step in range(400):
                        ids = np.concatenate([rng.choice(pool, 16, replace=True) for pool in pools])
                        hh = model(xt[ids]).reshape(3, 16, 128)
                        zz = (hh - model.contact_mean) / model.contact_scale
                        mean_z = zz.mean(1)
                        log10 = hh @ w.T + b
                        log4 = api.grouping(log10)
                        ce = F.cross_entropy(log4.reshape(-1, 4), yi.repeat_interleave(16))
                        mse = F.mse_loss(mean_z, target)
                        sims = F.normalize(mean_z, dim=1) @ F.normalize(targets, dim=1).T / 0.2
                        con = (torch.logsumexp(sims, 1) - torch.logsumexp(sims.masked_fill(~positive, -torch.inf), 1)).mean()
                        p10 = (log10 / 2).softmax(-1)
                        p4 = torch.stack([p10[..., g].sum(-1) for g in api.GROUPS], -1).mean(1)
                        kd = 4 * (qt * (qt.clamp_min(1e-10).log() - p4.clamp_min(1e-10).log())).sum(1).mean()
                        loss = ce if method == 'ce_only' else ce + mse + 0.1 * con + 0.5 * kd if method == 'full' else mse
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        opt.step()
                        if step in [0, 99, 199, 299, 399]:
                            traces.append(dict(fold=fold, heldout_label=heldout, method=method, seed=seed, step=step + 1, loss=float(loss.detach()), ce=float(ce.detach()), mse=float(mse.detach())))
                    model.eval()
                    with torch.inference_mode():
                        hh = model(xt[rte])
                        zz = (hh - model.contact_mean) / model.contact_scale
                        log10 = hh @ w.T + b
                        prob = api.grouping(log10).softmax(-1).numpy()
                    hh = hh.numpy()
                    zz = zz.numpy()
                    meta = rm.loc[rte].reset_index(drop=True)
                    key = f'{ident}__{method}__seed{seed}'
                    common = dict(fold=fold, heldout_label=heldout, method=method, seed=seed)
                    np.savez_compressed(out / 'windows' / f'{key}.npz', hidden=hh, standardized_hidden=zz, probabilities4=prob, logits10=log10.numpy(), bag_id=meta.bag_id.to_numpy(str), label=meta.label.to_numpy(), source_metadata_row=np.flatnonzero(rte), is_unseen=meta.label.to_numpy() == heldout)
                    rows.extend(metrics(meta.label.to_numpy(), prob, heldout, common, 'window'))
                    truth = []
                    bagprob = []
                    for bag in test_bags:
                        ids = meta.bag_id.eq(bag).to_numpy()
                        ci = cm.bag_id.eq(bag).to_numpy()
                        p = prob[ids].mean(0)
                        label = int(meta.loc[ids, 'label'].iloc[0])
                        mean = zz[ids].mean(0)
                        teacher = hz[ci].mean(0)
                        truth.append(label)
                        bagprob.append(p)
                        bags.append(dict(**common, bag_id=bag, label=label, scope='unseen_one' if label == heldout else 'seen_three', prediction=int(p.argmax()), n_windows=int(ids.sum()), window_correct=int((prob[ids].argmax(1) == label).sum()), standardized_mean_mse=float(np.mean((mean - teacher) ** 2)), standardized_mean_cosine=api.cosine(mean, teacher), **{f'p{k}': float(p[k]) for k in range(4)}))
                    rows.extend(metrics(np.array(truth), np.array(bagprob), heldout, common, 'recording'))
                    checkpoint = dict(encoder=model.state_dict(), radar_mean=torch.tensor(rsc.mean_), radar_scale=torch.tensor(rsc.scale_), contact_mean=torch.tensor(csc.mean_), contact_scale=torch.tensor(csc.scale_), head_weight=w, head_bias=b, method=method, seed=seed, heldout_label=heldout, steps=400, train_bags=train_bags, test_bags=test_bags, source_head=sp['shared_head'], source_head_sha256=sha(sp['shared_head']), protocol=str(ROOT / 'protocol_before_results.json'))
                    path = out / 'models' / f'{key}.pt'
                    torch.save(checkpoint, path)
                    loaded = torch.load(path, map_location='cpu', weights_only=True)
                    reloaded = api.Encoder(loaded['contact_mean'].numpy(), loaded['contact_scale'].numpy())
                    reloaded.load_state_dict(loaded['encoder'])
                    reloaded.eval()
                    with torch.inference_mode():
                        p_re = api.grouping(reloaded(xt[rte]) @ loaded['head_weight'].T + loaded['head_bias']).softmax(-1).numpy()
                    error = float(abs(prob - p_re).max())
                    assert error < 1e-06
                    assert torch.equal(w, wbefore) and torch.equal(b, bbefore)
                    checks.append(dict(**common, probability_reload_max_error=error, head_unchanged=True, no_heldout_radar_in_fit=True, scalers_fit_seen_train_only=True, training_recordings=3, test_recordings=4, code_targets_seen_classes_only=True))
                    print(key, 'seen', rows[-2]['accuracy'], 'unseen', rows[-1]['accuracy'], flush=True)
            pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
            pd.DataFrame(bags).to_csv(out / 'file_predictions.csv', index=False)
    assert len(checks) == 96
    assert hashes == {str(p): sha(p) for p in files}
    savej(out / 'splits.json', split_records)
    savej(out / 'verification.json', dict(models=checks, model_count=96, source_files_unchanged=True, source_head_prior_knows_all_four_contact_classes=True, whole_system_zero_shot=False, excluded_class_radar_not_used_for_fitting_or_selection=True, elapsed_seconds=time.time() - started, script_sha256=sha(__file__)))
    pd.DataFrame(traces).to_csv(out / 'training_trace.csv', index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['method', 'scope', 'unit'], as_index=False).agg(n=('n', 'sum'), correct=('correct', 'sum'), predicted_heldout=('predicted_heldout', 'sum'), mean_accuracy=('accuracy', 'mean'), mean_macro_f1=('macro_f1', 'mean'))
    summary['pooled_accuracy'] = summary.correct / summary.n
    summary['pooled_predicted_heldout_rate'] = summary.predicted_heldout / summary.n
    summary.to_csv(out / 'summary.csv', index=False)
    byclass = frame.groupby(['heldout_label', 'method', 'scope', 'unit'], as_index=False).agg(n=('n', 'sum'), correct=('correct', 'sum'), predicted_heldout=('predicted_heldout', 'sum'))
    byclass['accuracy'] = byclass.correct / byclass.n
    byclass['predicted_heldout_rate'] = byclass.predicted_heldout / byclass.n
    byclass.to_csv(out / 'summary_by_heldout_class.csv', index=False)
    print(summary.to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
