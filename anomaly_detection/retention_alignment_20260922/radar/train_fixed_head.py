"""Fixed original contact-head radar alignment: source-only file holdout audit."""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import argparse, hashlib, json, random, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
HERE = Path(_c2r_resolve_path(__file__)).resolve().parent
OLD = HERE.parents[1] / 'four_class_chain_20260922'
HEAD = Path(_c2r_resolve_path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn/classifier.pth'))
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]
METHODS = ['embedding_only', 'ce_only', 'ce_feat_0p1', 'ce_feat_1', 'ce_feat_10', 'ce_feat_1_kd', 'phase_ce_feat_1', 'smooth_ce_feat_1']

def writej(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def weights(m):
    sizes = m.groupby('bag_id').size()
    return np.array([1 / sizes[b] for b in m.bag_id]) * len(m) / len(sizes)

def grouping(logits):
    return torch.stack([torch.logsumexp(logits[..., g], -1) for g in GROUPS], -1)

def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

class Encoder(nn.Module):

    def __init__(self, nin, mean, scale):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 128))
        self.register_buffer('contact_mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('contact_scale', torch.tensor(scale, dtype=torch.float32))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.relu(self.contact_mean + self.contact_scale * self.net(x))

def main(args):
    torch.set_num_threads(2)
    if 'envs/m2vllm' not in __import__('sys').executable:
        raise RuntimeError('m2vllm required')
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Output must be new/empty')
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    writej(out / 'protocol_used.json', protocol)
    cm = pd.read_csv(args.contact_dir / 'metadata.csv')
    rm = pd.read_csv(OLD / 'radar/metadata.csv')
    cz = np.load(args.contact_dir / 'contact_features.npz')
    rz = np.load(OLD / 'radar/features.npz')
    h = cz['hidden_mean'].astype(np.float32)
    head = torch.load(HEAD, map_location='cpu', weights_only=True)
    w = head['linear2.weight']
    b = head['linear2.bias']
    assert h.min() >= 0 and h.shape == (len(cm), 128)
    assert np.array_equal(rz['bag_id'], rm.bag_id)
    x = {k: rz[k] for k in ['frame_shape', 'shape_phase']}
    centers = (rz['band_edges_hz'][:-1] + rz['band_edges_hz'][1:]) / 2
    kernel = np.exp(-0.5 * ((centers[:, None] - centers[None, :]) / 5.0) ** 2)
    kernel /= kernel.sum(1, keepdims=True)
    sm = x['frame_shape'] @ kernel.T
    sm -= sm.mean(1, keepdims=True)
    x['smooth_shape'] = sm.astype(np.float32)
    with torch.inference_mode():
        original_logits = torch.tensor(h) @ w.T + b
        cp = grouping(original_logits).softmax(-1).numpy()
        test_logits = torch.randn(20, 10)
        exact = grouping(test_logits).softmax(-1)
        target = torch.stack([test_logits.softmax(-1)[:, g].sum(-1) for g in GROUPS], -1)
        group_error = float((exact - target).abs().max())
        assert group_error < 1e-06
    rows = []
    scores = []
    traces = []
    split_records = []
    teacher_rows = []
    checks = []
    start = time.time()
    for task, classes in protocol['tasks'].items():
        for source, target in [(115200, 460800), (460800, 115200)] if args.skip_all_known else [(115200, 460800), (460800, 115200), (0, 0)]:
            fold = f'{source}_to_{target}' if source else 'all_known'
            ctr = cm.label.isin(classes) & (cm.baud == source if source else True)
            rtr = rm.label.isin(classes) & (rm.baud_candidate == source if source else True)
            train_bags = sorted(cm.loc[ctr, 'bag_id'].unique().tolist())
            test_bags = sorted(cm.loc[cm.label.isin(classes) & (cm.baud == target), 'bag_id'].unique().tolist()) if source else []
            assert set(train_bags) == set(rm.loc[rtr, 'bag_id']) and (not set(train_bags) & set(test_bags))
            csc = StandardScaler().fit(h[ctr], sample_weight=weights(cm[ctr]))
            hz = csc.transform(h).astype(np.float32)
            wcenter = w.numpy() - w.numpy().mean(0, keepdims=True)
            _, singular, vt = np.linalg.svd(wcenter * csc.scale_[None, :], full_matrices=False)
            rank = int((singular > singular.max() * 1e-06).sum())
            basis = vt[:rank]
            ct = np.array([hz[(cm.bag_id == bag).to_numpy()].mean(0) for bag in train_bags], dtype=np.float32)
            ct_raw = np.array([h[(cm.bag_id == bag).to_numpy()].mean(0) for bag in train_bags], dtype=np.float32)
            labels = np.array([int(cm.loc[cm.bag_id == bag, 'label'].iloc[0]) for bag in train_bags])
            qt = []
            for bag in train_bags:
                ids = (cm.bag_id == bag).to_numpy()
                p10 = (original_logits[ids] / 2).softmax(-1)
                qt.append(torch.stack([p10[:, g].sum(-1) for g in GROUPS], -1).mean(0))
            qt = torch.stack(qt)
            ct_t = torch.tensor(ct)
            yi = torch.tensor(labels)
            positive = torch.tensor(labels[:, None] == labels[None, :])
            pools = [np.flatnonzero((rm.bag_id == bag).to_numpy()) for bag in train_bags]
            split_records.append(dict(task=task, fold=fold, classes=classes, train_bags=train_bags, test_bags=test_bags))
            for bag, idx in cm.groupby('bag_id').indices.items():
                role = 'train' if bag in train_bags else 'test' if bag in test_bags else 'external' if cm.iloc[idx[0]].label < 0 else 'excluded_class'
                if role == 'excluded_class' or (role == 'external' and source):
                    continue
                p = cp[idx].mean(0)
                teacher_rows.append(dict(task=task, fold=fold, bag_id=bag, role=role, label=int(cm.iloc[idx[0]].label), state=cm.iloc[idx[0]].state, prediction=int(p.argmax()), **{f'p{k}': float(p[k]) for k in range(4)}))
            for method in args.methods:
                feature = 'shape_phase' if method.startswith('phase') else 'smooth_shape' if method.startswith('smooth') else 'frame_shape'
                rsc = StandardScaler().fit(x[feature][rtr], sample_weight=weights(rm[rtr]))
                xt = torch.tensor(rsc.transform(x[feature]).astype(np.float32))
                for seed in [42] if not source else args.seeds:
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    model = Encoder(xt.shape[1], csc.mean_, csc.scale_)
                    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                    rng = np.random.default_rng(seed)
                    lam = 0.1 if method == 'ce_feat_0p1' else 10 if method == 'ce_feat_10' else 1
                    for step in range(args.steps):
                        ids = np.concatenate([rng.choice(p, 16, replace=True) for p in pools])
                        hh = model(xt[ids]).reshape(len(pools), 16, 128)
                        z = (hh - model.contact_mean) / model.contact_scale
                        mean_z = z.mean(1)
                        log10 = hh @ w.T + b
                        log4 = grouping(log10)
                        ce = F.cross_entropy(log4.reshape(-1, 4), yi.repeat_interleave(16))
                        mse = F.mse_loss(mean_z, ct_t)
                        sims = F.normalize(mean_z, dim=1) @ F.normalize(ct_t, dim=1).T / 0.2
                        contrast = (torch.logsumexp(sims, 1) - torch.logsumexp(sims.masked_fill(~positive, -torch.inf), 1)).mean()
                        p10 = (log10 / 2).softmax(-1)
                        p4 = torch.stack([p10[..., g].sum(-1) for g in GROUPS], -1).mean(1)
                        kd = 4 * (qt * (qt.clamp_min(1e-10).log() - p4.clamp_min(1e-10).log())).sum(1).mean()
                        loss = mse if method == 'embedding_only' else ce if method == 'ce_only' else ce + lam * mse + 0.1 * contrast + (0.5 * kd if method == 'ce_feat_1_kd' else 0)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 5)
                        opt.step()
                        if step % 100 == 0 or step == args.steps - 1:
                            traces.append(dict(task=task, fold=fold, method=method, seed=seed, step=step + 1, loss=float(loss.detach()), ce=float(ce.detach()), mse=float(mse.detach()), contrast=float(contrast.detach()), kd=float(kd.detach())))
                    model.eval()
                    with torch.inference_mode():
                        hh = model(xt)
                        zz = (hh - model.contact_mean) / model.contact_scale
                        lp = grouping(hh @ w.T + b)
                        prob = lp.softmax(-1).numpy()
                        hh = hh.numpy()
                        zz = zz.numpy()
                        lp = lp.numpy()
                    predictions = []
                    for bag, idx in rm.groupby('bag_id').indices.items():
                        info = rm.iloc[idx[0]]
                        role = 'train' if bag in train_bags else 'test' if bag in test_bags else 'external' if info.label < 0 else 'excluded_class'
                        if role == 'excluded_class' or (role == 'external' and source):
                            continue
                        ci = np.flatnonzero((cm.bag_id == bag).to_numpy())
                        assert len(ci) > 0
                        pred = prob[idx].mean(0)
                        eh = hh[idx].mean(0)
                        ez = zz[idx].mean(0)
                        th = h[ci].mean(0)
                        tz = hz[ci].mean(0)
                        truth = int(info.label) if info.label >= 0 else 0 if info.state == 'bigNormal' else -1
                        delta = ez - tz
                        relevant = delta @ basis.T @ basis
                        null = delta - relevant
                        margin = float(lp[idx, truth].mean() - np.max(np.delete(lp[idx].mean(0), truth))) if truth >= 0 else None
                        tmargin = float(grouping(torch.tensor(th) @ w.T + b)[truth] - torch.max(torch.cat([grouping(torch.tensor(th) @ w.T + b)[:truth], grouping(torch.tensor(th) @ w.T + b)[truth + 1:]]))) if truth >= 0 else None
                        row = dict(task=task, fold=fold, method=method, seed=seed, feature=feature, role=role, bag_id=bag, state=info.state, label=truth, prediction=int(pred.argmax()), n_windows=len(idx), geometry_valid=not bool(info.geometry_roi_mismatch), mse_standardized=float(np.mean((ez - tz) ** 2)), mse_raw=float(np.mean((eh - th) ** 2)), cos_standardized=cosine(ez, tz), cos_raw=cosine(eh, th), classifier_rowspace_rank=rank, classifier_rowspace_error_energy=float(relevant @ relevant), classifier_nullspace_error_energy=float(null @ null), centered_logit_shift_l2=float(np.linalg.norm((eh - th) @ wcenter.T)), true_class_margin=margin, contact_true_class_margin=tmargin, **{f'p{k}': float(pred[k]) for k in range(4)})
                        rows.append(row)
                        predictions.append(row)
                    pdf = pd.DataFrame(predictions)
                    for role in ['train', 'test']:
                        part = pdf[pdf.role == role]
                        if len(part):
                            scores.append(dict(task=task, fold=fold, method=method, seed=seed, role=role, n_files=len(part), accuracy=accuracy_score(part.label, part.prediction), macro_f1=f1_score(part.label, part.prediction, labels=classes, average='macro', zero_division=0), mse_standardized=part.mse_standardized.mean(), cos_standardized=part.cos_standardized.mean(), cos_raw=part.cos_raw.mean(), mean_margin=part.true_class_margin.mean()))
                    if seed == 42:
                        dest = out / 'models' / task / fold
                        dest.mkdir(parents=True, exist_ok=True)
                        ckpt = dict(encoder=model.state_dict(), feature=feature, input_dim=xt.shape[1], radar_mean=rsc.mean_, radar_scale=rsc.scale_, contact_mean=csc.mean_, contact_scale=csc.scale_, head_weight=w, head_bias=b, groups=GROUPS, train_bags=train_bags, method=method, seed=seed, steps=args.steps, original_head_sha256=sha(HEAD), source_contact_path=str(args.contact_dir / 'contact_features.npz'), geometry_status='outer and keep invalid ROI', architecture='nonnegative_fixed_original_head_v1')
                        path = dest / f'{method}.pt'
                        torch.save(ckpt, path)
                        ck = torch.load(path, map_location='cpu', weights_only=False)
                        again = Encoder(ck['input_dim'], ck['contact_mean'], ck['contact_scale'])
                        again.load_state_dict(ck['encoder'])
                        again.eval()
                        with torch.inference_mode():
                            p2 = grouping(again(xt) @ ck['head_weight'].T + ck['head_bias']).softmax(-1).numpy()
                        err = float(np.max(abs(p2 - prob)))
                        assert err < 1e-06
                        checks.append(dict(path=str(path), probability_reload_max_error=err, nonnegative=bool(hh.min() >= 0), head_unchanged=bool(torch.equal(w, ck['head_weight']) and torch.equal(b, ck['head_bias']))))
                    print(task, fold, method, seed, 'test=', [(r['accuracy'], round(r['mse_standardized'], 4)) for r in scores[-2:] if r['role'] == 'test'], 'elapsed', round(time.time() - start, 1), flush=True)
                    pd.DataFrame(scores).to_csv(out / 'metrics.csv', index=False)
        pd.DataFrame(rows).to_csv(out / 'file_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'file_predictions.csv', index=False)
    pd.DataFrame(teacher_rows).to_csv(out / 'contact_original_predictions.csv', index=False)
    pd.DataFrame(traces).to_csv(out / 'training_trace.csv', index=False)
    writej(out / 'splits.json', split_records)
    writej(out / 'verification.json', dict(group_probability_max_error=group_error, models=checks, head_sha256=sha(HEAD), script_sha256=sha(Path(_c2r_resolve_path(__file__))), wall_seconds=time.time() - start))
    writej(out / 'runtime.json', dict(python=__import__('sys').version, torch=torch.__version__, numpy=np.__version__, cpu_threads=2, model_instances=len(set(((r['task'], r['fold'], r['method'], r['seed']) for r in rows))), original_head_untouched=True))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=HERE / 'stage_a')
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 42, 73])
    p.add_argument('--skip-all-known', action='store_true')
    p.add_argument('--contact-dir', type=Path, default=OLD / 'contact/retrained_fcn_fixed_external')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    main(p.parse_args())
