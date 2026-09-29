"""Predeclared radar -> frozen RotLLM hidden-space experiments, CPU threads 2."""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
B = Path(_c2r_resolve_path(__file__)).resolve().parent
OUT = B / 'radar'
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]

def sha(p):
    return hashlib.sha256(Path(_c2r_resolve_path(p)).read_bytes()).hexdigest()

def savej(p, d):
    Path(_c2r_resolve_path(p)).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def bagweights(md):
    sizes = md.groupby('bag_id').size()
    return np.array([1 / sizes[b] for b in md.bag_id]) * len(md) / len(sizes)

def cosine(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

def group4(logits):
    return torch.stack([torch.logsumexp(logits[..., idx], -1) for idx in GROUPS], -1)

def probs4(logits, headtype, temperature=1.0):
    scaled = logits / temperature
    return (group4(scaled) if headtype == 'native15' else scaled).softmax(-1)

class Encoder(nn.Module):

    def __init__(self, mean, scale):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 128))
        self.register_buffer('contact_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('contact_scale', torch.as_tensor(scale, dtype=torch.float32))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return F.relu(self.contact_mean + self.contact_scale * self.net(x))

def scores(df, pred, labels):
    return dict(n=len(df), accuracy=float(accuracy_score(df.label, pred)), macro_f1=float(f1_score(df.label, pred, labels=labels, average='macro', zero_division=0)))

def main():
    torch.set_num_threads(2)
    assert 'envs/m2vllm/' in sys.executable
    OUT.mkdir(exist_ok=True)
    if (OUT / 'verification.json').exists():
        raise RuntimeError('Completed output already exists, do not silently overwrite')
    root_protocol = json.loads((B.parent / 'protocol.json').read_text())
    cfg = root_protocol['radar_training']
    assert cfg['steps'] == 400 and cfg['seeds'] == [17, 42, 73]
    cm = pd.read_csv(B / 'metadata.csv')
    cz = np.load(B / 'teacher_features.npz', allow_pickle=False)
    h = cz['hidden_mean'].astype(np.float32)
    rm = pd.read_csv(root_protocol['data']['radar_metadata'])
    rz = np.load(root_protocol['data']['radar_features'], allow_pickle=False)
    x = rz['frame_shape']
    assert np.array_equal(rz['bag_id'], rm.bag_id.to_numpy(str))
    assert np.array_equal(cz['bag_id'], cm.bag_id.to_numpy(str))
    assert np.array_equal(cz['labels'], cm.label)
    assert h.min() >= 0 and np.isfinite(h).all() and np.isfinite(x).all()
    files_to_hash = [B / 'teacher_features.npz', B / 'metadata.csv', B / 'native_head15.npz', B / 'vendor/RotLLM/weights/encoder_weights.pth', B / 'vendor/RotLLM/weights/proj_weights.pth', Path(_c2r_resolve_path(root_protocol['data']['radar_features'])), Path(_c2r_resolve_path(root_protocol['data']['radar_metadata']))]
    files_to_hash += [B / 'heads' / f / 'head.npz' for f in ['115200_to_460800', '460800_to_115200']]
    hashes_before = {str(p): sha(p) for p in files_to_hash}
    savej(OUT / 'protocol.json', {**root_protocol, 'rotllm_radar_specific': {'primary_contact_features': str(B / 'teacher_features.npz'), 'primary_variant': 'raw_rms001', 'head_modes': ['native15', 'new_contact4'], 'native15_output': 'conditional probabilities restricted to four bearing groups; report full15 probabilities and gear mass separately', 'new_contact4_output': 'new contact-only four-class head; additionally save original15 diagnostic probabilities from same hidden', 'teacher_KD_target': 'mean of contact-window T2 conditional four-class probabilities computed on mean-XYZ hidden', 'no_target_checkpoint_selection': True, 'geometry_valid_macro_f1_labels': [0, 1, 3], 'saved_hidden': 'one row per existing 2-second radar window, all records including external'}})
    rm.to_csv(OUT / 'radar_metadata.csv', index=False)
    native = np.load(B / 'native_head15.npz', allow_pickle=False)
    nw = torch.tensor(native['weight15'], dtype=torch.float32)
    nb = torch.tensor(native['bias15'], dtype=torch.float32)
    ht = torch.tensor(h)
    metrics = []
    file_rows = []
    traces = []
    checks = []
    splits = []
    teacherrows = []
    start = time.monotonic()
    for source, target in root_protocol['data']['folds']:
        fold = f'{source}_to_{target}'
        ctr = (cm.label >= 0) & (cm.baud == source)
        rtr = (rm.label >= 0) & (rm.baud_candidate == source)
        trainbags = sorted(cm.loc[ctr, 'bag_id'].unique())
        testbags = sorted(cm.loc[(cm.label >= 0) & (cm.baud == target), 'bag_id'].unique())
        assert set(trainbags) == set(rm.loc[rtr, 'bag_id']) and (not set(trainbags) & set(testbags))
        csc = StandardScaler().fit(h[ctr], sample_weight=bagweights(cm[ctr]))
        rsc = StandardScaler().fit(x[rtr], sample_weight=bagweights(rm[rtr]))
        hz = csc.transform(h).astype(np.float32)
        xt = torch.tensor(rsc.transform(x).astype(np.float32))
        labels = np.array([int(cm.loc[cm.bag_id == b, 'label'].iloc[0]) for b in trainbags])
        yi = torch.tensor(labels)
        pools = [np.flatnonzero((rm.bag_id == b).to_numpy()) for b in trainbags]
        ct = torch.tensor(np.stack([hz[cm.bag_id == b].mean(0) for b in trainbags]))
        positive = torch.tensor(labels[:, None] == labels[None, :])
        splits.append(dict(fold=fold, train_bags=trainbags, test_bags=testbags, contact_train_windows=int(ctr.sum()), radar_train_windows=int(rtr.sum()), radar_test_windows=int(((rm.label >= 0) & (rm.baud_candidate == target)).sum())))
        for headmode in ['native15', 'new_contact4']:
            if headmode == 'native15':
                w, b = (nw.clone(), nb.clone())
                headpath = B / 'native_head15.npz'
            else:
                headpath = B / 'heads' / fold / 'head.npz'
                d = np.load(headpath, allow_pickle=False)
                assert set(d['train_bag_ids']) == set(trainbags)
                w = torch.tensor(d['weight4'], dtype=torch.float32)
                b = torch.tensor(d['bias4'], dtype=torch.float32)
            w0 = w.clone()
            b0 = b.clone()
            with torch.no_grad():
                teacherlogits = ht @ w.T + b
                cp = probs4(teacherlogits, headmode).numpy()
                qt = torch.stack([probs4(teacherlogits[(cm.bag_id == bag).to_numpy()], headmode, 2).mean(0) for bag in trainbags])
            for bag, ids in cm.groupby('bag_id').indices.items():
                info = cm.iloc[ids[0]]
                role = 'train' if bag in trainbags else 'test' if bag in testbags else 'external'
                pp = cp[ids].mean(0)
                teacherrows.append(dict(fold=fold, head=headmode, bag_id=bag, role=role, label=int(info.label), state=info.state, prediction=int(pp.argmax()), **{f'p{i}': float(pp[i]) for i in range(4)}))
            for method in cfg['methods']:
                for seed in cfg['seeds']:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    rng = np.random.default_rng(seed)
                    model = Encoder(csc.mean_, csc.scale_)
                    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
                    for step in range(cfg['steps']):
                        ids = np.concatenate([rng.choice(p, cfg['windows_per_training_bag_per_step'], replace=True) for p in pools])
                        hh = model(xt[ids]).reshape(len(pools), 16, 128)
                        z = (hh - model.contact_mean) / model.contact_scale
                        mean_z = z.mean(1)
                        log = hh @ w.T + b
                        log4 = group4(log) if headmode == 'native15' else log
                        ce = F.cross_entropy(log4.reshape(-1, 4), yi.repeat_interleave(16))
                        mse = F.mse_loss(mean_z, ct)
                        sims = F.normalize(mean_z, dim=1) @ F.normalize(ct, dim=1).T / 0.2
                        contrast = (torch.logsumexp(sims, 1) - torch.logsumexp(sims.masked_fill(~positive, -torch.inf), 1)).mean()
                        p4t = probs4(log, headmode, 2).mean(1)
                        kd = 4 * (qt * (qt.clamp_min(1e-10).log() - p4t.clamp_min(1e-10).log())).sum(1).mean()
                        loss = ce if method == 'ce_only' else mse if method == 'embedding_only' else ce + mse + 0.1 * contrast + 0.5 * kd
                        assert torch.isfinite(loss)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
                        opt.step()
                        if step % 100 == 0 or step == cfg['steps'] - 1:
                            traces.append(dict(fold=fold, head=headmode, method=method, seed=seed, step=step + 1, loss=float(loss.detach()), ce=float(ce.detach()), mse=float(mse.detach()), contrast=float(contrast.detach()), kd=float(kd.detach())))
                    model.eval()
                    with torch.inference_mode():
                        out_h = model(xt)
                        log = out_h @ w.T + b
                        prob = probs4(log, headmode).numpy()
                        p15 = (out_h @ nw.T + nb).softmax(-1).numpy()
                        out_z = ((out_h - model.contact_mean) / model.contact_scale).numpy()
                        out_h = out_h.numpy()
                    tag = f'{fold}__{headmode}__{method}__seed{seed}'
                    dest = OUT / 'models' / tag
                    dest.mkdir(parents=True, exist_ok=True)
                    ckpt = dict(encoder=model.state_dict(), radar_mean=torch.tensor(rsc.mean_), radar_scale=torch.tensor(rsc.scale_), contact_mean=torch.tensor(csc.mean_), contact_scale=torch.tensor(csc.scale_), head_weight=w, head_bias=b, native15_weight=nw, native15_bias=nb, head_mode=headmode, head_sha256=sha(headpath), input_dim=128, groups=GROUPS, train_bags=trainbags, test_bags=testbags, method=method, seed=seed, steps=cfg['steps'], architecture='relu(mean + scale * MLP128-LN-GELU-dropout0.1-64-GELU-128)', contact_feature_sha256=sha(B / 'teacher_features.npz'))
                    torch.save(ckpt, dest / 'model.pt')
                    ck = torch.load(dest / 'model.pt', map_location='cpu', weights_only=True)
                    again = Encoder(ck['contact_mean'], ck['contact_scale'])
                    again.load_state_dict(ck['encoder'], strict=True)
                    again.eval()
                    xt_again = torch.tensor(((x - ck['radar_mean'].numpy()) / ck['radar_scale'].numpy()).astype(np.float32))
                    xs = x.astype(np.float32).copy()
                    xs -= ck['radar_mean'].numpy()
                    xs /= ck['radar_scale'].numpy()
                    xt_again = torch.tensor(xs)
                    with torch.inference_mode():
                        h2 = again(xt_again)
                        p2 = probs4(h2 @ ck['head_weight'].T + ck['head_bias'], headmode).numpy()
                    reload_err = float(np.abs(prob - p2).max())
                    assert reload_err < 1e-05
                    assert torch.equal(w, w0) and torch.equal(b, b0)
                    np.savez_compressed(dest / 'predictions.npz', hidden=out_h, standardized_hidden=out_z, probs4=prob, probs15=p15, native15_diagnostic_only=np.array(headmode != 'native15'), bag_id=rm.bag_id.to_numpy(str), labels=rm.label.to_numpy(), geometry_valid=~rm.geometry_roi_mismatch.to_numpy(bool), radar_row=np.arange(len(rm)))
                    checks.append(dict(tag=tag, reload_probability_max_abs=reload_err, head_unchanged=True, weights_only_reload=True, nonnegative_hidden=bool(out_h.min() >= 0)))
                    model_files = []
                    for bag, ids in rm.groupby('bag_id').indices.items():
                        info = rm.iloc[ids[0]]
                        role = 'train' if bag in trainbags else 'test' if bag in testbags else 'external'
                        ci = np.flatnonzero((cm.bag_id == bag).to_numpy())
                        assert len(ci)
                        pp = prob[ids].mean(0)
                        q15 = p15[ids].mean(0)
                        ez = out_z[ids].mean(0)
                        tz = hz[ci].mean(0)
                        row = dict(fold=fold, head=headmode, method=method, seed=seed, role=role, bag_id=bag, label=int(info.label), state=info.state, n_windows=len(ids), prediction=int(pp.argmax()), geometry_valid=not bool(info.geometry_roi_mismatch), mse_standardized=float(np.square(ez - tz).mean()), cosine_standardized=cosine(ez, tz), cosine_raw=cosine(out_h[ids].mean(0), h[ci].mean(0)), native15_pred=int(q15.argmax()), native15_gear_mass=float(q15[10:].sum()), native15_top_other_window_rate=float((p15[ids].argmax(1) >= 10).mean()), native15_diagnostic_only=headmode != 'native15', **{f'p{i}': float(pp[i]) for i in range(4)})
                        file_rows.append(row)
                        model_files.append(row)
                    pdf = pd.DataFrame(model_files)
                    for role in ['train', 'test']:
                        for subset in ['all_four_provisional_roi', 'geometry_valid_subset']:
                            ff = pdf[pdf.role == role]
                            mask = rm.bag_id.isin(ff.bag_id).to_numpy()
                            labs = [0, 1, 2, 3]
                            if subset == 'geometry_valid_subset':
                                ff = ff[ff.geometry_valid]
                                mask &= ~rm.geometry_roi_mismatch.to_numpy(bool)
                                labs = [0, 1, 3]
                            rbase = dict(fold=fold, head=headmode, method=method, seed=seed, role=role, subset=subset, macro_f1_labels=','.join(map(str, labs)))
                            metrics.append(dict(**rbase, unit='file', **scores(ff, ff.prediction, labs), mse_standardized=float(ff.mse_standardized.mean()), cosine_standardized=float(ff.cosine_standardized.mean()), native15_gear_mass=float(ff.native15_gear_mass.mean()), native15_top_other_rate=float((ff.native15_pred >= 10).mean())))
                            metrics.append(dict(**rbase, unit='window', **scores(rm[mask], prob[mask].argmax(1), labs), native15_gear_mass=float(p15[mask, 10:].sum(1).mean()), native15_top_other_rate=float((p15[mask].argmax(1) >= 10).mean())))
                    testm = [r for r in metrics[-8:] if r['role'] == 'test' and r['subset'] == 'all_four_provisional_roi']
                    print(tag, [(r['unit'], round(r['accuracy'], 4)) for r in testm], round(time.monotonic() - start, 1), flush=True)
                    pd.DataFrame(metrics).to_csv(OUT / 'metrics.csv', index=False)
    pd.DataFrame(file_rows).to_csv(OUT / 'file_predictions.csv', index=False)
    pd.DataFrame(teacherrows).to_csv(OUT / 'contact_teacher_file_predictions.csv', index=False)
    pd.DataFrame(traces).to_csv(OUT / 'training_trace.csv', index=False)
    savej(OUT / 'splits.json', splits)
    assert hashes_before == {str(p): sha(p) for p in files_to_hash}
    savej(OUT / 'verification.json', dict(models=checks, models_completed=len(checks), input_hashes=hashes_before, input_hashes_unchanged=True, head_and_teacher_frozen=True, all_seeds_reported=True, fixed_step_no_heldout_selection=True, script_sha256=sha(__file__), root_protocol_sha256=sha(B.parent / 'protocol.json'), seconds=time.monotonic() - start, python=sys.executable, torch=torch.__version__))
    d = pd.DataFrame(metrics)
    test = d[d.role == 'test']
    summary = test.groupby(['head', 'method', 'subset', 'unit'], as_index=False).agg(accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'), macro_f1_mean=('macro_f1', 'mean'), n_models=('seed', 'size'), mse_standardized=('mse_standardized', 'mean'), cosine_standardized=('cosine_standardized', 'mean'))
    summary.to_csv(OUT / 'summary_mean_folds_seeds.csv', index=False)
    print(summary.to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
