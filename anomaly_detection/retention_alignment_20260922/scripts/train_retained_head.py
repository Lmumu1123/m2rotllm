"""Source-rehearsed BearLLM head adaptation with exact four-type grouping.

Encoder and linear1 remain frozen. A zero-initialized residual of linear2 is
trained using source replay, source ten-type distillation, and local four-type
labels. Candidate selection never reads source test or local held-out labels.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
ROOT = Path(_c2r_resolve_path(__file__)).resolve().parents[1]
OLD = ROOT.parent / 'four_class_chain_20260922'
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]
VARIANTS = ['v0', 'query_no_demean', 'query_no_demean_ref2k']

def sha(p):
    return hashlib.sha256(Path(_c2r_resolve_path(p)).read_bytes()).hexdigest()

def writej(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')

def group_logits(l):
    return torch.stack([torch.logsumexp(l[..., g], dim=-1) for g in GROUPS], -1)

def group_probs(p):
    return np.stack([p[..., g].sum(-1) for g in GROUPS], -1)

def score(y, p, n):
    return dict(accuracy=float(accuracy_score(y, p.argmax(-1))), macro_f1=float(f1_score(y, p.argmax(-1), labels=list(range(n)), average='macro', zero_division=0)), n=len(y))

class ResidualHead(nn.Module):

    def __init__(self, w, b, mean, scale):
        super().__init__()
        for name, x in [('original_w', w), ('original_b', b), ('mean', mean), ('scale', scale)]:
            self.register_buffer(name, torch.as_tensor(x, dtype=torch.float32))
        self.delta = nn.Linear(128, 10)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, h):
        return F.linear(h, self.original_w, self.original_b) + self.delta((h - self.mean) / self.scale)

    def effective(self):
        w = self.original_w + self.delta.weight / self.scale
        b = self.original_b + self.delta.bias - self.delta.weight @ (self.mean / self.scale)
        return (w.detach().numpy().copy(), b.detach().numpy().copy())

def source_prob(h, w, b):
    z = np.asarray(h, dtype=np.float32) @ np.asarray(w, dtype=np.float32).T + np.asarray(b, dtype=np.float32)
    return softmax(z, axis=-1).mean(1)

def file_prob(p, md):
    rows = []
    for bag, ids in md.groupby('bag_id', sort=True).indices.items():
        r = md.iloc[ids[0]]
        q = p[ids].mean(0)
        rows.append(dict(bag_id=bag, state=r.state, label=int(r.label), prediction=int(q.argmax()), n_windows=len(ids), **{f'p{i}': float(q[i]) for i in range(4)}))
    return pd.DataFrame(rows)

def file_score(p, md):
    f = file_prob(p, md)
    return score(f.label.to_numpy(), f[['p0', 'p1', 'p2', 'p3']].to_numpy(), 4)

def run(args):
    if '/envs/m2vllm/' not in __import__('sys').executable:
        raise ValueError('Use m2vllm')
    torch.set_num_threads(2)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = dict(user_max_accuracy_drop_percentage_points=0.5, representation='Frozen original FCN + linear1 ReLU, 128D; only residual linear2 is trained', four_type_interface='group logsumexp of 10 logits; internal ten logits retained for source knowledge', variants=VARIANTS, folds=['115200_to_460800', '460800_to_115200', 'all_known'], source_weights=[0, 1, 10, 100], steps=[200, 400, 800], optimizer='AdamW lr .003, weight_decay 0', losses='local grouped CE + lambda*(source ten CE + 2*T^2 KL(original_T||new_T)); T=2; + .001 delta parameter MSE', source_batch='256 query-reference draws from fixed source training replay only', local_batch='16 windows per training recording, equal recording weight', selection='Among rehearsal lambda>0 candidates passing source-val ten/four accuracy drop<=.005, maximize local TRAIN file macro-F1, then minimize local TRAIN CE, then maximize source-val ten accuracy, then minimum step/lambda. No local heldout metric participates.', fallback='If no trained candidate qualifies, unchanged original head. Source test evaluated only after selection is written.', source_test_acceptance='Post-selection ten and four accuracy drop<=.005; failure is reported/rejected, not used to retune/select another candidate.', external='keep/bigNormal excluded from all training/scalers/selection', source_protocol_limit='Original MBHM references may be from val/test and self references; this is an inherited regression benchmark, not independent physical generalization.', seed=42)
    writej(args.output / 'protocol.json', protocol)
    z = np.load(ROOT / 'source/cache.npz')
    oh = np.load(ROOT / 'source/original_head.npz')
    w0 = oh['weight10']
    b0 = oh['bias10']
    h = z['hidden_refs']
    y10 = z['label10']
    y4 = z['label4']
    split = z['split']
    orig = z['probs10_mean']
    srcnames = z['source']
    query = z['query_id']
    tr = np.flatnonzero(split == 'train')
    va = np.flatnonzero(split == 'val')
    te = np.flatnonzero(split == 'test')
    hs = h[tr].reshape(-1, 128)
    ys = np.repeat(y10[tr], 3)
    mean = hs.mean(0)
    scale = np.maximum(hs.std(0), 0.01)
    xs = torch.tensor(hs, dtype=torch.float32)
    yt = torch.tensor(ys, dtype=torch.long)
    baseline = {}
    for tag, ids in [('train', tr), ('val', va), ('test', te)]:
        baseline[tag] = {'ten': score(y10[ids], orig[ids], 10), 'four': score(y4[ids], group_probs(orig[ids]), 4)}
    writej(args.output / 'source_baseline.json', baseline)
    source_baseline_match = source_prob(h[va], w0, b0)
    if not np.array_equal(source_baseline_match.argmax(1), orig[va].argmax(1)):
        raise ValueError('Source head does not reproduce cached original predictions')
    results = []
    candidates = []
    predictions = []
    sourcegroups = []
    selections = []
    trace = []
    retention = []
    previous = []
    for variant in VARIANTS:
        vd = ROOT / 'preprocessing/variants' / variant / 'retrained_fcn'
        c = np.load(vd / 'contact_features.npz')
        md = pd.read_csv(vd / 'metadata.csv')
        ch = c['hidden_mean']
        folds = [('115200_to_460800', 115200), ('460800_to_115200', 460800), ('all_known', None)]
        for fold, baud in folds:
            local_train = (md.label >= 0).to_numpy() if baud is None else ((md.label >= 0) & (md.baud == baud)).to_numpy()
            local_test = np.zeros(len(md), bool) if baud is None else ((md.label >= 0) & (md.baud != baud)).to_numpy()
            ext = (md.label < 0).to_numpy()
            bags = sorted(md.loc[local_train, 'bag_id'].unique())
            pools = [np.flatnonzero(local_train & md.bag_id.eq(b).to_numpy()) for b in bags]
            by = np.array([int(md.loc[md.bag_id == b, 'label'].iloc[0]) for b in bags])
            base = dict(variant=variant, fold=fold)
            dest = args.output / 'models' / variant / fold
            dest.mkdir(parents=True)
            train_p0 = group_probs(softmax(ch[local_train] @ w0.T + b0, axis=1))
            candidates_fold = []
            saved = {}
            local_only = None
            for lam in [0, 1, 10, 100]:
                torch.manual_seed(42)
                rng = np.random.default_rng(42)
                model = ResidualHead(w0, b0, mean, scale)
                opt = torch.optim.AdamW(model.delta.parameters(), lr=0.003, weight_decay=0)
                cx = torch.tensor(ch, dtype=torch.float32)
                for step in range(1, 801):
                    si = rng.integers(len(xs), size=256)
                    li = np.concatenate([rng.choice(pool, size=16, replace=True) for pool in pools])
                    ly = torch.tensor(np.repeat(by, 16), dtype=torch.long)
                    lc = F.cross_entropy(group_logits(model(cx[li])), ly)
                    new = model(xs[si])
                    old = F.linear(xs[si], model.original_w, model.original_b)
                    ce = F.cross_entropy(new, yt[si])
                    T = 2.0
                    kd = F.kl_div(F.log_softmax(new / T, -1), F.softmax(old / T, -1), reduction='batchmean') * T * T
                    reg = model.delta.weight.square().mean() + model.delta.bias.square().mean()
                    loss = lc + lam * (ce + 2 * kd) + 0.001 * reg
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite adaptation loss')
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.delta.parameters(), 5.0)
                    opt.step()
                    if step in [200, 400, 800]:
                        w, b = model.effective()
                        p10 = source_prob(h[va], w, b)
                        sv10 = score(y10[va], p10, 10)
                        sv4 = score(y4[va], group_probs(p10), 4)
                        lp = group_probs(softmax(ch[local_train] @ w.T + b, axis=1))
                        ls = file_score(lp, md[local_train].reset_index(drop=True))
                        lce = float(-np.log(np.clip(lp[np.arange(len(lp)), md.loc[local_train, 'label'].to_numpy()], 1e-12, 1)).mean())
                        passed = sv10['accuracy'] >= baseline['val']['ten']['accuracy'] - 0.005 - 1e-12 and sv4['accuracy'] >= baseline['val']['four']['accuracy'] - 0.005 - 1e-12
                        rec = dict(**base, source_weight=lam, step=step, val_acc10=sv10['accuracy'], val_acc4=sv4['accuracy'], val_macro_f1_10=sv10['macro_f1'], val_macro_f1_4=sv4['macro_f1'], train_file_macro_f1=ls['macro_f1'], train_ce=lce, retention_gate=passed)
                        candidates.append(rec)
                        candidates_fold.append(rec)
                        saved[lam, step] = (w, b)
                        trace.append(dict(**base, source_weight=lam, step=step, total_loss=float(loss.detach()), local_CE=float(lc.detach()), source_CE=float(ce.detach()), source_KD=float(kd.detach())))
                        if lam == 0 and step == 800:
                            local_only = (w, b)
            valid = [r for r in candidates_fold if r['source_weight'] > 0 and r['retention_gate']]
            if valid:
                selected = sorted(valid, key=lambda r: (-r['train_file_macro_f1'], r['train_ce'], -r['val_acc10'], r['step'], r['source_weight']))[0]
                sw, sb = saved[selected['source_weight'], selected['step']]
                selection = dict(**selected, selection_used_local_test=False, selection_used_source_test=False, selection_status='trained_candidate')
            else:
                sw, sb = (w0.copy(), b0.copy())
                selection = dict(**base, source_weight=None, step=0, selection_status='unchanged_original_fallback', selection_used_local_test=False, selection_used_source_test=False)
            writej(dest / 'selection_before_test.json', selection)
            selections.append(selection)
            np.savez_compressed(dest / 'retained_head.npz', weight10=sw, bias10=sb, original_weight10=w0, original_bias10=b0, group_ids=np.array([0, 1, 1, 1, 3, 3, 3, 2, 2, 2]), source_scaler_mean=mean, source_scaler_scale=scale, local_training_bags=np.array(bags), variant=np.array(variant), fold=np.array(fold))
            methods = {'original_grouped': (w0, b0), 'local_only_residual': local_only, 'retained_head': (sw, sb)}
            if variant == 'v0':
                priorfold = 'all_known' if baud is None else '0000' if baud == 115200 else '1111'
                prior = np.load(OLD / 'results/chain_v1/models/four_class_provisional_roi' / priorfold / 'contact_head.npz')
                pp = softmax((h[te] - prior['mean']) / prior['scale'] @ prior['coef'].T + prior['intercept'], axis=-1).mean(1)
                prior4 = np.zeros_like(pp)
                prior4[:, prior['classes']] = pp
                previous.append(dict(fold=fold, method='previous_local_four_head', **score(y4[te], prior4, 4)))
            for method, (w, b) in methods.items():
                sp = source_prob(h[te], w, b)
                st10 = score(y10[te], sp, 10)
                st4 = score(y4[te], group_probs(sp), 4)
                d10 = 100 * (st10['accuracy'] - baseline['test']['ten']['accuracy'])
                d4 = 100 * (st4['accuracy'] - baseline['test']['four']['accuracy'])
                pass_test = d10 >= -0.5 - 1e-10 and d4 >= -0.5 - 1e-10
                retention.append(dict(**base, method=method, test_acc10=st10['accuracy'], test_acc4=st4['accuracy'], test_macro_f1_10=st10['macro_f1'], test_macro_f1_4=st4['macro_f1'], delta_acc10_pp=d10, delta_acc4_pp=d4, meets_user_global_retention=pass_test, test_n=len(te)))
                for source in sorted(set(srcnames[te])):
                    ids = np.flatnonzero(srcnames[te] == source)
                    sc10 = score(y10[te][ids], sp[ids], 10)
                    sc4 = score(y4[te][ids], group_probs(sp[ids]), 4)
                    oldacc = float(np.mean(orig[te][ids].argmax(1) == y10[te][ids]))
                    sourcegroups.append(dict(**base, method=method, source=source, acc10=sc10['accuracy'], acc4=sc4['accuracy'], delta_acc10_pp=100 * (sc10['accuracy'] - oldacc), n=len(ids)))
                localp = group_probs(softmax(ch @ w.T + b, axis=1))
                for scope, mask in [('local_train', local_train), ('local_heldout', local_test), ('external', ext)]:
                    if not mask.any():
                        continue
                    pr = file_prob(localp[mask], md[mask].reset_index(drop=True))
                    for k, v in dict(**base, method=method, scope=scope).items():
                        pr[k] = v
                    predictions.extend(pr.to_dict('records'))
                    if scope != 'external':
                        results.append(dict(**base, method=method, scope=scope, **score(pr.label.to_numpy(), pr[['p0', 'p1', 'p2', 'p3']].to_numpy(), 4)))
                if method == 'retained_head':
                    np.savez_compressed(dest / 'local_teacher_targets.npz', hidden_mean=ch, probs10=softmax(ch @ w.T + b, axis=1), probs4=localp, bag_id=md.bag_id.to_numpy(dtype=str), labels=md.label.to_numpy())
                    writej(dest / 'acceptance.json', dict(**base, source_test_acc10=st10['accuracy'], source_test_acc4=st4['accuracy'], delta_acc10_pp=d10, delta_acc4_pp=d4, meets_user_global_retention=pass_test, status='passes_global_regression_gate' if pass_test else 'rejected_do_not_replace_original', per_source_declines_must_be_reviewed=True, encoder_and_linear1_unchanged=True))
            print(json.dumps(dict(**base, selection=selection)), flush=True)
            for name, rows in [('candidates', candidates), ('contact_metrics', results), ('contact_predictions', predictions), ('source_retention', retention), ('source_retention_by_dataset', sourcegroups), ('previous_local_head_regression', previous), ('training_trace', trace)]:
                pd.DataFrame(rows).to_csv(args.output / (name + '.csv'), index=False)
            writej(args.output / 'selections.json', selections)
    writej(args.output / 'completion.json', dict(status='complete', original_weights_modified=False, source_test_for_selection=False, local_test_for_selection=False, python=__import__('sys').executable, script_sha256=sha(__file__), source_cache_sha256=sha(ROOT / 'source/cache.npz'), variants=VARIANTS, n_models=len(selections), n_candidate_snapshots=len(candidates)))
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'results/retained_head_v1')
    run(ap.parse_args())
