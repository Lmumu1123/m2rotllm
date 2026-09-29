"""Contact-only Bear -> UniFault mapping; all 36 radar encoders remain frozen.

Replay existing corrected-ROI development recordings. No hyperparameter search,
no new radar training and no overwrites of old files. Run with m2vllm.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / 'encoder_validation_20260923_v2'
BEAR = ROOT.parent / 'retention_alignment_20260922/preprocessing/variants/v0/retrained_fcn'
CTRL = V2 / 'controls/results'
UNI = V2 / 'teachers/unifault'
RADAR = V2 / 'geometry_corrected/radar'
METHODS = ['ce_only', 'matched_bag_mse', 'full', 'label_code_mse', 'label_code_ce_mse', 'wrong_class_mse']
SEEDS = [17, 42, 73]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def balanced_weights(meta):
    counts = meta.groupby('bag_id').size()
    return np.array([1 / counts[b] for b in meta.bag_id]) * len(meta) / len(counts)


def cosine(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


class FrozenEncoder(nn.Module):
    """Exactly the existing Bear student inference architecture, including ReLU."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 128))
        self.register_buffer('contact_mean', torch.zeros(128))
        self.register_buffer('contact_scale', torch.ones(128))

    def forward(self, x):
        return torch.relu(self.contact_mean + self.contact_scale * self.net(x))


def load_local_checkpoint(path):
    # Existing local training files contain numpy arrays, not only tensors.
    # Restricted loading; allow only those known serialization primitives.
    with torch.serialization.safe_globals([
        np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype('float64'))
    ]):
        return torch.load(path, map_location='cpu', weights_only=True)


def main():
    if 'envs/m2vllm/' not in sys.executable:
        raise RuntimeError('Run with the m2vllm interpreter')
    torch.set_num_threads(2)
    out = ROOT / 'results'
    if out.exists():
        raise RuntimeError('Refusing to overwrite existing results; preserve failed/completed evidence')
    out.mkdir()
    for name in ['mappings', 'predictions', 'fit_rows']:
        (out / name).mkdir()
    start = time.time()
    protocol = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        task='Frozen Bear radar encoders -> contact-only affine conversion -> frozen local UniFault four-class head',
        teacher_variant='predeclared paper_100ms; official UniFault Tiny pretrained backbone frozen',
        mapping='Same TRAIN contact windows, recording-balanced source and target StandardScaler, Ridge(alpha=1,solver=svd)',
        mapping_labels_used=False, mapping_radar_used=False, mapping_heldout_contact_used=False,
        mapping_output='signed affine output; NO ReLU or clipping, because UniFault features are signed',
        folds=['115200_to_460800', '460800_to_115200'], radar_methods=METHODS, seeds=SEEDS,
        radar_retraining=False, head_training=False, trained_artifacts='Exactly two contact-only affine mappings',
        radar_input='existing geometry-corrected frame_shape 128; replay original saved radar scaler and weights',
        head='existing per-fold UniFault local-contact-trained four-class head; its StandardScaler is already folded into weight4/bias4',
        native_wording='native_Uni_contact means direct Uni FEATURES with the local FOUR-CLASS head, not original 16-way head',
        primary_metrics='All fixed methods, per-window and mean-probability recording accuracy/macro-F1; no selection',
        secondary_metrics='recording-mean standardized feature MSE/cosine against true heldout Uni contact features',
        inference_verification='Reload and run all 36 frozen radar checkpoints; compare against archived hidden arrays',
        test_status='Previously inspected development recordings, not new blind data',
        physical_units='8 main recordings, 58 contact windows, 412 radar windows; seeds do not increase physical sample size',
        restrictions=['No same-class multiple-training-recording evidence', 'One physical bearing per class',
                      'No proof of class-beyond knowledge or unseen-fault recognition',
                      'New four-class Uni head has local label supervision; original public-task retention not established',
                      'Old saved controls mention invalid ROI historically; actual inputs here are v2 geometry_corrected'],
        no_hyperparameter_search=True, script_sha256=sha(__file__),
    )
    write_json(out / 'protocol_before_results.json', protocol)
    cm = pd.read_csv(BEAR / 'metadata.csv')
    a = np.load(BEAR / 'contact_features.npz')['hidden_mean'].astype(np.float64)
    rm = pd.read_csv(RADAR / 'metadata.csv')
    rz = np.load(RADAR / 'features.npz')
    radar_x = rz['frame_shape'].astype(np.float32)
    assert np.array_equal(rz['bag_id'], rm.bag_id.to_numpy(str))
    assert not rm.loc[rm.label >= 0, 'geometry_roi_mismatch'].astype(bool).any()
    assert (cm.label >= 0).sum() == 58 and (rm.label >= 0).sum() == 412
    assert cm.loc[cm.label >= 0, 'bag_id'].nunique() == rm.loc[rm.label >= 0, 'bag_id'].nunique() == 8
    splits = [s for s in json.loads((CTRL / 'splits.json').read_text()) if s['test_bags']]
    source_files = [BEAR / 'metadata.csv', BEAR / 'contact_features.npz', RADAR / 'metadata.csv',
                    RADAR / 'features.npz', CTRL / 'splits.json', UNI / 'verification.json', UNI / 'provenance.json']
    for sp in splits:
        td = UNI / 'paper_100ms' / sp['fold']
        source_files += [td / 'metadata.csv', td / 'teacher_features.npz', td / 'head.npz']
        for method in METHODS:
            for seed in SEEDS:
                tag = f"{sp['fold']}__{method}__seed{seed}"
                source_files += [CTRL / 'models' / f'{tag}.pt', CTRL / 'windows' / f'{tag}.npz']
    hashes = {str(p): sha(p) for p in source_files}
    write_json(out / 'input_hashes_before.json', hashes)
    all_windows, all_files, all_metrics, mapping_checks, model_checks = [], [], [], [], []
    key = ['bag_id', 'window_row']
    assert not cm.duplicated(key).any()
    for sp in splits:
        fold, train_bags, test_bags = sp['fold'], sp['train_bags'], sp['test_bags']
        assert not set(train_bags) & set(test_bags)
        td = UNI / 'paper_100ms' / fold
        um = pd.read_csv(td / 'metadata.csv')
        assert not um.duplicated(key).any()
        indexer = pd.MultiIndex.from_frame(um[key]).get_indexer(pd.MultiIndex.from_frame(cm[key]))
        assert (indexer >= 0).all() and len(set(indexer)) == len(cm)
        b = np.load(td / 'teacher_features.npz')['hidden_mean'][indexer].astype(np.float64)
        assert np.array_equal(cm.label, um.iloc[indexer].label)
        head = np.load(td / 'head.npz')
        assert set(head['train_bag_ids']) == set(train_bags) and set(head['test_bag_ids']) == set(test_bags)
        assert np.array_equal(head['classes'], np.arange(4))
        head_w, head_b = head['weight4'].copy(), head['bias4'].copy()
        train = cm.bag_id.isin(train_bags).to_numpy()
        test = cm.bag_id.isin(test_bags).to_numpy()
        sw = balanced_weights(cm[train])
        sa = StandardScaler().fit(a[train], sample_weight=sw)
        sb = StandardScaler().fit(b[train], sample_weight=sw)
        reg = Ridge(alpha=1., fit_intercept=True, solver='svd').fit(
            sa.transform(a[train]), sb.transform(b[train]), sample_weight=sw)
        wmap = sb.scale_[:, None] * reg.coef_ / sa.scale_[None, :]
        bmap = sb.mean_ + sb.scale_ * reg.intercept_ - wmap @ sa.mean_
        mapped = a @ wmap.T + bmap
        original_form = sb.inverse_transform(reg.predict(sa.transform(a)))
        affine_error = float(np.max(np.abs(mapped - original_form)))
        assert affine_error < 1e-7
        mp = out / 'mappings' / f'{fold}.npz'
        np.savez_compressed(mp, weight=wmap, bias=bmap, post_relu=np.array(False),
                            source_mean=sa.mean_, source_scale=sa.scale_, target_mean=sb.mean_, target_scale=sb.scale_,
                            ridge_alpha=np.array(1.), train_bags=np.array(train_bags), test_bags=np.array(test_bags))
        reloaded = np.load(mp)
        assert np.max(np.abs(a @ reloaded['weight'].T + reloaded['bias'] - mapped)) < 1e-10
        fit_rows = cm.loc[train, key + ['file', 'label']].copy()
        fit_rows['label_used_in_ridge'] = False
        fit_rows['recording_balanced_weight'] = sw
        fit_rows.to_csv(out / 'fit_rows' / f'{fold}.csv', index=False)
        check = dict(fold=fold, train_bags=train_bags, test_bags=test_bags, n_train_windows=int(train.sum()),
                     n_test_windows=int(test.sum()), target_negative_fraction=float((b[train] < 0).mean()),
                     mapped_test_negative_fraction=float((mapped[test] < 0).mean()), signed_output=True,
                     train_standardized_mse=float(np.mean(((mapped[train] - b[train]) / sb.scale_) ** 2)),
                     test_standardized_mse=float(np.mean(((mapped[test] - b[test]) / sb.scale_) ** 2)),
                     affine_serialization_error=affine_error, fit_uses_only_train_contact=True,
                     recording_weight_sums=fit_rows.groupby('bag_id').recording_balanced_weight.sum().to_dict())
        mapping_checks.append(check)

        def evaluate(meta, hidden, path, method='none', seed=-1):
            probs = softmax(hidden @ head_w.T + head_b, axis=1)
            pred = probs.argmax(1)
            role = np.array(['train' if bag in train_bags else 'test' if bag in test_bags else 'external' for bag in meta.bag_id])
            tag = f'{fold}__{path}__{method}__seed{seed}'
            np.savez_compressed(out / 'predictions' / f'{tag}.npz', hidden=hidden, probabilities4=probs,
                                bag_id=meta.bag_id.to_numpy(str), label=meta.label.to_numpy(), role=role)
            for i, row in enumerate(meta.itertuples()):
                all_windows.append(dict(fold=fold, path=path, method=method, seed=seed, row_index=i,
                                        bag_id=row.bag_id, role=role[i], label=int(row.label), prediction=int(pred[i]),
                                        **{f'p{j}':float(probs[i,j]) for j in range(4)}))
            bag_rows = []
            for bag, ids in meta.groupby('bag_id', sort=False).indices.items():
                true_label = int(meta.iloc[ids[0]].label)
                pp = probs[ids].mean(0)
                ci = cm.bag_id.eq(bag).to_numpy()
                assert ci.any()
                guessed = sb.transform(hidden[ids]).mean(0)
                target = sb.transform(b[ci]).mean(0)
                row = dict(fold=fold, path=path, method=method, seed=seed, bag_id=bag,
                           role=role[ids[0]], label=true_label, prediction=int(pp.argmax()), n_windows=len(ids),
                           mse_standardized=float(np.mean((guessed-target)**2)), cos_standardized=cosine(guessed,target),
                           **{f'p{j}':float(pp[j]) for j in range(4)})
                bag_rows.append(row)
                all_files.append(row)
            frame = pd.DataFrame(bag_rows)
            for rr in ['train', 'test']:
                m = role == rr
                ff = frame[frame.role == rr]
                for unit, true, estimated in [('window',meta.loc[m,'label'].to_numpy(),pred[m]),
                                               ('recording',ff.label.to_numpy(),ff.prediction.to_numpy())]:
                    metrics = dict(fold=fold, path=path, method=method, seed=seed, role=rr, unit=unit,
                                   n=len(true), correct=int((true==estimated).sum()),
                                   accuracy=float(accuracy_score(true,estimated)),
                                   macro_f1=float(f1_score(true,estimated,labels=range(4),average='macro',zero_division=0)),
                                   confusion_matrix=json.dumps(confusion_matrix(true,estimated,labels=range(4)).tolist()))
                    if unit == 'recording':
                        metrics.update(mse_standardized=float(ff.mse_standardized.mean()),cos_standardized=float(ff.cos_standardized.mean()))
                    all_metrics.append(metrics)

        # Complete the two contact controls before radar evaluation.
        evaluate(cm, b, 'direct_Uni_contact_local4')
        evaluate(cm, mapped, 'Bear_contact_mapped_to_Uni_local4')
        print(fold, 'contact map train/test MSE', check['train_standardized_mse'], check['test_standardized_mse'], flush=True)
        for method in METHODS:
            for seed in SEEDS:
                tag = f'{fold}__{method}__seed{seed}'
                model_path = CTRL / 'models' / f'{tag}.pt'
                ck = load_local_checkpoint(model_path)
                assert ck['method'] == method and ck['seed'] == seed and ck['feature'] == 'frame_shape'
                assert set(ck['train_bags']) == set(train_bags) and set(ck['test_bags']) == set(test_bags)
                net = FrozenEncoder()
                net.load_state_dict(ck['encoder'], strict=True)
                net.eval().requires_grad_(False)
                before_state = {k:v.clone() for k,v in net.state_dict().items()}
                scaler = StandardScaler()
                scaler.mean_, scaler.scale_ = ck['radar_mean'], ck['radar_scale']
                scaler.n_features_in_ = 128
                xx = torch.tensor(scaler.transform(radar_x).astype(np.float32))
                with torch.inference_mode():
                    hh = net(xx).numpy()
                old = np.load(CTRL / 'windows' / f'{tag}.npz')
                assert np.array_equal(old['bag_id'], rm.bag_id.to_numpy(str))
                assert np.array_equal(old['role'] == 'test', rm.bag_id.isin(test_bags).to_numpy())
                replay_error = float(np.max(np.abs(hh - old['hidden'])))
                assert replay_error < 1e-6, (tag,replay_error)
                assert all(torch.equal(v,before_state[k]) for k,v in net.state_dict().items())
                assert all(not v.requires_grad for v in net.parameters())
                output = hh.astype(np.float64) @ wmap.T + bmap
                evaluate(rm, output, 'frozen_Bear_radar_mapped_to_Uni_local4', method, seed)
                model_checks.append(dict(tag=tag, model_path=str(model_path), model_sha256=hashes[str(model_path)],
                                         replay_hidden_max_abs=replay_error, radar_parameters_bitwise_unchanged=True,
                                         head_frozen=True, output_negative_fraction=float((output<0).mean())))
        assert np.array_equal(head_w,head['weight4']) and np.array_equal(head_b,head['bias4'])
    pd.DataFrame(all_windows).to_csv(out/'window_predictions.csv',index=False)
    pd.DataFrame(all_files).to_csv(out/'file_predictions.csv',index=False)
    met = pd.DataFrame(all_metrics)
    met.to_csv(out/'metrics.csv',index=False)
    held = met[met.role == 'test']
    grouped = held.groupby(['path','method','unit'],as_index=False)
    summary = grouped[['correct','n']].sum().rename(columns={'correct':'correct_repeated','n':'n_repeated'})
    summary['accuracy'] = summary.correct_repeated / summary.n_repeated
    # Two-fold pooled macro-F1 per seed, not a mean over unequal fold sizes.
    per_seed=[]
    for unit, rows in [('window',pd.DataFrame(all_windows)),('recording',pd.DataFrame(all_files))]:
        rows=rows[rows.role=='test']
        for (path,method,seed),gg in rows.groupby(['path','method','seed']):
            per_seed.append(dict(path=path,method=method,seed=int(seed),unit=unit,n=len(gg),correct=int(gg.label.eq(gg.prediction).sum()),
                                 accuracy=float(gg.label.eq(gg.prediction).mean()),
                                 macro_f1=float(f1_score(gg.label,gg.prediction,labels=range(4),average='macro',zero_division=0))))
    ps=pd.DataFrame(per_seed)
    ps.to_csv(out/'per_seed_pooled_metrics.csv',index=False)
    st=ps.groupby(['path','method','unit'],as_index=False).agg(macro_f1_seed_mean=('macro_f1','mean'),
                                                            accuracy_seed_std=('accuracy',lambda x:float(np.std(x))),seeds=('seed','nunique'))
    summary=summary.merge(st,on=['path','method','unit'],validate='one_to_one')
    summary['unique_main_recordings']=8
    summary['unique_main_windows']=summary['path'].str.startswith('frozen_Bear_radar').map({True:412,False:58})
    summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(all_files).query("role == 'test'").groupby(['path','method'],as_index=False)[['mse_standardized','cos_standardized']].mean().to_csv(out/'feature_fidelity_summary.csv',index=False)
    assert len(model_checks)==36 and len(mapping_checks)==2
    assert hashes == {p:sha(p) for p in hashes}
    audit=dict(completed=True,elapsed_seconds=time.time()-start,models_evaluated=36,mappings_fitted=2,
               unique_main_recordings=8,contact_windows=58,radar_windows=412,seeds_increase_sample_size=False,
               exact_contact_row_join=True,fit_uses_only_train_contact=True,map_uses_labels=False,map_uses_radar=False,
               original_files_unchanged=True,head_weights_unchanged=True,no_radar_training=True,
               no_test_based_hyperparameter_selection=True,heldout_is_previously_seen_development_data=True,
               corrected_ROI_checked=True,mapping_checks=mapping_checks,model_checks=model_checks,
               protocol_sha256=sha(out/'protocol_before_results.json'),script_sha256=sha(__file__))
    write_json(out/'verification.json',audit)
    write_json(out/'input_hashes_after.json',{p:sha(p) for p in hashes})
    write_json(out/'splits.json',splits)
    print(summary.to_string(index=False),flush=True)
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()
