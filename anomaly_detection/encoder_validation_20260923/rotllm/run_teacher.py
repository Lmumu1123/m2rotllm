"""Frozen official RotLLM encoder audit plus contact-only four-class heads.

Read protocol_before_results.json for the preregistered primary/sensitivity branches.
Only new LogisticRegression heads are trained. Vendor weights remain untouched.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import hashlib, importlib.util, json, sqlite3, subprocess, sys, time, warnings
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
BASE = Path(_c2r_resolve_path(__file__)).resolve().parent
VENDOR = BASE / 'vendor/RotLLM'
DATA = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results'))
HELPER = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_chain_20260922/contact/evaluate_contact.py'))
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]
VARIANTS = [('raw_rms001', False, 0.01), ('demean_rms001', True, 0.01), ('raw_rms1', False, 1.0), ('demean_rms1', True, 1.0)]

def sha(p):
    h = hashlib.sha256()
    with open(_c2r_resolve_path(p), 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def savej(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def loadmod(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def metrics(y, p):
    return {'n': len(y), 'accuracy': float(accuracy_score(y, p)), 'macro_f1': float(f1_score(y, p, labels=[0, 1, 2, 3], average='macro', zero_division=0)), 'confusion_matrix': confusion_matrix(y, p, labels=[0, 1, 2, 3, -1]).tolist()}

def aggregate(md, probs):
    out = md.copy()
    for k in range(probs.shape[1]):
        out[f'p{k}'] = probs[:, k]
    cols = [f'p{k}' for k in range(probs.shape[1])]
    out['pred4'] = probs[:, :4].argmax(1)
    files = out.groupby(['file', 'label', 'state', 'baud', 'bag_id'], as_index=False)[cols].mean()
    files['pred4'] = files[cols[:4]].to_numpy().argmax(1)
    return (out, files)

def full_native_prediction(p15):
    p4 = np.stack([p15[:, idx].sum(1) for idx in GROUPS], 1)
    p9 = np.concatenate([p4, p15[:, 10:15]], 1)
    result = p9.argmax(1)
    return np.where(result >= 4, -1, result)

def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads((BASE / 'protocol_before_results.json').read_text())
    hashes_before = {str(p.relative_to(VENDOR)): sha(p) for p in [VENDOR / 'weights/encoder_weights.pth', VENDOR / 'weights/proj_weights.pth']}
    helper = loadmod(HELPER, 'bear_input_reader')
    _, md, manifest = helper.read_inputs(DATA)
    raw = np.stack([np.load(DATA / row.file, allow_pickle=False)['raw_xyz'][row.window_row] for row in md.itertuples()])
    assert raw.shape == (88, 4000, 3)
    assert len(md[md.label >= 0]) == 58
    md.to_csv(BASE / 'metadata.csv', index=False)
    enc_state = torch.load(VENDOR / 'weights/encoder_weights.pth', map_location='cpu', weights_only=True)
    proj_state = torch.load(VENDOR / 'weights/proj_weights.pth', map_location='cpu', weights_only=True)
    sfn = loadmod(VENDOR / 'code/models/SFN.py', 'official_rot_sfn')
    model = sfn.SpecFoldNet()
    assert model.encode_len == 720
    state = {'encoder.' + k: v for k, v in enc_state.items()}
    for k in ['0.weight', '0.bias', '2.weight', '2.bias']:
        state['fc.' + k] = proj_state['proj.' + k]
    assert set(state) == set(model.state_dict())
    model.load_state_dict(state, strict=True)
    np.savez_compressed(BASE / 'native_head15.npz', weight15=state['fc.2.weight'].numpy(), bias15=state['fc.2.bias'].numpy(), group4_indices=np.array([0, 1, 2, 3, 7, 8, 9, 4, 5, 6]), group4_offsets=np.array([0, 1, 4, 7, 10]), other_indices=np.arange(10, 15))
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to('cuda:0')
    state_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    with sqlite3.connect(f'file:{VENDOR}/datasets/vibration_metadata.sqlite?mode=ro', uri=True) as c:
        notes = c.execute('SELECT label,note FROM label_note ORDER BY label').fetchall()
    assert notes[0][1] == 'Normal State'
    assert all(('Inner Ring' in notes[k][1] for k in [1, 2, 3]))
    assert all(('Ball' in notes[k][1] for k in [4, 5, 6]))
    assert all(('Outer Ring' in notes[k][1] for k in [7, 8, 9]))
    assert all(('Gear' in notes[k][1] for k in [10, 11, 12, 13, 14]))
    savej(BASE / 'provenance.json', {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=VENDOR, text=True).strip(), 'source_url': 'https://github.com/SIA-IDE/RotLLM', 'vendor_weights_sha256': hashes_before, 'script_sha256': sha(__file__), 'protocol_sha256': sha(BASE / 'protocol_before_results.json'), 'safe_loading': 'torch.load(weights_only=True)', 'strict_state_keys': len(state), 'expected_state_keys': len(model.state_dict()), 'encoder_state_keys': len(enc_state), 'classifier_state_keys': 4, 'classifier_mapping': 'proj.0.* -> fc.0.*, proj.2.* -> fc.2.*', 'classification_path_coverage': 1.0, 'not_used_projection_keys': [k for k in proj_state if not k.startswith(('proj.0.', 'proj.2.'))], 'label_notes': notes, 'input_manifest': manifest, 'python': sys.executable, 'torch': torch.__version__})
    rows = []
    checks = []
    splits = []
    for name, demean, rms in VARIANTS:
        out = BASE / 'variants' / name
        out.mkdir(parents=True, exist_ok=True)
        x = raw.astype(np.float32)
        if demean:
            x = x - x.mean(axis=1, keepdims=True)
        z = dct(x, axis=1, type=2, norm='backward').transpose(0, 2, 1)
        z = np.pad(z, ((0, 0), (0, 0), (0, 20000)))
        z *= rms * np.sqrt(24000 / np.square(z).sum(axis=-1, keepdims=True))
        assert np.isfinite(z).all()
        np.testing.assert_allclose(np.sqrt(np.square(z).mean(-1)), rms, rtol=1e-06)
        hidden = []
        logits = []
        fmaps = []
        with torch.inference_mode():
            flat = z.reshape(-1, 24000)
            for start in range(0, len(flat), 48):
                t = torch.from_numpy(flat[start:start + 48]).to('cuda:0').reshape(-1, 3, 8000)
                f = model.encoder(t).flatten(1)
                h = model.fc[1](model.fc[0](f))
                l = model.fc[2](h)
                hidden.append(h.cpu().numpy())
                logits.append(l.cpu().numpy())
                fmaps.append(f.cpu().numpy())
        haxes = np.concatenate(hidden).reshape(88, 3, 128)
        h = haxes.mean(1)
        laxes = np.concatenate(logits).reshape(88, 3, 15)
        p15 = softmax(laxes, axis=-1).mean(1)
        p15_from_mean_hidden = softmax(h @ state['fc.2.weight'].numpy().T + state['fc.2.bias'].numpy(), axis=-1)
        p4unnorm = np.stack([p15[:, idx].sum(1) for idx in GROUPS], 1)
        p4 = p4unnorm / p4unnorm.sum(1, keepdims=True)
        featdata = dict(hidden_mean=h, hidden_axes=haxes, feature_map_mean=np.concatenate(fmaps).reshape(88, 3, 720).mean(1), logits15_axes=laxes, logits15_mean=laxes.mean(1), probs15_mean=p15, probs15_from_mean_hidden=p15_from_mean_hidden, probs4_mean=p4, labels=md.label.to_numpy(), label=md.label.to_numpy(), bag_id=md.bag_id.to_numpy(dtype=str), baud=md.baud.to_numpy(), states=md.state.to_numpy(dtype=str), state=md.state.to_numpy(dtype=str), file_names=md.file.to_numpy(dtype=str), window_rows=md.window_row.to_numpy())
        np.savez_compressed(out / 'teacher_features.npz', **featdata)
        md.to_csv(out / 'metadata.csv', index=False)
        native, files = aggregate(md, p4)
        native['gear_mass'] = p15[:, 10:].sum(1)
        native['pred15'] = p15.argmax(1)
        native['pred4_gear_errors'] = full_native_prediction(p15)
        mean_hidden_p4 = np.stack([p15_from_mean_hidden[:, idx].sum(1) for idx in GROUPS], 1)
        native['pred4_from_mean_hidden'] = mean_hidden_p4.argmax(1)
        native15, _ = aggregate(md, p15)
        files15 = native15.groupby('file')[[f'p{k}' for k in range(15)]].mean().loc[files.file].to_numpy()
        files['gear_mass'] = files15[:, 10:].sum(1)
        files['pred15'] = files15.argmax(1)
        files['pred4_gear_errors'] = full_native_prediction(files15)
        native.to_csv(out / 'native_window_predictions.csv', index=False)
        files.to_csv(out / 'native_file_predictions.csv', index=False)
        for unit, df in [('window', native), ('file', files)]:
            main_df = df[df.label >= 0]
            for mapname, col in [('bearing_conditional', 'pred4'), ('gear_as_errors', 'pred4_gear_errors')]:
                r = dict(variant=name, method='native_released_head', mapping=mapname, fold='all_evaluation', unit=unit, gear_mass_mean=float(main_df.gear_mass.mean()), pred15_other_rate=float((main_df.pred15 >= 10).mean()), **metrics(main_df.label, main_df[col]))
                rows.append(r)
        primarymain = md.label >= 0
        rows.append(dict(variant=name, method='native_head_on_mean_hidden', mapping='bearing_conditional', fold='all_evaluation', unit='window', **metrics(md.loc[primarymain, 'label'], mean_hidden_p4[primarymain].argmax(1))))
        checks.append(dict(variant=name, probability_pooling='native mean of three axis softmax; separate mean-hidden path also exported', native_mean_probability_vs_probability_of_mean_hidden_max_abs=float(np.max(np.abs(p15 - p15_from_mean_hidden))), main_four_class_argmax_disagreements=int((p4[primarymain].argmax(1) != mean_hidden_p4[primarymain].argmax(1)).sum())))
        for train_baud, test_baud in [(115200, 460800), (460800, 115200)]:
            fold = f'{train_baud}_to_{test_baud}'
            train = (md.label >= 0) & (md.baud == train_baud)
            test = (md.label >= 0) & (md.baud == test_baud)
            assert not set(md.loc[train, 'bag_id']) & set(md.loc[test, 'bag_id'])
            if name == VARIANTS[0][0]:
                splits.append(dict(fold=fold, train_bags=sorted(md.loc[train, 'bag_id'].unique()), test_bags=sorted(md.loc[test, 'bag_id'].unique()), train_windows=int(train.sum()), test_windows=int(test.sum())))
            scaler = StandardScaler().fit(h[train])
            lr = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                lr.fit(scaler.transform(h[train]), md.loc[train, 'label'])
            assert list(lr.classes_) == [0, 1, 2, 3]
            weight = lr.coef_ / scaler.scale_[None, :]
            bias = lr.intercept_ - weight @ scaler.mean_
            rawprob = softmax(h.astype(np.float64) @ weight.T + bias, axis=1)
            refprob = lr.predict_proba(scaler.transform(h))
            err = float(np.max(np.abs(rawprob - refprob)))
            assert err < 2e-05
            foldout = out / 'heads' / fold
            foldout.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(foldout / 'head.npz', weight4=weight, bias4=bias, classes=lr.classes_, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, train_bag_ids=md.loc[train, 'bag_id'].unique().astype(str), test_bag_ids=md.loc[test, 'bag_id'].unique().astype(str))
            pred, predfile = aggregate(md, rawprob)
            pred['split'] = np.where(train, 'train', np.where(test, 'test', 'external'))
            predfile['split'] = np.where((predfile.label >= 0) & (predfile.baud == train_baud), 'train', np.where(predfile.label >= 0, 'test', 'external'))
            pred.to_csv(foldout / 'window_predictions.csv', index=False)
            predfile.to_csv(foldout / 'file_predictions.csv', index=False)
            for unit, df in [('window', pred), ('file', predfile)]:
                for split in ['train', 'test']:
                    dd = df[df.split == split]
                    rows.append(dict(variant=name, method='new_contact_only_linear_head', mapping='four_class', fold=fold, split=split, unit=unit, **metrics(dd.label, dd.pred4)))
                ext = df[df.state == 'bigNormal']
                rows.append(dict(variant=name, method='new_contact_only_linear_head', mapping='external_normal_only', fold=fold, split='external', unit=unit, n=len(ext), accuracy=float((ext.pred4 == 0).mean()), macro_f1=None))
            checks.append(dict(variant=name, fold=fold, head_foldback_max_probability_error=err, fitting_warnings=[str(w.message) for w in caught], iterations=lr.n_iter_.tolist(), training_samples=int(train.sum()), test_samples=int(test.sum())))
        print(name, [(r['method'], r['fold'], r.get('split', 'native'), r['unit'], r['accuracy']) for r in rows if r['variant'] == name and r['unit'] == 'file'], flush=True)
        if name == protocol['primary_variant']:
            np.savez_compressed(BASE / 'teacher_features.npz', **featdata)
            primary = BASE / 'heads'
            primary.mkdir(exist_ok=True)
            for fold in ['115200_to_460800', '460800_to_115200']:
                target = primary / fold
                target.mkdir(exist_ok=True)
                (target / 'head.npz').write_bytes((out / 'heads' / fold / 'head.npz').read_bytes())
    for k, v in model.state_dict().items():
        assert torch.equal(state_before[k], v.cpu()), k
    hashes_after = {k: sha(VENDOR / k) for k in hashes_before}
    assert hashes_before == hashes_after
    pd.DataFrame(rows).to_csv(BASE / 'metrics.csv', index=False)
    savej(BASE / 'splits.json', splits)
    savej(BASE / 'verification.json', dict(encoder_and_native_classifier_parameters_and_buffers_unchanged=True, vendor_weights_unchanged=True, old_task_regression_measured=False, old_task_input_pipeline_not_overwritten=True, all_features_finite=True, strict_loading_coverage=1.0, all_checks=checks, elapsed_seconds=time.monotonic() - started, no_heldout_labels_used_for_scaler_or_head_fit=True, all_predeclared_variants_reported=True))
    print(pd.DataFrame(rows).drop(columns='confusion_matrix').to_string(index=False), flush=True)
if __name__ == '__main__':
    main()
