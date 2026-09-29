"""Time-ordered within-session separability, explicitly not generalization."""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '2'
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from diagnose_loso import HERE, sha, writej, weights, metrics

def main():
    assert sys.version_info >= (3, 10)
    out = HERE / 'within_session_diagnostic.csv'
    if out.exists():
        raise FileExistsError('Do not overwrite held-tail diagnostic')
    z = np.load(HERE / 'contact_features.npz', allow_pickle=False)
    psd = np.load(HERE / 'diagnostic_contact_psd.npz', allow_pickle=False)
    md = pd.read_csv(HERE / 'metadata.csv')
    train = np.zeros(len(md), bool)
    for session, part in md.groupby('session_id'):
        ordered = part.sort_values('packet_index').index.to_numpy()
        ntrain = max(1, min(len(ordered) - 1, 2 * len(ordered) // 3))
        train[ordered[:ntrain]] = True
        assert md.loc[ordered[:ntrain], 'byte_end_exclusive'].max() <= md.loc[ordered[ntrain:], 'byte_start'].min()
    test = ~train
    split = md.copy()
    split['role'] = np.where(train, 'train_early', 'test_late')
    split.to_csv(HERE / 'within_session_split.csv', index=False)
    writej(HERE / 'within_session_protocol_before_results.json', dict(evaluation_scope='Within the same 12 contact sessions; explicitly not independent-recording or unseen-RPM generalization', splitting='Sort complete packets by packet_index independently within each session; first floor(2*N/3) train and remaining tail test; no random shuffle; no overlapping raw samples', caveat='Packet gaps and wall-clock times unknown; early/late means observed complete-packet order, not equal wall-time fractions', algorithms='Fixed C=1 train-session-weighted StandardScaler+LogisticRegression for Bear128 and PSD128, XYZ and XY; fixed C=1 gamma=scale RBF SVC for physical6', inputs_sha256=sha(HERE / 'contact_features.npz'), no_model_or_hyperparameter_selection=True, training_packets=int(train.sum()), test_packets=int(test.sum()), script_sha256=sha(__file__)))
    data = dict(Bear128=z['embedding'], Bear128_XY_sensitivity=z['embedding_xy_sensitivity'], ContactPSD128=psd['psd_shape'], ContactPSD128_XY_sensitivity=psd['psd_shape_xy'], Physical6_RBF_SVM=z['physical_targets'])
    y = z['labels']
    sw = weights(z['session_id'][train])
    rows, summary, session_predictions = ([], [], [])
    for name, x in data.items():
        x = x.astype(float)
        sc = StandardScaler().fit(x[train], sample_weight=sw)
        if name == 'Physical6_RBF_SVM':
            model = SVC(C=1.0, gamma='scale', kernel='rbf').fit(sc.transform(x[train]), y[train], sample_weight=sw)
            pred = model.predict(sc.transform(x[test]))
            score = np.eye(4)[pred]
            pooling_type = 'fraction_window_hard_votes'
        else:
            model = LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs', random_state=42)
            model.fit(sc.transform(x[train]), y[train], sample_weight=sw)
            score = model.predict_proba(sc.transform(x[test]))
            pred = score.argmax(-1)
            pooling_type = 'mean_window_probabilities'
        dd = md[test].copy()
        dd['model'] = name
        dd['prediction'] = pred
        dd['pooling_type'] = pooling_type
        for k in range(4):
            dd[f'score{k}'] = score[:, k]
        rows.append(dd)
        sm = dd.groupby(['session_id', 'label', 'rpm'], as_index=False)[[f'score{k}' for k in range(4)]].mean()
        sm['prediction'] = sm[[f'score{k}' for k in range(4)]].to_numpy().argmax(-1)
        sm['model'] = name
        sm['pooling_type'] = pooling_type
        session_predictions.append(sm)
        summary.append(dict(model=name, level='packet', **metrics(y[test], pred)))
        summary.append(dict(model=name, level='session_tail', **metrics(sm.label, sm.prediction)))
    pd.concat(rows).to_csv(out, index=False)
    pd.concat(session_predictions).to_csv(HERE / 'within_session_tail_predictions.csv', index=False)
    pd.DataFrame(summary).to_csv(HERE / 'within_session_diagnostic_summary.csv', index=False)
    assert sha(HERE / 'contact_features.npz') == json.loads((HERE / 'within_session_protocol_before_results.json').read_text())['inputs_sha256']
    print(pd.DataFrame(summary).to_string(index=False))
if __name__ == '__main__':
    main()
