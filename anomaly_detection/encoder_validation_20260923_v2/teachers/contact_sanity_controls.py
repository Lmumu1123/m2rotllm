"""Post-inference diagnostic: can simple contact statistics explain high scores?

These controls are not used to choose UniFault parameters or input branches.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
BASE = Path(_c2r_resolve_path(__file__)).resolve().parent
DATA = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results'))

def main():
    out = BASE / 'contact_sanity_controls'
    out.mkdir(exist_ok=True)
    assert not (out / 'metrics.csv').exists()
    protocol = dict(timing='Declared after UniFault 58/58 observed; diagnostic only, no input/model selection', purpose='Check whether simple sensor offset, RMS or spectrum already separates this development set', methods=['raw_dc3', 'ac_rms3', 'ac_stats12', 'ac_spectral_shape64'], readout='same fixed train-only StandardScaler + balanced LogisticRegression C=1', confounds='raw_dc includes gravity, orientation and sensor bias; not a fault-specific label')
    (out / 'protocol_before_controls.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    md = pd.read_csv(BASE / 'unifault/metadata.csv')
    x = np.stack([np.load(DATA / r.file)['raw_xyz'][r.window_row] for r in md.itertuples()]).astype(float)
    dc = x.mean(1)
    ac = x - dc[:, None, :]
    rms = np.sqrt(np.mean(ac ** 2, 1))
    crest = np.max(np.abs(ac), 1) / np.maximum(rms, 1e-12)
    stats = np.concatenate([rms, crest, kurtosis(ac, axis=1, fisher=False, bias=False), np.mean(abs(ac), 1)], 1)
    ps = (abs(np.fft.rfft(ac * np.hanning(4000)[None, :, None], axis=1)) ** 2).mean(2)
    freq = np.fft.rfftfreq(4000, 1 / 4000)
    edges = np.linspace(10, 1950, 65)
    power = np.stack([ps[:, (freq >= lo) & (freq < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    shape = np.log10(np.maximum(power / np.maximum(power.sum(1, keepdims=True), 1e-24), 1e-12))
    methods = dict(raw_dc3=dc, ac_rms3=rms, ac_stats12=stats, ac_spectral_shape64=shape)
    md.to_csv(out / 'metadata.csv', index=False)
    np.savez_compressed(out / 'features.npz', **methods)
    counts = []
    predictions = []
    for method, h in methods.items():
        for trbaud, tebaud in [(115200, 460800), (460800, 115200)]:
            fold = f'{trbaud}_to_{tebaud}'
            tr = (md.label >= 0) & (md.baud == trbaud)
            te = (md.label >= 0) & (md.baud == tebaud)
            sc = StandardScaler().fit(h[tr])
            clf = LogisticRegression(C=1, max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=42).fit(sc.transform(h[tr]), md.loc[tr, 'label'])
            pp = clf.predict_proba(sc.transform(h))
            df = md.copy()
            cols = [f'p{k}' for k in range(4)]
            for k, col in enumerate(cols):
                df[col] = pp[:, k]
            ff = df.groupby(['file', 'label', 'state', 'baud', 'bag_id'], as_index=False)[cols].mean()
            for unit, frame in [('window', df), ('recording', ff)]:
                d = frame[(frame.label >= 0) & (frame.baud == tebaud)].copy()
                yp = d[cols].to_numpy().argmax(1)
                counts.append(dict(method=method, fold=fold, unit=unit, n=len(d), correct=int((d.label.to_numpy() == yp).sum()), accuracy=float(accuracy_score(d.label, yp)), macro_f1=float(f1_score(d.label, yp, labels=[0, 1, 2, 3], average='macro', zero_division=0))))
                d['prediction'] = yp
                d['method'] = method
                d['fold'] = fold
                d['unit'] = unit
                predictions.append(d)
    met = pd.DataFrame(counts)
    met.to_csv(out / 'metrics.csv', index=False)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(out / 'predictions.csv', index=False)
    pooled = []
    for (method, unit), d in pred.groupby(['method', 'unit']):
        pooled.append(dict(method=method, unit=unit, n=len(d), correct=int((d.label == d.prediction).sum()), accuracy=float(accuracy_score(d.label, d.prediction)), macro_f1=float(f1_score(d.label, d.prediction, labels=[0, 1, 2, 3], average='macro', zero_division=0))))
    pd.DataFrame(pooled).to_csv(out / 'pooled_metrics.csv', index=False)
    print(pd.DataFrame(pooled).to_string(index=False))
if __name__ == '__main__':
    main()
