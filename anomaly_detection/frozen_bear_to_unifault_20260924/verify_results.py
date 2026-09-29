"""Independent read-only verification of saved transfer arrays and metric counts."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
V2=ROOT.parent/'encoder_validation_20260923_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    hashes=json.loads((OUT/'input_hashes_before.json').read_text())
    assert all(sha(p)==v for p,v in hashes.items())
    s=pd.read_csv(OUT/'summary.csv');ww=pd.read_csv(OUT/'window_predictions.csv');ff=pd.read_csv(OUT/'file_predictions.csv')
    checks=[]
    for path in sorted((OUT/'predictions').glob('*.npz')):
        fold,route,method,seed=path.stem.split('__');seed=int(seed.removeprefix('seed'))
        z=np.load(path);head=np.load(V2/'teachers/unifault/paper_100ms'/fold/'head.npz')
        prob=softmax(z['hidden']@head['weight4'].T+head['bias4'],axis=1)
        assert np.isfinite(z['hidden']).all() and np.max(abs(prob-z['probabilities4']))<1e-12
        rr=ww[(ww.fold==fold)&(ww.path==route)&(ww.method==method)&(ww.seed==seed)].sort_values('row_index')
        assert np.array_equal(rr.bag_id.to_numpy(str),z['bag_id'])
        assert np.array_equal(rr.prediction,prob.argmax(1))
        bf=ff[(ff.fold==fold)&(ff.path==route)&(ff.method==method)&(ff.seed==seed)]
        for row in bf.itertuples():
            ids=z['bag_id']==row.bag_id
            assert row.prediction==prob[ids].mean(0).argmax()
        if route.startswith('frozen_Bear_radar'):
            m=np.load(OUT/'mappings'/f'{fold}.npz')
            original=np.load(V2/'controls/results/windows'/f'{fold}__{method}__seed{seed}.npz')
            mapped=original['hidden'].astype(float)@m['weight'].T+m['bias']
            assert not bool(m['post_relu'])
            assert np.max(abs(mapped-z['hidden']))<1e-7
        checks.append(dict(file=str(path),rows=len(prob),head_probability_error=float(np.max(abs(prob-z['probabilities4'])))))
    assert len(checks)==40
    for row in s.itertuples():
        d=ww if row.unit=='window' else ff
        d=d[(d.path==row.path)&(d.method==row.method)&(d.role=='test')]
        assert len(d)==row.n_repeated and sum(d.label==d.prediction)==row.correct_repeated
        assert d.bag_id.nunique()==8
    for fold in ['115200_to_460800','460800_to_115200']:
        fit=pd.read_csv(OUT/'fit_rows'/f'{fold}.csv');m=np.load(OUT/'mappings'/f'{fold}.npz')
        assert set(fit.bag_id)==set(m['train_bags']) and not set(fit.bag_id)&set(m['test_bags'])
        assert not fit.label_used_in_ridge.any()
        sums=fit.groupby('bag_id').recording_balanced_weight.sum()
        assert np.max(sums)-np.min(sums)<1e-12
    result=dict(passed=True,prediction_files_checked=len(checks),source_hashes_unchanged=True,
                metric_counts_recomputed=True,mapping_reapplication_verified=True,
                head_outputs_recomputed=True,train_test_separation_checked=True,checks=checks)
    dest=OUT/'independent_verification.json'
    if dest.exists():raise RuntimeError('Refuse verification overwrite')
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print('Independent verification passed:',len(checks),'prediction files')

if __name__=='__main__':main()
