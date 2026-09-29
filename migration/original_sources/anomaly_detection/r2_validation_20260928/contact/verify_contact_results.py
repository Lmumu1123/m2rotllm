#!/usr/bin/env python3
"""Independent arithmetic, provenance, grouping and held-out-session checks."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score,f1_score
HERE=Path(__file__).resolve().parent
GROUPS=[[0],[1,2,3],[7,8,9],[4,5,6]]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    z=np.load(HERE/'contact_features.npz',allow_pickle=False)
    md=pd.read_csv(HERE/'metadata.csv')
    assert z['embedding'].shape==(343,128)
    assert np.isfinite(z['embedding']).all() and z['embedding'].min()>=0
    assert np.array_equal(z['labels'],md.label) and np.array_equal(z['session_id'],md.session_id)
    assert np.array_equal(z['embedding'],z['hidden_axes'].mean(1))
    assert np.array_equal(z['embedding_xy_sensitivity'],z['hidden_axes'][:,:2].mean(1))
    p10=softmax(z['embedding'].astype(float)@z['retained_weight10'].astype(float).T+z['retained_bias10'],axis=-1)
    p4=np.stack([p10[:,g].sum(-1) for g in GROUPS],-1)
    diff=float(np.max(abs(p4-z['retained_probabilities4'])))
    assert diff<2e-5
    packets=pd.read_csv(HERE/'packet_predictions.csv')
    sessions=pd.read_csv(HERE/'session_predictions.csv')
    metrics=pd.read_csv(HERE/'metrics.csv')
    for item in metrics[metrics.scope=='all'].itertuples():
        rows=(packets if item.level=='packet' else sessions)
        rows=rows[(rows.variant==item.variant)&(rows.model==item.model)]
        assert len(rows)==item.n
        assert abs(accuracy_score(rows.label,rows.prediction)-item.accuracy)<1e-12
        assert abs(f1_score(rows.label,rows.prediction,labels=range(4),average='macro',zero_division=0)-item.macro_f1)<1e-12
    frozen=json.loads((HERE/'verification.json').read_text())
    for check in frozen['checks']:
        for path,checksum in check['model_provenance']['weights'].items():assert sha(path)==checksum
    for head in frozen['heads']:assert sha(head['head_path'])==head['sha256']
    checks=[]
    preds=pd.read_csv(HERE/'diagnostic_heads/packet_predictions.csv')
    for holdout in (1000,2000,3000):
        with np.load(HERE/f'diagnostic_heads/holdout_{holdout}.npz',allow_pickle=False) as ck:
            assert not bool(ck['accepted_for_source_retention'])
            train=set(ck['train_sessions'].tolist());test=set(ck['test_sessions'].tolist())
            assert len(train)==8 and len(test)==4 and not train&test
            ids=np.flatnonzero(z['rpm']==holdout)
            pred=softmax(z['embedding'][ids]@ck['weight4'].T+ck['bias4'],axis=-1)
            pp=preds[(preds.holdout_rpm==holdout)&(preds.model=='Bear128')]
            assert np.array_equal(pp.packet_index,md.iloc[ids].packet_index)
            assert np.array_equal(pp.session_id,md.iloc[ids].session_id)
            err=float(np.max(abs(pred-pp[[f'score{k}' for k in range(4)]].to_numpy())))
            assert err<1e-5
            assert np.array_equal(pred.argmax(-1),pp.prediction)
            checks.append(dict(heldout_rpm=holdout,raw_head_float32_reload_probability_max_error=err,
                               train_sessions=sorted(train),test_sessions=sorted(test)))
    output=dict(ok=True,primary_packets=343,primary_sessions=12,primary_head_float64_recompute_max_error=diff,
                all_frozen_metric_rows_recomputed=True,source_weights_still_unchanged=True,
                diagnostic_fold_reload=checks,physical_targets_finite=bool(np.isfinite(z['physical_targets']).all()),
                primary_feature_sha256=sha(HERE/'contact_features.npz'),script_sha256=sha(__file__))
    within_split=pd.read_csv(HERE/'within_session_split.csv')
    assert len(within_split)==343
    assert (within_split.role=='train_early').sum()==224
    assert (within_split.role=='test_late').sum()==119
    for _,part in within_split.groupby('session_id'):
        tr=part[part.role=='train_early'];te=part[part.role=='test_late']
        assert tr.packet_index.max()<te.packet_index.min()
        assert tr.byte_end_exclusive.max()<=te.byte_start.min()
    wp=pd.read_csv(HERE/'within_session_diagnostic.csv')
    ws=pd.read_csv(HERE/'within_session_tail_predictions.csv')
    wm=pd.read_csv(HERE/'within_session_diagnostic_summary.csv')
    for item in wm.itertuples():
        dd=(wp if item.level=='packet' else ws)
        dd=dd[dd.model==item.model]
        assert len(dd)==item.n
        assert abs(accuracy_score(dd.label,dd.prediction)-item.accuracy)<1e-12
        assert abs(f1_score(dd.label,dd.prediction,labels=range(4),average='macro',zero_division=0)-item.macro_f1)<1e-12
    output['within_session_time_order_and_metric_checks']=True
    (HERE/'independent_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
