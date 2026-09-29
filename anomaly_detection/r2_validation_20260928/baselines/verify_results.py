#!/usr/bin/env python3
"""Independent recomputation of reported metrics and split/aggregation invariants."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from run_baselines import LABELS, record_equal_weights, sha256

OUT=Path(__file__).resolve().parent

def read(name):
    return list(csv.DictReader(open(OUT/name,encoding='utf-8-sig')))

def main():
    windows=read('window_predictions.csv');records=read('recording_predictions.csv')
    grouped_w=defaultdict(list);grouped_r=defaultdict(list)
    for r in windows:grouped_w[(r['feature'],r['method'],r['recording_id'])].append(r)
    aggregate_checks=0
    for r in records:
        grouped_r[(r['feature'],r['method'])].append(r)
        if r['feature']=='qc_only':continue
        ww=grouped_w[(r['feature'],r['method'],r['recording_id'])]
        score=np.array([[float(w['score_'+la]) for la in LABELS] for w in ww]).mean(0)
        assert np.allclose(score,[float(r['score_'+la]) for la in LABELS],rtol=0,atol=1e-12)
        assert int(r['y_pred'])==int(score.argmax())
        assert int(r['n_windows'])==len(ww)
        assert len(set(w['y_true'] for w in ww))==1
        aggregate_checks+=1
    metric_checks=0
    for filename in ['metrics_by_fold.csv','metrics_pooled.csv']:
        for m in read(filename):
            source=windows if m['level']=='window' else records
            rr=[r for r in source if r['feature']==m['feature'] and r['method']==m['method']]
            if m['held_out_rpm']!='pooled_three_folds':rr=[r for r in rr if r['held_out_rpm']==m['held_out_rpm']]
            if m['distance_cm']!='all':rr=[r for r in rr if float(r['distance_cm'])==float(m['distance_cm'])]
            yy=np.array([int(r['y_true']) for r in rr]);pp=np.array([int(r['y_pred']) for r in rr])
            assert len(rr)==int(m['n'])
            assert np.isclose(accuracy_score(yy,pp),float(m['accuracy']),atol=1e-12)
            assert np.isclose(f1_score(yy,pp,labels=np.arange(4),average='macro',zero_division=0),float(m['macro_f1']),atol=1e-12)
            metric_checks+=1
    folds=json.loads((OUT/'splits.json').read_text())
    for f in folds:
        assert len(f['train_recordings'])==96 and len(f['test_recordings'])==48
        assert len(f['train_contact_ids'])==8 and len(f['test_contact_ids'])==4
        assert not set(f['train_contact_ids'])&set(f['test_contact_ids'])
        assert not set(f['train_recordings'])&set(f['test_recordings'])
    for key,rr in grouped_r.items():
        assert len(rr)==144 and len(set(r['recording_id'] for r in rr))==144,key
    # Unequal windows must not cause the scaler to overweight the longer recording.
    example=np.array([[0.],[0.],[0.],[100.]])
    ww=record_equal_weights(np.array(['a','a','a','b']))
    ss=StandardScaler().fit(example,sample_weight=ww)
    assert np.isclose(ss.mean_[0],50) and np.isclose(ww.mean(),1)
    e=json.loads((OUT/'evidence.json').read_text())
    assert sha256(Path(e['feature_file']))==e['feature_sha256']
    assert sha256(OUT/'protocol.json')==e['protocol_sha256']
    assert sha256(OUT/'run_baselines.py')==e['script_sha256']
    summary=dict(passed=True,window_prediction_rows=len(windows),recording_prediction_rows=len(records),
                 independently_recomputed_metric_rows=metric_checks,recording_aggregation_checks=aggregate_checks,
                 train_test_contact_disjoint_folds=len(folds),record_equal_weight_scaler_test=True,
                 feature_and_code_hashes_match=True,
                 svm_window_score_argmax_differs_from_standard_predict=sum(int(r['score_argmax_pred'])!=int(r['y_pred']) for r in windows))
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
