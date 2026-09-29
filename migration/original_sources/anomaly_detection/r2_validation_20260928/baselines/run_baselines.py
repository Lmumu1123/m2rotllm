#!/usr/bin/env python3
"""Fixed-hyperparameter radar baselines with held-out RPM/contact-session groups."""
import csv
import hashlib
import json
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from collections import Counter
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

BASE=Path('/home/huangyating/anomaly_detection/r2_validation_20260928')
OUT=BASE/'baselines'
LABELS=['normal','inBroken','outBroken','roll']
METHODS=[('complex_shape','linear_svc'),('complex_shape','rbf_svc'),
         ('complex_shape','knn5_distance'),('complex_shape','random_forest'),
         ('phase_shape','rbf_svc'),('geometry_complex_shape','rbf_svc'),('nuisance','rbf_svc')]

def save_csv(name, rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def record_equal_weights(recording_ids):
    count=Counter(recording_ids.tolist())
    # Normalize mean weight to 1, so C=1 retains the usual scale.
    return np.array([len(recording_ids)/(len(count)*count[r]) for r in recording_ids])

def classifier(name):
    if name=='linear_svc':return LinearSVC(C=1,class_weight='balanced',dual='auto',max_iter=20000,random_state=42)
    if name=='rbf_svc':return SVC(C=1,gamma='scale',probability=False,decision_function_shape='ovr')
    if name=='knn5_distance':return KNeighborsClassifier(n_neighbors=5,weights='distance',n_jobs=2)
    if name=='random_forest':return RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42,n_jobs=2)
    raise ValueError(name)

def scores(model, x):
    assert np.array_equal(model.classes_,np.arange(4))
    if hasattr(model,'decision_function'):
        d=model.decision_function(x)
        return softmax(d,axis=1),d,'softmax_of_OvR_decision_uncalibrated'
    p=model.predict_proba(x)
    return p,p,'predict_proba_uncalibrated'

def metrics_rows(feature,method,test_rpm,level,yy,pp,distances):
    rows=[]
    for dist in ['all',20,40,80]:
        ix=np.ones(len(yy),dtype=bool) if dist=='all' else distances==dist
        if not np.any(ix):continue
        rows.append(dict(feature=feature,method=method,held_out_rpm=test_rpm,level=level,
                         distance_cm=dist,n=int(ix.sum()),accuracy=float(accuracy_score(yy[ix],pp[ix])),
                         macro_f1=float(f1_score(yy[ix],pp[ix],labels=np.arange(4),average='macro',zero_division=0)),
                         confusion_matrix=json.dumps(confusion_matrix(yy[ix],pp[ix],labels=np.arange(4)).tolist())))
    return rows

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    protocol=dict(protocol='fixed_leave_one_RPM_out_v1',held_out_rpms=[1000,2000,3000],
                  labels=LABELS,methods=METHODS,hyperparameter_search='none',
                  model_parameters={n:classifier(n).get_params() for n in ['linear_svc','rbf_svc','knn5_distance','random_forest']},
                  scaler='StandardScaler fitted only on training windows; each recording equal total weight',
                  fit_weights='each training recording equal total weight, normalized mean sample weight 1; KNN cannot accept fit sample_weight',
                  record_prediction='argmax arithmetic mean of window class scores; SVM uses softmax of OVR margins, not calibrated probabilities',
                  window_prediction='standard classifier.predict; recording SVM score aggregation may resolve ties differently from individual predict votes',
                  group_isolation='held-out RPM removes all four associated contact sessions from training',
                  original_windows='all retained; no quality-based filtering',
                  quality_control_features=['nominal_duration_s','nominal_center_minus_declared_m','adc_near_limit_fraction','pilot_negative_positive_ratio','edge'],
                  quality_control_classifier='RBF SVC C=1 gamma=scale, record-level StandardScaler fitted on train only',
                  limitations=['single collection day; repeat independence unverified','not contact-model inference; these are radar-only baselines',
                               'post-acquisition development evaluation; not a sealed final benchmark'],
                  sklearn_version=sklearn.__version__)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False))
    feature_path=BASE/'radar/features.npz'
    with np.load(feature_path,allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    y=data['labels'];rpm=data['rpm'];dist=data['distance_cm'];rid=data['recording_id'];cid=data['contact_id']
    assert len(y)==2158 and len(set(rid))==144
    assert set(y.tolist())==set(range(4))
    windows=[];records=[];metrics=[];folds=[];runtimes=[]
    for test_rpm in [1000,2000,3000]:
        train=rpm!=test_rpm;test=~train
        train_contact=sorted(set(cid[train]));test_contact=sorted(set(cid[test]))
        assert not set(train_contact)&set(test_contact)
        assert not set(rid[train])&set(rid[test])
        folds.append(dict(held_out_rpm=test_rpm,train_recordings=sorted(set(rid[train])),
                          test_recordings=sorted(set(rid[test])),train_contact_ids=train_contact,test_contact_ids=test_contact,
                          train_windows=int(train.sum()),test_windows=int(test.sum())))
        w=record_equal_weights(rid[train])
        assert np.isclose(w.mean(),1)
        for feature,method in METHODS:
            tic=time.time();x=data[feature]
            assert np.isfinite(x).all(),feature
            scaler=StandardScaler().fit(x[train],sample_weight=w)
            tx=scaler.transform(x[train]);vx=scaler.transform(x[test])
            model=classifier(method)
            if method=='knn5_distance':model.fit(tx,y[train])
            else:model.fit(tx,y[train],sample_weight=w)
            p,d,kind=scores(model,vx);yp=model.predict(vx)
            metrics.extend(metrics_rows(feature,method,test_rpm,'window',y[test],yp,dist[test]))
            orig=np.flatnonzero(test)
            for i,ii in enumerate(orig):
                row=dict(feature=feature,method=method,held_out_rpm=test_rpm,window_index=int(ii),recording_id=rid[ii],
                         contact_id=cid[ii],distance_cm=float(dist[ii]),repeat=int(data['repeat'][ii]),
                         window_first_frame=int(data['window_first_frame'][ii]),y_true=int(y[ii]),y_pred=int(yp[i]),
                         score_argmax_pred=int(p[i].argmax()),score_kind=kind)
                row.update({f'score_{name}':float(p[i,k]) for k,name in enumerate(LABELS)})
                row.update({f'raw_decision_{name}':float(d[i,k]) for k,name in enumerate(LABELS)})
                windows.append(row)
            ry=[];rp=[];rd=[]
            for r in sorted(set(rid[test])):
                sel=rid[test]==r;yy=np.unique(y[test][sel]);assert len(yy)==1
                prob=p[sel].mean(0);raw=d[sel].mean(0)
                row=dict(feature=feature,method=method,held_out_rpm=test_rpm,recording_id=r,contact_id=cid[test][sel][0],
                         distance_cm=float(dist[test][sel][0]),repeat=int(data['repeat'][test][sel][0]),
                         n_windows=int(sel.sum()),y_true=int(yy[0]),y_pred=int(prob.argmax()),score_kind=kind)
                row.update({f'score_{name}':float(prob[k]) for k,name in enumerate(LABELS)})
                row.update({f'mean_raw_decision_{name}':float(raw[k]) for k,name in enumerate(LABELS)})
                records.append(row);ry.append(row['y_true']);rp.append(row['y_pred']);rd.append(row['distance_cm'])
            metrics.extend(metrics_rows(feature,method,test_rpm,'recording',np.array(ry),np.array(rp),np.array(rd)))
            runtimes.append(dict(feature=feature,method=method,held_out_rpm=test_rpm,seconds=time.time()-tic))
            print(f'{test_rpm} {feature} {method}: record {accuracy_score(ry,rp):.4f}, window {accuracy_score(y[test],yp):.4f}',flush=True)
    # A metadata-only diagnostic. No radar spectral features, no contact data.
    qc=list(csv.DictReader(open(BASE/'audit/radar_qc.csv',encoding='utf-8-sig')))
    qc_by_id={r['recording_id']:r for r in qc}
    qrid=np.array(sorted(set(rid)));qrpm=np.array([int(qc_by_id[r]['rpm']) for r in qrid])
    qdist=np.array([int(qc_by_id[r]['distance_cm']) for r in qrid])
    qy=np.array([LABELS.index(qc_by_id[r]['label']) for r in qrid])
    qx=np.array([[float(qc_by_id[r]['nominal_duration_s']),float(qc_by_id[r]['nominal_center_minus_declared_m']),
                  float(qc_by_id[r]['adc_near_limit_fraction']),float(qc_by_id[r]['pilot_negative_positive_ratio']),
                  float(qc_by_id[r]['edge']=='True')] for r in qrid])
    for test_rpm in [1000,2000,3000]:
        train=qrpm!=test_rpm;test=~train
        ss=StandardScaler().fit(qx[train]);model=classifier('rbf_svc').fit(ss.transform(qx[train]),qy[train])
        p,d,kind=scores(model,ss.transform(qx[test]));yp=model.predict(ss.transform(qx[test]))
        metrics.extend(metrics_rows('qc_only','rbf_svc',test_rpm,'recording',qy[test],yp,qdist[test]))
        for k,ii in enumerate(np.flatnonzero(test)):
            row=dict(feature='qc_only',method='rbf_svc',held_out_rpm=test_rpm,recording_id=qrid[ii],
                     contact_id=qc_by_id[qrid[ii]]['contact_id'],distance_cm=int(qdist[ii]),
                     repeat=int(qc_by_id[qrid[ii]]['repeat']),n_windows='not_applicable',
                     y_true=int(qy[ii]),y_pred=int(yp[k]),score_kind=kind)
            row.update({f'score_{name}':float(p[k,j]) for j,name in enumerate(LABELS)})
            row.update({f'mean_raw_decision_{name}':float(d[k,j]) for j,name in enumerate(LABELS)})
            records.append(row)
        print(f'{test_rpm} qc_only rbf_svc: record {accuracy_score(qy[test],yp):.4f}',flush=True)
    pooled=[]
    for source,level in [(windows,'window'),(records,'recording')]:
        pairs=sorted(set((r['feature'],r['method']) for r in source))
        for feature,method in pairs:
            rr=[r for r in source if r['feature']==feature and r['method']==method]
            pooled.extend(metrics_rows(feature,method,'pooled_three_folds',level,
                                       np.array([r['y_true'] for r in rr]),np.array([r['y_pred'] for r in rr]),
                                       np.array([r['distance_cm'] for r in rr])))
    save_csv('window_predictions.csv',windows);save_csv('recording_predictions.csv',records)
    save_csv('metrics_by_fold.csv',metrics);save_csv('metrics_pooled.csv',pooled);save_csv('runtimes.csv',runtimes)
    (OUT/'splits.json').write_text(json.dumps(folds,indent=2,ensure_ascii=False))
    evidence=dict(feature_file=str(feature_path),feature_sha256=sha256(feature_path),
                  audit_qc_sha256=sha256(BASE/'audit/radar_qc.csv'),
                  script_sha256=sha256(Path(__file__)),protocol_sha256=sha256(OUT/'protocol.json'),
                  n_windows=len(y),n_recordings=len(set(rid)),
                  no_train_test_contact_overlap=True,no_train_test_recording_overlap=True,
                  fit_only_train_scales=True,all_windows_retained=True)
    (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2))
    print('DONE',flush=True)

if __name__=='__main__':main()
