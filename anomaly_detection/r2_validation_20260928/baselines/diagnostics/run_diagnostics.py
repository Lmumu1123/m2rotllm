#!/usr/bin/env python3
"""Post-primary developmental diagnostics; separate from held-out RPM evidence."""
import csv
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_baselines import BASE, LABELS, classifier, record_equal_weights, scores, sha256

OUT=Path(__file__).resolve().parent
METHODS=[('complex_shape','rbf_svc'),('complex_shape','random_forest'),('nuisance','rbf_svc')]

def save_csv(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def metrics(protocol,fold,feature,method,level,yy,pp,rpm,dist):
    rows=[]
    groups=[('all','all',np.ones(len(yy),bool))]
    groups += [('rpm',str(r),rpm==r) for r in [1000,2000,3000]]
    groups += [('distance_cm',str(d),dist==d) for d in [20,40,80]]
    for group,value,ix in groups:
        if not ix.any():continue
        rows.append(dict(protocol=protocol,fold=fold,feature=feature,method=method,level=level,
                         subgroup=group,subgroup_value=value,n=int(ix.sum()),
                         accuracy=float(accuracy_score(yy[ix],pp[ix])),
                         macro_f1=float(f1_score(yy[ix],pp[ix],labels=np.arange(4),average='macro',zero_division=0)),
                         confusion_matrix=json.dumps(confusion_matrix(yy[ix],pp[ix],labels=np.arange(4)).tolist())))
    return rows

def main():
    declaration=dict(stage='developmental_diagnostic_added_after_primary_RPM_holdout_results',
                     purpose='separate within-condition discriminability and distance transfer from RPM transfer',
                     protocols=dict(recording_rep4='rep1/2/3 train, rep4 test; same speed/distance/session conditions remain in train',
                                    leave_one_distance='train other two distances; test one distance; all RPM values occur in train'),
                     no_merging_with_primary_LOSO=True,methods=METHODS,
                     hyperparameter_search='none; identical classifiers/scaler/record-aggregation as main baselines',
                     contact_data_usage='none; shared contact IDs retained only as provenance, not a model input',
                     interpretation_limit='pure radar condition-transfer diagnostics, not independent bearing/environment/campaign generalization; repeat numbers are timestamp rank',
                     weights='each training recording equal total weight; normalized mean sample weight 1',
                     window_predictions='standard classifier.predict',
                     recording_predictions='argmax arithmetic average of window scores; SVC softmax OVR decisions are uncalibrated')
    (OUT/'protocol.json').write_text(json.dumps(declaration,indent=2,ensure_ascii=False))
    with np.load(BASE/'radar/features.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    y=data['labels'];rpm=data['rpm'];dist=data['distance_cm'];rid=data['recording_id'];repeat=data['repeat'];cid=data['contact_id']
    tests=[('recording_rep4','rep4',repeat==4)]
    tests += [('leave_one_distance',str(d)+'cm',dist==d) for d in [20,40,80]]
    rows_w=[];rows_r=[];rows_m=[];splits=[]
    for protocol,fold,test in tests:
        train=~test
        assert not set(rid[train])&set(rid[test])
        splits.append(dict(protocol=protocol,fold=fold,train_recordings=sorted(set(rid[train])),test_recordings=sorted(set(rid[test])),
                           shared_contact_ids=sorted(set(cid[train])&set(cid[test])),
                           train_windows=int(train.sum()),test_windows=int(test.sum()),
                           explanation='shared contact IDs only document acquisition linkage; no contact data fed to these classifiers'))
        weights=record_equal_weights(rid[train])
        for feature,method in METHODS:
            x=data[feature];ss=StandardScaler().fit(x[train],sample_weight=weights)
            tx=ss.transform(x[train]);vx=ss.transform(x[test]);model=classifier(method)
            model.fit(tx,y[train],sample_weight=weights)
            p,d,kind=scores(model,vx);yp=model.predict(vx)
            rows_m.extend(metrics(protocol,fold,feature,method,'window',y[test],yp,rpm[test],dist[test]))
            for i,ii in enumerate(np.flatnonzero(test)):
                r=dict(protocol=protocol,fold=fold,feature=feature,method=method,window_index=int(ii),recording_id=rid[ii],
                       contact_id=cid[ii],rpm=int(rpm[ii]),distance_cm=int(dist[ii]),repeat=int(repeat[ii]),
                       y_true=int(y[ii]),y_pred=int(yp[i]),score_kind=kind)
                r.update({'score_'+la:float(p[i,k]) for k,la in enumerate(LABELS)})
                rows_w.append(r)
            ty=[];tp=[];trpm=[];td=[]
            for name in sorted(set(rid[test])):
                ix=rid[test]==name;yy=np.unique(y[test][ix]);assert len(yy)==1
                prob=p[ix].mean(0)
                r=dict(protocol=protocol,fold=fold,feature=feature,method=method,recording_id=name,contact_id=cid[test][ix][0],
                       rpm=int(rpm[test][ix][0]),distance_cm=int(dist[test][ix][0]),repeat=int(repeat[test][ix][0]),
                       n_windows=int(ix.sum()),y_true=int(yy[0]),y_pred=int(prob.argmax()),score_kind=kind)
                r.update({'score_'+la:float(prob[k]) for k,la in enumerate(LABELS)})
                rows_r.append(r);ty.append(r['y_true']);tp.append(r['y_pred']);trpm.append(r['rpm']);td.append(r['distance_cm'])
            rows_m.extend(metrics(protocol,fold,feature,method,'recording',np.array(ty),np.array(tp),np.array(trpm),np.array(td)))
            print(protocol,fold,feature,method,'window',accuracy_score(y[test],yp),'record',accuracy_score(ty,tp),flush=True)
    # Pool only the three disjoint distance folds; never average them with rep4 or RPM holdout.
    for source,level in [(rows_w,'window'),(rows_r,'recording')]:
        for feature,method in METHODS:
            rr=[r for r in source if r['protocol']=='leave_one_distance' and r['feature']==feature and r['method']==method]
            rows_m.extend(metrics('leave_one_distance','pooled_three_distances',feature,method,level,
                                  np.array([r['y_true'] for r in rr]),np.array([r['y_pred'] for r in rr]),
                                  np.array([r['rpm'] for r in rr]),np.array([r['distance_cm'] for r in rr])))
    save_csv('window_predictions.csv',rows_w);save_csv('recording_predictions.csv',rows_r);save_csv('metrics.csv',rows_m)
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2,ensure_ascii=False))
    (OUT/'evidence.json').write_text(json.dumps(dict(feature_sha256=sha256(BASE/'radar/features.npz'),
                                                   script_sha256=sha256(Path(__file__)),protocol_sha256=sha256(OUT/'protocol.json'),
                                                   training_radar_ids_disjoint_from_test=True,contact_data_never_loaded=True),indent=2))
    print('DONE',flush=True)

if __name__=='__main__':main()
