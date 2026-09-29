#!/usr/bin/env python3
"""Fixed-hyperparameter contact diagnostics; no replacement of retained head."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '2'
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

HERE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,o): Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def weights(session):
    names, counts = np.unique(session, return_counts=True)
    size = dict(zip(names,counts))
    return np.array([len(session)/(len(names)*size[s]) for s in session])
def metrics(y,p):
    return dict(n=len(y),correct=int(np.sum(np.asarray(y)==np.asarray(p))),
        accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,p)),confusion_matrix=confusion_matrix(y,p,labels=range(4)).tolist())
def spectrum(raw, axes):
    x = raw[:,:,axes].astype(np.float64)
    x -= x.mean(1,keepdims=True)
    f,p = welch(x,fs=4000,window='hann',nperseg=1024,noverlap=512,axis=1,detrend=False,scaling='density')
    p = p.mean(-1)
    edges = np.linspace(20,1800,129)
    feat = []
    for lo,hi in zip(edges[:-1],edges[1:]):
        ids = (f>=lo)&(f<hi)
        assert ids.any()
        feat.append(np.log(np.maximum(p[:,ids].sum(-1)*(f[1]-f[0]),1e-20)))
    feat = np.stack(feat,-1)
    feat -= feat.mean(-1,keepdims=True)
    return feat,edges

def main():
    assert 'envs/m2vllm' in sys.executable
    out = HERE/'diagnostic_heads'
    if out.exists(): raise FileExistsError('Existing diagnostic heads are immutable')
    out.mkdir()
    z = np.load(HERE/'contact_features.npz',allow_pickle=False)
    raw = np.load(HERE/'raw_contact_windows.npz',allow_pickle=False)['raw_xyz']
    x_psd,edges = spectrum(raw,[0,1,2])
    x_psd_xy,_ = spectrum(raw,[0,1])
    md = pd.read_csv(HERE/'metadata.csv')
    y,rpm,sid = z['labels'],z['rpm'],z['session_id']
    assert np.array_equal(md.session_id,sid) and np.array_equal(md.label,y)
    datasets = dict(Bear128=z['embedding'],Bear128_XY_sensitivity=z['embedding_xy_sensitivity'],
                    ContactPSD128=x_psd,ContactPSD128_XY_sensitivity=x_psd_xy,
                    Physical6_RBF_SVM=z['physical_targets'])
    writej(out/'protocol_before_results.json',dict(
        task='diagnostic_only_not_source_accuracy_retention_accepted',
        train_test='leave one entire RPM out; 8 contact sessions train and 4 unseen sessions test in each fold; no random window split',
        head='StandardScaler weighted by session, LogisticRegression C=1 max_iter=3000, all fixed before results',
        physical_baseline='same train-session-weighted StandardScaler then RBF SVC C=1 gamma=scale; fixed before results',
        signal_baseline='Welch fs4000, Hann1024, overlap512, demean, mean XYZ axis PSD; log integrated PSD in 128 equal-width bands 20-1800Hz; subtract per-window mean logpower',
        axis_sensitivity='XY removes Z in every class and fold, motivated by pre-inference clipping audit',
        source_acc_retention='unknown for these new diagnostic heads; they must not replace the previously validated retained head',
        no_hyperparameter_selection_on_test=True,existing_checkpoints_modified=False,
        feature_input_sha256=sha(HERE/'contact_features.npz'),raw_input_sha256=sha(HERE/'raw_contact_windows.npz'),
        script_sha256=sha(__file__)))
    np.savez_compressed(HERE/'diagnostic_contact_psd.npz',psd_shape=x_psd,psd_shape_xy=x_psd_xy,band_edges_hz=edges,
                        labels=y,rpm=rpm,session_id=sid,packet_index=z['packet_index'])
    metrics_rows,packets,sessions,checks=[],[],[],[]
    for holdout in sorted(np.unique(rpm)):
        train,test=rpm!=holdout,rpm==holdout
        assert set(sid[train]).isdisjoint(sid[test])
        sw=weights(sid[train])
        for name,xx in datasets.items():
            xx = xx.astype(np.float64)
            sc=StandardScaler().fit(xx[train],sample_weight=sw)
            xtr,xte=sc.transform(xx[train]),sc.transform(xx[test])
            if name=='Physical6_RBF_SVM':
                model=SVC(C=1.,gamma='scale',kernel='rbf',decision_function_shape='ovr')
                model.fit(xtr,y[train],sample_weight=sw)
                score=model.decision_function(xte)
                # SVC's argmax ovrs decision scores may differ from predict when
                # votes tie; use native predict for window metrics and vote
                # fractions for session pooling. No fictitious calibrated p.
                pred=model.predict(xte)
                pooling=np.eye(4)[pred]
                pooling_type='fraction_of_window_hard_votes'
            else:
                model=LogisticRegression(C=1.,max_iter=3000,solver='lbfgs',random_state=42)
                model.fit(xtr,y[train],sample_weight=sw)
                pooling=model.predict_proba(xte)
                pred=pooling.argmax(-1)
                pooling_type='mean_window_probabilities'
                w=model.coef_/sc.scale_[None,:]
                b=model.intercept_-w@sc.mean_
                err=float(np.max(abs((xx[test]@w.T+b)-model.decision_function(xte))))
                assert err<1e-8
                if name=='Bear128':
                    np.savez_compressed(out/f'holdout_{holdout}.npz',weight4=w.astype(np.float32),bias4=b.astype(np.float32),
                        scaler_mean=sc.mean_,scaler_scale=sc.scale_,classes=model.classes_,
                        train_sessions=np.unique(sid[train]),test_sessions=np.unique(sid[test]),heldout_rpm=holdout,
                        accepted_for_source_retention=np.array(False),native_feature_key=np.array('embedding'))
                checks.append(dict(holdout_rpm=int(holdout),model=name,folded_raw_linear_max_error=err,
                    train_sessions=np.unique(sid[train]).tolist(),test_sessions=np.unique(sid[test]).tolist()))
            item=md[test].copy()
            item['holdout_rpm'],item['model'],item['prediction']=holdout,name,pred
            item['pooling_type']=pooling_type
            for k in range(4):item[f'score{k}']=pooling[:,k]
            packets.append(item)
            sm=item.groupby(['session_id','label','rpm'],as_index=False)[[f'score{k}' for k in range(4)]].mean()
            sm['prediction']=sm[[f'score{k}' for k in range(4)]].to_numpy().argmax(-1)
            sm['holdout_rpm'],sm['model'],sm['pooling_type']=holdout,name,pooling_type
            sessions.append(sm)
            metrics_rows.append(dict(holdout_rpm=int(holdout),model=name,level='packet',**metrics(y[test],pred)))
            metrics_rows.append(dict(holdout_rpm=int(holdout),model=name,level='session',**metrics(sm.label,sm.prediction)))
    packet=pd.concat(packets);session=pd.concat(sessions)
    for name in datasets:
        p=packet[packet.model==name];s=session[session.model==name]
        metrics_rows.append(dict(holdout_rpm='pooled_oof',model=name,level='packet',**metrics(p.label,p.prediction)))
        metrics_rows.append(dict(holdout_rpm='pooled_oof',model=name,level='session',**metrics(s.label,s.prediction)))
    pd.DataFrame(metrics_rows).to_csv(out/'metrics.csv',index=False)
    packet.to_csv(out/'packet_predictions.csv',index=False)
    session.to_csv(out/'session_predictions.csv',index=False)
    writej(out/'verification.json',dict(checks=checks,source_retention_not_evaluated=True,
        all_folds_have_disjoint_contact_sessions=True,fixed_hyperparameters_no_test_tuning=True,
        source_feature_hash_unchanged=sha(HERE/'contact_features.npz')==json.loads((out/'protocol_before_results.json').read_text())['feature_input_sha256']))
    print(pd.DataFrame(metrics_rows).query('holdout_rpm == "pooled_oof"').to_string(index=False))

if __name__=='__main__':main()
