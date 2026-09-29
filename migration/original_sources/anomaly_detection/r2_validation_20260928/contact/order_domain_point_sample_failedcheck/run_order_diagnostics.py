#!/usr/bin/env python3
"""Development-only order-domain iteration after the primary evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,hilbert,periodogram
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

OUT=Path(__file__).resolve().parent
CONTACT=OUT.parent
FS=4000
GRID=np.linspace(.5,20.,128)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def weights(session):
    names,counts=np.unique(session,return_counts=True)
    count=dict(zip(names,counts))
    return np.array([len(session)/(len(names)*count[s]) for s in session])
def metrics(y,p):
    return dict(n=len(y),correct=int(np.sum(np.asarray(y)==np.asarray(p))),
        accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y,p)),confusion_matrix=confusion_matrix(y,p,labels=range(4)).tolist())
def axis_spectrum(x):
    x=x-x.mean(1,keepdims=True)
    rms=np.sqrt(np.mean(x*x,1,keepdims=True))
    xn=x/np.maximum(rms,1e-12)
    f,p=periodogram(xn,fs=FS,window='hann',detrend=False,scaling='density',axis=1)
    return f,p,int(np.sum(rms<1e-12))
def to_order(f,p,rpm,axes):
    # Change of density variable: S_order(o)=f_rot*S_hz(f_rot*o).
    # Linear interpolation changes coordinates, not information or resolution.
    p=p[:,:,axes].mean(-1)
    out=[]
    for row,rr in zip(p,rpm):
        fr=float(rr)/60.
        wanted=GRID*fr
        assert wanted[0]>=f[0] and wanted[-1]<=f[-1]
        s=np.interp(wanted,f,row)*fr
        z=np.log(np.maximum(s,1e-20))
        out.append(z-z.mean())
    return np.stack(out)

def main():
    assert 'envs/m2vllm' in sys.executable
    if (OUT/'protocol_before_results.json').exists():raise FileExistsError('Development protocol/results are immutable')
    OUT.mkdir(parents=True,exist_ok=True)
    source_npz=CONTACT/'raw_contact_windows.npz'
    source_teacher=CONTACT/'contact_features.npz'
    methods=[f'{axis}_{feature}_{model}' for axis in ['xyz','xy'] for feature in ['raw_order','envelope_order','concat_order'] for model in ['LogReg_C1','RBF_SVM_C1']]
    protocol=dict(
        status='development_iteration_after_observing_primary_results_not_a_new_blind_test',
        motivating_observation='Primary contact raw PSD LOSO 41.69% versus within-session 99.16%; fixed front ends test whether rotor-frequency shifts explain part of the gap',
        primary_results_not_replaced=True,no_bear_weight_or_preprocessing_modification=True,
        source_retention_not_certified=True,source_accuracy_not_evaluated=True,
        prior='true recorded RPM supplied for every training and test packet; deployment would need known RPM or separately validated RPM estimation',
        raw_input='all 343 one-second packets at4000Hz, no clipping deletion or score-based filtering',
        axis_rule='XYZ main and XY uniform sensitivity, all classes treated identically',
        amplitude_rule='each packet and axis demeaned then divided by its own RMS; no population statistics fitted before split',
        raw_spectrum='1s Hann periodogram, nfft=4000, 1Hz Fourier grid; Hann equivalent noise bandwidth1.5Hz and broader spectral response',
        envelope='fourth-order Butterworth bandpass500-1800Hz designed by scipy.signal.butter(4), forward-backward SOS filtering, Hilbert envelope, remove400 samples each edge, demean and axiswise RMS normalize remaining0.8s',
        envelope_spectrum='0.8s Hann periodogram, nfft=3200, 1.25Hz Fourier grid; no claim of sub-grid information',
        order_coordinate='128 equally spaced points inclusive0.5..20 shaft orders; f_rot=rpm/60; linear PSD interpolation at f=order*f_rot, density multiplied by f_rot; log, then subtract per-window feature-block mean',
        order_resolution='raw Fourier-grid orders=1/f_rot; envelope=1.25/f_rot; interpolated128-point grid does not improve physical resolution',
        concatenation='concatenate raw128 and envelope128 after their independent per-window log-centering',
        classifier='fold-only session-weighted StandardScaler then LogisticRegression C1 max_iter3000 or RBF SVC C1 gamma=scale; no tuning',
        split='three leave-entire-RPM-out folds; four held-out contact sessions each; every packet tested exactly once per method',
        session_pooling='LogReg mean window probabilities; SVM hard-class vote fractions',
        configurations=methods,configuration_count=12,model_fit_count=36,
        input_hashes={str(source_npz):sha(source_npz),str(source_teacher):sha(source_teacher)},
        script_sha256=sha(__file__))
    writej(OUT/'protocol_before_results.json',protocol)
    started=time.monotonic()
    raw=np.load(source_npz,allow_pickle=False)['raw_xyz'].astype(float)
    z=np.load(source_teacher,allow_pickle=False)
    y,rpm,sid=z['labels'],z['rpm'],z['session_id']
    md=pd.read_csv(CONTACT/'metadata.csv')
    assert raw.shape==(343,4000,3) and np.array_equal(md.session_id,sid)
    f,p,nzero=axis_spectrum(raw)
    sos=butter(4,[500,1800],fs=FS,btype='bandpass',output='sos')
    filtered=sosfiltfilt(sos,raw-raw.mean(1,keepdims=True),axis=1)
    envelope=np.abs(hilbert(filtered,axis=1))[:,400:-400]
    ef,ep,ezero=axis_spectrum(envelope)
    feats={}
    for axis,axes in [('xyz',[0,1,2]),('xy',[0,1])]:
        feats[f'{axis}_raw_order']=to_order(f,p,rpm,axes)
        feats[f'{axis}_envelope_order']=to_order(ef,ep,rpm,axes)
        feats[f'{axis}_concat_order']=np.concatenate([feats[f'{axis}_raw_order'],feats[f'{axis}_envelope_order']],axis=1)
    assert all(np.isfinite(v).all() for v in feats.values())
    np.savez_compressed(OUT/'order_features.npz',**{k:v.astype(np.float32) for k,v in feats.items()},
        order_grid=GRID,labels=y,rpm=rpm,session_id=sid,packet_index=z['packet_index'],
        raw_frequency_grid_hz=f,envelope_frequency_grid_hz=ef,source_uses_true_rpm=np.array(True))
    extraction_elapsed=time.monotonic()-started
    rows,packets,sessions,checks=[],[],[],[]
    for holdout in (1000,2000,3000):
        train,test=rpm!=holdout,rpm==holdout
        sw=weights(sid[train])
        assert set(sid[train]).isdisjoint(sid[test])
        for feat,x in feats.items():
            sc=StandardScaler().fit(x[train],sample_weight=sw)
            xtr,xte=sc.transform(x[train]),sc.transform(x[test])
            for model_name in ['LogReg_C1','RBF_SVM_C1']:
                begin=time.monotonic();name=f'{feat}_{model_name}'
                if model_name=='LogReg_C1':
                    model=LogisticRegression(C=1.,max_iter=3000,random_state=42,solver='lbfgs')
                    model.fit(xtr,y[train],sample_weight=sw)
                    score=model.predict_proba(xte);pred=score.argmax(-1);pool='mean_window_probabilities'
                else:
                    model=SVC(C=1.,gamma='scale',kernel='rbf')
                    model.fit(xtr,y[train],sample_weight=sw)
                    pred=model.predict(xte);score=np.eye(4)[pred];pool='window_hard_vote_fraction'
                runtime=time.monotonic()-begin
                item=md[test].copy();item['holdout_rpm']=holdout;item['method']=name;item['prediction']=pred;item['pooling']=pool
                for k in range(4):item[f'score{k}']=score[:,k]
                packets.append(item)
                sm=item.groupby(['session_id','label','rpm'],as_index=False)[[f'score{k}' for k in range(4)]].mean()
                sm['prediction']=sm[[f'score{k}' for k in range(4)]].to_numpy().argmax(-1)
                sm['holdout_rpm']=holdout;sm['method']=name;sm['pooling']=pool;sessions.append(sm)
                rows.append(dict(holdout_rpm=holdout,method=name,level='packet',**metrics(y[test],pred)))
                rows.append(dict(holdout_rpm=holdout,method=name,level='session',**metrics(sm.label,sm.prediction)))
                checks.append(dict(holdout_rpm=holdout,method=name,train_sessions=np.unique(sid[train]).tolist(),
                    test_sessions=np.unique(sid[test]).tolist(),training_n=int(train.sum()),test_n=int(test.sum()),
                    fit_predict_seconds=runtime,scaler_train_only=True,source_retention_certified=False))
    packet=pd.concat(packets);session=pd.concat(sessions)
    for method in methods:
        d=packet[packet.method==method];s=session[session.method==method]
        assert len(d)==343 and len(s)==12 and not d.duplicated(['session_id','packet_index']).any()
        rows.append(dict(holdout_rpm='pooled_oof',method=method,level='packet',**metrics(d.label,d.prediction)))
        rows.append(dict(holdout_rpm='pooled_oof',method=method,level='session',**metrics(s.label,s.prediction)))
    scores=pd.DataFrame(rows)
    scores.to_csv(OUT/'metrics.csv',index=False)
    scores[scores.holdout_rpm=='pooled_oof'].to_csv(OUT/'summary.csv',index=False)
    packet.to_csv(OUT/'packet_predictions.csv',index=False);session.to_csv(OUT/'session_predictions.csv',index=False)
    assert all(sha(path)==value for path,value in protocol['input_hashes'].items())
    writej(OUT/'verification.json',dict(ok=True,inputs_unchanged=True,bear_model_unchanged=True,
        no_test_hyperparameter_selection=True,development_after_primary_disclosed=True,model_checks=checks,
        feature_extraction_seconds=extraction_elapsed,total_seconds=time.monotonic()-started,
        raw_zero_energy_axes=nzero,envelope_zero_energy_axes=ezero,all_features_finite=True,
        feature_sha256=sha(OUT/'order_features.npz')))
    print(scores[scores.holdout_rpm=='pooled_oof'].to_string(index=False))

if __name__=='__main__':main()
