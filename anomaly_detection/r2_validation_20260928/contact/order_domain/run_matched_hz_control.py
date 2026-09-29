#!/usr/bin/env python3
"""Two fixed-Hz controls matched to the corrected order-bin pipeline."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
import time,json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,hilbert
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from run_order_diagnostics import OUT,CONTACT,FS,EDGES,axis_spectrum,to_order,sha,writej,weights,metrics

def main():
    proto=OUT/'hz_control_protocol_before_results.json'
    if proto.exists():raise FileExistsError('Do not overwrite matched-Hz controls')
    source=CONTACT/'raw_contact_windows.npz';teacher=CONTACT/'contact_features.npz'
    writej(proto,dict(status='development_after_primary_and_order_results_not_blind_test',
        purpose='two controlled configurations: original+envelope concatenation with XYZ and XY, RBF SVC C1 gamma scale',
        change='constant reference RPM2000 for every packet solely to define frequency-bin edges: 128 fixed Hz bands spanning16.6666667..666.6666667Hz; actual RPM still used to define held-out folds',
        everything_else='identical axiswise per-window RMS, periodogram,500-1800Hz envelope filter,Hilbert,0.1s edge trim,128 bin-energy integration,log-centering,scaler,session weights,classifier,folds',
        comparison_caveat='order-dependent physical frequency coverage varies with trueRPM; fixed-Hz coverage cannot be identical at all RPM, and this is part of the tested coordinate normalization',
        source_retention_certified=False,input_sha256={str(source):sha(source),str(teacher):sha(teacher)},script_sha256=sha(__file__)))
    start=time.monotonic()
    raw=np.load(source,allow_pickle=False)['raw_xyz'].astype(float)
    z=np.load(teacher,allow_pickle=False);y,rpm,sid=z['labels'],z['rpm'],z['session_id']
    md=pd.read_csv(CONTACT/'metadata.csv')
    f,p,_=axis_spectrum(raw)
    sos=butter(4,[500,1800],fs=FS,btype='bandpass',output='sos')
    env=np.abs(hilbert(sosfiltfilt(sos,raw-raw.mean(1,keepdims=True),axis=1),axis=1))[:,400:-400]
    ef,ep,_=axis_spectrum(env)
    fixed=np.full(len(raw),2000)
    feats={axis:np.concatenate([to_order(f,p,fixed,axes),to_order(ef,ep,fixed,axes)],1)
           for axis,axes in [('xyz',[0,1,2]),('xy',[0,1])]}
    np.savez_compressed(OUT/'hz_control_features.npz',**{f'{k}_concat_fixed_hz':v.astype(np.float32) for k,v in feats.items()},
        fixed_frequency_edges_hz=EDGES*(2000/60),labels=y,rpm=rpm,session_id=sid,packet_index=z['packet_index'])
    rows,packets,sessions,checks=[],[],[],[]
    for holdout in (1000,2000,3000):
        tr,te=rpm!=holdout,rpm==holdout;sw=weights(sid[tr])
        for axis,x in feats.items():
            name=f'{axis}_concat_fixed_hz_RBF_SVM_C1'
            sc=StandardScaler().fit(x[tr],sample_weight=sw)
            model=SVC(C=1.,gamma='scale',kernel='rbf').fit(sc.transform(x[tr]),y[tr],sample_weight=sw)
            pred=model.predict(sc.transform(x[te]));score=np.eye(4)[pred]
            dd=md[te].copy();dd['holdout_rpm']=holdout;dd['method']=name;dd['prediction']=pred
            for k in range(4):dd[f'score{k}']=score[:,k]
            packets.append(dd)
            sm=dd.groupby(['session_id','label','rpm'],as_index=False)[[f'score{k}' for k in range(4)]].mean()
            sm['prediction']=sm[[f'score{k}' for k in range(4)]].to_numpy().argmax(-1)
            sm['holdout_rpm']=holdout;sm['method']=name;sessions.append(sm)
            rows.append(dict(holdout_rpm=holdout,method=name,level='packet',**metrics(y[te],pred)))
            rows.append(dict(holdout_rpm=holdout,method=name,level='session',**metrics(sm.label,sm.prediction)))
            checks.append(dict(holdout_rpm=holdout,method=name,train_sessions=np.unique(sid[tr]).tolist(),
                test_sessions=np.unique(sid[te]).tolist(),source_retention_certified=False))
    pk=pd.concat(packets);ss=pd.concat(sessions)
    for method in pk.method.unique():
        pp=pk[pk.method==method];s=ss[ss.method==method]
        assert len(pp)==343 and len(s)==12
        rows.append(dict(holdout_rpm='pooled_oof',method=method,level='packet',**metrics(pp.label,pp.prediction)))
        rows.append(dict(holdout_rpm='pooled_oof',method=method,level='session',**metrics(s.label,s.prediction)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'hz_control_metrics.csv',index=False)
    result[result.holdout_rpm=='pooled_oof'].to_csv(OUT/'hz_control_summary.csv',index=False)
    pk.to_csv(OUT/'hz_control_packet_predictions.csv',index=False)
    ss.to_csv(OUT/'hz_control_session_predictions.csv',index=False)
    assert all(sha(path)==h for path,h in json.loads(proto.read_text())['input_sha256'].items())
    writej(OUT/'hz_control_verification.json',dict(ok=True,inputs_unchanged=True,checks=checks,total_seconds=time.monotonic()-start,
        configurations=2,model_fits=6,fixed_hyperparameters=True,no_test_tuning=True))
    print(result[result.holdout_rpm=='pooled_oof'].to_string(index=False))

if __name__=='__main__':main()
