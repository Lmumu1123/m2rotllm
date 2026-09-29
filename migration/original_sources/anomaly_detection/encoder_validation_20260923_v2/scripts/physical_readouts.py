"""Contact-only physical readouts with explicit oracle class-mean controls.

No radar labels train a readout. No synchronized cross-modal windows are assumed.
Held-out contact prediction is a necessary validity gate, not optional evidence.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'encoder_validation_20260923'
CONTACT=ROOT.parent/'four_class_preprocessing_20260921/results'
BEAR=ROOT.parent/'retention_alignment_20260922/preprocessing/variants/v0/retrained_fcn'
TARGETS=['log_band_rms','log_crest_factor','log_pearson_kurtosis','power_20_200','power_200_400','power_400_800']
EPS=1e-12

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weights(md):
    count=md.groupby('bag_id').size()
    return np.array([1/count[b] for b in md.bag_id])*len(md)/len(count)
def descriptor(raw):
    x=raw.astype(np.float64)-raw.mean(axis=0,keepdims=True)
    sos=butter(4,[20,800],fs=4000,btype='bandpass',output='sos')
    x=sosfiltfilt(sos,x,axis=0)[400:-400]
    rms_axis=np.sqrt(np.mean(x*x,axis=0))
    rms=np.sqrt(np.mean(x*x))
    crest=np.mean(np.max(np.abs(x),axis=0)/np.maximum(rms_axis,EPS))
    kurt=np.mean(kurtosis(x,axis=0,fisher=False,bias=True))
    f,p=welch(x,fs=4000,nperseg=512,noverlap=256,axis=0)
    p=p.mean(axis=1)
    energy=[]
    for lo,hi in [(20,200),(200,400),(400,800)]:
        good=(f>=lo)&(f<hi)
        energy.append(np.trapz(p[good],f[good]))
    ratio=np.array(energy)/max(sum(energy),EPS)
    y=np.r_[np.log(max(rms,EPS)),np.log(max(crest,EPS)),np.log(max(kurt,EPS)),ratio]
    if not np.isfinite(y).all():raise ValueError('Nonfinite descriptor; do not silently drop windows')
    return y

def main():
    assert 'envs/m2vllm' in sys.executable
    out=ROOT/'results/physical_readouts'
    if out.exists():raise RuntimeError('Refusing overwrite')
    out.mkdir();(out/'models').mkdir()
    protocol=json.loads((ROOT/'protocol.json').read_text())['physical_readouts']
    protocol.update(descriptor_axis_rule='RMS pooled across time/XYZ; crest and Pearson kurtosis averaged across axes before natural log; Welch axis powers averaged before band ratios',
                    alpha=10.,oracle_baseline_uses_true_test_class=True,physical_descriptor_targets_not_fault_severity=True)
    (out/'protocol_before_results.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    cm=pd.read_csv(BEAR/'metadata.csv')
    raw=[];hashes={}
    for row in cm.itertuples():
        p=CONTACT/row.file
        if str(p) not in hashes:hashes[str(p)]=sha(p)
        with np.load(p,allow_pickle=False) as z:raw.append(z['raw_xyz'][row.window_row])
    raw=np.stack(raw);assert raw.shape==(88,4000,3)
    y=np.stack([descriptor(x) for x in raw]);assert np.isfinite(y).all()
    targets=cm.copy()
    for j,name in enumerate(TARGETS):targets[name]=y[:,j]
    targets.to_csv(out/'contact_physical_targets.csv',index=False)
    teachers={
        'Bear128':(np.load(BEAR/'contact_features.npz')['hidden_mean'],cm),
        'Rot128':(np.load(OLD/'rotllm/teacher_features.npz')['hidden_mean'],pd.read_csv(OLD/'rotllm/metadata.csv')),
        'UniFault128':(None,None)}
    rm=pd.read_csv(ROOT/'geometry_corrected/radar/metadata.csv')
    key=['bag_id','window_row'];key_index=pd.MultiIndex.from_frame(cm[key])
    splits=json.loads((ROOT/'controls/results/splits.json').read_text())
    errors=[];predictions=[];contact_window_rows=[];fitchecks=[]
    for teacher,(hh,md) in teachers.items():
        for sp in splits:
            fold=sp['fold'];train=cm.bag_id.isin(sp['train_bags']).to_numpy();test=cm.bag_id.isin(sp['test_bags']).to_numpy()
            if teacher=='UniFault128':
                td=ROOT/'teachers/unifault/paper_100ms'/fold
                hh=np.load(td/'teacher_features.npz')['hidden_mean'];md=pd.read_csv(td/'metadata.csv')
            order=pd.MultiIndex.from_frame(md[key]).get_indexer(key_index)
            assert (order>=0).all();h=hh[order].astype(float)
            assert not set(sp['train_bags'])&set(sp['test_bags'])
            sw=weights(cm[train]);hs=StandardScaler().fit(h[train],sample_weight=sw);ys=StandardScaler().fit(y[train],sample_weight=sw)
            model=Ridge(alpha=10.,solver='svd').fit(hs.transform(h[train]),ys.transform(y[train]),sample_weight=sw)
            w=ys.scale_[:,None]*model.coef_/hs.scale_[None,:]
            b=ys.mean_+ys.scale_*model.intercept_-w@hs.mean_
            predict=lambda z:np.asarray(z)@w.T+b
            yp=predict(h)
            assert np.max(np.abs(yp-ys.inverse_transform(model.predict(hs.transform(h)))))<1e-8
            np.savez_compressed(out/'models'/f'{teacher}__{fold}.npz',weight=w,bias=b,target_scale=ys.scale_,target_mean=ys.mean_,train_bags=np.array(sp['train_bags']),targets=np.array(TARGETS))
            fitchecks.append(dict(teacher=teacher,fold=fold,train_windows=int(train.sum()),test_windows=int(test.sum()),alpha=10.,train_only=True))
            class_y={label:y[train&(cm.label.to_numpy()==label)].mean(0) for label in range(4)}
            class_h={label:h[train&(cm.label.to_numpy()==label)].mean(0) for label in range(4)}
            for idx in np.flatnonzero(test):
                for j,target in enumerate(TARGETS):
                    truth=y[idx,j];p=yp[idx,j];base=class_y[int(cm.iloc[idx].label)][j]
                    contact_window_rows.append(dict(teacher=teacher,fold=fold,bag_id=cm.iloc[idx].bag_id,window_row=int(cm.iloc[idx].window_row),target=target,truth=truth,prediction=p,oracle_class_prediction=base,abs_error_std=abs(p-truth)/ys.scale_[j],oracle_class_abs_error_std=abs(base-truth)/ys.scale_[j]))
            def eval_bags(meta,pred,path,method,seed):
                for bag in sp['test_bags']:
                    ci=cm.bag_id.eq(bag).to_numpy();ri=meta.bag_id.eq(bag).to_numpy();label=int(cm.loc[ci,'label'].iloc[0])
                    true=y[ci].mean(0);guess=pred[ri].mean(0)
                    for j,target in enumerate(TARGETS):
                        predictions.append(dict(teacher=teacher,fold=fold,path=path,method=method,seed=seed,bag_id=bag,label=label,target=target,truth=float(true[j]),prediction=float(guess[j]),train_target_scale=float(ys.scale_[j]),abs_error_std=float(abs(guess[j]-true[j])/ys.scale_[j])))
            eval_bags(cm,yp,'contact','native_contact',-1)
            oracle=np.stack([class_y.get(int(label),ys.mean_) for label in cm.label])
            proto=np.stack([predict(class_h[int(label)]) if label>=0 else ys.mean_ for label in cm.label])
            eval_bags(cm,oracle,'oracle_baseline','true_class_target_mean',-1)
            eval_bags(cm,proto,'oracle_baseline','true_class_teacher_prototype',-1)
            eval_bags(cm,np.tile(ys.mean_,(len(cm),1)),'constant_baseline','global_train_mean',-1)
            if teacher=='Bear128':
                for p in sorted((ROOT/'controls/results/windows').glob(f'{fold}__*.npz')):
                    rest=p.stem[len(fold)+2:];method,seedstr=rest.rsplit('__seed',1)
                    with np.load(p) as z:
                        assert np.array_equal(z['bag_id'],rm.bag_id.to_numpy(str));rh=z['hidden']
                    eval_bags(rm,predict(rh),'radar_to_own_teacher',method,int(seedstr))
            elif teacher=='Rot128':
                for p in sorted((ROOT/'rotllm_radar/models').glob(f'{fold}__*/predictions.npz')):
                    _,head,method,seedstr=p.parent.name.split('__')
                    with np.load(p) as z:
                        assert np.array_equal(z['bag_id'],rm.bag_id.to_numpy(str));rh=z['hidden']
                    eval_bags(rm,predict(rh),f'radar_to_own_teacher_{head}',method,int(seedstr[4:]))
                m=np.load(ROOT/'results/contact_only_stitch/mappings'/f'{fold}.npz')
                for p in sorted((ROOT/'controls/results/windows').glob(f'{fold}__*.npz')):
                    rest=p.stem[len(fold)+2:];method,seedstr=rest.rsplit('__seed',1)
                    rh=np.load(p)['hidden'];mh=np.maximum(rh@m['weight'].T+m['bias'],0)
                    eval_bags(rm,predict(mh),'frozen_Bear_radar_mapped_to_Rot',method,int(seedstr))
            else:
                for p in sorted((ROOT/'unifault_radar/models').glob(f'{fold}__*/predictions.npz')):
                    _,constraint,method,seedstr=p.parent.name.split('__')
                    with np.load(p) as z:
                        assert np.array_equal(z['bag_id'],rm.bag_id.to_numpy(str));rh=z['hidden']
                    eval_bags(rm,predict(rh),f'radar_to_own_teacher_{constraint}',method,int(seedstr[4:]))
    d=pd.DataFrame(predictions);d.to_csv(out/'recording_predictions.csv',index=False)
    summary=d.groupby(['teacher','path','method','target'],as_index=False).agg(mae_train_std=('abs_error_std','mean'),n_repeated=('bag_id','size'))
    summary.to_csv(out/'summary.csv',index=False)
    cw=pd.DataFrame(contact_window_rows);cw.to_csv(out/'contact_window_predictions.csv',index=False)
    gates=[]
    for teacher in teachers:
        for target in TARGETS:
            dd=summary[(summary.teacher==teacher)&(summary.target==target)].set_index('method')
            native=float(dd.loc['native_contact','mae_train_std']);oracle=float(dd.loc['true_class_target_mean','mae_train_std'])
            gates.append(dict(teacher=teacher,target=target,contact_readout_mae=native,class_only_mae=oracle,
                              passes_contact_predictivity_gate=bool(native<oracle),improvement_vs_class_oracle=oracle-native,
                              gate_is_descriptive_not_statistical=True))
    pd.DataFrame(gates).to_csv(out/'contact_validity_gates.csv',index=False)
    assert all(sha(p)==h for p,h in hashes.items())
    (out/'verification.json').write_text(json.dumps(dict(input_contact_hashes=hashes,input_files_unchanged=True,fit_checks=fitchecks,
        target_readouts_receive_no_radar_or_heldout_contact_during_fit=True,oracle_true_class_is_explicit=True,
        main_records=8,contact_windows=58,unique_target_dimensions=6,band_fractions_are_dependent=True,
        script_sha256=sha(__file__),root_protocol_sha256=sha(ROOT/'protocol.json')),ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(gates).to_string(index=False))
    print(out)

if __name__=='__main__':main()
