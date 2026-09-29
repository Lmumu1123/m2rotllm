"""Freeze existing radar students; calibrate BearLLM->RotLLM using contact only.

Fixed alpha and preprocessing, all methods/seeds reported. Previously seen development
recordings, not a blind test. Original checkpoints and arrays are never modified.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
import pandas as pd
from scipy.special import softmax, logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
BEAR = Path('/home/huangyating/anomaly_detection/retention_alignment_20260922/preprocessing/variants/v0/retrained_fcn')
ROT = ROOT/'rotllm'
CONTROL = ROOT/'controls/results'
GROUPS = [[0], [1,2,3], [7,8,9], [4,5,6]]
METHODS = ['ce_only','matched_bag_mse','full','label_code_mse','label_code_ce_mse','wrong_class_mse']
SEEDS = [17,42,73]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savej(p, data): p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def balanced_weights(md):
    counts=md.groupby('bag_id').size()
    return np.array([1/counts[b] for b in md.bag_id])*len(md)/len(counts)

def head_probs(h, w, b, native):
    logits=h.astype(np.float64)@w.T+b
    if native:
        p15=softmax(logits,axis=1)
        l4=np.stack([logsumexp(logits[:,g],axis=1) for g in GROUPS],axis=1)
        return softmax(l4,axis=1),p15[:,10:].sum(1),(p15.argmax(1)>=10).astype(float)
    return softmax(logits,axis=1),np.zeros(len(h)),np.zeros(len(h))

def main():
    assert 'envs/m2vllm' in sys.executable
    out=ROOT/'results/contact_only_stitch'
    if out.exists(): raise RuntimeError('Refusing to overwrite stitching results')
    out.mkdir(parents=True)
    (out/'mappings').mkdir()
    start=time.time()
    protocol=dict(primary_target_variant='raw_rms001',
        map='contact training only: recording-balanced StandardScaler for both spaces, Ridge alpha=1',
        output_constraint='ReLU applied to mapped hidden because both native penultimate spaces are nonnegative',
        head_policies=['untouched native 15-way head with explicit bearing-conditioned output','separate contact-trained 4-way readout'],
        ridge_alpha=1.0,no_hyperparameter_or_method_selection=True,radar_encoder_retrained=False,
        map_uses_class_labels=False,map_uses_radar=False,
        new_four_class_head_used_local_contact_labels=True,
        states='all methods and seeds from BearLLM-only control training; no RotLLM feature or score informed their losses',
        aggregation='head(mean across contact axes) per short contact window, then mean probabilities over each recording; radar likewise per-window probabilities',
        caveats=['same previously inspected 8 recordings','outer radar ROI invalid','not a new-task test','contact readout has seen local training labels'])
    savej(out/'protocol_before_results.json',protocol)
    cm=pd.read_csv(BEAR/'metadata.csv'); bm=pd.read_csv(ROT/'metadata.csv')
    rm=pd.read_csv(CONTROL/'radar_window_metadata.csv')
    key=['bag_id','window_row']
    assert not cm.duplicated(key).any() and not bm.duplicated(key).any()
    a=np.load(BEAR/'contact_features.npz')['hidden_mean'].astype(np.float64)
    b=np.load(ROT/'teacher_features.npz')['hidden_mean'].astype(np.float64)
    order=pd.MultiIndex.from_frame(bm[key]).get_indexer(pd.MultiIndex.from_frame(cm[key]))
    assert (order>=0).all(); b=b[order]
    assert np.array_equal(cm.label,bm.iloc[order].label)
    native=np.load(ROT/'native_head15.npz')
    sources=[BEAR/'metadata.csv',BEAR/'contact_features.npz',ROT/'metadata.csv',ROT/'teacher_features.npz',
        ROT/'native_head15.npz',CONTROL/'radar_window_metadata.csv',CONTROL/'splits.json']
    hashes={str(p):sha(p) for p in sources}
    allmetrics=[]; bags=[]; checks=[]; correspondence=[]
    splits=json.loads((CONTROL/'splits.json').read_text())
    # Controls preserve the original stage_b split schema.
    for sp in splits:
        fold=sp['fold']; train_bags=sp['train_bags']; test_bags=sp['test_bags']
        if not test_bags: continue
        assert not set(train_bags)&set(test_bags)
        ctr=cm.bag_id.isin(train_bags).to_numpy(); cte=cm.bag_id.isin(test_bags).to_numpy()
        weights=balanced_weights(cm[ctr])
        sa=StandardScaler().fit(a[ctr],sample_weight=weights)
        sb=StandardScaler().fit(b[ctr],sample_weight=weights)
        x=sa.transform(a); y=sb.transform(b)
        mapper=Ridge(alpha=1.0,fit_intercept=True,solver='svd').fit(x[ctr],y[ctr],sample_weight=weights)
        def mapped(h): return np.maximum(sb.inverse_transform(mapper.predict(sa.transform(h))),0)
        b_from_a=mapped(a)
        # Fold the two fixed scalers and ridge into one affine map; final ReLU is explicit.
        wmap=sb.scale_[:,None]*mapper.coef_/sa.scale_[None,:]
        bmap=sb.mean_+sb.scale_*mapper.intercept_-wmap@sa.mean_
        af=np.maximum(a@wmap.T+bmap,0)
        maperr=float(np.max(np.abs(af-b_from_a))); assert maperr<1e-7
        path=out/'mappings'/f'{fold}.npz'
        np.savez_compressed(path,weight=wmap,bias=bmap,post_relu=np.array(True),train_bags=np.array(train_bags),
            source_mean=sa.mean_,source_scale=sa.scale_,target_mean=sb.mean_,target_scale=sb.scale_,alpha=np.array(1.))
        chk=np.load(path)
        assert np.array_equal(chk['train_bags'],np.array(train_bags))
        assert np.max(np.abs(np.maximum(a@chk['weight'].T+chk['bias'],0)-b_from_a))<1e-7
        checks.append(dict(fold=fold,n_contact_train_windows=int(ctr.sum()),n_contact_train_recordings=len(train_bags),
            map_serialization_max_abs=maperr,
            train_standardized_mse=float(np.mean((sb.transform(b_from_a[ctr])-y[ctr])**2)),
            test_standardized_mse=float(np.mean((sb.transform(b_from_a[cte])-y[cte])**2)),
            test_is_development=True))
        for idx in np.flatnonzero(cte):
            correspondence.append(dict(fold=fold,bag_id=cm.iloc[idx].bag_id,window_row=int(cm.iloc[idx].window_row),
                mse_B_standardized=float(np.mean((sb.transform(b_from_a[idx:idx+1])[0]-y[idx])**2))))
        headpath=ROT/'heads'/fold/'head.npz'; hashes[str(headpath)]=sha(headpath)
        newhead=np.load(headpath)
        assert set(newhead['train_bag_ids'].tolist())==set(train_bags)
        heads=[('native15_conditioned',native['weight15'],native['bias15'],True),
               ('contact_only_new4',newhead['weight4'],newhead['bias4'],False)]
        def evaluate(meta,hidden,path_name,method,seed):
            take=meta.bag_id.isin(test_bags).to_numpy(); mm=meta[take].reset_index(drop=True)
            hh=hidden[take]
            assert len(mm)>0
            for hn,w,bb,isnative in heads:
                prob,gear,other=head_probs(hh,w,bb,isnative)
                for scope in ['all_four_development','known_roi_valid_subset']:
                    good=np.ones(len(mm),bool) if scope=='all_four_development' else ~mm.bag_id.isin(
                        rm.loc[rm.geometry_roi_mismatch,'bag_id'].unique()).to_numpy()
                    labels=[0,1,2,3] if scope=='all_four_development' else [0,1,3]
                    yp=prob.argmax(1); ym=mm.label.to_numpy()
                    shared=dict(fold=fold,head=hn,path=path_name,radar_method=method,seed=seed,scope=scope)
                    allmetrics.append(dict(**shared,unit='window',n=int(good.sum()),correct=int((yp[good]==ym[good]).sum()),
                        accuracy=float(accuracy_score(ym[good],yp[good])),
                        macro_f1=float(f1_score(ym[good],yp[good],labels=labels,average='macro',zero_division=0)),
                        mean_gear_mass=float(gear[good].mean()),pred15_other_rate=float(other[good].mean())))
                    yt=[];pred=[]
                    for bag in mm.loc[good,'bag_id'].unique():
                        ids=mm.bag_id.eq(bag).to_numpy(); pp=prob[ids].mean(0); truth=int(mm.loc[ids,'label'].iloc[0])
                        yt.append(truth);pred.append(int(pp.argmax()))
                        bags.append(dict(**shared,bag_id=bag,label=truth,prediction=int(pp.argmax()),n_windows=int(ids.sum()),
                            **{f'p{i}':float(pp[i]) for i in range(4)}))
                    allmetrics.append(dict(**shared,unit='recording',n=len(yt),correct=int((np.array(yt)==np.array(pred)).sum()),
                        accuracy=float(accuracy_score(yt,pred)),
                        macro_f1=float(f1_score(yt,pred,labels=labels,average='macro',zero_division=0))))
        evaluate(cm,b,'native_B_contact','none',-1)
        evaluate(cm,b_from_a,'A_contact_to_B','none',-1)
        for method in METHODS:
            for seed in SEEDS:
                p=CONTROL/'windows'/f'{fold}__{method}__seed{seed}.npz'
                hashes[str(p)]=sha(p)
                with np.load(p) as z:
                    assert np.array_equal(z['bag_id'],rm.bag_id.to_numpy(str))
                    assert np.array_equal(z['role']=='test',rm.bag_id.isin(test_bags).to_numpy())
                    rh=z['hidden'].astype(np.float64)
                evaluate(rm,mapped(rh),'frozen_radar_A_to_B',method,seed)
        print(fold,'contact map train/test MSE',checks[-1]['train_standardized_mse'],checks[-1]['test_standardized_mse'],flush=True)
    pd.DataFrame(allmetrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(bags).to_csv(out/'file_predictions.csv',index=False)
    pd.DataFrame(correspondence).to_csv(out/'contact_mapping_errors.csv',index=False)
    met=pd.DataFrame(allmetrics)
    met.groupby(['head','path','radar_method','scope','unit'],dropna=False)[['accuracy','macro_f1']].mean().to_csv(out/'summary.csv')
    assert all(sha(p)==h for p,h in hashes.items())
    savej(out/'verification.json',dict(input_hashes_unchanged=True,inputs=hashes,map_checks=checks,
        contact_row_join_complete=True,fit_uses_only_contact_train=True,radar_encoder_retrained=False,
        target_labels_not_used_for_mapping=True,new_B_readout_local_contact_labels_used=True,
        elapsed_seconds=time.time()-start,protocol_sha256=sha(out/'protocol_before_results.json'),script_sha256=sha(__file__)))
    print(met[(met.scope=='all_four_development')&(met.unit=='recording')].groupby(['head','path','radar_method'])[['accuracy','macro_f1']].mean().to_string())

if __name__=='__main__': main()
