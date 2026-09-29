"""Source-only file-held-out contact adaptation and radar bag distillation.

All hyperparameters are fixed in protocol.json. Geometry-incompatible outer/keep
data are retained only as explicitly provisional diagnostics. No raw input edits.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import time
import random
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT.parent/'four_class_preprocessing_20260921/results'
OUT=ROOT/'results'
METHODS=['supervised','head_only','distill','no_contrast','no_radar_CE','shuffled','distill_phase','original_head']


def save_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def file_weights(meta):
    counts=meta.groupby('bag_id').size()
    return np.array([1/counts[b] for b in meta.bag_id])*len(meta)/len(counts)


def fit_linear(x,meta):
    w=file_weights(meta)
    sc=StandardScaler().fit(x,sample_weight=w)
    clf=LogisticRegression(C=1.,max_iter=3000,solver='lbfgs',random_state=42)
    clf.fit(sc.transform(x),meta.label,sample_weight=w)
    return sc,clf


def global_probs(prob,classes):
    result=np.zeros((len(prob),4))
    result[:,np.asarray(classes,dtype=int)]=prob
    return result


def aggregate(prob,meta):
    rows=[]
    for bag,idx in meta.groupby('bag_id',sort=True).indices.items():
        info=meta.iloc[idx[0]]
        p=prob[idx].mean(0)
        rows.append(dict(bag_id=bag,label=int(info.label),state=info.state,n_windows=len(idx),
                         prediction=int(np.argmax(p)),**{'p'+str(i):float(p[i]) for i in range(4)}))
    return pd.DataFrame(rows)


def metrics(pred,classes):
    return dict(n_files=len(pred),accuracy=float(accuracy_score(pred.label,pred.prediction)),
                macro_f1=float(f1_score(pred.label,pred.prediction,labels=classes,average='macro',zero_division=0)),
                balanced_accuracy=float(np.mean([np.mean(pred.loc[pred.label==c,'prediction']==c) for c in classes])))


def contact_spectral(meta,edges):
    cache={};result=[];stats=[]
    for r in meta.itertuples():
        if r.file not in cache:
            with np.load(INPUT/r.file) as z:
                cache[r.file]={k:z[k] for k in ['common_power','common_frequency_hz','raw_xyz']}
        z=cache[r.file];f=z['common_frequency_hz'];p=z['common_power'][r.window_row]
        bp=np.stack([p[:,(f>=a)&(f<b)].mean(1) for a,b in zip(edges[:-1],edges[1:])],axis=1)
        x=np.log10(bp+1e-20);x-=x.mean(1,keepdims=True);result.append(x.ravel())
        raw=z['raw_xyz'][r.window_row].astype(float);raw-=raw.mean(0)
        rms=np.sqrt((raw**2).mean(0));crest=np.max(abs(raw),axis=0)/np.maximum(rms,1e-12)
        stats.append(np.r_[np.log(rms+1e-12),kurtosis(raw,axis=0,fisher=False),crest,np.mean(abs(raw),axis=0)])
    return np.array(result),np.array(stats)


class RadarEncoder(nn.Module):
    def __init__(self,nin):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(nin,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),
                               nn.Linear(128,64),nn.GELU(),nn.Linear(64,128))
    def forward(self,x):return self.net(x)


def train_student(x,rm,hm,contact_x,csc,cmodel,train_bags,classes,method,seed,steps,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    tr=rm.bag_id.isin(train_bags).to_numpy()
    sc=StandardScaler().fit(x[tr],sample_weight=file_weights(rm[tr]))
    xs=sc.transform(x).astype(np.float32)
    if not np.isfinite(xs).all():raise ValueError('Nonfinite radar feature transform')
    xt=torch.tensor(xs,device=device)
    # The bag order is deterministic; no target labels or contact rows are used.
    bag_labels=np.array([int(rm[rm.bag_id==b].label.iloc[0]) for b in train_bags])
    class_index={c:i for i,c in enumerate(classes)}
    yi=torch.tensor([class_index[v] for v in bag_labels],device=device)
    pools=[np.flatnonzero((rm.bag_id==b).to_numpy()) for b in train_bags]
    teacher_z=csc.transform(contact_x).astype(np.float32)
    centers=[];teacher_prob=[]
    T=2.
    for b in train_bags:
        ids=(hm.bag_id==b).to_numpy()
        assert ids.any()
        centers.append(teacher_z[ids].mean(0))
        teacher_prob.append(softmax((teacher_z[ids]@cmodel.coef_.T+cmodel.intercept_)/T,axis=1).mean(0))
    centers=np.array(centers);teacher_prob=np.array(teacher_prob)
    if method=='shuffled':
        # A predeclared wrong-correspondence control; not random target labels.
        indices=np.roll(np.arange(len(train_bags)),len(train_bags)//len(classes))
        centers=centers[indices];teacher_prob=teacher_prob[indices]
    ct=torch.tensor(centers,dtype=torch.float32,device=device)
    qt=torch.tensor(teacher_prob,dtype=torch.float32,device=device)
    w=torch.tensor(cmodel.coef_,dtype=torch.float32,device=device)
    bias=torch.tensor(cmodel.intercept_,dtype=torch.float32,device=device)
    positives=torch.tensor(bag_labels[:,None]==bag_labels[None,:],device=device)
    encoder=RadarEncoder(x.shape[1]).to(device)
    own=nn.Linear(128,len(classes)).to(device) if method=='supervised' else None
    params=list(encoder.parameters())+(list(own.parameters()) if own is not None else [])
    opt=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    rng=np.random.default_rng(seed)
    trace=[]
    nbag=len(pools);nwin=16
    started=time.monotonic()
    for step in range(steps):
        ids=np.concatenate([rng.choice(p,size=nwin,replace=True) for p in pools])
        h=encoder(xt[ids]).reshape(nbag,nwin,128)
        logits=own(h) if own is not None else h@w.T+bias
        ce=F.cross_entropy(logits.reshape(-1,len(classes)),yi.repeat_interleave(nwin))
        mean_h=h.mean(1)
        feat=F.mse_loss(mean_h,ct)
        mean_p=(logits/T).softmax(-1).mean(1)
        kd=(qt*(qt.clamp_min(1e-10).log()-mean_p.clamp_min(1e-10).log())).sum(1).mean()*T*T
        sim=F.normalize(mean_h,dim=1)@F.normalize(ct,dim=1).T/.2
        contrast=(torch.logsumexp(sim,1)-torch.logsumexp(sim.masked_fill(~positives,-torch.inf),1)).mean()
        if method in ['supervised','head_only']:loss=ce
        elif method=='original_head':loss=feat
        else:
            loss=feat+.5*kd
            if method!='no_radar_CE':loss=loss+ce
            if method!='no_contrast':loss=loss+.1*contrast
        if not torch.isfinite(loss):raise ValueError((method,step,'nonfinite loss'))
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(params,5.);opt.step()
        if step%50==0 or step==steps-1:
            trace.append(dict(step=step+1,total=float(loss.detach()),CE=float(ce.detach()),
                              feature=float(feat.detach()),KL=float(kd.detach()),contrast=float(contrast.detach())))
    encoder.eval()
    if own is not None:own.eval()
    with torch.inference_mode():
        embedded=encoder(xt)
        logits=own(embedded) if own is not None else embedded@w.T+bias
        probability=logits.softmax(-1).cpu().numpy()
        if method=='original_head':
            original=torch.load('/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn/classifier.pth',map_location='cpu',weights_only=True)
            raw_h=embedded.cpu().numpy()*csc.scale_+csc.mean_
            p10=softmax(raw_h@original['linear2.weight'].numpy().T+original['linear2.bias'].numpy(),axis=1)
            p4=np.stack([p10[:,0],p10[:,1:4].sum(1),p10[:,7:10].sum(1),p10[:,4:7].sum(1)],axis=1)
            probability=p4
    checkpoint=dict(encoder={k:v.detach().cpu() for k,v in encoder.state_dict().items()},
        own_head=None if own is None else {k:v.detach().cpu() for k,v in own.state_dict().items()},
        radar_mean=sc.mean_,radar_scale=sc.scale_,contact_mean=csc.mean_,contact_scale=csc.scale_,
        contact_coef=cmodel.coef_,contact_intercept=cmodel.intercept_,classes=np.array(classes),
        method=method,seed=seed,steps=steps,input_dim=x.shape[1],training_bags=train_bags,
        status='provisional geometry; not a deployment-validated fault model')
    checkpoint['output_head_type']='original_fcn_10class' if method=='original_head' else 'learned_radar_head' if method=='supervised' else 'frozen_adapted_contact_head'
    if method=='original_head':
        checkpoint['original_coef']=original['linear2.weight'].numpy()
        checkpoint['original_intercept']=original['linear2.bias'].numpy()
        checkpoint['original_head_sha256']=hashlib.sha256(Path('/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn/classifier.pth').read_bytes()).hexdigest()
    output_prob=probability if method=='original_head' else global_probs(probability,classes)
    return output_prob,embedded.cpu().numpy(),checkpoint,trace,time.monotonic()-started


def make_folds(rm,classes):
    grouped={c:rm[rm.label==c].drop_duplicates('bag_id').sort_values('baud_candidate').bag_id.tolist() for c in classes}
    assert all(len(v)==2 for v in grouped.values())
    folds=[]
    for bits in itertools.product([0,1],repeat=len(classes)):
        tr=[grouped[c][bits[i]] for i,c in enumerate(classes)]
        te=[grouped[c][1-bits[i]] for i,c in enumerate(classes)]
        folds.append(dict(fold=''.join(map(str,bits)),train_bags=tr,test_bags=te,
                          direction='115200_to_460800' if sum(bits)==0 else '460800_to_115200' if sum(bits)==len(bits) else 'mixed_recording_holdout'))
    folds.append(dict(fold='all_known',train_bags=sum(grouped.values(),[]),test_bags=[],direction='external_only'))
    return folds


def run(args):
    torch.set_num_threads(2)
    if '/envs/m2vllm/' not in __import__('sys').executable:raise ValueError('Use m2vllm')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Choose new nonempty-free output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    cm=pd.read_csv(ROOT/'contact/retrained_fcn_fixed_external/metadata.csv')
    cz=np.load(ROOT/'contact/retrained_fcn_fixed_external/contact_features.npz')
    rm=pd.read_csv(ROOT/'radar/metadata.csv')
    rz=np.load(ROOT/'radar/features.npz')
    assert len(cm)==len(cz['hidden_mean']) and len(rm)==len(rz['frame_shape'])
    primary=cz['hidden_mean']; spectral,stats=contact_spectral(cm,rz['band_edges_hz'])
    contact_inputs={'contact_hidden':primary,'contact_xyz_hidden':cz['hidden_axes'].reshape(len(cm),-1),
                    'contact_spectral':spectral,'contact_statistics':stats}
    finalmeta=pd.read_csv(ROOT/'contact/retrained_final_fixed_external/metadata.csv')
    assert finalmeta[['file','window_row']].equals(cm[['file','window_row']])
    contact_inputs['contact_final_hidden']=np.load(ROOT/'contact/retrained_final_fixed_external/contact_features.npz')['hidden_mean']
    original=torch.load('/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn/classifier.pth',map_location='cpu',weights_only=True)
    ph=softmax(primary@original['linear2.weight'].numpy().T+original['linear2.bias'].numpy(),axis=1)
    original_mean_hidden=np.stack([ph[:,0],ph[:,1:4].sum(1),ph[:,7:10].sum(1),ph[:,4:7].sum(1)],axis=1)
    results=[];filepreds=[];external=[];traces=[];split_records=[];llm=[];checks=[]
    tasks={'four_class_provisional_roi':[0,1,2,3],'three_class_geometry_compatible':[0,1,3]}
    if args.task!='both':tasks={args.task:tasks[args.task]}
    for task,classes in tasks.items():
        folds=make_folds(rm,classes)
        if args.quick:folds=[folds[0],folds[-2],folds[-1]]
        for fold in folds:
            tr=cm.bag_id.isin(fold['train_bags']).to_numpy()
            te=cm.bag_id.isin(fold['test_bags']).to_numpy()
            rt=rm.bag_id.isin(fold['train_bags']).to_numpy()
            rv=rm.bag_id.isin(fold['test_bags']).to_numpy()
            assert set(fold['train_bags']).isdisjoint(fold['test_bags'])
            assert set(cm[tr].bag_id)==set(rm[rt].bag_id)
            assert set(cm[te].bag_id)==set(rm[rv].bag_id)
            split_records.append(dict(task=task,**fold,classes=classes,
                train_contact_rows=np.flatnonzero(tr).tolist(),test_contact_rows=np.flatnonzero(te).tolist(),
                train_radar_rows=np.flatnonzero(rt).tolist(),test_radar_rows=np.flatnonzero(rv).tolist()))
            csc,cmodel=fit_linear(primary[tr],cm[tr])
            # Contact head parameters are frozen before any radar optimization.
            model_dir=args.output/'models'/task/fold['fold'];model_dir.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(model_dir/'contact_head.npz',mean=csc.mean_,scale=csc.scale_,
                coef=cmodel.coef_,intercept=cmodel.intercept_,classes=cmodel.classes_)

            def record(prob,meta,mask,method,seed,modality):
                base=dict(task=task,fold=fold['fold'],direction=fold['direction'],method=method,seed=seed,modality=modality)
                if mask.any():
                    p=aggregate(prob[mask],meta[mask]);scores=metrics(p,classes)
                    wp=np.argmax(prob[mask],1)
                    results.append(dict(**base,**scores,window_accuracy=float(np.mean(wp==meta[mask].label.to_numpy()))))
                    for k,v in base.items():p[k]=v
                    filepreds.append(p)
                    if fold['direction']!='mixed_recording_holdout' and seed in [-1,42] and method in ['contact_hidden','radar_frame_shape','supervised','distill','no_radar_CE']:
                        for row in p.to_dict('records'):llm.append(dict(**row,id=f'{task}_{fold["fold"]}_{method}_{seed}_{row["bag_id"]}',true_label=row['label']))
                if fold['fold']=='all_known':
                    ex=(meta.label<0).to_numpy()
                    p=aggregate(prob[ex],meta[ex])
                    for k,v in base.items():p[k]=v
                    p['healthy_probability']=p.p0
                    p['max_probability']=p[['p0','p1','p2','p3']].max(axis=1)
                    p['unknown_score']=1-p.max_probability
                    p['known_class_prediction_only']=True
                    external.append(p)
                    if seed in [-1,42] and method in ['contact_hidden','radar_frame_shape','supervised','distill','no_radar_CE']:
                        for row in p.to_dict('records'):llm.append(dict(**row,id=f'{task}_{fold["fold"]}_{method}_{seed}_{row["bag_id"]}',true_label=0 if row['state']=='bigNormal' else -1))

            for name,x in contact_inputs.items():
                sc,mo=fit_linear(x[tr],cm[tr])
                p=global_probs(mo.predict_proba(sc.transform(x)),classes)
                record(p,cm,te,name,-1,'contact')
            # Original ten-class probabilities are kept as a failed baseline.
            orig=cz['probs4_mean'].copy()
            record(orig,cm,te,'contact_original',-1,'contact')
            record(original_mean_hidden,cm,te,'contact_original_mean_hidden',-1,'contact')
            for key in ['frame_shape','shape_phase','coherent_shape','time_stats','confound_distance','confound_raw_amp']:
                x=rz[key];sc,mo=fit_linear(x[rt],rm[rt]);p=global_probs(mo.predict_proba(sc.transform(x)),classes)
                record(p,rm,rv,'radar_'+key,-1,'radar')
            seeds=[42] if args.quick or fold['fold']=='all_known' else [17,42,73]
            for method in METHODS:
                if args.quick and method not in ['supervised','head_only','distill','shuffled','original_head']:continue
                for seed in seeds:
                    key='shape_phase' if method=='distill_phase' else 'frame_shape'
                    prob,embedding,checkpoint,trace,seconds=train_student(rz[key],rm,cm,primary,csc,cmodel,
                        fold['train_bags'],classes,method,seed,args.steps,args.device)
                    record(prob,rm,rv,method,seed,'radar')
                    for row in trace:traces.append(dict(task=task,fold=fold['fold'],method=method,seed=seed,**row))
                    if fold['direction']!='mixed_recording_holdout' and seed==42:
                        path=model_dir/f'{method}_seed{seed}.pt'
                        checkpoint.update(input_feature=key,task=task,protocol_file=str(ROOT/'protocol.json'))
                        torch.save(checkpoint,path)
                        np.savez_compressed(path.with_suffix('.predictions.npz'),probabilities=prob,embedding=embedding,
                                            bag_id=rm.bag_id.to_numpy(dtype=str))
                    print(json.dumps(dict(task=task,fold=fold['fold'],method=method,seed=seed,seconds=round(seconds,2))),flush=True)
            pd.DataFrame(results).to_csv(args.output/'metrics.csv',index=False)
            if filepreds:pd.concat(filepreds,ignore_index=True).to_csv(args.output/'file_predictions.csv',index=False)
            if external:pd.concat(external,ignore_index=True).to_csv(args.output/'external_predictions.csv',index=False)
            pd.DataFrame(traces).to_csv(args.output/'training_trace.csv',index=False)
            save_json(args.output/'splits.json',split_records)
            pd.DataFrame(llm).to_csv(args.output/'llm_bridge_inputs.csv',index=False)
    save_json(args.output/'runtime.json',dict(python=__import__('sys').executable,torch=torch.__version__,device=args.device,
        seeds=[42] if args.quick else [17,42,73],steps=args.steps,quick=args.quick,
        protocol_sha256=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='No outcome-based tuning; candidate ROI four-class outputs remain confounded'))
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=OUT/'chain_v1')
    ap.add_argument('--device',default='cuda:1')
    ap.add_argument('--steps',type=int,default=200)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--task',default='both',choices=['both','four_class_provisional_roi','three_class_geometry_compatible'])
    run(ap.parse_args())
