"""Predeclared R2 leave-one-speed-out distillation; no test-driven selection."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1]
GROUPS=[[0],[1,2,3],[7,8,9],[4,5,6]]
METHODS=['head_CE','contact_session_MSE','full','class_prototype_MSE','class_code_MSE','same_class_wrong_speed_MSE']
SEEDS=[17,42,73]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def balanced(ids):
    _,index,count=np.unique(ids,return_inverse=True,return_counts=True)
    return len(ids)/len(count)/count[index]

def group(logits):
    if logits.shape[-1]==4:return logits
    return torch.stack([torch.logsumexp(logits[...,g],-1) for g in GROUPS],-1)

def grouped_tempered(logits,T=2.):
    p=(logits/T).softmax(-1)
    if p.shape[-1]==4:return p
    return torch.stack([p[...,g].sum(-1) for g in GROUPS],-1)

def cos(a,b):
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))

class Encoder(nn.Module):
    def __init__(self,mean,scale):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),
                               nn.Linear(128,64),nn.GELU(),nn.Linear(64,128))
        self.register_buffer('contact_mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('contact_scale',torch.tensor(scale,dtype=torch.float32))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        return F.relu(self.contact_mean+self.contact_scale*self.net(x))

class Direct(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),
                               nn.Linear(128,64),nn.GELU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,4))
    def forward(self,x):return self.net(x)

def make_codes(w,b,mean,scale,device):
    m=torch.tensor(mean,dtype=torch.float32,device=device)
    s=torch.tensor(scale,dtype=torch.float32,device=device)
    z=nn.Parameter(torch.zeros(4,128,device=device))
    opt=torch.optim.Adam([z],lr=.05)
    for _ in range(400):
        h=F.relu(m+s*z)
        loss=F.cross_entropy(group(h@w.T+b),torch.arange(4,device=device))+.001*((h-m)/s).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        h=F.relu(m+s*z)
        probs=group(h@w.T+b).softmax(-1)
    return ((h-m)/s).detach(),probs.cpu().numpy()

def metric(y,p):
    return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=range(4),average='macro',zero_division=0)),
                correct=int(np.sum(y==p)),n=len(y))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head',choices=['archived','diagnostic'],default='archived')
    parser.add_argument('--feature',default='complex_shape')
    parser.add_argument('--methods',nargs='+',default=None)
    parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(2)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out=ROOT/'experiments'/f'{args.head}__{args.feature}'
    if out.exists():raise RuntimeError(f'Refusing overwrite {out}')
    out.mkdir();(out/'predictions').mkdir();(out/'models').mkdir();(out/'targets').mkdir()
    cp=ROOT/'contact/contact_features.npz';rp=ROOT/'radar/features.npz'
    c=np.load(cp);r=np.load(rp)
    x=r[args.feature].astype(np.float32);h=c['embedding'].astype(np.float32)
    y=r['labels'].astype(int);cy=c['labels'].astype(int)
    rpm=r['rpm'].astype(int);crpm=c['rpm'].astype(int)
    rid=r['recording_id'].astype(str);sid=r['contact_id'].astype(str);csid=c['session_id'].astype(str)
    yy=c['physical_targets'].astype(float);names=c['physical_target_names'].astype(str)
    methods=args.methods or METHODS
    if set(methods)-set(METHODS):raise ValueError('Unknown method')
    hashes={str(p):sha(p) for p in [cp,rp,ROOT/'protocol_before_results.json']}
    config=dict(head=args.head,feature=args.feature,methods=methods,seeds=args.seeds,steps=400,
                primary=(args.head=='archived' and args.feature=='complex_shape'),
                head_certification='historical retained frozen' if args.head=='archived' else 'new contact-only LOSO diagnostic; source retention NOT certified',
                heldout_test_not_used_for_optimization=True,source_hashes=hashes,
                physical_readout_alpha=10,fit_weighting='contact sessions / radar recordings equal total weight',
                program_sha256=sha(__file__))
    write_json(out/'run_protocol.json',config)
    metrics=[];records=[];sessions=[];physical=[];contact_probes=[];traces=[];split_rows=[]
    for hold in [1000,2000,3000]:
        fold=f'holdout_{hold}'
        rt=rpm!=hold;ct=crpm!=hold;test=~rt;ctest=~ct
        assert not set(sid[rt])&set(csid[ctest])
        assert set(sid[rt])==set(csid[ct])
        assert not set(rid[rt])&set(rid[test])
        train_sessions=sorted(set(sid[rt]));test_sessions=sorted(set(sid[test]))
        split_rows.append(dict(fold=fold,train_contact_sessions=sorted(set(csid[ct])),test_contact_sessions=sorted(set(csid[ctest])),
                               train_recordings=sorted(set(rid[rt])),test_recordings=sorted(set(rid[test])),
                               train_radar_windows=int(rt.sum()),test_radar_windows=int(test.sum()),
                               train_contact_packets=int(ct.sum()),test_contact_packets=int(ctest.sum())))
        csc=StandardScaler().fit(h[ct],sample_weight=balanced(csid[ct]));rsc=StandardScaler().fit(x[rt],sample_weight=balanced(rid[rt]))
        hz=csc.transform(h).astype(np.float32);xx=torch.tensor(rsc.transform(x),dtype=torch.float32,device=device)
        all_targets={s:hz[csid==s].mean(0) for s in set(csid)}
        labels=np.array([cy[csid==s][0] for s in train_sessions])
        speeds=np.array([crpm[csid==s][0] for s in train_sessions])
        targets=np.stack([all_targets[s] for s in train_sessions])
        prototypes={cl:targets[labels==cl].mean(0) for cl in range(4)}
        raw_class_prototypes={cl:prototypes[cl]*csc.scale_+csc.mean_ for cl in range(4)}
        if args.head=='archived':
            wn=c['retained_weight10'];bn=c['retained_bias10']
        else:
            hp=ROOT/'contact/diagnostic_heads'/f'{fold}.npz'
            hd=np.load(hp);wn=hd['weight4'];bn=hd['bias4']
            config['source_hashes'][str(hp)]=sha(hp)
        w=torch.tensor(wn,dtype=torch.float32,device=device);b=torch.tensor(bn,dtype=torch.float32,device=device)
        mean=torch.tensor(csc.mean_,dtype=torch.float32,device=device);scale=torch.tensor(csc.scale_,dtype=torch.float32,device=device)
        target_t=torch.tensor(targets,device=device);label_t=torch.tensor(labels,device=device)
        positive=torch.tensor(labels[:,None]==labels[None,:],device=device)
        logits_h=torch.tensor(h,dtype=torch.float32,device=device)@w.T+b
        q_all=grouped_tempered(logits_h).detach()
        qt=torch.stack([q_all[torch.tensor(csid==s,device=device)].mean(0) for s in train_sessions])
        codes,code_probs=make_codes(w,b,csc.mean_,csc.scale_,device)
        _,sv,vt=np.linalg.svd((wn-wn.mean(0))*csc.scale_[None,:],full_matrices=False)
        rank=int(np.sum(sv>max(sv.max(),1e-12)*1e-6));basis=vt[:rank]
        pools=[[np.flatnonzero(rid==record) for record in sorted(set(rid[sid==s]))] for s in train_sessions]
        def draw(rng):
            return np.array([rng.choice(group_pool[rng.integers(len(group_pool))]) for group_pool in pools for _ in range(16)])
        # The oracle class baselines below use held-out TRUE labels only to score controls.
        ys=StandardScaler().fit(yy[ct],sample_weight=balanced(csid[ct]))
        readout=Ridge(alpha=10,solver='svd').fit(hz[ct],ys.transform(yy[ct]),sample_weight=balanced(csid[ct]))
        def read(hz_in):return ys.inverse_transform(readout.predict(hz_in))
        class_phys={cl:np.stack([yy[csid==s].mean(0) for s in train_sessions if cy[csid==s][0]==cl]).mean(0) for cl in range(4)}
        for s in test_sessions:
            ci=csid==s;cl=int(cy[ci][0]);truth=yy[ci].mean(0)
            for m,pred in [('direct_contact_readout',read(hz[ci]).mean(0)),('oracle_true_class_target',class_phys[cl]),
                           ('oracle_true_class_embedding',read(prototypes[cl][None,:])[0])]:
                for j,name in enumerate(names):
                    contact_probes.append(dict(fold=fold,session_id=s,method=m,target=name,truth=float(truth[j]),prediction=float(pred[j]),
                                               scale=float(ys.scale_[j]),abs_error_std=float(abs(pred[j]-truth[j])/ys.scale_[j])))
        np.savez_compressed(out/'targets'/f'{fold}.npz',contact_mean=csc.mean_,contact_scale=csc.scale_,radar_mean=rsc.mean_,radar_scale=rsc.scale_,
                            train_sessions=train_sessions,test_sessions=test_sessions,targets=targets,class_prototypes=np.stack([prototypes[k] for k in range(4)]),
                            label_codes=codes.cpu().numpy(),label_code_probabilities=code_probs,head_weight=wn,head_bias=bn,
                            readout_coef=readout.coef_,readout_intercept=readout.intercept_,physical_mean=ys.mean_,physical_scale=ys.scale_,head_basis=basis)
        for method in methods+(['direct_MLP'] if args.head=='archived' and args.feature=='complex_shape' else []):
            target=target_t
            if method=='class_prototype_MSE': target=torch.tensor(np.stack([prototypes[cl] for cl in labels]),dtype=torch.float32,device=device)
            if method=='class_code_MSE': target=codes[label_t]
            if method=='same_class_wrong_speed_MSE':
                permutation=[np.flatnonzero((labels==cl)&(speeds!=sp))[0] for cl,sp in zip(labels,speeds)]
                target=target_t[permutation]
            for seed in args.seeds:
                started=time.time();torch.manual_seed(seed);np.random.seed(seed)
                direct=method=='direct_MLP'
                model=(Direct() if direct else Encoder(csc.mean_,csc.scale_)).to(device)
                opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                rng=np.random.default_rng(seed)
                for step in range(400):
                    ids=draw(rng);batch=xx[ids]
                    if direct:
                        loss=F.cross_entropy(model(batch),label_t.repeat_interleave(16))
                        ce=loss;mse=loss*0;kd=loss*0;contrast=loss*0
                    else:
                        hh=model(batch).reshape(len(pools),16,128)
                        zz=(hh-mean)/scale;bag=zz.mean(1);logits=hh@w.T+b
                        ce=F.cross_entropy(group(logits).reshape(-1,4),label_t.repeat_interleave(16))
                        mse=F.mse_loss(bag,target)
                        sim=F.normalize(bag,dim=1)@F.normalize(target_t,dim=1).T/.2
                        contrast=(torch.logsumexp(sim,1)-torch.logsumexp(sim.masked_fill(~positive,-torch.inf),1)).mean()
                        pp=grouped_tempered(logits).mean(1)
                        kd=4*(qt*(qt.clamp_min(1e-10).log()-pp.clamp_min(1e-10).log())).sum(1).mean()
                        loss=ce if method=='head_CE' else ce+mse+.1*contrast+.5*kd if method=='full' else mse
                    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    if step%100==0 or step==399:
                        traces.append(dict(fold=fold,method=method,seed=seed,step=step+1,loss=float(loss.detach()),ce=float(ce.detach()),
                                           mse=float(mse.detach()),kd=float(kd.detach()),contrast=float(contrast.detach())))
                model.eval()
                with torch.no_grad():
                    output=model(xx)
                    probs=(output if direct else group(output@w.T+b)).softmax(-1).cpu().numpy()
                    hh_all=None if direct else output.cpu().numpy()
                key=f'{fold}__{method}__seed{seed}'
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},feature=args.feature,
                                head=args.head,method=method,fold=fold,radar_mean=rsc.mean_,radar_scale=rsc.scale_),out/'models'/f'{key}.pt')
                save=dict(probabilities4=probs[test],labels=y[test],recording_id=rid[test],contact_id=sid[test],row_index=np.flatnonzero(test))
                if not direct:save['hidden']=hh_all[test]
                np.savez_compressed(out/'predictions'/f'{key}.npz',**save)
                test_preds=[];test_labels=[]
                for record in sorted(set(rid)):
                    ix=rid==record;cl=int(y[ix][0]);guess=probs[ix].mean(0);prediction=int(guess.argmax())
                    info=dict(fold=fold,method=method,seed=seed,recording_id=record,contact_id=sid[ix][0],label=cl,prediction=prediction,
                              role='test' if rpm[ix][0]==hold else 'train',rpm=int(rpm[ix][0]),distance_cm=int(r['distance_cm'][ix][0]),
                              n_windows=int(ix.sum()),window_accuracy=float((probs[ix].argmax(-1)==cl).mean()),**{f'p{k}':float(guess[k]) for k in range(4)})
                    records.append(info)
                    if info['role']=='test':test_preds.append(prediction);test_labels.append(cl)
                for level,truth,pred in [('window',y[test],probs[test].argmax(-1)),('recording',np.array(test_labels),np.array(test_preds))]:
                    metrics.append(dict(fold=fold,method=method,seed=seed,level=level,**metric(truth,pred)))
                for cm in [20,40,80]:
                    ii=test&(r['distance_cm'].astype(int)==cm)
                    metrics.append(dict(fold=fold,method=method,seed=seed,level=f'window_{cm}cm',**metric(y[ii],probs[ii].argmax(-1))))
                if not direct:
                    zz_all=(hh_all-csc.mean_)/csc.scale_
                    for s in test_sessions:
                        ri=sid==s;ci=csid==s;cl=int(cy[ci][0]);tz=all_targets[s]
                        # Equal weight for each physical radar recording, not each window count.
                        ez=np.stack([zz_all[rid==rec].mean(0) for rec in sorted(set(rid[ri]))]).mean(0)
                        delta=ez-tz;visible=delta@basis.T@basis;null=delta-visible
                        oracle_delta=prototypes[cl]-tz
                        sessions.append(dict(fold=fold,method=method,seed=seed,session_id=s,label=cl,
                                             mse=float(np.mean(delta**2)),cosine=cos(ez,tz),head_row_mse=float(np.mean(visible**2)),
                                             head_null_mse=float(np.mean(null**2)),oracle_class_embedding_mse=float(np.mean(oracle_delta**2)),
                                             improvement_vs_class_mse=float(np.mean(oracle_delta**2)-np.mean(delta**2)),head_rank=rank))
                        truth=yy[ci].mean(0);pred=read(ez[None,:])[0]
                        for j,name in enumerate(names):
                            physical.append(dict(fold=fold,method=method,seed=seed,session_id=s,label=cl,target=name,truth=float(truth[j]),
                                                 prediction=float(pred[j]),scale=float(ys.scale_[j]),abs_error_std=float(abs(pred[j]-truth[j])/ys.scale_[j]),
                                                 oracle_class_abs_error_std=float(abs(class_phys[cl][j]-truth[j])/ys.scale_[j])))
                print(f'{args.head} {args.feature} {fold} {method} seed{seed}: window={metrics[-5]["accuracy"]:.3f} recording={metrics[-4]["accuracy"]:.3f} ({time.time()-started:.1f}s)',flush=True)
                pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        pd.DataFrame(records).to_csv(out/'recording_predictions.csv',index=False)
        pd.DataFrame(sessions).to_csv(out/'session_fidelity.csv',index=False)
        pd.DataFrame(physical).to_csv(out/'physical_predictions.csv',index=False)
        pd.DataFrame(contact_probes).to_csv(out/'contact_probe_controls.csv',index=False)
        pd.DataFrame(traces).to_csv(out/'training_trace.csv',index=False)
        write_json(out/'splits.json',split_rows)
    for p,value in config['source_hashes'].items():
        assert sha(p)==value,f'Input changed: {p}'
    write_json(out/'run_protocol.json',config)
    write_json(out/'verification.json',dict(input_hashes_unchanged=True,folds=len(split_rows),leakage_checks_passed=True,
                                         heldout_contact_targets_used_only_in_evaluation=True,source_head_not_modified=True,
                                         radar_recordings=len(set(rid)),contact_sessions=len(set(csid)),
                                         repeated_seeds_do_not_increase_physical_sample_count=True,completed=True))
    print(out,flush=True)

if __name__=='__main__':main()
