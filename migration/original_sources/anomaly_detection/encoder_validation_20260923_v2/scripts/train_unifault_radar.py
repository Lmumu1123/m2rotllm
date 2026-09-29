"""Frozen signed UniFault teacher: correct signed student vs legacy ReLU control."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'unifault_radar'
SEEDS=[17,42,73]
METHODS=['ce_only','embedding_only','full']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def weights(md):
    count=md.groupby('bag_id').size();return np.array([1/count[b] for b in md.bag_id])*len(md)/len(count)
def cosine(a,b):return float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
class Encoder(nn.Module):
    def __init__(self,mean,scale,nonnegative=False):
        super().__init__();self.nonnegative=nonnegative
        self.net=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,128))
        self.register_buffer('contact_mean',torch.as_tensor(mean,dtype=torch.float32));self.register_buffer('contact_scale',torch.as_tensor(scale,dtype=torch.float32))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        h=self.contact_mean+self.contact_scale*self.net(x)
        return F.relu(h) if self.nonnegative else h

def main():
    torch.set_num_threads(2);assert 'envs/m2vllm/' in sys.executable
    if OUT.exists():raise RuntimeError('Refusing overwrite')
    OUT.mkdir();(OUT/'models').mkdir()
    protocol=dict(teacher='Official UniFault Tiny pretrained Transformer; frozen backbone, per-fold contact-only four-class readout',
        teacher_input='predeclared paper_100ms branch, 10 time blocks x 3 axes averaged per 1s contact window',
        radar_input='geometry-corrected frame_shape 128',constraints=['signed_linear','legacy_nonnegative_relu'],
        constraints_reason='UniFault LayerNorm features contain negative values; legacy ReLU is a deliberate incorrect-support control',
        methods=METHODS,seeds=SEEDS,steps=400,windows_per_bag=16,optimizer='AdamW lr.001 wd.01 clip5',
        loss='CE + MSE + .1 contrast(tau.2) + .5 KD(T2)',
        fixed_final_step_no_target_selection=True,test_is_seen_development_data=True,
        source_public_retention='Untouched original model; new readout has not been evaluated on original public task',
        no_new_physical_samples=True)
    (OUT/'protocol_before_results.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    rm=pd.read_csv(ROOT/'geometry_corrected/radar/metadata.csv');rz=np.load(ROOT/'geometry_corrected/radar/features.npz');x=rz['frame_shape']
    assert np.array_equal(rz['bag_id'],rm.bag_id)
    metrics=[];files=[];checks=[];hashes={};splits=[]
    started=time.time()
    for source,target in [(115200,460800),(460800,115200)]:
        fold=f'{source}_to_{target}';td=ROOT/'teachers/unifault/paper_100ms'/fold
        cm=pd.read_csv(td/'metadata.csv');h=np.load(td/'teacher_features.npz')['hidden_mean'].astype(np.float32)
        hd=np.load(td/'head.npz');w=torch.tensor(hd['weight4'],dtype=torch.float32);b=torch.tensor(hd['bias4'],dtype=torch.float32)
        ctr=(cm.label>=0)&(cm.baud==source);rtr=(rm.label>=0)&(rm.baud_candidate==source)
        trainbags=sorted(cm.loc[ctr,'bag_id'].unique());testbags=sorted(cm.loc[(cm.label>=0)&(cm.baud==target),'bag_id'].unique())
        assert set(hd['train_bag_ids'])==set(trainbags) and not set(trainbags)&set(testbags)
        for p in [td/'teacher_features.npz',td/'head.npz',td/'metadata.csv']:hashes[str(p)]=sha(p)
        cs=StandardScaler().fit(h[ctr],sample_weight=weights(cm[ctr]));rs=StandardScaler().fit(x[rtr],sample_weight=weights(rm[rtr]))
        hz=cs.transform(h).astype(np.float32);xt=torch.tensor(rs.transform(x).astype(np.float32))
        labels=np.array([int(cm.loc[cm.bag_id==bag,'label'].iloc[0]) for bag in trainbags]);yi=torch.tensor(labels)
        ct=torch.tensor(np.stack([hz[cm.bag_id==bag].mean(0) for bag in trainbags]));pools=[np.flatnonzero(rm.bag_id.eq(bag).to_numpy()) for bag in trainbags]
        positive=torch.tensor(labels[:,None]==labels[None,:]);th=torch.tensor(h)
        with torch.no_grad():qt=torch.stack([((th[cm.bag_id.eq(bag).to_numpy()]@w.T+b)/2).softmax(-1).mean(0) for bag in trainbags])
        splits.append(dict(fold=fold,train_bags=trainbags,test_bags=testbags))
        for constraint in protocol['constraints']:
            for method in METHODS:
                for seed in SEEDS:
                    torch.manual_seed(seed);rng=np.random.default_rng(seed)
                    model=Encoder(cs.mean_,cs.scale_,constraint=='legacy_nonnegative_relu');opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                    for step in range(400):
                        ids=np.concatenate([rng.choice(pool,16,replace=True) for pool in pools]);hh=model(xt[ids]).reshape(4,16,128)
                        zz=(hh-model.contact_mean)/model.contact_scale;mz=zz.mean(1);logits=hh@w.T+b
                        ce=F.cross_entropy(logits.reshape(-1,4),yi.repeat_interleave(16));mse=F.mse_loss(mz,ct)
                        sim=F.normalize(mz,dim=1)@F.normalize(ct,dim=1).T/.2
                        contrast=(torch.logsumexp(sim,1)-torch.logsumexp(sim.masked_fill(~positive,-torch.inf),1)).mean()
                        q=(logits/2).softmax(-1).mean(1);kd=4*(qt*(qt.clamp_min(1e-10).log()-q.clamp_min(1e-10).log())).sum(1).mean()
                        loss=ce if method=='ce_only' else mse if method=='embedding_only' else ce+mse+.1*contrast+.5*kd
                        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    model.eval()
                    with torch.inference_mode():rh=model(xt).numpy();pr=torch.tensor(rh@w.numpy().T+b.numpy()).softmax(-1).numpy()
                    tag=f'{fold}__{constraint}__{method}__seed{seed}';dest=OUT/'models'/tag;dest.mkdir()
                    torch.save(dict(encoder=model.state_dict(),nonnegative=model.nonnegative,head_weight=w,head_bias=b,
                        radar_mean=torch.tensor(rs.mean_),radar_scale=torch.tensor(rs.scale_),train_bags=trainbags,test_bags=testbags,
                        teacher_feature_sha256=sha(td/'teacher_features.npz'),method=method,seed=seed),dest/'model.pt')
                    ck=torch.load(dest/'model.pt',map_location='cpu',weights_only=True);again=Encoder(cs.mean_,cs.scale_,ck['nonnegative']);again.load_state_dict(ck['encoder']);again.eval()
                    with torch.no_grad():err=float(np.max(np.abs(again(xt).numpy()-rh)))
                    assert err<1e-6
                    np.savez_compressed(dest/'predictions.npz',hidden=rh,probs4=pr,bag_id=rm.bag_id.to_numpy(str),labels=rm.label.to_numpy())
                    rz=(rh-cs.mean_)/cs.scale_;bagrows=[]
                    for bag,ids in rm.groupby('bag_id').indices.items():
                        info=rm.iloc[ids[0]];ci=cm.bag_id.eq(bag).to_numpy();truth=int(info.label);pp=pr[ids].mean(0)
                        role='train' if bag in trainbags else 'test' if bag in testbags else 'external'
                        row=dict(fold=fold,constraint=constraint,method=method,seed=seed,bag_id=bag,label=truth,role=role,prediction=int(pp.argmax()),n_windows=len(ids),
                            mse_standardized=float(np.mean((rz[ids].mean(0)-hz[ci].mean(0))**2)),cos_standardized=cosine(rz[ids].mean(0),hz[ci].mean(0)),**{f'p{i}':float(pp[i]) for i in range(4)})
                        files.append(row);bagrows.append(row)
                    df=pd.DataFrame(bagrows)
                    for role,bags in [('train',trainbags),('test',testbags)]:
                        mask=rm.bag_id.isin(bags).to_numpy();ff=df[df.role==role]
                        base=dict(fold=fold,constraint=constraint,method=method,seed=seed,role=role)
                        metrics.append(dict(**base,unit='window',n=int(mask.sum()),correct=int((pr[mask].argmax(1)==rm.loc[mask,'label'].to_numpy()).sum()),accuracy=float(accuracy_score(rm.loc[mask,'label'],pr[mask].argmax(1))),macro_f1=float(f1_score(rm.loc[mask,'label'],pr[mask].argmax(1),labels=range(4),average='macro',zero_division=0))))
                        metrics.append(dict(**base,unit='recording',n=len(ff),correct=int(ff.label.eq(ff.prediction).sum()),accuracy=float(ff.label.eq(ff.prediction).mean()),macro_f1=float(f1_score(ff.label,ff.prediction,labels=range(4),average='macro',zero_division=0)),mse_standardized=float(ff.mse_standardized.mean()),cos_standardized=float(ff.cos_standardized.mean())))
                    checks.append(dict(tag=tag,reload_hidden_max_abs=err,head_frozen=True,negative_hidden_fraction=float((rh<0).mean())))
                    print(tag,round(metrics[-2]['accuracy'],4),flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(files).to_csv(OUT/'file_predictions.csv',index=False)
    (OUT/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    assert hashes=={p:sha(p) for p in hashes}
    (OUT/'verification.json').write_text(json.dumps(dict(models=checks,models_completed=len(checks),teacher_hashes=hashes,teacher_files_unchanged=True,elapsed_seconds=time.time()-started,script_sha256=sha(__file__)),indent=2)+'\n')
    d=pd.DataFrame(metrics);t=d[d.role.eq('test')];q=t.groupby(['constraint','method','unit'])[['correct','n']].sum();q['accuracy']=q.correct/q.n;q.to_csv(OUT/'summary.csv');print(q.to_string())

if __name__=='__main__':main()
