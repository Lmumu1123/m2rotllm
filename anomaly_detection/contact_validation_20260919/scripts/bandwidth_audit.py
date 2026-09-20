"""Exploratory frozen-checkpoint bandwidth audit on a fixed MBHM test subset.

This masks existing one-second DCN coefficients. It is not a simulation of a
real accelerometer's antialias filter, noise, mount, housing or aliasing.
"""
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.metrics import accuracy_score,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from contact_pipeline import ROOT,write_json
from teacher_bridge import REPO,ASSETS,CHECKPOINTS,load_teacher,embed
from scripts.evaluate_mbhm import read_metadata,make_protocol,LABELS


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset=ASSETS/'mbhm_dataset'
    allpairs,_=make_protocol(read_metadata(dataset),42,3)
    candidates=[r for r in allpairs if r['split']=='test' and r['reference_index']==0]
    rng=np.random.default_rng(20260919);chosen=[]
    for label in range(10):
        pool=[r for r in candidates if r['label']==label]
        chosen.extend(pool[i] for i in sorted(rng.choice(len(pool),size=min(50,len(pool)),replace=False)))
    chosen=sorted(chosen,key=lambda r:r['query_id'] if 'query_id' in r else r['file_id'])
    meta=pd.DataFrame(chosen)
    print('Columns:',list(meta.columns),'rows',len(meta),flush=True)
    # Released protocol uses vib_id/ref_id; explicitly identify actual schema.
    query_key=next(k for k in ['file_id','vib_id','query_id'] if k in meta)
    ref_key=next(k for k in ['ref_id','reference_id'] if k in meta)
    qids=meta[query_key].to_numpy();rids=meta[ref_key].to_numpy()
    ids=np.unique(np.r_[qids,rids]);lookup={v:i for i,v in enumerate(ids)}
    with h5py.File(dataset/'data.hdf5','r') as f:
        store=np.stack([f['vibration'][int(i)] for i in ids])
    x=np.stack([store[[lookup[i] for i in qids]],store[[lookup[i] for i in rids]]],axis=1)
    assert x.shape==(len(meta),2,24000)
    meta.to_csv(ROOT/'results/bandwidth_subset.csv',index=False)
    original_energy=np.sum(x*x,axis=2)
    bands={'native_DCN':None,'below_2000Hz':4000,'below_800Hz':1600}
    results=[];predictions=[];representations={};energy=[];start=time.monotonic()
    for band,cut in bands.items():
        if cut is not None:
            retained=np.sum(x[:,:,:cut]**2,axis=2)/original_energy
            for i,row in meta.iterrows():
                energy.append(dict(query_id=int(qids[i]),label=int(row.label),band=band,
                    query_energy_fraction=float(retained[i,0]),reference_energy_fraction=float(retained[i,1])))
    for name in CHECKPOINTS:
        model=load_teacher(name,device)
        initial_buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
        for band,cut in bands.items():
            current=x.copy()
            if cut is not None:
                current[:,:,cut:]=0
                denominator=np.sum(current**2,axis=2,keepdims=True)
                if np.any(denominator<=0):raise ValueError('Empty retained band')
                current*=.01*np.sqrt(24000/denominator)
            logits=[];hidden=[]
            for i in range(0,len(x),32):
                _,h,l=embed(model,torch.as_tensor(current[i:i+32],device=device))
                logits.append(l.cpu().numpy());hidden.append(h.cpu().numpy())
            logits=np.concatenate(logits);hidden=np.concatenate(hidden)
            p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
            pred=p.argmax(1);y=meta.label.to_numpy()
            row=dict(model=name,band=band,queries=len(y),accuracy=float(accuracy_score(y,pred)),
                     macro_f1=float(f1_score(y,pred,labels=list(range(10)),average='macro',zero_division=0)),
                     healthy_false_alarm=float(np.mean(pred[y==0]!=0)),fault_miss=float(np.mean(pred[y!=0]==0)))
            results.append(row)
            for i in range(len(meta)):
                predictions.append(dict(model=name,band=band,query_id=int(qids[i]),ref_id=int(rids[i]),
                    label=int(y[i]),prediction=int(pred[i]),p_healthy=float(p[i,0])))
            representations[name+'__'+band]=hidden
            print(row,flush=True)
        assert all(torch.equal(v.detach().cpu(),initial_buffers[k]) for k,v in model.named_buffers()),'BatchNorm mutated'
        del model
        if device!='cpu':torch.cuda.empty_cache()
    results=pd.DataFrame(results)
    results.to_csv(ROOT/'results/bandwidth_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(ROOT/'results/bandwidth_predictions.csv',index=False)
    pd.DataFrame(energy).to_csv(ROOT/'results/retained_energy.csv',index=False)
    np.savez_compressed(ROOT/'results/mbhm_hidden_embeddings.npz',**representations)
    savehash={}
    for name,d in CHECKPOINTS.items():
        savehash[name]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.suffix in ['.pth','.safetensors','.json']}
    write_json(ROOT/'results/bandwidth_protocol.json',dict(device=device,torch=torch.__version__,python=sys.executable,
        seed=20260919,source_protocol_seed=42,query_count=len(meta),per_label=meta.label.value_counts().sort_index().to_dict(),
        self_reference_pairs=int(np.sum(qids==rids)),query_role_split='Test under reproduced split; original official checkpoint training membership is unknown',
        reference_pool='Original protocol: all healthy same-condition records; can cross split boundaries',
        band_mapping='Assumes original 1-second DCN, f[k]=k/2 Hz; retained band re-normalized to RMS 0.01',
        is_hardware_simulation=False,is_current_contact_dataset=False,hyperparameter_tuning=False,
        frozen_batchnorm_buffers_verified=True,checkpoints=savehash,wall_seconds=time.monotonic()-start))
    fig,ax=plt.subplots(figsize=(10,4.5))
    results.pivot(index='model',columns='band',values='macro_f1')[list(bands)].plot.bar(ax=ax,rot=10,color=['#2875B8','#D78428','#168C80'])
    ax.set(ylabel='Macro-F1 (fixed ten classes)',xlabel='',ylim=(0,1.05),title='Exploratory MBHM bandwidth audit: fixed query subset, frozen models')
    ax.legend(title='DCT frequency band');fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/'figures'/('02_bandwidth_audit.'+ext),dpi=170,bbox_inches='tight')
    print('Completed bandwidth audit',flush=True)


if __name__=='__main__':main()
