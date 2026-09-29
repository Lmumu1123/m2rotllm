"""No training: check the time-order identifiability of the actual bag loss.

Permute the window feature multiset within each recording. This changes neither
per-window label CE nor any of the three bag-aggregation constraints. This is a
statement about the objective, not a test of physical waveform alignment.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax, logsumexp
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/home/huangyating/anomaly_detection/retention_alignment_20260922')
OLD=ROOT.parent/'four_class_chain_20260922'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'radar'))
from train_fixed_head import Encoder
GROUPS=[[0],[1,2,3],[7,8,9],[4,5,6]]


def group_probs(p):return np.stack([p[...,g].sum(-1) for g in GROUPS],-1)


def loss_values(cb,rb,labels,w,b,mean,scale):
    ct=np.stack([((h-mean)/scale).mean(0) for h in cb])
    rt=np.stack([((h-mean)/scale).mean(0) for h in rb])
    q=np.stack([group_probs(softmax((h@w.T+b)/2,axis=1)).mean(0) for h in cb])
    p=np.stack([group_probs(softmax((h@w.T+b)/2,axis=1)).mean(0) for h in rb])
    ce=np.mean([-np.log(group_probs(softmax(h@w.T+b,axis=1))[:,lab]).mean() for h,lab in zip(rb,labels)])
    mse=np.mean((rt-ct)**2)
    sims=(rt/np.maximum(np.linalg.norm(rt,axis=1,keepdims=True),1e-12))@(ct/np.maximum(np.linalg.norm(ct,axis=1,keepdims=True),1e-12)).T/.2
    con=(logsumexp(sims,axis=1)-logsumexp(np.where(labels[:,None]==labels[None,:],sims,-np.inf),axis=1)).mean()
    kd=4*np.mean(np.sum(q*(np.log(np.maximum(q,1e-10))-np.log(np.maximum(p,1e-10))),axis=1))
    return dict(CE=float(ce),bag_MSE=float(mse),bag_contrast=float(con),bag_KD=float(kd),
                total=float(ce+mse+.1*con+.5*kd))


def main():
    torch.set_num_threads(2)
    cm=pd.read_csv(ROOT/'preprocessing/variants/v0/retrained_fcn/metadata.csv')
    ch=np.load(ROOT/'preprocessing/variants/v0/retrained_fcn/contact_features.npz')['hidden_mean']
    rm=pd.read_csv(OLD/'radar/metadata.csv');rz=np.load(OLD/'radar/features.npz')
    rng=np.random.default_rng(20260922);rows=[]
    for fold in ['115200_to_460800','460800_to_115200']:
        path=ROOT/'radar/stage_b_v0/models/four_class_provisional_roi'/fold/'ce_feat_1_kd.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        enc=Encoder(ck['input_dim'],ck['contact_mean'],ck['contact_scale']).eval();enc.load_state_dict(ck['encoder'])
        xx=torch.tensor(((rz[ck['feature']]-ck['radar_mean'])/ck['radar_scale']).astype(np.float32))
        with torch.inference_mode():rh=enc(xx).numpy().astype(np.float64)
        cb=[ch[cm.bag_id.eq(b)].astype(np.float64) for b in ck['train_bags']]
        rb=[rh[rm.bag_id.eq(b)] for b in ck['train_bags']]
        labels=np.array([int(cm.loc[cm.bag_id.eq(b),'label'].iloc[0]) for b in ck['train_bags']])
        args=(labels,ck['head_weight'].numpy().astype(np.float64),ck['head_bias'].numpy().astype(np.float64),ck['contact_mean'],ck['contact_scale'])
        original=loss_values(cb,rb,*args)
        for i in range(50):
            val=loss_values([h[rng.permutation(len(h))] for h in cb],[h[rng.permutation(len(h))] for h in rb],*args)
            for name in original:rows.append(dict(fold=fold,permutation=i,term=name,original=original[name],permuted=val[name],absolute_difference=abs(original[name]-val[name])))
    df=pd.DataFrame(rows);df.to_csv(OUT/'actual_loss_permutation_checks.csv',index=False)
    assert df.absolute_difference.max()<1e-10
    n=200;t=np.arange(n)/n
    hc=np.stack([2+np.sin(2*np.pi*t),2+np.cos(2*np.pi*t)],1)
    hr=np.roll(hc,50,axis=0)
    bag_mse=float(np.mean((hc.mean(0)-hr.mean(0))**2));paired_mse=float(np.mean((hc-hr)**2))
    # A separate counterexample: equal first moments do not imply equal distributions.
    first_moments=dict(P=[0.,2.],Q=[1.,1.],mean_P=1.,mean_Q=1.,variance_P=1.,variance_Q=0.)
    write=dict(status='passed',training=False,actual_frozen_models=2,permutations_per_model=50,
       max_loss_change=float(df.absolute_difference.max()),
       loss_check_scope='All windows per training bag, equally weighted bag CE; same mathematical loss terms as training, no random 16-window minibatch/dropout',
       what_is_permuted='Order of complete window embeddings after feature extraction; not waveform samples inside a window',
       exact_theoretical_invariance=True,synthetic_quarter_cycle_shift=dict(bag_mean_MSE=bag_mse,paired_time_MSE=paired_mse),
       equal_mean_unequal_distribution=first_moments,
       conclusion='Current bag objective cannot identify time order or certify event alignment. No claim that real bearing diagnosis never needs time alignment.')
    (OUT/'verification.json').write_text(json.dumps(write,ensure_ascii=False,indent=2)+'\n')
    fig,ax=plt.subplots(1,2,figsize=(11,3.8))
    ax[0].plot(t,hc[:,0],label='Contact synthetic feature',color='#177f8c',lw=2)
    ax[0].plot(t,hr[:,0],label='Radar: quarter-cycle shift',color='#be5a53',lw=2)
    ax[0].axhline(2,color='gray',ls='--',label='Identical mean')
    ax[0].set(xlabel='Normalized time',ylabel='Synthetic feature value',title='Different event timing; identical mean');ax[0].legend(fontsize=8)
    ax[1].bar(['Bag mean MSE','Paired-time MSE'],[bag_mse,paired_mse],color=['#177f8c','#be5a53'])
    ax[1].set(ylabel='Mean squared error',title='The bag objective misses the time shift',ylim=(0,1.2))
    ax[1].text(0,.035,'0',ha='center');ax[1].text(1,1.035,f'{paired_mse:.1f}',ha='center')
    fig.suptitle('Mathematical counterexample — not a physical sensor experiment',fontsize=12)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'time_identifiability_counterexample.{ext}',dpi=180)
    print(json.dumps(write,ensure_ascii=False))


if __name__=='__main__':main()
