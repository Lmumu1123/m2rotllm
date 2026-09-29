"""Conservative all-known head: additionally gate every source validation set.

This is a disclosed second development iteration, after v1's subgroup losses
were inspected. Source test is reused for regression, not a new blind test.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.special import softmax
from train_retained_head import ROOT,ResidualHead,group_logits,group_probs,source_prob,score,file_prob,file_score,writej,sha

OUT=ROOT/'results/retained_head_conservative'

def main():
    torch.set_num_threads(2)
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    writej(OUT/'protocol.json',dict(iteration='second development iteration after v1 subgroup review',
        source_test_is_reused_regression_not_new_blind_test=True,
        source_weights=[10,100],steps=[400,800],variants=['v0','query_no_demean'],seed=42,
        model_and_loss='Identical to train_retained_head.py; frozen FCN and original linear1',
        acceptance='source val each source dataset and global ten/four accuracy drop <=.5 percentage points',
        selection='Among passing candidates maximize local TRAIN file macro-F1, then source-val ten accuracy, then source-val four accuracy, then minimum delta parameter norm; no new source test scores used.',
        deployment_rejection='Global and every source test ten/four regression checked after selection; report failures without selecting another checkpoint from test outcomes.',
        reason='Global aggregate v1 tolerance hid >.5pp declines in some datasets. Stronger conservative constraint rather than more local CE fitting.'))
    z=np.load(ROOT/'source/cache.npz');o=np.load(ROOT/'source/original_head.npz')
    h=z['hidden_refs'];split=z['split'];y10=z['label10'];y4=z['label4'];names=z['source'];orig=z['probs10_mean']
    tr=np.flatnonzero(split=='train');va=np.flatnonzero(split=='val');te=np.flatnonzero(split=='test')
    hs=h[tr].reshape(-1,128);mean=hs.mean(0);scale=np.maximum(hs.std(0),.01)
    xs=torch.tensor(hs);ys=torch.tensor(np.repeat(y10[tr],3));w0=o['weight10'];b0=o['bias10']
    rows=[];groups=[];preds=[];chosen=[]
    for variant in ['v0','query_no_demean']:
        v=ROOT/'preprocessing/variants'/variant/'retrained_fcn'
        ch=np.load(v/'contact_features.npz')['hidden_mean'];md=pd.read_csv(v/'metadata.csv')
        mask=md.label.ge(0).to_numpy();bags=sorted(md.loc[mask,'bag_id'].unique())
        pools=[np.flatnonzero(mask&md.bag_id.eq(b).to_numpy()) for b in bags]
        by=np.array([int(md.loc[md.bag_id==b,'label'].iloc[0]) for b in bags]);cx=torch.tensor(ch)
        snapshots={};records=[]
        for lam in [10,100]:
            torch.manual_seed(42);rng=np.random.default_rng(42)
            model=ResidualHead(w0,b0,mean,scale);opt=torch.optim.AdamW(model.delta.parameters(),lr=.003,weight_decay=0)
            for step in range(1,801):
                si=rng.integers(len(xs),size=256);li=np.concatenate([rng.choice(p,size=16,replace=True) for p in pools])
                lc=F.cross_entropy(group_logits(model(cx[li])),torch.tensor(np.repeat(by,16)))
                new=model(xs[si]);old=F.linear(xs[si],model.original_w,model.original_b)
                ce=F.cross_entropy(new,ys[si]);kd=F.kl_div(F.log_softmax(new/2,-1),F.softmax(old/2,-1),reduction='batchmean')*4
                reg=model.delta.weight.square().mean()+model.delta.bias.square().mean()
                loss=lc+lam*(ce+2*kd)+.001*reg
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.delta.parameters(),5);opt.step()
                if step not in [400,800]:continue
                w,b=model.effective();p=source_prob(h[va],w,b);passed=True
                for source in ['ALL']+sorted(set(names[va])):
                    ids=np.arange(len(va)) if source=='ALL' else np.flatnonzero(names[va]==source)
                    old10=score(y10[va][ids],orig[va][ids],10);old4=score(y4[va][ids],group_probs(orig[va][ids]),4)
                    s10=score(y10[va][ids],p[ids],10);s4=score(y4[va][ids],group_probs(p[ids]),4)
                    d10=s10['accuracy']-old10['accuracy'];d4=s4['accuracy']-old4['accuracy']
                    ok=d10>=-.005-1e-12 and d4>=-.005-1e-12;passed&=ok
                    groups.append(dict(variant=variant,lambda_source=lam,step=step,split='val',source=source,
                        acc10=s10['accuracy'],acc4=s4['accuracy'],delta_acc10_pp=100*d10,delta_acc4_pp=100*d4,passes=ok,n=len(ids)))
                lp=group_probs(softmax(ch[mask]@w.T+b,axis=1));lf=file_score(lp,md[mask].reset_index(drop=True))
                rec=dict(variant=variant,lambda_source=lam,step=step,all_source_val_gate=passed,
                    val_acc10=score(y10[va],p,10)['accuracy'],val_acc4=score(y4[va],group_probs(p),4)['accuracy'],
                    local_train_macro_f1=lf['macro_f1'],delta_norm=float(reg.detach()))
                records.append(rec);rows.append(rec);snapshots[(lam,step)]=(w,b)
        valid=[r for r in records if r['all_source_val_gate']]
        if valid:
            sel=sorted(valid,key=lambda r:(-r['local_train_macro_f1'],-r['val_acc10'],-r['val_acc4'],r['delta_norm']))[0]
            w,b=snapshots[(sel['lambda_source'],sel['step'])]
        else:sel=dict(variant=variant,lambda_source=0,step=0,status='original_fallback');w,b=w0,b0
        dest=OUT/'models'/variant/'all_known';dest.mkdir(parents=True)
        writej(dest/'selection_before_test.json',sel);chosen.append(sel)
        np.savez_compressed(dest/'retained_head.npz',weight10=w,bias10=b,original_weight10=w0,original_bias10=b0,
            local_training_bags=np.array(bags),variant=np.array(variant),fold=np.array('all_known'))
        p=source_prob(h[te],w,b);group_ok=[]
        for source in ['ALL']+sorted(set(names[te])):
            ids=np.arange(len(te)) if source=='ALL' else np.flatnonzero(names[te]==source)
            s10=score(y10[te][ids],p[ids],10);s4=score(y4[te][ids],group_probs(p[ids]),4)
            d10=s10['accuracy']-score(y10[te][ids],orig[te][ids],10)['accuracy']
            d4=s4['accuracy']-score(y4[te][ids],group_probs(orig[te][ids]),4)['accuracy']
            ok=d10>=-.005-1e-12 and d4>=-.005-1e-12;group_ok.append(ok)
            groups.append(dict(variant=variant,lambda_source=sel['lambda_source'],step=sel['step'],split='test',source=source,
                acc10=s10['accuracy'],acc4=s4['accuracy'],delta_acc10_pp=100*d10,delta_acc4_pp=100*d4,passes=ok,n=len(ids)))
        lp=group_probs(softmax(ch@w.T+b,axis=1))
        pp=file_prob(lp,md);pp['variant']=variant;preds.extend(pp.to_dict('records'))
        np.savez_compressed(dest/'local_teacher_targets.npz',hidden_mean=ch,probs10=softmax(ch@w.T+b,axis=1),
            probs4=lp,bag_id=md.bag_id.to_numpy(dtype=str),labels=md.label.to_numpy())
        writej(dest/'acceptance.json',dict(variant=variant,all_known=True,passes_global_and_every_source_test=all(group_ok),
            source_test_is_reused_regression_not_new_blind_test=True,local_train_accuracy=file_score(lp[mask],md[mask].reset_index(drop=True))['accuracy'],
            status='passes_stronger_source_regression' if all(group_ok) else 'rejected_for_subgroup_decline',original_encoder_unchanged=True))
        print(json.dumps(dict(selection=sel,passes_every_test_source=all(group_ok))),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'candidates.csv',index=False)
    pd.DataFrame(groups).to_csv(OUT/'source_group_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(OUT/'contact_predictions.csv',index=False)
    writej(OUT/'selections.json',chosen)
    writej(OUT/'completion.json',dict(status='complete',script_sha256=sha(__file__),modified_original_weights=False))

if __name__=='__main__':main()
