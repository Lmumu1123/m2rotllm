#!/usr/bin/env python3
"""Audit a released, frozen UniFault Tiny checkpoint and fit contact-only readouts.

The reviewed official network is reused after removing optional training-only
dependencies. No pretrained model is trained here; only sklearn linear heads.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import hashlib, importlib.util, json, subprocess, sys, time, warnings
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE=Path(__file__).resolve().parent
VENDOR=BASE/'vendor/UniFault'
WEIGHTS=BASE/'vendor/weights/pretrain-epoch=1.ckpt'
CONTACT=Path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results')
MD=Path('/home/huangyating/anomaly_detection/encoder_validation_20260923/rotllm/metadata.csv')
VARIANTS=['paper_100ms','code_native1024']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savej(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load_official():
    """Transform only dependency plumbing, keeping checkpoint names/operations."""
    text=(VENDOR/'model/model.py').read_text()
    substitutions={
        'import pytorch_lightning as L':'# Dependency-only adaptation: nn.Module below.',
        'from einops import rearrange':'# Single rearrange below replaced by exact reshape.',
        'from timm.models.layers import DropPath':'# DropPath occurs only in unused AttentionBlock; never called.',
        'L.LightningModule':'nn.Module',
        'from .Transformer_utils import':'from model.Transformer_utils import',
        "x = rearrange(x, 'b m n p -> (b m) n p')":"x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])"
    }
    counts={}
    for old,new in substitutions.items():
        counts[old]=text.count(old); assert counts[old]>0,old
        text=text.replace(old,new)
    target=BASE/'official_model_inference_compat.py';target.write_text(text)
    sys.path.insert(0,str(VENDOR))
    spec=importlib.util.spec_from_file_location('official_unifault_compat',target)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=SimpleNamespace(seq_len=1024,patch_size=64,embed_dim=128,dropout=.3,heads=4,depth=4,num_classes=16,num_channels=3)
    model=module.Transformer_bkbone(args)
    unsupported=torch.serialization.get_unsafe_globals_in_checkpoint(WEIGHTS)
    assert not unsupported
    ck=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    state={k.removeprefix('model.'):v for k,v in ck['state_dict'].items()}
    assert set(state)==set(model.state_dict())
    model.load_state_dict(state,strict=True)
    for p in model.parameters():p.requires_grad_(False)
    model.eval().to('cuda:0')
    info=dict(official_repository='https://github.com/emadeldeen24/UniFault',
        official_download_link='https://tinyurl.com/mtdytebs',
        resolved_official_folder='https://www.dropbox.com/scl/fo/kolaib1t6o2q96b7qxbsn/AFEUxkRP54i3xg6OsOQzhJ4?rlkey=ve3mt6w9tfbi5khkn8s8eqvwt&dl=1',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=VENDOR,text=True).strip(),
        checkpoint_sha256=sha(WEIGHTS),zip_sha256=sha(BASE/'vendor/UniFault_Tiny.zip'),
        checkpoint_epoch=ck['epoch'],checkpoint_global_step=ck['global_step'],
        state_keys=len(state),strict_coverage=1.,parameters=sum(p.numel() for p in model.parameters()),
        dependency_only_substitutions=counts,unsafe_checkpoint_globals=unsupported,
        native_head=dict(outputs=16,semantic_mapping_verified=False,reported_four_class_accuracy=False),
        forward_features='signed 128-dimensional LayerNorm features, mean over 16 tokens and 3 axes',
        source_sha256={str(p.relative_to(VENDOR)):sha(p) for p in [VENDOR/'model/model.py',VENDOR/'model/Transformer_utils.py',VENDOR/'data_preprocessing/preprocess_general.py',VENDOR/'datalaoders/train_dataloader.py']})
    return model,info

def get_features(model,x):
    # x: existing one-second windows x chunks x XYZ x samples.
    n,k,c,l=x.shape;assert (c,l)==(3,1024)
    h=[];native=[]
    with torch.inference_mode():
        for start in range(0,n*k,64):
            batch=torch.tensor(x.reshape(-1,c,l)[start:start+64],device='cuda:0').to(torch.bfloat16).float()
            tokens=model(batch)
            axes=tokens.reshape(-1,c,16,128).mean(2)
            h.append(axes.cpu().numpy())
            native.append(model.predict(tokens).cpu().numpy())
    hs=np.concatenate(h).reshape(n,k,c,128)
    native=np.concatenate(native).reshape(n,k,16)
    return hs,native

def rows_metrics(df,cols):
    p=df[cols].to_numpy();yp=p.argmax(1);y=df.label.to_numpy()
    return dict(n=len(y),correct=int((y==yp).sum()),accuracy=float(accuracy_score(y,yp)),
        macro_f1=float(f1_score(y,yp,labels=[0,1,2,3],average='macro',zero_division=0)),
        confusion_matrix=confusion_matrix(y,yp,labels=[0,1,2,3]).tolist())

def main():
    assert 'envs/m2vllm/' in sys.executable
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();protocol=json.loads((BASE/'protocol_before_results.json').read_text())
    out=BASE/'unifault';out.mkdir(exist_ok=True)
    assert not (out/'metrics.csv').exists(),'Refuse overwrite finished results'
    model,provenance=load_official()
    state_before={k:v.cpu().clone() for k,v in model.state_dict().items()}
    md=pd.read_csv(MD);md.to_csv(out/'metadata.csv',index=False)
    raw=np.stack([np.load(CONTACT/row.file,allow_pickle=False)['raw_xyz'][row.window_row] for row in md.itertuples()]).astype(np.float32)
    assert raw.shape==(88,4000,3) and (md.label>=0).sum()==58
    hashes={str(CONTACT/f):sha(CONTACT/f) for f in md.file.unique()}
    hashes[str(WEIGHTS)]=sha(WEIGHTS)
    provenance.update(protocol_sha256=sha(BASE/'protocol_before_results.json'),script_sha256=sha(__file__),input_hashes=hashes)
    savej(out/'provenance.json',provenance)
    np.savez_compressed(out/'native_head16_unmapped.npz',weight16=model.head.weight.cpu().numpy(),bias16=model.head.bias.cpu().numpy())
    metrics=[];checks=[];pooled=[];splits=[]
    for variant in VARIANTS:
        if variant=='paper_100ms':
            z=raw.reshape(88,10,400,3).transpose(0,1,3,2)
            z=resample_poly(z,64,25,axis=-1).astype(np.float32)
        else:
            z=raw[:,:3072].reshape(88,3,1024,3).transpose(0,1,3,2)
        assert z.shape[-2:]==(3,1024)
        for trainbaud,testbaud in [(115200,460800),(460800,115200)]:
            fold=f'{trainbaud}_to_{testbaud}';dest=out/variant/fold;dest.mkdir(parents=True,exist_ok=True)
            tr=((md.label>=0)&(md.baud==trainbaud)).to_numpy();te=((md.label>=0)&(md.baud==testbaud)).to_numpy()
            assert not set(md.loc[tr,'bag_id'])&set(md.loc[te,'bag_id'])
            lo=z[tr].min((0,1,3));hi=z[tr].max((0,1,3));assert (hi>lo).all()
            normalized=(z-lo[None,None,:,None])/(hi-lo+1e-6)[None,None,:,None]
            haxes,l16=get_features(model,normalized)
            h=haxes.mean((1,2));assert h.shape==(88,128) and np.isfinite(h).all()
            scalar=StandardScaler().fit(h[tr])
            clf=LogisticRegression(C=1.,max_iter=2000,solver='lbfgs',class_weight='balanced',random_state=42)
            with warnings.catch_warnings(record=True) as ww:
                clf.fit(scalar.transform(h[tr]),md.loc[tr,'label'])
            assert list(clf.classes_)==[0,1,2,3]
            w=clf.coef_/scalar.scale_[None,:];b=clf.intercept_-w@scalar.mean_
            probs=softmax(h.astype(float)@w.T+b,axis=1)
            err=float(abs(probs-clf.predict_proba(scalar.transform(h))).max());assert err<2e-5
            np.savez_compressed(dest/'teacher_features.npz',hidden_mean=h,hidden_axes=haxes.mean(1),
                hidden_subwindows=haxes.mean(2),hidden_subwindows_axes=haxes,logits16_subwindows=l16,
                labels=md.label.to_numpy(),label=md.label.to_numpy(),baud=md.baud.to_numpy(),
                bag_id=md.bag_id.to_numpy(str),file_names=md.file.to_numpy(str),window_rows=md.window_row.to_numpy(),
                states=md.state.to_numpy(str),input_min=lo,input_max=hi,output_signed=np.array(True))
            np.savez_compressed(dest/'head.npz',weight4=w,bias4=b,classes=clf.classes_,
                scaler_mean=scalar.mean_,scaler_scale=scalar.scale_,train_bag_ids=md.loc[tr,'bag_id'].unique().astype(str),
                test_bag_ids=md.loc[te,'bag_id'].unique().astype(str))
            cols=[f'p{k}' for k in range(4)];df=md.copy()
            for k in range(4):df[f'p{k}']=probs[:,k]
            df['pred4']=probs.argmax(1);df['split']=np.where(tr,'train',np.where(te,'test','external'))
            ff=df.groupby(['file','label','state','baud','bag_id','split'],as_index=False)[cols].mean();ff['pred4']=ff[cols].to_numpy().argmax(1)
            df.to_csv(dest/'window_predictions.csv',index=False);ff.to_csv(dest/'file_predictions.csv',index=False)
            md.to_csv(dest/'metadata.csv',index=False)
            for unit,frame in [('window',df),('recording',ff)]:
                for split in ['train','test']:
                    metrics.append(dict(variant=variant,fold=fold,unit=unit,split=split,**rows_metrics(frame[frame.split==split],cols)))
                sub=frame[(frame.split=='external')&(frame.state=='bigNormal')]
                metrics.append(dict(variant=variant,fold=fold,unit=unit,split='external_normal_only',n=len(sub),correct=int((sub.pred4==0).sum()),accuracy=float((sub.pred4==0).mean()),macro_f1=None))
                temp=frame[frame.split=='test'].copy();temp['variant']=variant;temp['fold']=fold;temp['unit']=unit;pooled.append(temp)
            checks.append(dict(variant=variant,fold=fold,folded_head_max_probability_error=err,
                normalized_min=float(normalized.min()),normalized_max=float(normalized.max()),
                signed_hidden_fraction=float((h<0).mean()),training_windows=int(tr.sum()),test_windows=int(te.sum()),
                warnings=[str(w.message) for w in ww],head_iterations=clf.n_iter_.tolist()))
            splits.append(dict(variant=variant,fold=fold,train_bags=md.loc[tr,'bag_id'].unique().tolist(),test_bags=md.loc[te,'bag_id'].unique().tolist(),
                features=str(dest/'teacher_features.npz'),head=str(dest/'head.npz')))
            print(variant,fold,[r for r in metrics[-6:] if r['split']=='test'],flush=True)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    allpred=pd.concat(pooled,ignore_index=True);allpred.to_csv(out/'pooled_test_predictions.csv',index=False)
    pm=[]
    for (variant,unit),df in allpred.groupby(['variant','unit']):
        pm.append(dict(variant=variant,unit=unit,**rows_metrics(df,cols)))
    pd.DataFrame(pm).to_csv(out/'pooled_metrics.csv',index=False)
    savej(out/'splits.json',splits)
    for k,v in model.state_dict().items():assert torch.equal(state_before[k],v.cpu()),k
    assert all(sha(p)==s for p,s in hashes.items())
    savej(out/'verification.json',dict(encoder_parameters_and_buffers_unchanged=True,inputs_and_checkpoint_unchanged=True,
        complete_strict_checkpoint_loaded=True,native_16_class_semantics_unknown=True,
        no_native_four_class_accuracy_claim=True,no_local_encoder_training=True,
        contact_normalization_and_head_use_train_only=True,all_predeclared_variants_reported=True,
        all_feature_arrays_finite=True,checks=checks,elapsed_seconds=time.monotonic()-start))
    print(pd.DataFrame(pm).drop(columns='confusion_matrix').to_string(index=False),flush=True)

if __name__=='__main__':main()
