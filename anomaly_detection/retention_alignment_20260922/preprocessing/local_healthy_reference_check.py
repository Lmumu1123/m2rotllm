#!/usr/bin/env python3
"""Explicit extra healthy-calibration protocol, separate from zero-calibration CV."""
from pathlib import Path
import importlib.util, json, time
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('variants_api',HERE/'export_variants.py')
variants=importlib.util.module_from_spec(spec);spec.loader.exec_module(variants)
api=variants.api

def main():
    out=HERE/'local_healthy_reference';out.mkdir(exist_ok=False)
    api.savej(out/'protocol.json',dict(created_before_inference=True,
        purpose='Frozen original FCN with extra local healthy calibration; not zero-shot or main no-calibration holdout',
        query_scope='8 known-class main-rig files only; 58 windows',
        reference_selection='For every query recording, choose normal recording with opposite baud, independent of query fault label',
        reference_windows='Deterministic linspace indices, at most 4 per reference recording; no outcome search',
        axis_pairing='x-to-x, y-to-y, z-to-z; mean probabilities and hidden across reference windows then axes',
        variants=['v0','query_no_demean'],model='retrained_fcn',weights_frozen=True,
        calibration_limit='Same rig/cfg/distance user-confirmed, but RPM/load unconfirmed. Do not claim exact same-condition reference.',
        external_files='Neither keep nor bigNormal is used as reference or query',
        leakage_boundary='Reference is disjoint recording from each query; this protocol may use the other baud normal for calibration and is not the prior holdout protocol. Training/CV scores must not be pooled with this diagnostic.'))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    _,md,_=api.read_inputs(variants.DATA)
    md=md[md.label>=0].copy().reset_index(drop=True)
    raw=np.stack([np.load(variants.DATA/r.file,allow_pickle=False)['raw_xyz'][r.window_row] for r in md.itertuples()])
    model,prov=api.load_model('retrained_fcn');model=model.to('cuda:1')
    frozen={k:v.cpu().clone() for k,v in model.state_dict().items()}
    results=[];refs_manifest=[]
    for name,demean in [('v0',True),('query_no_demean',False)]:
        q=variants.transformed_queries(raw,demean)
        hidden=np.zeros((len(md),3,128),np.float32);p10=np.zeros((len(md),3,10),np.float32)
        for file,rows in md.groupby('file',sort=True).groups.items():
            idx=np.asarray(list(rows));baud=int(md.iloc[idx[0]].baud)
            normal_md=md[(md.label==0)&(md.baud!=baud)]
            assert normal_md.file.nunique()==1 and file!=normal_md.file.iloc[0]
            pick=np.linspace(0,len(normal_md)-1,min(4,len(normal_md)),dtype=int)
            ri=normal_md.index.to_numpy()[pick]
            refs_manifest.append(dict(variant=name,query_file=file,reference_file=normal_md.file.iloc[0],
                reference_window_rows=md.iloc[ri].window_row.to_list(),query_baud=baud,reference_baud=int(normal_md.baud.iloc[0])))
            with torch.inference_mode():
                for axis in range(3):
                    for ref_i in ri:
                        x=np.stack([q[idx,axis],np.broadcast_to(q[ref_i,axis],(len(idx),24000))],axis=1)
                        feat=model.encoder(torch.from_numpy(x).to('cuda:1'))
                        h=torch.relu(model.classifier.linear1(feat.flatten(1)))
                        pp=model.classifier.linear2(h).softmax(-1)
                        hidden[idx,axis]+=h.cpu().numpy()/len(ri)
                        p10[idx,axis]+=pp.cpu().numpy()/len(ri)
        avg=p10.mean(1);p4=np.stack([avg[:,g].sum(1) for g in api.MAP],axis=1)
        pred=md.copy();pred['pred4']=p4.argmax(1)
        for i in range(4):pred[f'p4_{i}']=p4[:,i]
        pred.to_csv(out/f'{name}_window_predictions.csv',index=False)
        bag=pred.groupby(['file','label','state','baud'],as_index=False)[[f'p4_{i}' for i in range(4)]].mean()
        bag['pred4']=bag[[f'p4_{i}' for i in range(4)]].to_numpy().argmax(1)
        bag.to_csv(out/f'{name}_file_predictions.csv',index=False)
        np.savez_compressed(out/f'{name}_features.npz',hidden_mean=hidden.mean(1),hidden_axes=hidden,
            probs10_mean=avg,probs10_axes=p10,probs4_mean=p4,labels=md.label.to_numpy(),
            file_names=md.file.to_numpy(dtype=str),window_rows=md.window_row.to_numpy(),
            baud=md.baud.to_numpy(),bag_id=md.bag_id.to_numpy(dtype=str),states=md.state.to_numpy(dtype=str))
        for scope,frame in [('all8',bag),('queries115200',bag[bag.baud==115200]),('queries460800',bag[bag.baud==460800])]:
            results.append(dict(variant=name,scope=scope,**api.metrics(frame)))
        print(name,results[-3:],flush=True)
    assert all(torch.equal(frozen[k],v.cpu()) for k,v in model.state_dict().items())
    api.savej(out/'reference_manifest.json',refs_manifest)
    api.savej(out/'metrics.json',results)
    pd.DataFrame([{k:v for k,v in r.items() if k!='confusion_matrix'} for r in results]).to_csv(out/'metrics.csv',index=False)
    api.savej(out/'provenance.json',{**prov,'all_weights_and_bn_buffers_unchanged':True,
        'source_sha256':api.sha(__file__),'mean_rule':'query and local healthy reference both receive same demean variant',
        'parent_export_sha256':api.sha(HERE/'export_variants.py')})
    md.to_csv(out/'metadata.csv',index=False)

if __name__=='__main__':main()
