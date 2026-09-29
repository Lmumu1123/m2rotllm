"""Radar-only inference from exported features and a saved student checkpoint.

Contact samples and labels are not required. Probabilities from a known-wrong
target range are explicitly marked invalid for diagnosis.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from run_chain import RadarEncoder,global_probs,aggregate


def predict(model_path,features_path,device='cpu'):
    # These files are locally produced by run_chain.py and include NumPy scaler
    # arrays. Only load our trusted checkpoints, never arbitrary downloaded PT.
    ck=torch.load(model_path,map_location='cpu',weights_only=False)
    z=np.load(features_path,allow_pickle=False)
    x=np.asarray(z[ck['input_feature']],np.float32)
    if x.ndim!=2 or x.shape[1]!=ck['input_dim'] or not np.isfinite(x).all():
        raise ValueError('Invalid radar feature matrix')
    x=(x-ck['radar_mean'])/ck['radar_scale']
    encoder=RadarEncoder(ck['input_dim']);encoder.load_state_dict(ck['encoder'],strict=True)
    encoder.to(device).eval()
    with torch.inference_mode():
        h=encoder(torch.tensor(x,dtype=torch.float32,device=device)).cpu().numpy()
    if ck['output_head_type']=='original_fcn_10class':
        raw=h*ck['contact_scale']+ck['contact_mean']
        p10=softmax(raw@ck['original_coef'].T+ck['original_intercept'],axis=1)
        p=np.stack([p10[:,0],p10[:,1:4].sum(1),p10[:,7:10].sum(1),p10[:,4:7].sum(1)],axis=1)
    else:
        if ck['output_head_type']=='learned_radar_head':
            w=ck['own_head']['weight'].numpy();b=ck['own_head']['bias'].numpy()
        else:
            w=ck['contact_coef'];b=ck['contact_intercept']
        p=global_probs(softmax(h@w.T+b,axis=1),ck['classes'])
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=2e-6):
        raise ValueError('Invalid inference probabilities')
    return p,h,ck


def file_output(prob,meta):
    # group by bag and average probabilities; labels/state are not needed.
    temp=pd.DataFrame(prob,columns=['p0','p1','p2','p3'])
    temp['bag_id']=meta.bag_id.to_numpy()
    out=temp.groupby('bag_id',sort=True).mean().reset_index()
    out['candidate_class']=out[['p0','p1','p2','p3']].to_numpy().argmax(1)
    if 'geometry_roi_mismatch' in meta:
        flag=meta.groupby('bag_id').geometry_roi_mismatch.any()
        out['geometry_roi_mismatch']=out.bag_id.map(flag)
    else:out['geometry_roi_mismatch']=True
    out['diagnosis_status']=np.where(out.geometry_roi_mismatch,'invalid_or_unverified_target_roi','experimental_not_deployment_validated')
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--features',type=Path,required=True)
    ap.add_argument('--metadata',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cpu')
    a=ap.parse_args();torch.set_num_threads(2)
    p,h,ck=predict(a.model,a.features,a.device)
    meta=pd.read_csv(a.metadata)
    if len(meta)!=len(p):raise ValueError('Metadata and feature row count differs')
    out=file_output(p,meta);a.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print(out.to_string(index=False))
