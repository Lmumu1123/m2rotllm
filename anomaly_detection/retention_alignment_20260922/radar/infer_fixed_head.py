"""Radar-only inference into the frozen original BearLLM head.

No contact features or ground-truth labels are read by this script. All learned
scalers, encoder and original ten-class head are in the trusted local checkpoint.
The ten-class probabilities can feed the original linear3 language projector.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from train_fixed_head import Encoder,grouping

def geometry_status(meta):
    """Only explicit boolean false flags verify compatible geometry.

    Missing flags, NaN, numeric guesses and unrecognized strings are unknown,
    never evidence of a valid target range gate.
    """
    if 'geometry_roi_mismatch' not in meta:
        return False,'unverified_geometry_do_not_issue_diagnosis'
    flags=[]
    for value in meta['geometry_roi_mismatch']:
        if isinstance(value,(bool,np.bool_)):flags.append(bool(value))
        elif isinstance(value,str) and value.strip().lower() in ('true','false'):flags.append(value.strip().lower()=='true')
        else:flags.append(None)
    if any(value is True for value in flags):return False,'invalid_geometry_do_not_issue_diagnosis'
    if not flags or any(value is None for value in flags):return False,'unverified_geometry_do_not_issue_diagnosis'
    return True,'experimental_recording_prediction'

def infer(args):
    torch.set_num_threads(2)
    if args.output.exists() and any(args.output.iterdir()):raise RuntimeError('Use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.model,map_location='cpu',weights_only=False)
    z=np.load(args.features,allow_pickle=False)
    if ck['feature']=='smooth_shape':
        centers=(z['band_edges_hz'][:-1]+z['band_edges_hz'][1:])/2
        kernel=np.exp(-.5*((centers[:,None]-centers[None,:])/5.)**2);kernel/=kernel.sum(1,keepdims=True)
        x=z['frame_shape']@kernel.T;x-=x.mean(1,keepdims=True)
    else:x=z[ck['feature']]
    # Deliberately select only geometry and recording identity columns.
    meta=pd.read_csv(args.metadata,usecols=lambda c:c in ['bag_id','recording_row','geometry_roi_mismatch','range_center_m'])
    assert len(x)==len(meta)
    model=Encoder(ck['input_dim'],ck['contact_mean'],ck['contact_scale'])
    model.load_state_dict(ck['encoder']);model.eval()
    xt=torch.tensor(((x-ck['radar_mean'])/ck['radar_scale']).astype(np.float32))
    with torch.inference_mode():
        h=model(xt);l10=h@ck['head_weight'].T+ck['head_bias'];p10=l10.softmax(-1).numpy();p4=grouping(l10).softmax(-1).numpy()
    np.savez_compressed(args.output/'window_outputs.npz',embedding=h.numpy(),probability10=p10,probability4=p4,bag_id=meta.bag_id.to_numpy(str))
    rows=[]
    for bag,idx in meta.groupby('bag_id').indices.items():
        a=p10[idx].mean(0);b=p4[idx].mean(0)
        valid,status=geometry_status(meta.iloc[idx])
        rows.append(dict(bag_id=bag,n_windows=len(idx),geometry_valid=valid,diagnosis_status=status,prediction=int(b.argmax()),**{f'p4_{i}':float(b[i]) for i in range(4)},**{f'p10_{i}':float(a[i]) for i in range(10)}))
    pd.DataFrame(rows).to_csv(args.output/'file_probabilities.csv',index=False)
    (args.output/'manifest.json').write_text(json.dumps(dict(model=str(args.model),features=str(args.features),metadata=str(args.metadata),labels_read=False,contact_data_read=False,head=ck.get('head_status','unchanged_original_linear2'),shared_head_path=ck.get('shared_head_path'),source_retention_acceptance=ck.get('source_retention_acceptance'),language_interface='mean p10 per bag may be passed directly to the original linear3; p4 is exact sum over groups',severity_validation=False),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    infer(p.parse_args())
