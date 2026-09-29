#!/usr/bin/env python3
"""One-command radar-only classification -> actual frozen LLM -> output guard.

Required metadata: bag_id in feature row order. Optional geometry fields:
geometry_roi_mismatch, range_center_m, intended_distance_m. Labels, state and
contact data are never read. This is inference, not an accuracy evaluation.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,time
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def bool_value(value):
    if pd.isna(value):return None
    if isinstance(value,str):
        if value.lower() not in ['true','false']:raise ValueError(f'Invalid geometry flag {value!r}')
        return value.lower()=='true'
    if value in (0,1,False,True):return bool(value)
    raise ValueError(f'Invalid geometry flag {value!r}')

def geometry_quality(meta):
    """Use only measured metadata; unavailable quality stays unverified."""
    rows=[]
    for bag,g in meta.groupby('bag_id',sort=True):
        flags=[bool_value(v) for v in g['geometry_roi_mismatch']] if 'geometry_roi_mismatch' in g else [None]
        mismatch=True if any(v is True for v in flags) else None if any(v is None for v in flags) else False
        rows.append(dict(recording_id=bag,geometry_roi_mismatch=mismatch,
            range_center_m=float(g.range_center_m.median()) if 'range_center_m' in g and g.range_center_m.notna().any() else None,
            intended_distance_m=float(g.intended_distance_m.median()) if 'intended_distance_m' in g and g.intended_distance_m.notna().any() else None,
            geometry_source='actual_inference_metadata' if mismatch is not None else 'unavailable_or_incomplete_inference_metadata'))
    return pd.DataFrame(rows)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--model',type=Path,required=True)
    a.add_argument('--features',type=Path,required=True)
    a.add_argument('--metadata',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--device',default='cuda:0')
    args=a.parse_args()
    for p in [args.model,args.features,args.metadata]:
        if not p.is_file():raise FileNotFoundError(p)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing nonempty output directory: {args.output}')
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();torch.set_num_threads(2)
    # Reading selected columns means label/state/contact fields are not loaded.
    available=pd.read_csv(args.metadata,nrows=0).columns
    if 'bag_id' not in available:raise ValueError('Metadata needs bag_id in feature row order')
    allowed=['bag_id','geometry_roi_mismatch','range_center_m','intended_distance_m']
    meta=pd.read_csv(args.metadata,usecols=[k for k in allowed if k in available],dtype={'bag_id':str})
    if meta.bag_id.isna().any():raise ValueError('Every feature row needs a bag_id')
    meta.to_csv(args.output/'metadata_used.csv',index=False)
    from infer_radar import predict,file_output
    p,h,ck=predict(args.model,args.features,args.device)
    if len(p)!=len(meta):raise ValueError('Metadata/features row count mismatch')
    np.savez_compressed(args.output/'window_predictions.npz',probabilities4=p,radar_embedding=h,
                        bag_id=meta.bag_id.to_numpy(dtype=str))
    safe=meta.copy()
    if 'geometry_roi_mismatch' in safe:
        safe['geometry_roi_mismatch']=safe.geometry_roi_mismatch.map(lambda v: True if bool_value(v) is None else bool_value(v))
    files=file_output(p,safe)
    files.to_csv(args.output/'radar_file_probabilities.csv',index=False)
    quality=geometry_quality(meta)
    quality_path=args.output/'actual_radar_geometry.csv';quality.to_csv(quality_path,index=False)
    rows=[]
    for r in files.to_dict(orient='records'):
        # These literal fields satisfy the existing evaluator interface; none
        # is a measured ground truth, speed, training split or contact input.
        rows.append(dict(id='radar_inference_'+r['bag_id'],bag_id=r['bag_id'],method='radar_only_saved_model',
            task='interface_control',fold='inference_no_labels',direction='no_evaluation',seed=ck.get('seed',42),
            modality='radar',true_label=-1,**{f'p{i}':r[f'p{i}'] for i in range(4)}))
    bridge_inputs=args.output/'llm_bridge_inputs.csv';pd.DataFrame(rows).to_csv(bridge_inputs,index=False)
    bridge_output=args.output/'llm'
    bridge_cmd=[sys.executable,str(ROOT/'contact/run_llm_bridge.py'),'--inputs',str(bridge_inputs),
                '--no-controls','--device',args.device,'--output',str(bridge_output)]
    with (args.output/'llm_generation.log').open('w') as log:
        subprocess.run(bridge_cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    guard_cmd=[sys.executable,str(ROOT/'contact/guard_llm_output.py'),'--directory',str(bridge_output),
               '--radar-quality',str(quality_path)]
    with (args.output/'guard.log').open('w') as log:
        subprocess.run(guard_cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    guarded=pd.read_csv(bridge_output/'guarded_predictions.csv')
    chosen=['bag_id','p0','p1','p2','p3','structured_pred_label','structured_pred_name',
            'diagnosis_status','geometry_roi_mismatch','radar_range_center_m',
            'generated_text','parsed_coarse_label','llm_class_conflict','language_fallback_applied',
            'guarded_text','guarded_output_source','known_class_candidate_only',
            'unknown_detection_supported','unknown_status']
    final=guarded[chosen].rename(columns={'generated_text':'raw_llm_text'})
    final['n_windows']=final.bag_id.map(meta.groupby('bag_id').size())
    final.to_csv(args.output/'predictions.csv',index=False)
    raw_metrics=pd.read_csv(bridge_output/'generation_metrics.csv')
    assert raw_metrics.labelled_n.eq(0).all()
    assert raw_metrics.accuracy.isna().all() and raw_metrics.macro_f1.isna().all()
    assert len(final)==meta.bag_id.nunique()
    manifest=dict(status='complete',purpose='radar-only interface demonstration, not held-out accuracy evaluation',
       windows=len(meta),recordings=len(final),python=sys.executable,device=args.device,
       input_feature=ck['input_feature'],metadata_columns_used=list(meta.columns),
       original_label_state_or_contact_fields_read=False,contact_feature_files_read=False,
       inference_true_label='constant -1 solely for legacy evaluator interface; never measured truth',
       placeholder_fields={'task':'interface_control','fold':'inference_no_labels','direction':'no_evaluation','modality':'radar'},
       diagnostic_accuracy_reported=False,unknown_detection_supported=False,
       raw_llm_probability_fidelity=float(guarded.probability_faithful.mean()),
       language_fallback_count=int(guarded.language_fallback_applied.sum()),
       invalid_or_unverified_roi_count=int(final.diagnosis_status.str.startswith('invalid').sum()),
       diagnosis_status_counts=final.diagnosis_status.value_counts().to_dict(),
       hashes={str(path):sha(path) for path in [args.model,args.features,args.metadata,Path(__file__)]},
       bridge_command=bridge_cmd,guard_command=guard_cmd,elapsed_seconds=time.monotonic()-start)
    (args.output/'inference_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','windows','recordings','language_fallback_count','invalid_or_unverified_roi_count']}))
    print(str(args.output/'predictions.csv'))

if __name__=='__main__':main()
