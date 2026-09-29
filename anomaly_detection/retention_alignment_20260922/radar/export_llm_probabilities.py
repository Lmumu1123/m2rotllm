"""Export held-out/external radar p10 for actual original language projector.

Inference reads no contact data. Ground-truth metadata is attached only after
inference for scoring; it never enters the encoder or probability computation.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import torch
from train_fixed_head import Encoder,grouping

R=Path(__file__).resolve().parent
OLD=R.parents[1]/'four_class_chain_20260922'
torch.set_num_threads(2)
parser=argparse.ArgumentParser();parser.add_argument('--include-conservative',action='store_true');args=parser.parse_args()
z=np.load(OLD/'radar/features.npz');meta=pd.read_csv(OLD/'radar/metadata.csv')
runs=[('A','v0','stage_a_seed42','ce_feat_1'),('A','query_no_demean','stage_a_no_demean','ce_feat_1'),('B','v0','stage_b_v0','ce_feat_1_kd'),('B','query_no_demean','stage_b_no_demean','ce_feat_1_kd')]
if args.include_conservative:runs += [('B_conservative','v0','stage_b_conservative_v0','ce_feat_1_kd'),('B_conservative','query_no_demean','stage_b_conservative_no_demean','ce_feat_1_kd')]
rows=[];checks=[]
for stage,variant,run,full in runs:
    for fold in (['all_known'] if stage=='B_conservative' else ['115200_to_460800','460800_to_115200','all_known']):
      for method in ([full] if stage=='B_conservative' else ['embedding_only','ce_only',full]):
        path=R/run/'models/four_class_provisional_roi'/fold/f'{method}.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model=Encoder(ck['input_dim'],ck['contact_mean'],ck['contact_scale']);model.load_state_dict(ck['encoder']);model.eval()
        x=((z[ck['feature']]-ck['radar_mean'])/ck['radar_scale']).astype(np.float32)
        with torch.inference_mode():
            h=model(torch.tensor(x));l=h@ck['head_weight'].T+ck['head_bias'];p10=l.softmax(-1).numpy();p4=grouping(l).softmax(-1).numpy()
        selected=(meta.label<0) if fold=='all_known' else ((meta.label>=0)&(meta.baud_candidate==int(fold.split('_to_')[1])))
        # The split's source bags are forbidden in held-out inference exports.
        if fold!='all_known':assert set(meta.loc[selected,'bag_id']).isdisjoint(ck['train_bags'])
        old=pd.read_csv(R/run/'file_predictions.csv')
        old=old[(old.task=='four_class_provisional_roi')&(old.fold==fold)&(old.method==method)&(old.seed==42)].set_index('bag_id')
        for bag,idx0 in meta.loc[selected].groupby('bag_id').groups.items():
            idx=np.array(list(idx0));q10=p10[idx].mean(0);q4=p4[idx].mean(0);info=meta.loc[idx[0]]
            error=float(np.max(abs(q4-old.loc[bag,[f'p{k}' for k in range(4)]].to_numpy(float))));assert error<1e-6
            row=dict(id=f'{stage}_{variant}_{fold}_{method}_{bag}',stage=stage,variant=variant,fold=fold,method=method,seed=42,bag_id=bag,state=info.state,true_label=int(info.label) if info.label>=0 else 0 if info.state=='bigNormal' else -1,evaluation_scope='external' if fold=='all_known' else 'heldout',geometry_valid=not bool(info.geometry_roi_mismatch),prediction=int(q4.argmax()),model_path=str(path),**{f'p{k}':float(q4[k]) for k in range(4)},**{f'p10_{k}':float(q10[k]) for k in range(10)})
            rows.append(row);checks.append(error)
export=pd.DataFrame(rows);export['task']='four_class_provisional_roi'
export['head_version']=export.stage.map({'A':'original_unchanged','B':'shared_v1','B_conservative':'shared_conservative'})
export['source_scope']=np.where(export.stage=='A','unchanged_original_all_sources',np.where((export.stage=='B')&(export.fold=='all_known'),'global_gate_only_some_sources_declined','global_and_each_source_regression_gate'))
suffix='_plus_conservative' if args.include_conservative else ''
export.to_csv(R/f'llm_original_p10_inputs{suffix}.csv',index=False)
if args.include_conservative:export[export.stage=='B_conservative'].to_csv(R/'llm_conservative_p10_inputs.csv',index=False)
(R/f'llm_original_p10_manifest{suffix}.json').write_text(json.dumps(dict(n_rows=len(rows),n_models=38 if args.include_conservative else 36,checkpoint_reload_max_error=max(checks),contact_data_read=False,ground_truth_used_for_inference=False,language_projector_input='p10_0 through p10_9 in original ten-class order; do not uniformly redistribute p4',coarse_class_order=['normal','inner','outer','ball'],fault_severity_evaluation=False,geometry_invalid_outputs='research comparison only; do not issue diagnostic statements'),indent=2)+'\n')
print('exported',len(rows),'rows; max reload error',max(checks))
