from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from infer_fixed_head import geometry_status
from train_fixed_head import Encoder

R=Path(__file__).resolve().parent;OLD=R.parents[1]/'four_class_chain_20260922'
out=R/'radar_only_conservative_v0'
cases=[({},False),({'geometry_roi_mismatch':[np.nan,np.nan]},False),({'geometry_roi_mismatch':[False,np.nan]},False),({'geometry_roi_mismatch':['False','False']},True),({'geometry_roi_mismatch':['True','False']},False),({'geometry_roi_mismatch':[False,False]},True),({'geometry_roi_mismatch':[True]},False),({'geometry_roi_mismatch':['unknown']},False),({'geometry_roi_mismatch':[0]},False),({'geometry_roi_mismatch':[]},False)]
for frame,expected in cases:assert geometry_status(pd.DataFrame(frame))[0]==expected
actual=pd.read_csv(out/'file_probabilities.csv').set_index('bag_id')
old=pd.read_csv(R/'stage_b_conservative_v0/file_predictions.csv').set_index('bag_id')
assert set(old.index)==set(actual.index)
p4err=float(np.max(abs(actual.loc[old.index,[f'p4_{i}' for i in range(4)]].to_numpy()-old[[f'p{i}' for i in range(4)]].to_numpy())))
assert p4err<1e-6
ck=torch.load(R/'stage_b_conservative_v0/models/four_class_provisional_roi/all_known/ce_feat_1_kd.pt',map_location='cpu',weights_only=False)
model=Encoder(ck['input_dim'],ck['contact_mean'],ck['contact_scale']);model.load_state_dict(ck['encoder']);model.eval();torch.set_num_threads(2)
rz=np.load(OLD/'radar/features.npz');rm=pd.read_csv(OLD/'radar/metadata.csv')
with torch.inference_mode():
    h=model(torch.tensor(((rz['frame_shape']-ck['radar_mean'])/ck['radar_scale']).astype(np.float32)));p10=(h@ck['head_weight'].T+ck['head_bias']).softmax(-1).numpy()
p10errors=[]
for bag,idx in rm.groupby('bag_id').indices.items():p10errors.append(float(np.max(abs(p10[idx].mean(0)-actual.loc[bag,[f'p10_{i}' for i in range(10)]].to_numpy(float)))))
assert max(p10errors)<1e-6
meta=pd.read_csv(OLD/'results/radar_only_demo_inputs/metadata.csv');forbidden=set(meta.columns)&{'label','labels','state','contact','true_label'}
assert not forbidden
assert (~actual.loc[rm[rm.geometry_roi_mismatch].bag_id.unique(),'geometry_valid']).all()
report=dict(passed=True,strict_geometry_cases=len(cases),n_files=len(actual),input_metadata_columns=meta.columns.tolist(),labels_state_contact_absent=True,all_file_p4_error_vs_training=p4err,all_file_p10_error_vs_full_feature_inference=max(p10errors),geometry_invalid_files=int((~actual.geometry_valid).sum()),contact_data_required=False)
(out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
