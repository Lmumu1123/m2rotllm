"""Read-only independent re-evaluation of student/teacher supervision levels.

This script does not import the training Encoder or grouping implementation.
No optimizer is constructed and no old archive is modified.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

BASE=Path('/home/huangyating/anomaly_detection/retention_alignment_20260922')
OLD=BASE.parent/'four_class_chain_20260922'
OUT=Path(__file__).resolve().parent
GROUPS=[[0],[1,2,3],[7,8,9],[4,5,6]]

class IndependentEncoder(nn.Module):
    def __init__(self,ck):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(ck['input_dim'],128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,128))
        self.register_buffer('contact_mean',torch.tensor(ck['contact_mean'],dtype=torch.float32))
        self.register_buffer('contact_scale',torch.tensor(ck['contact_scale'],dtype=torch.float32))
    def forward(self,x):return F.relu(self.contact_mean+self.contact_scale*self.net(x))

def grouped(logits,temp=1.):
    p10=(logits/temp).softmax(-1)
    return torch.stack([p10[...,g].sum(-1) for g in GROUPS],-1)

def score_rows(prob,meta,mask,fold,method,modality):
    rows=[]
    for i in np.flatnonzero(mask):
        info=meta.iloc[i]
        rows.append(dict(fold=fold,method=method,modality=modality,bag_id=info.bag_id,row=int(i),state=info.state,label=int(info.label),prediction=int(prob[i].argmax()),correct=int(prob[i].argmax())==int(info.label),geometry_valid=True if modality=='contact' else not bool(info.geometry_roi_mismatch),**{f'p{k}':float(prob[i,k]) for k in range(4)}))
    return rows

def main():
    assert 'envs/m2vllm' in __import__('sys').executable
    torch.set_num_threads(2)
    cm=pd.read_csv(BASE/'preprocessing/variants/v0/retrained_fcn/metadata.csv')
    rm=pd.read_csv(OLD/'radar/metadata.csv')
    h=np.load(BASE/'preprocessing/variants/v0/retrained_fcn/contact_features.npz')['hidden_mean']
    x=np.load(OLD/'radar/features.npz')['frame_shape']
    windows=[];targets=[];radarbags=[];checks=[]
    for fold,target in [('115200_to_460800',460800),('460800_to_115200',115200)]:
        headpath=BASE/'results/retained_head_v1/models/v0'/fold/'retained_head.npz'
        hd=np.load(headpath);w=torch.tensor(hd['weight10']);b=torch.tensor(hd['bias10'])
        cl=torch.tensor(h)@w.T+b
        pc=grouped(cl).numpy();pt=grouped(cl,2.).numpy()
        cmask=((cm.label>=0)&(cm.baud==target)).to_numpy()
        windows.extend(score_rows(pc,cm,cmask,fold,'shared_contact_head','contact'))
        for bag,idx in cm[cm.label>=0].groupby('bag_id').indices.items():
            # Known rows precede external rows, but use stable original indices.
            idx=np.flatnonzero((cm.bag_id==bag).to_numpy())
            label=int(cm.iloc[idx[0]].label)
            mean_h=torch.tensor(h[idx].mean(0))
            p_mse=grouped(mean_h@w.T+b).numpy()
            p_kd=pt[idx].mean(0);p_mean=pc[idx].mean(0)
            role='source_train' if bag in hd['local_training_bags'] else 'heldout_diagnostic_only'
            targets.append(dict(fold=fold,bag_id=bag,role=role,state=cm.iloc[idx[0]].state,label=label,n_contact_windows=len(idx),window_correct=int((pc[idx].argmax(1)==label).sum()),mean_T1_prob_prediction=int(p_mean.argmax()),KD_T2_bag_target_prediction=int(p_kd.argmax()),head_of_MSE_bag_prototype_prediction=int(p_mse.argmax()),**{f'KD_target_p{k}':float(p_kd[k]) for k in range(4)}))
        for method in ['embedding_only','ce_only','ce_feat_1_kd']:
            path=BASE/'radar/stage_b_v0/models/four_class_provisional_roi'/fold/f'{method}.pt'
            ck=torch.load(path,map_location='cpu',weights_only=False)
            assert np.array_equal(ck['head_weight'].numpy(),hd['weight10'])
            assert np.array_equal(ck['head_bias'].numpy(),hd['bias10'])
            assert set(ck['train_bags'])==set(hd['local_training_bags'])
            assert not(set(cm.loc[cmask,'bag_id'])&set(ck['train_bags']))
            model=IndependentEncoder(ck);model.load_state_dict(ck['encoder']);model.eval()
            with torch.inference_mode():
                rh=model(torch.tensor(((x-ck['radar_mean'])/ck['radar_scale']).astype(np.float32)))
                pr=grouped(rh@ck['head_weight'].T+ck['head_bias']).numpy()
            mask=((rm.label>=0)&(rm.baud_candidate==target)).to_numpy()
            windows.extend(score_rows(pr,rm,mask,fold,method,'radar'))
            for bag in sorted(rm.loc[mask,'bag_id'].unique()):
                idx=np.flatnonzero((rm.bag_id==bag).to_numpy());info=rm.iloc[idx[0]];q=pr[idx].mean(0)
                radarbag=dict(fold=fold,method=method,bag_id=bag,state=info.state,label=int(info.label),prediction=int(q.argmax()),correct=int(q.argmax())==int(info.label),geometry_valid=not bool(info.geometry_roi_mismatch),n_windows=len(idx),window_correct=int((pr[idx].argmax(1)==info.label).sum()))
                radarbags.append(radarbag)
            checks.append(dict(fold=fold,method=method,radar_checkpoint=str(path),radar_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),contact_head=str(headpath),contact_head_sha256=hashlib.sha256(headpath.read_bytes()).hexdigest(),head_weights_bitwise_equal=True,head_bias_bitwise_equal=True,train_bag_sets_equal=True,heldout_bags_disjoint=True))
    ww=pd.DataFrame(windows);tt=pd.DataFrame(targets);rr=pd.DataFrame(radarbags)
    ww.to_csv(OUT/'window_predictions.csv',index=False);tt.to_csv(OUT/'contact_bag_targets.csv',index=False);rr.to_csv(OUT/'radar_bag_predictions.csv',index=False)
    totals=ww.groupby(['modality','method']).agg(correct=('correct','sum'),n=('correct','size')).reset_index();totals['accuracy']=totals.correct/totals.n;totals.to_csv(OUT/'window_totals.csv',index=False)
    perfile=ww.groupby(['modality','method','fold','state','bag_id','geometry_valid']).agg(correct=('correct','sum'),n=('correct','size')).reset_index();perfile['accuracy']=perfile.correct/perfile.n;perfile.to_csv(OUT/'window_per_file.csv',index=False)
    valid=ww[ww.geometry_valid].groupby(['modality','method']).agg(correct=('correct','sum'),n=('correct','size')).reset_index();valid['accuracy']=valid.correct/valid.n;valid.to_csv(OUT/'geometry_valid_window_totals.csv',index=False)
    byfold=ww.groupby(['modality','method','fold']).agg(correct=('correct','sum'),n=('correct','size')).reset_index();byfold['accuracy']=byfold.correct/byfold.n;byfold.to_csv(OUT/'window_by_fold.csv',index=False)
    target_counts=[]
    for role,part in tt.groupby('role'):
        for key in ['mean_T1_prob_prediction','KD_T2_bag_target_prediction','head_of_MSE_bag_prototype_prediction']:
            target_counts.append(dict(role=role,target_kind=key,correct=int((part[key]==part.label).sum()),n=len(part)))
    pd.DataFrame(target_counts).to_csv(OUT/'bag_target_totals.csv',index=False)
    old=pd.read_csv(OUT.parent/'window_predictions.csv');old=old[old.scope=='heldout_recordings']
    ours=ww[((ww.modality=='radar')&(ww.method=='ce_feat_1_kd'))|(ww.modality=='contact')]
    compare=ours.merge(old,on=['modality','fold','row'],suffixes=('_new','_old'))
    assert len(compare)==len(ours)==len(old)
    assert (compare.prediction_new==compare.prediction_old).all()
    manifest=dict(read_only=True,training=False,encoder_reimplemented_independently=True,probability_grouping_independent=True,six_student_heads_equal_their_contact_head=True,earlier_threshold_window_predictions_exactly_reproduced=True,checks=checks)
    (OUT/'verification.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(totals.to_string(index=False));print(byfold.to_string(index=False));print(pd.DataFrame(target_counts).to_string(index=False));print(valid.to_string(index=False))

if __name__=='__main__':main()
