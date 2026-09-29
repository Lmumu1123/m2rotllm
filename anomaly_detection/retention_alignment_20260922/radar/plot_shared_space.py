"""Source-contact-fitted PCA for qualitative held-out embedding comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_fixed_head import Encoder,grouping

R=Path(__file__).resolve().parent;OLD=R.parents[1]/'four_class_chain_20260922'
torch.set_num_threads(2)
cm=pd.read_csv(OLD/'contact/retrained_fcn_fixed_external/metadata.csv');ch=np.load(OLD/'contact/retrained_fcn_fixed_external/contact_features.npz')['hidden_mean']
rm=pd.read_csv(OLD/'radar/metadata.csv');rx=np.load(OLD/'radar/features.npz')['frame_shape']
fold='115200_to_460800'
specs=[('stage_a_seed42','embedding_only','A: original head, embedding only'),('stage_a_seed42','ce_feat_1','A: original head, CE + alignment'),('stage_b_v0','ce_feat_1_kd','B: retained head, CE + alignment')]
ck0=torch.load(R/specs[0][0]/'models/four_class_provisional_roi'/fold/f'{specs[0][1]}.pt',map_location='cpu',weights_only=False)
train=cm.bag_id.isin(ck0['train_bags']).to_numpy();cz=(ch-ck0['contact_mean'])/ck0['contact_scale']
pca=PCA(n_components=2,svd_solver='full').fit(cz[train])
bags=rm[(rm.baud_candidate==460800)&(rm.label>=0)].drop_duplicates('bag_id').sort_values('label').bag_id.tolist()
teacher=np.stack([cz[(cm.bag_id==bag).to_numpy()].mean(0) for bag in bags]);tp=pca.transform(teacher)
results=[];records=[]
for run,method,title in specs:
    ck=torch.load(R/run/'models/four_class_provisional_roi'/fold/f'{method}.pt',map_location='cpu',weights_only=False)
    assert np.allclose(ck['contact_mean'],ck0['contact_mean'],rtol=1e-5,atol=3e-6) and np.allclose(ck['contact_scale'],ck0['contact_scale'],rtol=1e-5,atol=3e-6)
    net=Encoder(128,ck['contact_mean'],ck['contact_scale']);net.load_state_dict(ck['encoder']);net.eval()
    with torch.inference_mode():
        h=net(torch.tensor(((rx-ck['radar_mean'])/ck['radar_scale']).astype(np.float32)));pr=grouping(h@ck['head_weight'].T+ck['head_bias']).softmax(-1).numpy();z=(h.numpy()-ck0['contact_mean'])/ck0['contact_scale']
    radar=np.stack([z[(rm.bag_id==bag).to_numpy()].mean(0) for bag in bags]);rp=pca.transform(radar);results.append((title,rp))
    for i,bag in enumerate(bags):
        mask=(rm.bag_id==bag).to_numpy();t=int(rm.loc[mask,'label'].iloc[0]);pred=int(pr[mask].mean(0).argmax());info=rm.loc[mask].iloc[0]
        records.append(dict(run=run,method=method,bag_id=bag,label=t,prediction=pred,geometry_valid=not bool(info.geometry_roi_mismatch),teacher_pc1=tp[i,0],teacher_pc2=tp[i,1],radar_pc1=rp[i,0],radar_pc2=rp[i,1],MSE_full128=float(np.mean((radar[i]-teacher[i])**2))))
pd.DataFrame(records).to_csv(R/'embedding_projection_points.csv',index=False)
colors=['#1D77B2','#E58B2D','#A84BA2','#3B9D6D'];names=['normal','inner','outer*','ball']
fig,axes=plt.subplots(1,3,figsize=(14,4.8))
allpoints=np.vstack([tp]+[r[1] for r in results]);lo=allpoints.min(0)-1;hi=allpoints.max(0)+1
for axis,(title,rp) in zip(axes,results):
    for i,(color,name) in enumerate(zip(colors,names)):
        axis.plot([tp[i,0],rp[i,0]],[tp[i,1],rp[i,1]],color=color,alpha=.6,linestyle='--' if i==2 else '-')
        axis.scatter(tp[i,0],tp[i,1],marker='o',s=95,facecolor='none',edgecolor=color,linewidth=2,label=name)
        axis.scatter(rp[i,0],rp[i,1],marker='^',s=75,c=color)
    axis.set_title(title,fontsize=10);axis.set_xlabel(f'Source-contact PC1 ({pca.explained_variance_ratio_[0]:.1%})');axis.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})');axis.set_xlim(lo[0],hi[0]);axis.set_ylim(lo[1],hi[1]);axis.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Held-out bag embeddings in one source-contact-fitted projection\nCircle: contact; triangle: radar.  * Outer radar ROI invalid; 2-D proximity is qualitative only.',fontsize=11)
fig.tight_layout();fig.savefig(R/'shared_embedding_projection.png',dpi=180,bbox_inches='tight');fig.savefig(R/'shared_embedding_projection.pdf',bbox_inches='tight');plt.close(fig)
(R/'embedding_projection_protocol.json').write_text(json.dumps(dict(fold=fold,seed=42,pca_fit='source contact windows only; no target or radar fit',common_coordinate_scaler='Stage A source contact statistics used for all displayed points',pc_variance_ratio=pca.explained_variance_ratio_.tolist(),source_bags=ck0['train_bags'],target_bags=bags,teacher_encoder_unchanged_between_A_B=True,recomputed_v0_hidden_max_difference=0.00017410517,recomputed_v0_hidden_mse=8.9893155e-11,limits='2-D projection does not substitute 128-D fidelity or classifier margin metrics; outer ROI invalid'),indent=2)+'\n')
print('source-fitted PCA plot complete')
