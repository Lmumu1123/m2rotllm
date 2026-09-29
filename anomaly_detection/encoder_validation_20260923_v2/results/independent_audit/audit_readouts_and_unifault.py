"""Independent read-only recomputation; no imports from the two audited scripts."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='2'
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,welch
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BEAR=ROOT.parent/'retention_alignment_20260922/preprocessing/variants/v0/retrained_fcn'
OLD=ROOT.parent/'encoder_validation_20260923'
RAW=ROOT.parent/'four_class_preprocessing_20260921/results'
TARGETS=['log_band_rms','log_crest_factor','log_pearson_kurtosis','power_20_200','power_200_400','power_400_800']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b,tol=1e-8):
 err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
 if not err<=tol:raise AssertionError(f'max difference {err} > {tol}')
 return err
def weights(md):
 counts=md.bag_id.value_counts();v=md.bag_id.map(lambda v:1/counts[v]).to_numpy();return v*len(v)/v.sum()
def desc(raw):
 # Independent Pearson kurtosis calculation by central moments.
 x=raw.astype('float64')-raw.mean(axis=0,keepdims=True)
 filt=butter(4,(20,800),fs=4000,btype='bandpass',output='sos')
 x=sosfiltfilt(filt,x,axis=0)[400:3600]
 rms=np.sqrt(np.mean(x*x));rmsaxes=np.sqrt(np.mean(x*x,axis=0))
 crest=np.mean(np.max(abs(x),axis=0)/rmsaxes)
 centered=x-x.mean(axis=0);m2=np.mean(centered**2,axis=0);m4=np.mean(centered**4,axis=0)
 kurt=np.mean(m4/m2**2)
 f,p=welch(x,fs=4000,nperseg=512,noverlap=256,axis=0);p=p.mean(axis=1)
 power=[]
 for a,b in ((20,200),(200,400),(400,800)):
  keep=(f>=a)&(f<b);power.append(np.trapz(p[keep],f[keep]))
 power=np.asarray(power);power/=power.sum()
 return np.r_[np.log(rms),np.log(crest),np.log(kurt),power]

class IndependentEncoder(nn.Module):
 def __init__(self,state,relu):
  super().__init__();self.relu=relu
  self.net=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,128))
  self.register_buffer('contact_mean',torch.zeros(128));self.register_buffer('contact_scale',torch.ones(128));self.load_state_dict(state)
 def forward(self,x):
  y=self.contact_mean+self.contact_scale*self.net(x)
  return y.clamp(min=0) if self.relu else y

def main():
 torch.set_num_threads(2)
 sources=[ROOT/'scripts/physical_readouts.py',ROOT/'scripts/train_unifault_radar.py']
 tracked={str(p):sha(p) for p in sources}
 cm=pd.read_csv(BEAR/'metadata.csv');assert not cm.duplicated(['bag_id','window_row']).any()
 raw=[]
 for row in cm.itertuples():
  p=RAW/row.file;tracked.setdefault(str(p),sha(p))
  with np.load(p) as z:
   assert int(z['labels'][row.window_row])==row.label;raw.append(z['raw_xyz'][row.window_row])
  side=json.loads(p.with_suffix('.json').read_text());assert side['windows'][row.window_row]['packet_ordinal_observed']==row.packet_ordinal_observed
 y=np.stack([desc(z) for z in raw]);pd_targets=pd.read_csv(ROOT/'results/physical_readouts/contact_physical_targets.csv')
 assert cm[['file','window_row','bag_id']].equals(pd_targets[['file','window_row','bag_id']])
 targeterr=eq(y,pd_targets[TARGETS].to_numpy(),tol=1e-12);eq(y[:,3:].sum(axis=1),np.ones(88),tol=1e-12)
 splits=json.loads((ROOT/'controls/results/splits.json').read_text())
 saved=pd.read_csv(ROOT/'results/physical_readouts/recording_predictions.csv');savedwin=pd.read_csv(ROOT/'results/physical_readouts/contact_window_predictions.csv')
 fits=[];contact_recomputed=[]
 for name in ('Bear128','Rot128','UniFault128'):
  for sp in splits:
   fold=sp['fold'];tr=cm.bag_id.isin(sp['train_bags']).to_numpy();te=cm.bag_id.isin(sp['test_bags']).to_numpy();assert not (tr&te).any()
   td=BEAR if name=='Bear128' else OLD/'rotllm' if name=='Rot128' else ROOT/'teachers/unifault/paper_100ms'/fold
   file='contact_features.npz' if name=='Bear128' else 'teacher_features.npz';h0=np.load(td/file)['hidden_mean'].astype(float);md=pd.read_csv(td/'metadata.csv')
   assert not md.duplicated(['bag_id','window_row']).any();idx=pd.MultiIndex.from_frame(md[['bag_id','window_row']]).get_indexer(pd.MultiIndex.from_frame(cm[['bag_id','window_row']]))
   assert (idx>=0).all();h=h0[idx]
   sw=weights(cm[tr]);sx=StandardScaler().fit(h[tr],sample_weight=sw);sy=StandardScaler().fit(y[tr],sample_weight=sw)
   # Refit only the already fixed diagnostic readout; this verifies an existing fit, no new model search.
   rr=Ridge(alpha=10,solver='svd').fit(sx.transform(h[tr]),sy.transform(y[tr]),sample_weight=sw)
   yp=sy.inverse_transform(rr.predict(sx.transform(h)))
   stored=np.load(ROOT/'results/physical_readouts/models'/f'{name}__{fold}.npz')
   assert set(stored['train_bags'])==set(sp['train_bags']);assert list(stored['targets'])==TARGETS
   ystored=h@stored['weight'].T+stored['bias'];perr=eq(yp,ystored,1e-8);serr=eq(stored['target_scale'],sy.scale_,1e-12)
   classmean={c:y[tr&(cm.label.to_numpy()==c)].mean(0) for c in range(4)}
   proto={c:h[tr&(cm.label.to_numpy()==c)].mean(0)@stored['weight'].T+stored['bias'] for c in range(4)}
   max_csv=0
   for bag in sp['test_bags']:
    mask=cm.bag_id.eq(bag).to_numpy();label=int(cm.loc[mask,'label'].iloc[0]);truth=y[mask].mean(0)
    for method,pred in [('native_contact',yp[mask].mean(0)),('true_class_target_mean',classmean[label]),('true_class_teacher_prototype',proto[label]),('global_train_mean',sy.mean_)]:
     q=saved[(saved.teacher==name)&(saved.fold==fold)&(saved.method==method)&(saved.bag_id==bag)].set_index('target').loc[TARGETS]
     max_csv=max(max_csv,eq(q.truth,truth),eq(q.prediction,pred),eq(q.abs_error_std,np.abs(pred-truth)/sy.scale_))
     for j,target in enumerate(TARGETS):contact_recomputed.append(dict(teacher=name,fold=fold,bag_id=bag,method=method,target=target,abs_error_std=float(abs(pred[j]-truth[j])/sy.scale_[j])))
   for row in savedwin[(savedwin.teacher==name)&(savedwin.fold==fold)].itertuples():
    idx=np.flatnonzero(cm.bag_id.eq(row.bag_id)&cm.window_row.eq(row.window_row));assert len(idx)==1;idx=idx[0];j=TARGETS.index(row.target)
    assert te[idx];max_csv=max(max_csv,eq(row.prediction,yp[idx,j]),eq(row.abs_error_std,abs(yp[idx,j]-y[idx,j])/sy.scale_[j]))
   fits.append(dict(teacher=name,fold=fold,train_windows=int(tr.sum()),test_windows=int(te.sum()),teacher_metadata_keys_unique=True,
    max_reloaded_readout_prediction_error=perr,max_target_scale_error=serr,max_contact_csv_error=max_csv))
 contact=pd.DataFrame(contact_recomputed);gates=pd.read_csv(ROOT/'results/physical_readouts/contact_validity_gates.csv');gatechecks=[]
 for r in gates.itertuples():
  dd=contact[(contact.teacher==r.teacher)&(contact.target==r.target)].groupby('method').abs_error_std.mean()
  eq(dd['native_contact'],r.contact_readout_mae);eq(dd['true_class_target_mean'],r.class_only_mae)
  flag=bool(dd['native_contact']<dd['true_class_target_mean']);assert flag==r.passes_contact_predictivity_gate
  gatechecks.append(dict(teacher=r.teacher,target=r.target,passes=flag))
 # Independently verify all exported recording losses, summary weighting and row grouping.
 eq(saved.abs_error_std,abs(saved.prediction-saved.truth)/saved.train_target_scale,1e-12)
 summaries=pd.read_csv(ROOT/'results/physical_readouts/summary.csv');computed=saved.groupby(['teacher','path','method','target']).abs_error_std.agg(['mean','size'])
 for row in summaries.itertuples():
  rr=computed.loc[(row.teacher,row.path,row.method,row.target)];eq(row.mae_train_std,rr['mean'],1e-12);assert row.n_repeated==rr['size']
 radar_meta=pd.read_csv(ROOT/'geometry_corrected/radar/metadata.csv');x=np.load(ROOT/'geometry_corrected/radar/features.npz')['frame_shape']
 modelrows=[];mf=pd.read_csv(ROOT/'unifault_radar/file_predictions.csv');lower=[]
 for sp in splits:
  fold=sp['fold'];td=ROOT/'teachers/unifault/paper_100ms'/fold
  md=pd.read_csv(td/'metadata.csv');h=np.load(td/'teacher_features.npz')['hidden_mean'].astype(np.float32);hd=np.load(td/'head.npz')
  ctr=md.bag_id.isin(sp['train_bags']).to_numpy();rtr=radar_meta.bag_id.isin(sp['train_bags']).to_numpy()
  sx=StandardScaler().fit(x[rtr],sample_weight=weights(radar_meta[rtr]));cs=StandardScaler().fit(h[ctr],sample_weight=weights(md[ctr]));xt=torch.tensor(sx.transform(x).astype(np.float32));hz=cs.transform(h).astype(np.float32)
  for bag in sp['train_bags']+sp['test_bags']:
   mu=h[md.bag_id.eq(bag)].mean(0);bound=float(np.mean((np.minimum(mu,0)/cs.scale_)**2));lower.append(dict(fold=fold,bag_id=bag,role='train' if bag in sp['train_bags'] else 'test',nonnegative_bag_MSE_lower_bound=bound,negative_teacher_mean_fraction=float((mu<0).mean())))
  checkpoints=sorted((ROOT/'unifault_radar/models').glob(f'{fold}__*/model.pt'));assert len(checkpoints)==18
  for path in checkpoints:
   tag=path.parent.name;_,constraint,method,seed=tag.split('__');ck=torch.load(path,map_location='cpu',weights_only=True);tracked[str(path)]=sha(path)
   assert set(ck['train_bags'])==set(sp['train_bags']);assert set(ck['test_bags'])==set(sp['test_bags']);assert ck['teacher_feature_sha256']==sha(td/'teacher_features.npz')
   assert ck['nonnegative']==(constraint=='legacy_nonnegative_relu')
   meanerr=eq(ck['radar_mean'].numpy(),sx.mean_,1e-8);scaleerr=eq(ck['radar_scale'].numpy(),sx.scale_,1e-8)
   eq(ck['encoder']['contact_mean'].numpy(),cs.mean_,1e-6);eq(ck['encoder']['contact_scale'].numpy(),cs.scale_,1e-6)
   eq(ck['head_weight'].numpy(),hd['weight4'],1e-5);eq(ck['head_bias'].numpy(),hd['bias4'],1e-5)
   model=IndependentEncoder(ck['encoder'],ck['nonnegative']).eval()
   with torch.no_grad():rh=model(xt).numpy()
   pred=np.load(path.parent/'predictions.npz');eq(pred['labels'],radar_meta.label.to_numpy());assert np.array_equal(pred['bag_id'],radar_meta.bag_id.to_numpy())
   herr=eq(rh,pred['hidden'],1e-6);prob=softmax(rh@ck['head_weight'].numpy().T+ck['head_bias'].numpy(),axis=1);prerr=eq(prob,pred['probs4'],1e-6)
   rz=(rh-cs.mean_)/cs.scale_;mseerr=0
   for bag,indices in radar_meta.groupby('bag_id').indices.items():
    diff=rz[indices].mean(0)-hz[md.bag_id.eq(bag)].mean(0);mse=float(np.dot(diff,diff)/len(diff));pmean=prob[indices].mean(0)
    row=mf[(mf.fold==fold)&(mf.constraint==constraint)&(mf.method==method)&(mf.seed==int(seed[4:]))&(mf.bag_id==bag)].iloc[0]
    mseerr=max(mseerr,eq(row.mse_standardized,mse,1e-7));eq(row[['p0','p1','p2','p3']].to_numpy(float),pmean,1e-6)
   modelrows.append(dict(model=tag,hidden_reload_error=herr,head_probability_error=prerr,recording_MSE_error=mseerr,
    radar_scaler_mean_error=meanerr,radar_scaler_scale_error=scaleerr,teacher_head_matches=True))
 assert len(modelrows)==36;assert all(sha(p)==v for p,v in tracked.items())
 pd.DataFrame(modelrows).to_csv(OUT/'unifault_36_reload_checks.csv',index=False);pd.DataFrame(lower).to_csv(OUT/'unifault_relu_feasible_error_lower_bounds.csv',index=False)
 # Radar readout rows are recomputed on one full fold across all saved teacher/student paths.
 sampled=[]
 for row in saved[(saved.fold==splits[0]['fold'])&saved.path.str.startswith('radar_to_own_teacher')].groupby(['teacher','path','method','seed','bag_id']):
  (teacher,pname,method,seed,bag),group=row;fold=splits[0]['fold'];model=np.load(ROOT/'results/physical_readouts/models'/f'{teacher}__{fold}.npz')
  if teacher=='Bear128':p=ROOT/'controls/results/windows'/f'{fold}__{method}__seed{seed}.npz'
  elif teacher=='Rot128':p=ROOT/'rotllm_radar/models'/f'{fold}__{pname[len("radar_to_own_teacher_"):]}__{method}__seed{seed}'/'predictions.npz'
  else:p=ROOT/'unifault_radar/models'/f'{fold}__{pname[len("radar_to_own_teacher_"):]}__{method}__seed{seed}'/'predictions.npz'
  z=np.load(p);mask=z['bag_id']==bag;prediction=(z['hidden'][mask]@model['weight'].T+model['bias']).mean(0)
  err=eq(group.set_index('target').loc[TARGETS].prediction,prediction,1e-8);sampled.append(err)
 report=dict(status='pass',audited_scripts={str(p):sha(p) for p in sources},all_tracked_inputs_unchanged=True,
  descriptor_windows=88,descriptor_max_abs_error=targeterr,teacher_readout_fits=fits,contact_gates=gatechecks,
  unifault_models_reloaded=36,unifault_largest_hidden_reload_error=max(r['hidden_reload_error'] for r in modelrows),
  unifault_largest_head_probability_error=max(r['head_probability_error'] for r in modelrows),
  unifault_largest_recording_MSE_error=max(r['recording_MSE_error'] for r in modelrows),
  physical_radar_prediction_groups_recomputed_first_fold=len(sampled),physical_radar_prediction_largest_error=max(sampled),
  no_new_encoder_training=True,no_main_scripts_modified=True,
  interpretation_limits=['Six generic signal statistics are not fault severity ground truth.','Three band fractions are dependent and sum to 1.',
   'Oracle class baselines explicitly receive true test labels and are diagnostic controls, not deployable predictors.',
   'Contact validity gates average errors over eight recordings and are descriptive, not significance tests; all gates must be reported.',
   'Each training class has one recording, so real bag means equal class prototypes.',
   'Radar and intermittent contact windows are matched at recording level only.',
   'Signed/ReLU comparison changes only output feasible domain under the frozen teacher head; it does not compare optimized arbitrary architectures.',
   'ReLU cannot match negative teacher class-mean coordinates; the positive MSE lower bound is algebraic and does not prove rich knowledge.'])
 (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
