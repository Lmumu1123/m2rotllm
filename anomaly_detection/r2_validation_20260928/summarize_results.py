"""Summarize fixed R2 runs, pooling out-of-fold predictions once per seed."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
seed_rows=[]
physical=[]
fidelity=[]
for folder in sorted((ROOT/'experiments').glob('*__*')):
    if not (folder/'verification.json').is_file():continue
    proto=json.loads((folder/'run_protocol.json').read_text())
    d=pd.read_csv(folder/'recording_predictions.csv')
    d=d[d.role=='test']
    for (method,seed),sub in d.groupby(['method','seed']):
        assert len(sub)==144 and sub.recording_id.nunique()==144
        truth=[];pred=[]
        for p in sorted((folder/'predictions').glob(f'holdout_*__{method}__seed{seed}.npz')):
            with np.load(p) as z:
                truth.extend(z['labels'].tolist());pred.extend(z['probabilities4'].argmax(-1).tolist())
        assert len(truth)==2158
        for level,yy,pp in [('recording',sub.label,sub.prediction),('window',truth,pred)]:
            seed_rows.append(dict(head=proto['head'],feature=proto['feature'],method=method,seed=seed,level=level,
                                  accuracy=accuracy_score(yy,pp),macro_f1=f1_score(yy,pp,average='macro',labels=range(4),zero_division=0),
                                  n=len(yy)))
    q=pd.read_csv(folder/'physical_predictions.csv');q['head']=proto['head'];q['feature']=proto['feature'];physical.append(q)
    q=pd.read_csv(folder/'session_fidelity.csv');q['head']=proto['head'];q['feature']=proto['feature'];fidelity.append(q)
seed_df=pd.DataFrame(seed_rows)
seed_df.to_csv(OUT/'neural_per_seed_pooled.csv',index=False)
summary=seed_df.groupby(['head','feature','method','level'],as_index=False).agg(
    accuracy_mean=('accuracy','mean'),accuracy_seed_sd=('accuracy','std'),macro_f1_mean=('macro_f1','mean'),
    macro_f1_seed_sd=('macro_f1','std'),seeds=('seed','nunique'),n_per_seed=('n','first'))
summary.to_csv(OUT/'neural_summary.csv',index=False)
f=pd.concat(fidelity,ignore_index=True)
f.groupby(['head','feature','method'],as_index=False).agg(
    standardized_mse=('mse','mean'),standardized_cosine=('cosine','mean'),
    oracle_class_mse=('oracle_class_embedding_mse','mean'),improvement_vs_class=('improvement_vs_class_mse','mean'),
    head_row_mse=('head_row_mse','mean'),head_null_mse=('head_null_mse','mean'),
    unique_contact_sessions=('session_id','nunique')).to_csv(OUT/'fidelity_summary.csv',index=False)
p=pd.concat(physical,ignore_index=True)
p.groupby(['head','feature','method','target'],as_index=False).agg(
    readout_MAE_train_std=('abs_error_std','mean'),oracle_true_class_MAE_train_std=('oracle_class_abs_error_std','mean'),
    unique_contact_sessions=('session_id','nunique')).to_csv(OUT/'physical_readout_summary.csv',index=False)
ctrl=pd.read_csv(ROOT/'experiments/archived__complex_shape/contact_probe_controls.csv')
gate=ctrl.groupby(['target','method']).abs_error_std.mean().unstack('method').reset_index()
gate['contact_better_than_true_class_mean']=gate.direct_contact_readout<gate.oracle_true_class_target
gate.to_csv(OUT/'physical_contact_validity_gates.csv',index=False)

fig,axs=plt.subplots(1,3,figsize=(15,4.7))
colors=['#4477AA','#EE7733','#228833']
# These are different evaluation protocols, deliberately shown separately.
axs[0].bar([0,1],[99.1596638655,41.6909620991],color=colors[:2])
axs[0].set_xticks([0,1],['Later packets,\nsame DAT','Unseen speed\n(entire DAT held out)'])
axs[0].set_title('Contact PSD: protocol matters')
axs[1].bar([0,1,2],[97.2222222222,37.5,27.0833333333],color=colors)
axs[1].set_xticks([0,1,2],['New recording,\nseen conditions','Unseen\nspeed','Unseen\ndistance'])
axs[1].set_title('Radar random forest: recording accuracy')
main=summary[(summary['head']=='archived')&(summary.feature=='complex_shape')&(summary.level=='recording')].set_index('method')
methods=['head_CE','contact_session_MSE','full','class_code_MSE']
values=[100*main.loc[m,'accuracy_mean'] for m in methods]
axs[2].bar(range(4),values,color=['#BBBBBB','#4477AA','#228833','#CC6677'],
            yerr=[100*main.loc[m,'accuracy_seed_sd'] for m in methods],capsize=3)
axs[2].set_xticks(range(4),['Head CE','Contact\nmean','Full\nalignment','Class\ncode'])
axs[2].set_title('Radar through frozen Bear head: unseen speed')
for ax in axs:
    ax.set_ylim(0,108);ax.set_ylabel('Accuracy (%)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{bar.get_height():.1f}',ha='center',fontsize=10)
fig.suptitle('R2 validation: high accuracy within a condition does not establish transfer',fontsize=14)
fig.text(.5,.015,'Error bars: SD across 3 seeds, not confidence intervals. Radar: 144 recordings; contact: 12 sessions.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,1))
fig.savefig(OUT/'main_results.png',dpi=180)
fig.savefig(OUT/'main_results.pdf')
plt.close(fig)
evidence=dict(contact_sessions=12,contact_packets=343,radar_recordings=144,radar_windows=2158,
              primary_head='archived frozen retained Bear classifier',main_feature='complex_shape',
              neural_models=len(seed_df[seed_df.level=='recording'])*3,
              per_seed_oof_predictions_exactly_once=True,seed_spread_not_physical_confidence_interval=True,
              source_models_not_modified=True,language_model_generation_rerun=False)
(OUT/'summary_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(summary[(summary['head']=='archived')&(summary.feature=='complex_shape')&(summary.level=='recording')].to_string(index=False))
print(OUT)
