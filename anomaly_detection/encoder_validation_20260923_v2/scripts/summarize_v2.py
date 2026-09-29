"""Read saved outputs; no model selection or parameter fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/summary';OUT.mkdir(exist_ok=True)
sources=[]
def read(rel):
    p=ROOT/rel;sources.append(p);return pd.read_csv(p)
def pool(d,keys,n='n',acc='accuracy',correct=None):
    d=d.copy();d['correct_count']=d[correct] if correct else np.rint(d[n]*d[acc]).astype(int)
    z=d.groupby(keys,dropna=False).agg(correct_repeated=('correct_count','sum'),n_repeated=(n,'sum')).reset_index()
    z['accuracy']=z.correct_repeated/z.n_repeated;return z

b=read('controls/results/metrics.csv');b=b[b.role.eq('test')]
bs=pool(b,['method'],'n_windows','window_accuracy')
bs=bs.merge(b.groupby('method')[['bag_accuracy','mse_standardized','cos_standardized']].mean(),on='method')
bs.to_csv(OUT/'bear_controls.csv',index=False)
r=read('rotllm_radar/metrics.csv');r=r[r.role.eq('test')&r.subset.eq('all_four_provisional_roi')]
rs=pool(r,['head','method','unit']);rs.to_csv(OUT/'rotllm_radar.csv',index=False)
s=read('results/contact_only_stitch/metrics.csv');s=s[s.scope.eq('all_four_development')]
ss=pool(s,['head','path','radar_method','unit'],correct='correct');ss.to_csv(OUT/'stitch.csv',index=False)
u=read('unifault_radar/metrics.csv');u=u[u.role.eq('test')]
us=pool(u,['constraint','method','unit'],correct='correct');us.to_csv(OUT/'unifault_radar.csv',index=False)
physical=read('results/physical_readouts/recording_predictions.csv')
oracle=physical[physical.method.eq('true_class_target_mean')][['teacher','fold','bag_id','target','prediction']].rename(columns={'prediction':'oracle_prediction'})
joined=physical.merge(oracle,on=['teacher','fold','bag_id','target'],validate='many_to_one')
joined['squared_error']=(joined.prediction-joined.truth)**2
joined['oracle_squared_error']=(joined.oracle_prediction-joined.truth)**2
skill=joined.groupby(['teacher','path','method','target']).agg(sse=('squared_error','sum'),oracle_sse=('oracle_squared_error','sum'),n_repeated=('bag_id','size')).reset_index()
skill['skill_vs_true_class_mean']=np.where(skill.oracle_sse>1e-12,1-skill.sse/skill.oracle_sse,np.nan)
skill.to_csv(OUT/'physical_secondary_sse_skill.csv',index=False)
p=read('results/physical_readouts/summary.csv')
g=read('results/physical_readouts/contact_validity_gates.csv')
held=read('heldout_radar_class/results/summary.csv')
held.to_csv(OUT/'heldout_radar_class.csv',index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(15,4.8))
tab=us[(us.method=='embedding_only')&(us.unit=='window')].set_index('constraint')
v=[100*tab.loc[k,'accuracy'] for k in ['legacy_nonnegative_relu','signed_linear']]
axs[0].bar([0,1],v,color=['#CB6560','#278D95'])
axs[0].set_xticks([0,1],['ReLU output','Signed output']);axs[0].set_ylim(0,115)
axs[0].set_ylabel('Radar window accuracy (%)');axs[0].set_title('A. UniFault needs a signed interface')
for i,a in enumerate(v):axs[0].text(i,a+2,f'{a:.2f}',ha='center')

targets=['log_band_rms','log_pearson_kurtosis','power_400_800'];labels=['RMS','Kurtosis','400-800 Hz fraction']
methods=['native_contact','full','label_code_mse'];names=['Actual contact','Real-feature radar','Artificial-code radar'];colors=['#555F71','#278D95','#D9A149']
q=p[p.teacher.eq('Bear128')].copy();base=q[q.method.eq('true_class_target_mean')].set_index('target').mae_train_std
x=np.arange(3);width=.24
for i,(method,name,color) in enumerate(zip(methods,names,colors)):
    d=q[q.method.eq(method)].set_index('target')
    vals=[d.loc[t,'mae_train_std']/base[t] for t in targets]
    axs[1].bar(x+(i-1)*width,vals,width,label=name,color=color)
axs[1].axhline(1,color='#AB4B49',linestyle='--',label='True-class mean baseline')
axs[1].set_xticks(x,labels,rotation=10);axs[1].set_ylabel('Error / class-only baseline (lower is better)')
axs[1].set_title('B. Physical readout is a stricter check');axs[1].legend(fontsize=8,frameon=False)

q=ss[(ss['head']=='contact_only_new4')&(ss.path=='frozen_radar_A_to_B')&(ss.unit=='recording')].set_index('radar_method')
ms=['ce_only','matched_bag_mse','full','label_code_mse'];ls=['Class only','Teacher mean','Full loss','Artificial codes']
v=[100*q.loc[m,'accuracy'] for m in ms]
axs[2].bar(range(4),v,color=['#778DA9','#278D95','#397F60','#D9A149'])
axs[2].set_xticks(range(4),ls,rotation=23,ha='right');axs[2].set_ylim(0,115)
axs[2].set_ylabel('Recording accuracy (%)');axs[2].set_title('C. Frozen radar + contact-only adapter')
for i,a in enumerate(v):axs[2].text(i,a+2,f'{a:.0f}',ha='center')
for ax in axs:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Corrected raw radar: useful interface transfer, insufficient evidence for rich fault knowledge',fontsize=13)
fig.text(.5,.015,'8 development recordings; 58 contact windows; 412 radar windows. Seeds repeat data. No new independent fault specimens.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.06,1,.94])
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'v2_key_results.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

(OUT/'verification.json').write_text(json.dumps(dict(input_sha256={str(z):hashlib.sha256(z.read_bytes()).hexdigest() for z in sources},
    aggregation='Within each seed pool the two complementary test folds; then mean seeds. Since per-seed denominator identical, pooled repeated counts equivalent.',
    physical_primary='Train-std-normalized absolute error, equal recording weights; contact validity gate fixed before evaluation',
    physical_secondary='SSE skill added after inspecting primary results as a complete descriptive table; not used for selection',
    unique_main_recordings=8,unique_contact_windows=58,unique_radar_windows=412,
    training_runs=dict(Bear_controls=36,Rot_radar=36,UniFault_signed_and_relu=36,radar_heldout_class=96),
    model_training_runs_total=204,no_bearing_severity_labels_available=True),ensure_ascii=False,indent=2)+'\n')
print(OUT)
