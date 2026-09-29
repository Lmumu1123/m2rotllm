"""Summarize fixed experiments without selecting methods, seeds or checkpoints."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'iteration_summary'
OUT.mkdir(exist_ok=True)
sources = []

def read(rel):
    p = ROOT / rel
    sources.append(p)
    return pd.read_csv(p)

def pooled(d, keys, ncol='n', acol='accuracy', ccol=None):
    d = d.copy()
    d['count_correct'] = d[ccol] if ccol else np.rint(d[ncol] * d[acol]).astype(int)
    out = d.groupby(keys, dropna=False).agg(correct_repeated=('count_correct', 'sum'),
                                           n_repeated=(ncol, 'sum')).reset_index()
    out['accuracy'] = out.correct_repeated / out.n_repeated
    return out

c = read('controls/results/metrics.csv')
c = c[c.role.eq('test')]
ct = pooled(c, ['method'], 'n_windows', 'window_accuracy')
ct = ct.merge(c.groupby('method')[['bag_accuracy','mse_standardized','cos_standardized']].mean(), on='method')
ct.to_csv(OUT/'bear_controls.csv', index=False)

r = read('rotllm/radar/metrics.csv')
rt = pooled(r[r.role.eq('test')], ['head','method','subset','unit'])
rt.to_csv(OUT/'rotllm_radar.csv', index=False)

s = read('results/contact_only_stitch/metrics.csv')
st = pooled(s, ['head','path','radar_method','scope','unit'], ccol='correct')
st.to_csv(OUT/'contact_only_stitch.csv', index=False)

cp = read('controls/results/contact_predictions.csv')
cp = cp[cp.role.eq('test')]
bear_c = dict(name='archived_Bear_stageB_contact',correct=int(np.rint(cp.n_windows*cp.window_accuracy).sum()),
              n=int(cp.n_windows.sum()),recordings_correct=int(cp.label.eq(cp.prediction).sum()),recordings_n=len(cp))
read('rotllm/pooled_metrics.csv')
(OUT/'contact_reference.json').write_text(json.dumps(bear_c,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1,3,figsize=(15,4.5))
colors=['#778DA9','#277DA8','#37A18E','#D69B45']
methods=['ce_only','matched_bag_mse','full','label_code_mse']
labels=['Class only','Teacher mean','Full loss','Artificial codes']
axes[0].bar(np.arange(4),[ct.set_index('method').loc[m,'bag_accuracy']*100 for m in methods],color=colors)
axes[0].set_xticks(np.arange(4),labels,rotation=22,ha='right')
axes[0].set_title('A. Same Bear head: all reach 100%')
axes[0].set_ylabel('Recording accuracy (%)')
axes[0].set_ylim(0,115)
for i in range(4): axes[0].text(i,102,'100',ha='center')

tab=rt[(rt['head']=='native15')&(rt.subset=='all_four_provisional_roi')&(rt.unit=='window')].set_index('method')
vals=[tab.loc[m,'accuracy']*100 for m in ['ce_only','embedding_only','full']]
axes[1].bar(np.arange(3),vals,color=colors[:3])
axes[1].set_xticks(np.arange(3),['Class only','Teacher mean','Full loss'],rotation=22,ha='right')
axes[1].set_title('B. Original Rot head is unsuitable here')
axes[1].set_ylabel('Radar window accuracy (%)')
axes[1].set_ylim(0,115)
for i,v in enumerate(vals): axes[1].text(i,v+2,f'{v:.1f}',ha='center')

tab=st[(st['head']=='contact_only_new4')&(st.path=='frozen_radar_A_to_B')&(st.unit=='recording')]
x=np.arange(4); width=.37
for off,scope,color,title in [(-width/2,'all_four_development','#277DA8','All 4 classes; outer ROI wrong'),
                            (width/2,'known_roi_valid_subset','#D69B45','Exclude known-wrong outer ROI')]:
    t=tab[tab.scope.eq(scope)].set_index('radar_method')
    vs=[t.loc[m,'accuracy']*100 for m in methods]
    axes[2].bar(x+off,vs,width,color=color,label=title)
axes[2].set_xticks(x,labels,rotation=22,ha='right')
axes[2].set_title('C. Frozen radar transferred to Rot readout')
axes[2].set_ylabel('Recording accuracy (%)')
axes[2].set_ylim(0,125)
axes[2].legend(loc='upper left',fontsize=8,frameon=False)
for ax in axes:
    ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
fig.suptitle('Development evidence: prediction accuracy alone does not establish feature transfer',fontsize=13)
fig.text(.5,.012,'8 existing recordings; 2 folds; 3 seeds. Repeats are not new samples. Known outer-range error remains unrepaired.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.055,1,.94])
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'validation_summary.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

verification = dict(
    aggregation='Pool complementary test folds within each seed, then average the 3 seeds. Identical denominators per seed allow equivalent pooled correct/total calculation.',
    warning='Repeated counts in CSV are computational totals, not independent physical sample counts.',
    unique_main_recordings=8,unique_main_contact_windows=58,unique_main_radar_windows=412,
    known_roi_valid_radar_windows=312,old_data_previously_used_for_development=True,
    inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
print(OUT)
print(ct.to_string(index=False))
