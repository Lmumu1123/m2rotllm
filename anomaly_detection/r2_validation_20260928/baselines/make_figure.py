#!/usr/bin/env python3
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
def read(p):return list(csv.DictReader(open(p,encoding='utf-8-sig')))
main=read(OUT/'metrics_pooled.csv')
diag=read(OUT/'diagnostics/metrics.csv')
models=[('complex_shape','rbf_svc','Complex spectrum + RBF'),('complex_shape','random_forest','Complex spectrum + RF'),('nuisance','rbf_svc','Amplitude/range + RBF')]
fig,ax=plt.subplots(figsize=(10,5))
xx=np.arange(3);width=.24
for i,(feature,method,title) in enumerate(models):
    a=next(float(r['accuracy']) for r in main if r['feature']==feature and r['method']==method and r['level']=='recording' and r['distance_cm']=='all')
    b=next(float(r['accuracy']) for r in diag if r['protocol']=='recording_rep4' and r['feature']==feature and r['method']==method and r['level']=='recording' and r['subgroup']=='all')
    c=next(float(r['accuracy']) for r in diag if r['fold']=='pooled_three_distances' and r['feature']==feature and r['method']==method and r['level']=='recording' and r['subgroup']=='all')
    bars=ax.bar(xx+(i-1)*width,100*np.array([b,a,c]),width,label=title)
    ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
ax.axhline(25,color='gray',ls=':',lw=1,label='Balanced 4-class chance level')
ax.set_xticks(xx,['Held-out recording rep4\nSame speed/distance seen; 36 records','Held-out speed\n144 out-of-fold records','Held-out distance\n144 out-of-fold records'])
ax.set_ylabel('Recording accuracy (%)');ax.set_ylim(0,116)
ax.set_title('Pure radar baselines: different protocols must not be averaged together')
ax.legend(ncol=2,loc='upper center',fontsize=9);ax.grid(axis='y',alpha=.15)
fig.tight_layout();fig.savefig(OUT/'protocol_comparison.png',dpi=180)
