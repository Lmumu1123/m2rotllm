#!/usr/bin/env python3
"""Export explicit head(mean XYZ hidden) native contact reference for stitching."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
B=Path(__file__).resolve().parent
m=pd.read_csv(B/'metadata.csv');z=np.load(B/'teacher_features.npz',allow_pickle=False)
p15=z['probs15_from_mean_hidden'];groups=[[0],[1,2,3],[7,8,9],[4,5,6]]
p4=np.stack([p15[:,g].sum(1) for g in groups],1);p4/=p4.sum(1,keepdims=True)
for i in range(4):m[f'p4_{i}']=p4[:,i]
for i in range(15):m[f'p15_{i}']=p15[:,i]
m['pred4']=p4.argmax(1);m['pred15']=p15.argmax(1);m['gear_mass']=p15[:,10:].sum(1)
cols=[f'p4_{i}' for i in range(4)]+[f'p15_{i}' for i in range(15)]
f=m.groupby(['file','label','state','baud','bag_id'],as_index=False)[cols].mean()
f['pred4']=f[[f'p4_{i}' for i in range(4)]].to_numpy().argmax(1)
f['pred15']=f[[f'p15_{i}' for i in range(15)]].to_numpy().argmax(1)
f['gear_mass']=f[[f'p15_{i}' for i in range(10,15)]].sum(1)
m.to_csv(B/'native_mean_hidden_window_predictions.csv',index=False);f.to_csv(B/'native_mean_hidden_file_predictions.csv',index=False)
out=[]
for unit,df in [('window',m),('file',f)]:
    d=df[df.label>=0]
    out.append(dict(unit=unit,n=len(d),correct=int((d.label==d.pred4).sum()),accuracy=float(accuracy_score(d.label,d.pred4)),macro_f1=float(f1_score(d.label,d.pred4,labels=range(4),average='macro',zero_division=0)),gear_mass_mean=float(d.gear_mass.mean()),pred15_other_rate=float((d.pred15>=10).mean())))
(B/'native_mean_hidden_metrics.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
