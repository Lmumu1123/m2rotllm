#!/usr/bin/env python3
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_baselines import LABELS,sha256

OUT=Path(__file__).resolve().parent
def read(name):return list(csv.DictReader(open(OUT/name,encoding='utf-8-sig')))

windows=read('window_predictions.csv');records=read('recording_predictions.csv')
groups=defaultdict(list)
for w in windows:groups[(w['protocol'],w['fold'],w['feature'],w['method'],w['recording_id'])].append(w)
for r in records:
    ww=groups[(r['protocol'],r['fold'],r['feature'],r['method'],r['recording_id'])]
    pp=np.mean([[float(w['score_'+la]) for la in LABELS] for w in ww],axis=0)
    assert np.allclose(pp,[float(r['score_'+la]) for la in LABELS],rtol=0,atol=1e-12)
    assert pp.argmax()==int(r['y_pred']) and len(ww)==int(r['n_windows'])
for m in read('metrics.csv'):
    source=windows if m['level']=='window' else records
    rr=[r for r in source if r['protocol']==m['protocol'] and r['feature']==m['feature'] and r['method']==m['method']]
    if m['fold']!='pooled_three_distances':rr=[r for r in rr if r['fold']==m['fold']]
    if m['subgroup']!='all':rr=[r for r in rr if r[m['subgroup']]==m['subgroup_value']]
    y=np.array([int(r['y_true']) for r in rr]);p=np.array([int(r['y_pred']) for r in rr])
    assert len(rr)==int(m['n'])
    assert np.isclose(accuracy_score(y,p),float(m['accuracy']),atol=1e-12)
    assert np.isclose(f1_score(y,p,labels=np.arange(4),average='macro',zero_division=0),float(m['macro_f1']),atol=1e-12)
splits=json.loads((OUT/'splits.json').read_text())
for f in splits:
    assert not set(f['train_recordings'])&set(f['test_recordings'])
    assert len(f['test_recordings'])==(36 if f['protocol']=='recording_rep4' else 48)
    assert len(f['shared_contact_ids'])==12
e=json.loads((OUT/'evidence.json').read_text())
assert e['script_sha256']==sha256(OUT/'run_diagnostics.py')
assert e['protocol_sha256']==sha256(OUT/'protocol.json')
summary=dict(passed=True,metric_rows_recomputed=len(read('metrics.csv')),recording_aggregates_recomputed=len(records),
             window_predictions=len(windows),protocol_fold_count=len(splits),no_radar_recording_overlap=True,
             contact_ids_shared_but_contact_data_unused=True)
(OUT/'verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
