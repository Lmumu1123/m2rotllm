#!/usr/bin/env python3
"""Join separate native-p10 generation batches; never regenerate/rewrite them."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from run_p10_llm import ROOT,GROUPS,summarize

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    batches=[ROOT/'results'/x for x in ['llm_native_p10_chain','llm_native_p10_conservative']]
    out=ROOT/'results/llm_native_p10_combined';out.mkdir(exist_ok=False)
    data=pd.concat([pd.read_csv(p/'generation_predictions.csv',dtype={'fold':str}) for p in batches],ignore_index=True)
    assert len(data)==152 and data.id.nunique()==152
    conflicts=[]
    for row in data.itertuples():
        p=np.array(json.loads(row.p10_used));g=np.array([p[ix].sum() for ix in GROUPS])
        assert np.isclose(p.sum(),1,atol=1e-6)
        assert np.allclose(g,json.loads(row.p4_grouped),atol=1e-6)
        assert row.probability_argmax==int(g.argmax())
        assert row.probability_faithful==(row.parsed_coarse_label==row.probability_argmax)
        assert not row.guard_uses_ground_truth
        if not row.geometry_valid:assert row.guarded_label==-1 and not row.diagnosis_usable
        if row.llm_class_conflict:conflicts.append(row.id)
    data.to_csv(out/'generation_predictions.csv',index=False)
    data[data.llm_class_conflict].to_csv(out/'raw_llm_class_conflicts.csv',index=False)
    summarize(data,out)
    held=data[data.evaluation_scope=='heldout_main'].copy()
    held['raw_correct']=held.parsed_coarse_label==held.true_label
    held['classifier_correct']=held.probability_argmax==held.true_label
    summary=held.groupby(['stage','variant','method'],as_index=False).agg(n=('id','size'),
        unique_recordings=('bag_id','nunique'),raw_accuracy=('raw_correct','mean'),
        classifier_accuracy=('classifier_correct','mean'),raw_fidelity=('probability_faithful','mean'),
        geometry_valid_count=('geometry_valid','sum'))
    summary.to_csv(out/'heldout_summary.csv',index=False)
    prov=dict(source_batches={str(p/'generation_predictions.csv'):sha(p/'generation_predictions.csv') for p in batches},
        source_provenance={str(p/'provenance.json'):sha(p/'provenance.json') for p in batches},
        rows=len(data),raw_faithful=int(data.probability_faithful.sum()),raw_fidelity=float(data.probability_faithful.mean()),
        raw_class_conflicts=int(data.llm_class_conflict.sum()),unparseable=int((data.parsed_coarse_label<0).sum()),
        truncated=int(data.truncated.sum()),invalid_geometry=int((~data.geometry_valid).sum()),
        p10_to_p4_checked=True,guard_truth_independent=True,controls_separate=4,
        inference_rows_not_independent_recordings=True,native_probabilities_used_without_p4_expansion=True)
    (out/'verification.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(prov,ensure_ascii=False))

if __name__=='__main__':main()
