#!/usr/bin/env python3
"""Deterministic consistency/ROI status guard; does not modify LLM outputs.

The decision function never reads true_label or state. Closed-set confidence
cannot certify detection of an unknown fault, for any input row.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

PHRASES=['Normal','Inner race fault','Outer race fault','Rolling element fault']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def guard(prob, parsed, generated):
    """Only model outputs enter the class-conflict decision."""
    p=np.asarray(prob,np.float64)
    if p.shape!=(4,) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1,atol=1e-4):
        raise ValueError('Invalid probability vector')
    candidate=int(p.argmax())
    conflict=int(parsed)!=candidate
    return candidate,conflict,PHRASES[candidate] if conflict else str(generated)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent/'llm_bridge_chain_v1')
    a.add_argument('--radar-quality',type=Path,default=Path(__file__).resolve().parents[1]/'radar/recording_quality.csv')
    args=a.parse_args()
    source=args.directory/'generation_predictions.csv'
    inputs=[source,args.directory/'generation_predictions.jsonl',args.radar_quality]
    before={str(p):sha(p) for p in inputs}
    df=pd.read_csv(source,dtype={'fold':str})
    quality=pd.read_csv(args.radar_quality)
    if quality.recording_id.duplicated().any():raise ValueError('Ambiguous radar quality metadata')
    q=quality.set_index('recording_id').to_dict(orient='index')
    rows=[]
    for row in df.to_dict(orient='records'):
        label,conflict,text=guard([row[f'p{i}'] for i in range(4)],row['parsed_coarse_label'],row['generated_text'])
        modality=str(row['modality']); info=q.get(str(row['bag_id']))
        mismatch=None
        if modality=='radar':
            if info is None or pd.isna(info.get('geometry_roi_mismatch')):
                status='invalid_or_unverified_target_roi'
            else:
                rawflag=info['geometry_roi_mismatch']
                if isinstance(rawflag,str):
                    if rawflag.lower() not in ['true','false']:raise ValueError('Invalid geometry mismatch flag')
                    mismatch=rawflag.lower()=='true'
                else:mismatch=bool(rawflag)
                status='invalid_target_roi' if mismatch else 'experimental_known_class_candidate'
        elif modality=='contact':status='experimental_known_class_candidate'
        else:raise ValueError(f'Unknown modality {modality}')
        rows.append({**row,'structured_pred_label':label,'structured_pred_name':PHRASES[label],
                     'llm_class_conflict':conflict,'language_fallback_applied':conflict,'guarded_text':text,
                     'guarded_output_source':'deterministic_class_phrase' if conflict else 'original_llm_generation',
                     'guarded_class_label':label if conflict else int(row['parsed_coarse_label']),
                     'geometry_roi_mismatch':mismatch,'diagnosis_status':status,
                     'known_class_candidate_only':True,'unknown_detection_supported':False,
                     'unknown_status':'not_assessed_by_closed_set_classifier',
                     'radar_range_center_m':info['range_center_m'] if modality=='radar' and info else None})
    out=pd.DataFrame(rows)
    # Persist every original probability exactly as read; no posterior correction.
    assert np.array_equal(out[[f'p{i}' for i in range(4)]].to_numpy(),df[[f'p{i}' for i in range(4)]].to_numpy())
    assert out.guarded_class_label.eq(out.structured_pred_label).all()
    # Deliberate truth/state perturbation checks that these cannot trigger fallback.
    mutated=df.copy();mutated['true_label']=-999;mutated['state']='arbitrary_unseen_state'
    for row, expected in zip(mutated.to_dict(orient='records'),rows):
        pred,conflict,text=guard([row[f'p{i}'] for i in range(4)],row['parsed_coarse_label'],row['generated_text'])
        assert (pred,conflict,text)==(expected['structured_pred_label'],expected['llm_class_conflict'],expected['guarded_text'])
    out.to_csv(args.directory/'guarded_predictions.csv',index=False)
    after={str(p):sha(p) for p in inputs};assert before==after
    n=len(out);fallback=int(out.language_fallback_applied.sum())
    audit=dict(rows=n,raw_llm_probability_fidelity_count=int(out.parsed_coarse_label.eq(out.structured_pred_label).sum()),
               raw_llm_probability_fidelity=float(out.parsed_coarse_label.eq(out.structured_pred_label).mean()),
               fallback_count=fallback,conflict_ids=out.loc[out.llm_class_conflict,'id'].tolist(),
               guarded_rule_consistency_count=int(out.guarded_class_label.eq(out.structured_pred_label).sum()),
               guarded_rule_consistency=float(out.guarded_class_label.eq(out.structured_pred_label).mean()),
               rule='if parsed_coarse_label differs from probability argmax, retain classifier probabilities and argmax; replace only language with fixed class phrase',
               interpretation='Guarded consistency is guaranteed by a deterministic rule, not improved LLM learned accuracy. Original generations and raw LLM metrics remain unchanged.',
               ground_truth_and_state_used_for_fallback=False,truth_state_perturbation_invariance_passed=True,
               unknown_detection_supported=False,
               unknown_note='All rows remain closed-set candidates; high confidence does not prove unknown detection. keep is not credited as a correct unknown classification.',
               roi_source=str(args.radar_quality),roi_rule='For radar rows join actual recording_id by bag_id, use geometry_roi_mismatch; missing quality is unverified. Contact rows have no radar ROI gate.',
               invalid_or_unverified_radar_rows=int(out.diagnosis_status.str.startswith('invalid').sum()),
               diagnosis_status_counts=out.diagnosis_status.value_counts().to_dict(),
               original_inputs_sha256=before,original_inputs_unchanged=True,script_sha256=sha(__file__))
    (args.directory/'guard_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:audit[k] for k in ['rows','raw_llm_probability_fidelity_count','fallback_count','guarded_rule_consistency_count','diagnosis_status_counts']}))

if __name__=='__main__':main()
