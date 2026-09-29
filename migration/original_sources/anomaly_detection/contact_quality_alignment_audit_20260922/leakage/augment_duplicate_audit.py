"""No-fit sensitivity after identifying exact source-query duplicates."""
from pathlib import Path
from collections import Counter
import hashlib,json,sqlite3
import numpy as np
import pandas as pd
from scipy.special import softmax

OUT=Path(__file__).resolve().parent
ROOT=Path('/home/huangyating/anomaly_detection/retention_alignment_20260922')
RAW=ROOT.parent/'four_class_preprocessing_20260921/results'
GROUPS=[[0],[1,2,3],[7,8,9],[4,5,6]]
def group(p):return np.stack([p[...,g].sum(-1) for g in GROUPS],-1)

def main():
    duplicates=json.loads((OUT/'source_cross_split_exact_duplicate_rows.json').read_text())
    counts=Counter();overlap={s:set() for s in ['val','test']};entities=[]
    for d in duplicates:
        sources=','.join(sorted(set(d['sources'])));counts[sources]+=1
        if 'train' in d['query_splits']:
            for s in ['val','test']:
                overlap[s].update(i for i,qsplit in zip(d['file_ids'],d['query_splits']) if qsplit==s)
            for i,split in zip(d['file_ids'],d['query_splits']):
                if split in ['val','test']:entities.append(dict(file_id=i,split=split,sha256=d['sha256'],exact_matching_training_query_ids=[q for q,s in zip(d['file_ids'],d['query_splits']) if s=='train'],source=sources))
    pd.DataFrame(entities).to_csv(OUT/'source_eval_queries_identical_to_train.csv',index=False)
    (OUT/'source_train_eval_duplicate_counts.json').write_text(json.dumps(dict(cross_groups_by_source=dict(counts),validation_queries_identical_to_training_query=len(overlap['val']),test_queries_identical_to_training_query=len(overlap['test']),test_query_ids=sorted(overlap['test']),cross_groups_with_inconsistent_labels=sum(len(set(d['labels']))>1 for d in duplicates)),indent=2)+'\n')
    z=np.load(ROOT/'source/cache.npz');test=(z['split']=='test');ids=z['query_id'][test];h=z['hidden_refs'][test];y10=z['label10'][test];y4=z['label4'][test];original=z['probs10_mean'][test]
    retained=~np.isin(ids,list(overlap['test']));assert retained.sum()==12248
    models=[('original','original','original',None)]
    models += [('v1',variant,fold,ROOT/'results/retained_head_v1/models'/variant/fold/'retained_head.npz') for variant in ['v0','query_no_demean'] for fold in ['115200_to_460800','460800_to_115200','all_known']]
    models += [('conservative',variant,'all_known',ROOT/'results/retained_head_conservative/models'/variant/'all_known/retained_head.npz') for variant in ['v0','query_no_demean']]
    rows=[]
    for version,variant,fold,path in models:
        if path is None:p=original
        else:
            head=np.load(path);p=softmax(h@head['weight10'].T+head['bias10'],axis=-1).mean(1)
        for scope,mask in [('same_original_test',np.ones(len(ids),bool)),('remove_31_exact_train_query_duplicates_only',retained)]:
            rows.append(dict(version=version,variant=variant,fold=fold,scope=scope,n=int(mask.sum()),correct10=int((p[mask].argmax(1)==y10[mask]).sum()),acc10=float((p[mask].argmax(1)==y10[mask]).mean()),correct4=int((group(p[mask]).argmax(1)==y4[mask]).sum()),acc4=float((group(p[mask]).argmax(1)==y4[mask]).mean())))
    df=pd.DataFrame(rows)
    for scope in df.scope.unique():
        bm=df[(df.version=='original')&(df.scope==scope)].iloc[0]
        for k in ['10','4']:df.loc[df.scope==scope,f'delta_acc{k}_pp']=100*(df.loc[df.scope==scope,f'acc{k}']-bm[f'acc{k}'])
    df['not_a_clean_protocol']=True;df.to_csv(OUT/'remove_exact_duplicates_sensitivity.csv',index=False)
    # Does the current train-replay query subset contain such duplicates?
    replayq=set(z['query_id'][z['split']=='train'].tolist());direct=set()
    for d in duplicates:
        if set(d['file_ids'])&replayq:direct.update(q for q,s in zip(d['file_ids'],d['query_splits']) if s=='test')
    # Cross-domain exact DCN match only, not a raw-signal independence claim.
    sh=set(pd.read_csv(OUT/'source_dcn_row_hashes.csv',usecols=['sha256']).sha256)
    matches=[];n=0
    for item in json.loads((RAW/'summary.json').read_text()):
        if item['modality']!='contact':continue
        a=np.load(RAW/item['output'])['dcn']
        for i,row in enumerate(a):
            for axis,v in enumerate(row):
                n+=1;digest=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                if digest in sh:matches.append(dict(file=item['output'],window=i,axis=axis,sha256=digest))
    extra=dict(current_replay_training_queries_with_test_DCN_duplicate_test_queries=sorted(direct),n_local_contact_axis_DCN_compared_to_full_source=n,n_exact_source_local_DCN_matches=len(matches),matches=matches,sensitivity_limit='Removing 31 query duplicates leaves cross-split healthy references, shared conditions, unknown original recordings/entities and reused test; not a clean test.')
    (OUT/'additional_duplicate_checks.json').write_text(json.dumps(extra,indent=2)+'\n')
    print(df.to_string(index=False));print(extra)

if __name__=='__main__':main()
