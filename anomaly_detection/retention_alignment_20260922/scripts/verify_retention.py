"""Independent reload/regression audit; no model training or selection."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score,f1_score

ROOT=Path(__file__).resolve().parents[1]
GROUPS=[[0],[1,2,3],[7,8,9],[4,5,6]]
def grouped(p):return np.stack([p[...,g].sum(-1) for g in GROUPS],axis=-1)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=ROOT/'results/retention_verification';out.mkdir(exist_ok=True)
    z=np.load(ROOT/'source/cache.npz');m=z['split']=='test';h=z['hidden_refs'][m]
    y10=z['label10'][m];y4=z['label4'][m];orig=z['probs10_mean'][m];conditions=z['condition_id'][m];sources=z['source'][m]
    a0=accuracy_score(y10,orig.argmax(1));a4=accuracy_score(y4,grouped(orig).argmax(1))
    entries=[];cg=[];changed=[];perclass=[]
    for path in sorted((ROOT/'results').glob('retained_head*/models/*/*/retained_head.npz')):
        ck=np.load(path);variant=str(ck['variant']);fold=str(ck['fold']);version=path.parents[3].name
        p10=softmax(h@ck['weight10'].T+ck['bias10'],axis=-1).mean(1);p4=grouped(p10)
        assert np.isfinite(p10).all() and np.allclose(p10.sum(1),1,atol=1e-6)
        s10=accuracy_score(y10,p10.argmax(1));s4=accuracy_score(y4,p4.argmax(1))
        record=dict(version=version,variant=variant,fold=fold,acc10=s10,acc4=s4,
                    macro_f1_10=f1_score(y10,p10.argmax(1),labels=list(range(10)),average='macro',zero_division=0),
                    macro_f1_4=f1_score(y4,p4.argmax(1),labels=list(range(4)),average='macro',zero_division=0),
                    delta_acc10_pp=100*(s10-a0),delta_acc4_pp=100*(s4-a4),model_sha256=sha(path))
        if version=='retained_head_v1':
            prior=pd.read_csv(ROOT/'results'/version/'source_retention.csv')
            r=prior[(prior.variant==variant)&(prior.fold==fold)&(prior.method=='retained_head')].iloc[0]
            assert abs(s10-r.test_acc10)<1e-10 and abs(s4-r.test_acc4)<1e-10
        else:
            prior=pd.read_csv(ROOT/'results'/version/'source_group_metrics.csv')
            r=prior[(prior.variant==variant)&(prior.split=='test')&(prior.source=='ALL')].iloc[0]
            assert abs(s10-r.acc10)<1e-10 and abs(s4-r.acc4)<1e-10
        local=ROOT/'preprocessing/variants'/variant/'retrained_fcn'
        lc=np.load(local/'contact_features.npz')['hidden_mean'];md=pd.read_csv(local/'metadata.csv')
        source_bags=set(ck['local_training_bags'].tolist())
        assert not any(md.loc[md.label<0,'bag_id'].isin(source_bags))
        expected=set(md.loc[(md.label>=0)&(md.baud==(115200 if fold=='115200_to_460800' else 460800)),'bag_id']) if fold!='all_known' else set(md.loc[md.label>=0,'bag_id'])
        assert source_bags==expected
        q=softmax(lc@ck['weight10'].T+ck['bias10'],axis=1)
        targets=np.load(path.parent/'local_teacher_targets.npz')
        assert np.max(abs(q-targets['probs10']))<1e-6
        record['local_target_max_error']=float(np.max(abs(q-targets['probs10'])))
        entries.append(record)
        if version!='retained_head_conservative':continue
        for nc,truth,new,old in [(10,y10,p10,orig),(4,y4,p4,grouped(orig))]:
            npred=new.argmax(1);opred=old.argmax(1)
            changed.append(dict(version=version,variant=variant,classes=nc,n=len(truth),
                prediction_changed=int((npred!=opred).sum()),old_correct_new_wrong=int(((opred==truth)&(npred!=truth)).sum()),
                old_wrong_new_correct=int(((opred!=truth)&(npred==truth)).sum())))
            for condition in sorted(set(conditions)):
                idx=np.flatnonzero(conditions==condition);na=float(np.mean(npred[idx]==truth[idx]));oa=float(np.mean(opred[idx]==truth[idx]))
                cg.append(dict(variant=variant,classes=nc,condition_id=int(condition),source=sources[idx[0]],n=len(idx),
                               old_accuracy=oa,new_accuracy=na,delta_accuracy_pp=100*(na-oa)))
            for label in range(nc):
                idx=np.flatnonzero(truth==label)
                perclass.append(dict(variant=variant,classes=nc,label=label,n=len(idx),old_recall=float(np.mean(opred[idx]==truth[idx])),new_recall=float(np.mean(npred[idx]==truth[idx]))))
    pd.DataFrame(entries).to_csv(out/'reloaded_source_metrics.csv',index=False)
    pd.DataFrame(cg).to_csv(out/'condition_regression.csv',index=False)
    pd.DataFrame(changed).to_csv(out/'changed_predictions.csv',index=False)
    pd.DataFrame(perclass).to_csv(out/'per_class_recall.csv',index=False)
    assert len(entries)==11
    for name in ['train_retained_head.py','refine_retention.py']:
        version='retained_head_v1' if name.startswith('train') else 'retained_head_conservative'
        assert sha(ROOT/'scripts'/name)==json.loads((ROOT/'results'/version/'completion.json').read_text())['script_sha256']
    report=dict(status='passed',head_models_reloaded=len(entries),query_reference_aggregation='softmax per reference, then average',
        local_metadata_training_bags_verified=True,external_excluded_from_training=True,
        archived_script_hash_matches=True,source_test_queries=len(y10),
        note='Per-condition results are disclosed; passing global/per-dataset gate is not a guarantee for every small condition or unseen speed/geometry.')
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report));print(pd.DataFrame(changed).to_string(index=False))
    print(pd.DataFrame(cg).sort_values('delta_accuracy_pp').head(8).to_string(index=False))

if __name__=='__main__':main()
