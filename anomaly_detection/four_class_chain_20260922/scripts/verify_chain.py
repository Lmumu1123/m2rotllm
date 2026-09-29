"""Check saved student inference, split isolation, and held-out bag alignment."""
from pathlib import Path
import ast
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from infer_radar import predict,file_output

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/chain_v1'


def main():
    torch.set_num_threads(2)
    cm=pd.read_csv(ROOT/'contact/retrained_fcn_fixed_external/metadata.csv')
    cx=np.load(ROOT/'contact/retrained_fcn_fixed_external/contact_features.npz')['hidden_mean']
    rm=pd.read_csv(ROOT/'radar/metadata.csv')
    splits=json.loads((RUN/'splits.json').read_text())
    split_map={(s['task'],s['fold']):s for s in splits}
    for s in splits:
        assert set(s['train_bags']).isdisjoint(s['test_bags'])
        assert set(cm.iloc[s['train_contact_rows']].bag_id)==set(rm.iloc[s['train_radar_rows']].bag_id)==set(s['train_bags'])
        assert set(cm.iloc[s['test_contact_rows']].bag_id)==set(rm.iloc[s['test_radar_rows']].bag_id)==set(s['test_bags'])
        assert (cm.iloc[s['train_contact_rows']].label>=0).all()
        assert (rm.iloc[s['train_radar_rows']].label>=0).all()
    checks=[];align=[]
    for path in sorted((RUN/'models').rglob('*_seed42.pt')):
        p,h,ck=predict(path,ROOT/'radar/features.npz')
        ref=np.load(path.with_suffix('.predictions.npz'))
        err=float(np.max(abs(p-ref['probabilities'])))
        assert err<2e-5,(str(path),err)
        assert np.array_equal(p.argmax(1),ref['probabilities'].argmax(1))
        assert np.allclose(h,ref['embedding'],atol=2e-5,rtol=2e-5)
        fold=path.parent.name;task=path.parent.parent.name
        sp=split_map[(task,fold)]
        assert ck['training_bags']==sp['train_bags']
        checks.append(dict(model=str(path.relative_to(ROOT)),max_probability_error=err,
                           labels_or_contact_samples_required_at_inference=False))
        if sp['test_bags'] and ck['method']!='supervised':
            targets=[];students=[]
            for bag in sp['test_bags']:
                ci=(cm.bag_id==bag).to_numpy();ri=(rm.bag_id==bag).to_numpy()
                targets.append(((cx[ci]-ck['contact_mean'])/ck['contact_scale']).mean(0))
                students.append(h[ri].mean(0))
            t=np.array(targets);s=np.array(students)
            sn=s/np.maximum(np.linalg.norm(s,axis=1,keepdims=True),1e-12)
            tn=t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12)
            sim=sn@tn.T
            align.append(dict(task=task,fold=fold,method=ck['method'],n_bags=len(s),
                 mse=float(np.mean((s-t)**2)),paired_cosine=float(np.diag(sim).mean()),
                 other_cosine=float(sim[~np.eye(len(s),dtype=bool)].mean()),
                 bag_retrieval_top1=float(np.mean(sim.argmax(1)==np.arange(len(s)))),
                 note='one held-out bag per class; retrieval cannot separate class matching from instance matching'))
        if task=='four_class_provisional_roi' and fold=='all_known' and ck['method']=='distill':
            output=file_output(p,rm)
            assert (output[output.geometry_roi_mismatch].diagnosis_status=='invalid_or_unverified_target_roi').all()
            output.to_csv(RUN/'radar_only_reloaded_predictions.csv',index=False)
    metrics=pd.read_csv(RUN/'metrics.csv',dtype={'fold':str})
    assert len(metrics)==888,len(metrics)
    assert len(checks)==48,len(checks)
    traces=pd.read_csv(RUN/'training_trace.csv',dtype={'fold':str})
    trained=traces[['task','fold','method','seed']].drop_duplicates()
    assert len(trained)==592,len(trained)
    assert traces.groupby(['task','fold','method','seed']).step.max().eq(200).all()
    pd.DataFrame(align).to_csv(RUN/'heldout_alignment.csv',index=False)
    for path in (ROOT/'scripts').glob('*.py'):ast.parse(path.read_text())
    result=dict(status='passed',python=sys.executable,model_reload_count=len(checks),
        trained_student_instances=len(trained),metric_rows=len(metrics),folds_including_external=len(splits),
        split_isolation=True,external_cases_absent_from_training=True,all_saved_models_match=checks,
        parameter_gradients_only_student='Original FCN frozen during export; adapted contact heads fitted only on source then constants in student loss',
        limitations=['known wrong ROI outside primary three-type subset', 'repeated file partitions not independent experiments',
                     'bag alignment is not synchronized window alignment', 'no calibrated unknown-fault detector'])
    (RUN/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='all_saved_models_match'},indent=2))


if __name__=='__main__':main()
