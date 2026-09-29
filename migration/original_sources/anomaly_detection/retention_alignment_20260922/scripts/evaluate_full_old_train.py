"""Full original-training-query regression for already frozen selected heads."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import pandas as pd
import h5py
import torch
from sklearn.metrics import accuracy_score,f1_score
from shared_contact_model import SharedFourClassModel,ORIGINAL,GROUPS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/full_old_train_regression'
DATA=Path('/media/nas_users/huangyating/bearllm-assets/mbhm_dataset')
def main():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,'/home/huangyating/BearLLM/scripts')
    from evaluate_mbhm import read_metadata,make_protocol,sha256
    allpairs,_=make_protocol(read_metadata(DATA),42,3)
    pairs=[r for r in allpairs if r['split']=='train']
    records=[pairs[i] for i in range(0,len(pairs),3)];assert len(records)==85955
    head=ROOT/'results/retained_head_conservative/models/v0/all_known/retained_head.npz'
    model=SharedFourClassModel(head).to('cuda:1').eval()
    old=np.load(ROOT/'source/original_head.npz')
    heads={'original':(old['weight10'],old['bias10'])}
    for variant in ['v0','query_no_demean']:
        z=np.load(ROOT/'results/retained_head_conservative/models'/variant/'all_known/retained_head.npz')
        heads[variant]=(z['weight10'],z['bias10'])
    weights={k:(torch.tensor(w,device='cuda:1'),torch.tensor(b,device='cuda:1')) for k,(w,b) in heads.items()}
    results={k:np.empty((len(records),10),np.float32) for k in heads}
    with h5py.File(DATA/'data.hdf5','r') as f:data=f['vibration'][:]
    with torch.inference_mode():
        for start in range(0,len(records),64):
            batch=pairs[start*3:min(start+64,len(records))*3]
            ids=np.array([[r['file_id'],r['ref_id']] for r in batch])
            h=model.contact_hidden(torch.tensor(data[ids],device='cuda:1'))
            for key,(w,b) in weights.items():
                results[key][start:start+len(batch)//3]=(h@w.T+b).softmax(-1).reshape(-1,3,10).mean(1).cpu().numpy()
    truth10=np.array([r['label'] for r in records]);mapping=np.array([0,1,1,1,3,3,3,2,2,2]);truth4=mapping[truth10]
    sources=np.array([r['source'] for r in records]);metrics=[]
    for key,p in results.items():
        p4=np.stack([p[:,g].sum(-1) for g in GROUPS],-1)
        for src in ['ALL']+sorted(set(sources)):
            idx=np.arange(len(p)) if src=='ALL' else np.flatnonzero(sources==src)
            orig=results['original'];op4=np.stack([orig[:,g].sum(-1) for g in GROUPS],-1)
            a10=accuracy_score(truth10[idx],p[idx].argmax(1));a4=accuracy_score(truth4[idx],p4[idx].argmax(1))
            d10=100*(a10-accuracy_score(truth10[idx],orig[idx].argmax(1)));d4=100*(a4-accuracy_score(truth4[idx],op4[idx].argmax(1)))
            metrics.append(dict(model=key,source=src,n=len(idx),accuracy10=a10,accuracy4=a4,
                macro_f1_10=f1_score(truth10[idx],p[idx].argmax(1),labels=list(range(10)),average='macro',zero_division=0),
                macro_f1_4=f1_score(truth4[idx],p4[idx].argmax(1),labels=list(range(4)),average='macro',zero_division=0),
                delta_acc10_pp=d10,delta_acc4_pp=d4,meets_half_pp=d10>=-.5-1e-10 and d4>=-.5-1e-10))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    np.savez_compressed(OUT/'predictions.npz',**{k+'_p10':v for k,v in results.items()},label10=truth10,label4=truth4,
        query_id=np.array([r['file_id'] for r in records]),source=sources)
    report=dict(status='complete',original_training_queries=len(records),reference_pairs=len(pairs),
        no_training_or_model_selection=True,already_frozen_conservative_heads=True,
        input_data_already_DCN=True,original_reference_protocol_preserved=True,
        user_constraint='Original training and test accuracies checked separately; not unseen-data generalization.',
        original_weights={str(ORIGINAL/f):sha256(ORIGINAL/f) for f in ['feature_encoder.pth','classifier.pth']})
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__=='__main__':main()
