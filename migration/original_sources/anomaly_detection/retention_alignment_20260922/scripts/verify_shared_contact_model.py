"""Real signal -> frozen FCN -> retained four-type module interface check."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import h5py
import importlib.util
from scipy.special import softmax
from shared_contact_model import SharedFourClassModel,ORIGINAL,GROUPS

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(2);device='cuda:1' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32=False
    head=ROOT/'results/retained_head_conservative/models/v0/all_known/retained_head.npz'
    paths=[ORIGINAL/'feature_encoder.pth',ORIGINAL/'classifier.pth'];before={str(p):sha(p) for p in paths}
    model=SharedFourClassModel(head).to(device).eval();z=np.load(ROOT/'source/cache.npz')
    # Fixed first query per source, source split TEST, after model selection;
    # this verifies implementation, not another model-choice experiment.
    ids=np.flatnonzero(z['split']=='test');chosen=[]
    for source in sorted(set(z['source'][ids])):chosen.append(int(ids[np.flatnonzero(z['source'][ids]==source)[0]]))
    with h5py.File('/media/nas_users/huangyating/bearllm-assets/mbhm_dataset/data.hdf5','r') as f:
        data=f['vibration'];pairs=np.stack([np.stack([np.stack([data[int(z['query_id'][i])],data[int(r)]]) for r in z['reference_ids'][i]]) for i in chosen]).astype(np.float32)
    t=torch.tensor(pairs,device=device)
    with torch.inference_mode():
        result=model.contact_views(t,pool='probability')
        h=model.contact_hidden(t.flatten(0,1)).cpu().numpy().reshape(len(chosen),3,128)
        spec=importlib.util.spec_from_file_location('original_fcn_check','/home/huangyating/BearLLM/models/FCN.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        original=mod.FaultClassificationNetwork()
        original.encoder.load_state_dict(torch.load(ORIGINAL/'feature_encoder.pth',map_location='cpu',weights_only=True))
        original.classifier.load_state_dict(torch.load(ORIGINAL/'classifier.pth',map_location='cpu',weights_only=True))
        original.to(device).eval()
        direct_h=torch.relu(original.classifier.linear1(original.encoder(t.flatten(0,1)).flatten(1))).cpu().numpy().reshape(h.shape)
        interface_hidden_error=float(np.max(abs(h-direct_h)))
        # Exact logits4 interface versus marginalization of internal probabilities.
        single=model.classify_embedding(torch.tensor(h[:,0],device=device))
        marginal=torch.stack([single['probabilities10'][:,g].sum(-1) for g in GROUPS],-1)
        ge=float((marginal-single['probabilities4']).abs().max().cpu())
    ck=np.load(head);expected=softmax(z['hidden_refs'][chosen]@ck['weight10'].T+ck['bias10'],axis=-1).mean(1)
    prob_error=float(np.max(abs(result['probabilities10'].cpu().numpy()-expected)))
    hidden_error=float(np.max(abs(h-z['hidden_refs'][chosen])))
    hidden_relative_l2=float(np.linalg.norm(h-z['hidden_refs'][chosen])/np.linalg.norm(z['hidden_refs'][chosen]))
    print(json.dumps(dict(probability_error=prob_error,hidden_error=hidden_error,hidden_relative_l2=hidden_relative_l2,original_interface_error=interface_hidden_error,group_error=ge)),flush=True)
    # Compare the unchanged FCN at the same batch size exactly. The cached
    # reference was exported in larger CUDA convolution batches; quantify its
    # numerical difference separately instead of claiming bitwise identity.
    assert interface_hidden_error==0 and prob_error<2e-5 and ge<2e-6
    assert np.array_equal(result['probabilities10'].cpu().numpy().argmax(1),expected.argmax(1))
    assert before=={str(p):sha(p) for p in paths}
    out=ROOT/'results/retention_verification'
    report=dict(status='passed',n_raw_query_reference_pairs=int(len(chosen)*3),device=device,
        source_queries=[int(z['query_id'][i]) for i in chosen],source_hidden_max_error=hidden_error,
        source_hidden_relative_l2_error=hidden_relative_l2,original_FCN_same_batch_hidden_max_error=interface_hidden_error,
        cached_batch_numeric_difference_disclosed=True,
        source_probability_max_error=prob_error,group_logsumexp_probability_identity_max_error=ge,
        public_forward_classes=4,retained_internal_logits=10,contact_and_radar_share_identical_head=True,
        original_weights_unchanged=True,original_weight_hashes=before)
    (out/'shared_model_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':main()
