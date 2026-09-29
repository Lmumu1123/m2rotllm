"""Independent disk-contract verification, no training and no score selection."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from scipy.special import softmax
import run_unifault as run

def main():
    torch.set_num_threads(2)
    base=Path(__file__).resolve().parent;root=base/'unifault'
    model,provenance=run.load_official();records=[]
    md=pd.read_csv(root/'metadata.csv')
    raw=np.stack([np.load(run.CONTACT/r.file,allow_pickle=False)['raw_xyz'][r.window_row] for r in md.itertuples()]).astype(np.float32)
    hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in raw]
    assert len(set(hashes))==len(hashes),'Unexpected exact duplicate contact window'
    for interface in json.loads((base/'teacher_interfaces.json').read_text()):
        feat=np.load(interface['feature_file']);head=np.load(interface['head_file'])
        m=pd.read_csv(interface['metadata_file']);assert m.equals(md)
        assert np.array_equal(feat['bag_id'],md.bag_id.to_numpy(str))
        assert np.array_equal(feat['window_rows'],md.window_row.to_numpy())
        train=md.bag_id.isin(head['train_bag_ids']).to_numpy();test=md.bag_id.isin(head['test_bag_ids']).to_numpy()
        assert not (train&test).any() and not (md.loc[train|test,'label']<0).any()
        z=resample_poly(raw.reshape(88,10,400,3).transpose(0,1,3,2),64,25,axis=-1).astype(np.float32)
        lo=z[train].min((0,1,3));hi=z[train].max((0,1,3))
        assert np.array_equal(lo,feat['input_min']) and np.array_equal(hi,feat['input_max'])
        pick=np.array([np.flatnonzero(train)[0],np.flatnonzero(test)[0],np.flatnonzero(md.label<0)[0]])
        zz=(z[pick]-lo[None,None,:,None])/(hi-lo+1e-6)[None,None,:,None]
        hs,logits=run.get_features(model,zz)
        error=float(np.abs(hs.mean((1,2))-feat['hidden_mean'][pick]).max());assert error<2e-6
        p=softmax(feat['hidden_mean'].astype(float)@head['weight4'].T+head['bias4'],axis=1)
        recorded=pd.read_csv(Path(interface['feature_file']).parent/'window_predictions.csv')
        prob_error=float(abs(p-recorded[[f'p{k}' for k in range(4)]].to_numpy()).max());assert prob_error<1e-12
        records.append(dict(fold=interface['fold'],metadata_identity=True,train_only_minmax_verified=True,
            reproduced_train_test_external_rows=pick.tolist(),feature_reload_max_abs=error,
            classifier_reload_max_probability_abs=prob_error,
            no_exact_contact_window_duplicates_across_recordings=True))
    (root/'independent_verification.json').write_text(json.dumps(dict(checks=records,scores_not_used_for_selection=True),indent=2)+'\n')
    print(records)

if __name__=='__main__':main()
