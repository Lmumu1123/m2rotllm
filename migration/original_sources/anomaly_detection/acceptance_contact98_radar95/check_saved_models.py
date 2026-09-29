"""Read-only scoring of saved models against the user's new accuracy targets.

No training or checkpoint selection. Window scores are secondary and correlated;
recording-level held-out scores remain the primary local evaluation.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

BASE = Path('/home/huangyating/anomaly_detection/retention_alignment_20260922')
OLD = BASE.parent/'four_class_chain_20260922'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE/'radar'))
from train_fixed_head import Encoder
GROUPS = [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]
NAMES = ['normal', 'inner', 'outer', 'ball']


def group(p10):
    return np.stack([p10[:, ids].sum(1) for ids in GROUPS], 1)


def main():
    assert 'envs/m2vllm' in sys.executable
    torch.set_num_threads(2)
    cm = pd.read_csv(BASE/'preprocessing/variants/v0/retrained_fcn/metadata.csv')
    h = np.load(BASE/'preprocessing/variants/v0/retrained_fcn/contact_features.npz')['hidden_mean']
    rm = pd.read_csv(OLD/'radar/metadata.csv')
    rx = np.load(OLD/'radar/features.npz')['frame_shape']
    summaries, records, windows, provenance = [], [], [], []
    for fold, target in [('115200_to_460800', 460800), ('460800_to_115200', 115200), ('all_known', None)]:
        version = 'retained_head_v1' if target else 'retained_head_conservative'
        scope = 'heldout_recordings' if target else 'training_recordings'
        cp = BASE/'results'/version/'models/v0'/fold/'retained_head.npz'
        cz = np.load(cp)
        pc = group(softmax(h@cz['weight10'].T + cz['bias10'], axis=1))
        radar_dir = 'stage_b_v0' if target else 'stage_b_conservative_v0'
        rp = BASE/'radar'/radar_dir/'models/four_class_provisional_roi'/fold/'ce_feat_1_kd.pt'
        ck = torch.load(rp, map_location='cpu', weights_only=False)
        assert np.array_equal(cz['weight10'], ck['head_weight'].numpy())
        assert np.array_equal(cz['bias10'], ck['head_bias'].numpy())
        model = Encoder(ck['input_dim'], ck['contact_mean'], ck['contact_scale'])
        model.load_state_dict(ck['encoder']); model.eval()
        with torch.inference_mode():
            rh = model(torch.tensor(((rx-ck['radar_mean'])/ck['radar_scale']).astype(np.float32)))
            pr = group((rh@ck['head_weight'].T+ck['head_bias']).softmax(1).numpy())
        for modality, md, prob, baudkey, threshold in [('contact', cm, pc, 'baud', .98), ('radar', rm, pr, 'baud_candidate', .95)]:
            mask = md.label.ge(0).to_numpy()
            if target:
                mask &= md[baudkey].eq(target).to_numpy()
                assert not set(md.loc[mask, 'bag_id']) & set(cz['local_training_bags'])
            y = md.loc[mask, 'label'].to_numpy(int)
            pred = prob[mask].argmax(1)
            sums = dict(modality=modality, version=version, fold=fold, scope=scope, threshold=threshold,
                window_correct=int((pred==y).sum()), window_n=len(y), window_accuracy=float((pred==y).mean()))
            file_correct = 0
            for bag, ids0 in md.loc[mask].groupby('bag_id').groups.items():
                ids = np.array(ids0)
                truth = int(md.loc[ids[0], 'label']); predfile = int(prob[ids].mean(0).argmax())
                valid = True if modality=='contact' else not bool(md.loc[ids, 'geometry_roi_mismatch'].any())
                file_correct += predfile == truth
                records.append(dict(modality=modality, version=version, fold=fold, scope=scope, bag_id=bag,
                    truth=NAMES[truth], prediction=NAMES[predfile], correct=predfile==truth,
                    geometry_valid=valid, n_windows=len(ids), window_correct=int((prob[ids].argmax(1)==truth).sum())))
                for idx in ids:
                    windows.append(dict(modality=modality, version=version, fold=fold, scope=scope, bag_id=bag,
                        row=int(idx), truth=truth, prediction=int(prob[idx].argmax()), geometry_valid=valid))
            n = md.loc[mask, 'bag_id'].nunique()
            summaries.append(dict(**sums, file_correct=int(file_correct), file_n=int(n), file_accuracy=file_correct/n,
                                  numerical_file_threshold_pass=file_correct/n>threshold))
        for p in [cp, rp]:
            provenance.append(dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    s = pd.DataFrame(summaries); s.to_csv(OUT/'metrics.csv', index=False)
    f = pd.DataFrame(records); f.to_csv(OUT/'file_predictions.csv', index=False)
    pd.DataFrame(windows).to_csv(OUT/'window_predictions.csv', index=False)
    class_rows=[]
    for (modality, scope, truth), a in f.groupby(['modality', 'scope', 'truth']):
        class_rows.append(dict(modality=modality,scope=scope,truth=truth,correct=int(a.correct.sum()),n=len(a),
                               geometry_valid_n=int(a.geometry_valid.sum())))
    pd.DataFrame(class_rows).to_csv(OUT/'per_class_recordings.csv',index=False)
    info = dict(status='complete', experiment='read-only saved-model evaluation; no training or selection',
        targets=dict(contact_accuracy_strictly_greater_than=.98,radar_accuracy_strictly_greater_than=.95,
                     original_accuracy_maximum_decline_percentage_points=.5),
        primary_unit='held-out recording, not individual correlated windows',
        original_models=provenance, conservative_all_known_scores_are_training=True,
        four_class_radar_acceptance=False, reason='Outer-class wrong ROI invalidates full physical four-class acceptance',
        external_bigNormal_previous_result='Both contact and radar classify 0/2 as normal; separate unseen-motor test fails',
        small_sample_limitation='Eight recordings from one rig are not evidence of stable >98%/>95% performance on future recordings')
    (OUT/'verification.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(s.to_string(index=False))


if __name__=='__main__':
    main()
