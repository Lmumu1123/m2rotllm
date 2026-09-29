"""Re-run contact DCN through the actual frozen FCN and shared adapted heads.

No training. Radar predictions are obtained with the very same module's
classify_embedding method, and compared to the previously scored checkpoint.
"""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import torch
BASE = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/retention_alignment_20260922'))
OLD = BASE.parent / 'four_class_chain_20260922'
OUT = Path(_c2r_resolve_path(__file__)).resolve().parent
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'radar'))
from shared_contact_model import SharedFourClassModel
from train_fixed_head import Encoder

def main():
    assert sys.version_info >= (3, 10)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda:0'
    cm = pd.read_csv(BASE / 'preprocessing/variants/v0/retrained_fcn/metadata.csv')
    rm = pd.read_csv(OLD / 'radar/metadata.csv')
    qz = np.load(BASE / 'preprocessing/queries_and_references.npz')
    q = qz['query_demean']
    refs = qz['references_original']
    flat = q.reshape(-1, 24000)
    cached = np.load(BASE / 'preprocessing/variants/v0/retrained_fcn/contact_features.npz')['hidden_mean']
    first = BASE / 'results/retained_head_v1/models/v0/115200_to_460800/retained_head.npz'
    model = SharedFourClassModel(first).to(device).eval()
    hsum = np.zeros((len(flat), 128), np.float64)
    with torch.inference_mode():
        for ref in refs:
            for start in range(0, len(flat), 48):
                qq = flat[start:start + 48]
                x = np.stack([qq, np.broadcast_to(ref, qq.shape)], 1)
                hh = model.contact_hidden(torch.tensor(x, device=device)).cpu().numpy()
                hsum[start:start + len(qq)] += hh
    hidden = (hsum / len(refs)).reshape(len(q), 3, 128).astype(np.float32).mean(1)
    hidden_error = float(np.max(abs(hidden - cached)))
    rows = []
    win = []
    rz = np.load(OLD / 'radar/features.npz')
    for fold, target in [('115200_to_460800', 460800), ('460800_to_115200', 115200), ('all_known', None)]:
        version = 'retained_head_v1' if target else 'retained_head_conservative'
        headpath = BASE / 'results' / version / 'models/v0' / fold / 'retained_head.npz'
        model = SharedFourClassModel(headpath).to(device).eval()
        rd = 'stage_b_v0' if target else 'stage_b_conservative_v0'
        checkpoint = BASE / 'radar' / rd / 'models/four_class_provisional_roi' / fold / 'ce_feat_1_kd.pt'
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert torch.equal(model.weight10.cpu(), ck['head_weight'])
        assert torch.equal(model.bias10.cpu(), ck['head_bias'])
        assert hashlib.sha256(headpath.read_bytes()).hexdigest() == ck['shared_head_sha256']
        enc = Encoder(ck['input_dim'], ck['contact_mean'], ck['contact_scale']).eval()
        enc.load_state_dict(ck['encoder'])
        x = ((rz[ck['feature']] - ck['radar_mean']) / ck['radar_scale']).astype(np.float32)
        with torch.inference_mode():
            cp = model.classify_embedding(torch.tensor(hidden, device=device))['probabilities4'].cpu().numpy()
            cached_p = model.classify_embedding(torch.tensor(cached, device=device))['probabilities4'].cpu().numpy()
            rh = enc(torch.tensor(x)).to(device)
            rp = model.classify_embedding(rh)['probabilities4'].cpu().numpy()
            original_p10 = (rh.cpu() @ ck['head_weight'].T + ck['head_bias']).softmax(1)
            original_p4 = torch.stack([original_p10[:, g].sum(1) for g in [[0], [1, 2, 3], [7, 8, 9], [4, 5, 6]]], 1).numpy()
        assert np.array_equal(cp.argmax(1), cached_p.argmax(1))
        assert np.array_equal(rp.argmax(1), original_p4.argmax(1))
        for modality, md, p, baud in [('contact', cm, cp, 'baud'), ('radar', rm, rp, 'baud_candidate')]:
            use = md.label.ge(0).to_numpy()
            if target:
                use &= md[baud].eq(target).to_numpy()
            ids = np.flatnonzero(use)
            y = md.label.to_numpy()[ids]
            est = p[ids].argmax(1)
            rows.append(dict(fold=fold, version=version, modality=modality, scope='heldout' if target else 'training', n=len(ids), correct=int((est == y).sum()), accuracy=float((est == y).mean()), identical_head_weights=True, head_probability_recompute_max_error=float(np.max(abs(p - (cached_p if modality == 'contact' else original_p4))))))
            for i in ids:
                win.append(dict(fold=fold, modality=modality, row=int(i), bag_id=md.iloc[i].bag_id, truth=int(md.iloc[i].label), prediction=int(p[i].argmax()), **{f'p{k}': float(p[i, k]) for k in range(4)}))
    pd.DataFrame(rows).to_csv(OUT / 'unified_module_metrics.csv', index=False)
    pd.DataFrame(win).to_csv(OUT / 'unified_module_window_predictions.csv', index=False)
    report = dict(status='passed', environment=sys.executable, device=device, contact_DCN_windows=len(q), axes=3, healthy_references=len(refs), FCN_query_reference_pairs=len(q) * 3 * len(refs), hidden_max_error_from_archived_cache=hidden_error, contact_predictions_identical_to_cache=True, radar_predictions_identical_to_checkpoint=True, same_head_weight_and_bias_for_both_modalities=True, same_shared_module_used_for_all_predictions=True, conclusion='Reported window scores are the adapted BearLLM diagnostic-head argmax, not Qwen text generation accuracy', interface='Contact DCN -> original FCN+linear1 -> adapted linear2; radar features -> learned radar encoder -> identical adapted linear2', no_training=True)
    (OUT / 'unified_module_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__ == '__main__':
    main()
