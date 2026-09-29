"""Server-side standalone FCN embedding export; requires PyTorch and BearLLM repo.

Only accepts independently recorded, same-condition, training-only normal references.
Does not load the LLM or substitute one final adapter's classification head for another.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from preprocess import save_json, sha256


def make_pairs(root, assignments):
    summary = json.loads((root/'summary.json').read_text())
    contacts = {r['output']: r for r in summary if r['modality'] == 'contact' and r['label'] >= 0}
    groups = {}
    for name, info in assignments.items():
        if name not in contacts:
            raise ValueError(f'Unknown or non-four-class contact file: {name}')
        if not info.get('condition_id') or not info.get('independent_run_id') or info.get('split') not in ('train', 'val', 'test', 'reference'):
            raise ValueError(f'Unconfirmed condition/run/split: {name}')
        group = info['independent_run_id']
        if group in groups and groups[group] != info['split']:
            raise ValueError('Same independent run crosses splits')
        groups[group] = info['split']
    pairs, missing = [], []
    for name, q in contacts.items():
        if name not in assignments:
            missing.append(dict(file=name, windows=q['windows'], reason='unassigned_metadata'))
            continue
        info = assignments[name]
        if info['split'] == 'reference':
            continue
        refs = [(rname, j) for rname, r in contacts.items() if r['label'] == 0 and rname in assignments
                and assignments[rname]['split'] in ('train', 'reference')
                and assignments[rname]['condition_id'] == info['condition_id']
                and assignments[rname]['independent_run_id'] != info['independent_run_id']
                for j in range(r['windows'])]
        if not refs:
            missing.append(dict(file=name, windows=q['windows'], reason='no_independent_training_normal_reference'))
            continue
        for i in range(q['windows']):
            rname, j = refs[i % len(refs)]
            pairs.append(dict(query=name, query_row=i, reference=rname, reference_row=j,
                              label=q['label'], **info))
    return pairs, missing


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--assignments', type=Path, required=True)
    a.add_argument('--bearllm-repo', type=Path, required=True)
    a.add_argument('--weights', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--device', default='cpu')
    args = a.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output directory')
    assignments = json.loads(args.assignments.read_text())
    pairs, missing = make_pairs(args.data, assignments)
    args.output.mkdir(parents=True)
    save_json(args.output/'pairing_coverage.json', dict(pairs=pairs, missing=missing))
    if not pairs:
        raise ValueError('No valid pairs. Fill verified run/condition/split metadata; see coverage report.')
    import torch
    source = args.bearllm_repo/'models/FCN.py'
    spec = importlib.util.spec_from_file_location('standalone_bearllm_fcn', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.FaultClassificationNetwork()
    model.encoder.load_state_dict(torch.load(args.weights/'feature_encoder.pth', map_location='cpu', weights_only=True))
    model.classifier.load_state_dict(torch.load(args.weights/'classifier.pth', map_location='cpu', weights_only=True))
    model.to(args.device).eval()
    for param in model.parameters(): param.requires_grad_(False)
    cache = {}
    def get(name):
        if name not in cache:
            with np.load(args.data/name, allow_pickle=False) as z:
                cache[name] = z['dcn']
        return cache[name]
    maps, hidden, logits = [], [], []
    before = {n: b.detach().cpu().clone() for n, b in model.named_buffers()}
    with torch.inference_mode():
        for row in pairs:
            q, r = get(row['query'])[row['query_row']], get(row['reference'])[row['reference_row']]
            x = torch.as_tensor(np.stack([q, r], axis=1), device=args.device)
            feature = model.encoder(x)
            h = torch.relu(model.classifier.linear1(feature.flatten(1)))
            logit = model.classifier.linear2(h)
            if feature.shape != (3, 128, 47) or h.shape != (3, 128) or logit.shape != (3, 10):
                raise ValueError('Server model interface differs from audited standalone FCN')
            if not all(torch.isfinite(v).all() for v in [feature, h, logit]):
                raise ValueError('Nonfinite model output')
            maps.append(feature.cpu().numpy()); hidden.append(h.cpu().numpy()); logits.append(logit.cpu().numpy())
    for name, buffer in model.named_buffers():
        if not torch.equal(before[name], buffer.cpu()): raise RuntimeError('Model buffers changed during export')
    h = np.stack(hidden)
    np.savez_compressed(args.output/'contact_embeddings.npz', feature_map=np.stack(maps), hidden=h,
                       hidden_l2=h/np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-12),
                       logits10=np.stack(logits), labels=np.asarray([r['label'] for r in pairs]))
    save_json(args.output/'provenance.json', dict(torch=torch.__version__, source_sha256=sha256(source),
        encoder_sha256=sha256(args.weights/'feature_encoder.pth'),
        classifier_sha256=sha256(args.weights/'classifier.pth'), assignments_sha256=sha256(args.assignments),
        mode='frozen eval', buffers_unchanged=True,
        limitation='embeddings only; no local classification or transfer accuracy inferred'))
    print(f'Saved {len(pairs)} query windows, 3 axis views each; missing files: {len(missing)}')


if __name__ == '__main__': main()
