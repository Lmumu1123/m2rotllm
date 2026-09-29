"""Apply exported state/speed models to cached features, without fitting.

This utility does not infer faults and does not parse new radar configurations.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def predict(model_path, feature_path, metadata_path, environment=None, exclude_off=False):
    with np.load(model_path, allow_pickle=False) as model:
        name = str(model['feature_name'].item())
        with np.load(feature_path, allow_pickle=False) as arrays:
            x = arrays[name]
        meta = pd.read_csv(metadata_path, dtype={'label': str})
        if len(x) != len(meta):
            raise ValueError('Feature and metadata rows differ')
        keep = np.ones(len(meta), dtype=bool)
        if environment is not None:
            keep &= (meta.environment == environment).to_numpy()
        if exclude_off:
            keep &= (meta.label != 'off').to_numpy()
        meta = meta.loc[keep].copy()
        if not len(meta):
            raise ValueError('No selected rows')
        x = x[keep]
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite input features')
        logits = ((x-model['mean'])/model['scale']) @ model['coef'].T + model['intercept']
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        classes = model['classes'].astype(str)
    temp = pd.DataFrame(prob, columns=['p_'+c for c in classes])
    temp['file'] = meta.file.to_numpy()
    p = temp.groupby('file', sort=True).mean()
    cols = [c for c in ['environment', 'distance_cm', 'label'] if c in meta.columns]
    info = meta.drop_duplicates('file').set_index('file').loc[p.index, cols]
    out = info.join(p)
    out['prediction'] = classes[np.argmax(p.to_numpy(), axis=1)]
    return out.reset_index()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--features', type=Path, default=ROOT/'results/features.npz')
    parser.add_argument('--metadata', type=Path, default=ROOT/'results/windows.csv')
    parser.add_argument('--environment', choices=['room', 'narrow'])
    parser.add_argument('--exclude-off', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = predict(args.model, args.features, args.metadata, args.environment, args.exclude_off)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'Saved {len(out)} file predictions to {args.output}')
