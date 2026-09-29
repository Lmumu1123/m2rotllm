"""Predeclared frozen-model DCN audit; never selects a variant on local outcomes."""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import argparse, importlib.util, json, sys, time
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct
BASE = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_chain_20260922/contact/evaluate_contact.py'))
DATA = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results'))
OUT = Path(_c2r_resolve_path(__file__)).resolve().parent
spec = importlib.util.spec_from_file_location('contact_eval_audit', BASE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
VARIANTS = {'v0': dict(query_demean=True, reference_demean=False, reference_bandlimit=False), 'reference_bandlimited2k': dict(query_demean=True, reference_demean=False, reference_bandlimit=True), 'query_no_demean': dict(query_demean=False, reference_demean=False, reference_bandlimit=False), 'query_no_demean_ref2k': dict(query_demean=False, reference_demean=False, reference_bandlimit=True), 'query_ref_demean': dict(query_demean=True, reference_demean=True, reference_bandlimit=False), 'query_ref_demean_ref2k': dict(query_demean=True, reference_demean=True, reference_bandlimit=True)}

def normalize(x):
    norm = np.linalg.norm(x.astype(np.float64), axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError('DCN energy vanished')
    return (x * (0.01 * np.sqrt(24000) / norm)).astype(np.float32)

def transformed_queries(raw, demean):
    x = raw.astype(np.float64)
    if demean:
        x = x - x.mean(axis=1, keepdims=True)
    c = dct(x, type=2, norm='backward', axis=1).transpose(0, 2, 1)
    z = np.zeros((len(raw), 3, 24000), np.float64)
    z[:, :, :4000] = c
    return normalize(z)

def transformed_references(refs, demean, bandlimit):
    z = refs.copy()
    if demean:
        z[:, 0] = 0
    if bandlimit:
        z[:, 4000:] = 0
    return normalize(z) if demean or bandlimit else z

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--batch', type=int, default=48)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'frozen_variant_metrics.csv').exists():
        raise FileExistsError('Existing audit is immutable; choose new output')
    protocol = dict(created_before_inference=True, variants=VARIANTS, query_sampling_rate_hz=4000, query_samples=4000, query_duration_s=1, dct_type=2, dct_normalization='backward', coefficient_frequency_hz='k / 2; 0 <= k < 4000 has measured support; padding does not add information', dcn='0.01 * sqrt(24000) * c / ||c||2; signed coefficients retained', reference_seed=42, reference_rule='one public MBHM source-training healthy row per dataset; unchanged choice across all variants', reference_condition='not matched to local rig; protocol deviation explicitly retained', forbidden='No test-normal reference; no local label or external bigNormal selection; no weight updates; no fabricated 4096-point data', smoothing='SFN is Spectral Folding Network. No smoothing variant is silently inserted into pretrained BearLLM; incompatible representation needs source replay training.', primary_model='retrained_fcn', audit_models=['retrained_fcn', 'retrained_final', 'official_final'], source_sha256=api.sha(__file__), feature_schema='same as prior contact_features.npz; metadata row order fixed')
    api.savej(a.output / 'protocol.json', protocol)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    current, md, manifest = api.read_inputs(DATA)
    refs, refmeta = api.external_references()
    raw = np.stack([np.load(DATA / r.file, allow_pickle=False)['raw_xyz'][r.window_row] for r in md.itertuples()])
    q0 = transformed_queries(raw, True)
    q1 = transformed_queries(raw, False)
    assert np.allclose(current, q0, atol=2e-05)
    api.savej(a.output / 'input_manifest.json', manifest)
    api.savej(a.output / 'reference_metadata.json', refmeta)
    np.savez_compressed(a.output / 'queries_and_references.npz', query_demean=q0, query_no_demean=q1, references_original=refs)
    diag = md.copy()
    for j, axis in enumerate('xyz'):
        diag[f'raw_mean_{axis}'] = raw.mean(1)[:, j]
        diag[f'raw_ac_rms_{axis}'] = raw.std(1)[:, j]
        diag[f'no_demean_dc_energy_fraction_{axis}'] = q1[:, j, 0] ** 2 / np.sum(q1[:, j] ** 2, axis=1)
    diag.to_csv(a.output / 'query_dc_diagnostics.csv', index=False)
    refdiag = pd.DataFrame(refmeta)
    refdiag['dc_energy_fraction'] = refs[:, 0] ** 2 / np.sum(refs ** 2, axis=1)
    refdiag['energy_above_2khz_fraction'] = np.sum(refs[:, 4000:] ** 2, axis=1) / np.sum(refs ** 2, axis=1)
    refdiag.to_csv(a.output / 'reference_dc_diagnostics.csv', index=False)
    allresults = []
    axes = []
    for kind in protocol['audit_models']:
        model, prov = api.load_model(kind)
        for name, cfg in VARIANTS.items():
            started = time.monotonic()
            dest = a.output / 'variants' / name / kind
            q = q0 if cfg['query_demean'] else q1
            r = transformed_references(refs, cfg['reference_demean'], cfg['reference_bandlimit'])
            result = api.evaluate(model, q, r, md, dest, a.device, a.batch)
            api.savej(dest / 'provenance.json', {**prov, **cfg, 'source_script_sha256': api.sha(__file__), 'elapsed_s': time.monotonic() - started, 'model_weights_and_bn_frozen': True, 'no_label_used_for_prediction': True})
            allresults.append(dict(model=kind, variant=name, **{f'file_{k}': v for k, v in result['files'].items() if k != 'confusion_matrix'}, **{f'window_{k}': v for k, v in result['windows'].items() if k != 'confusion_matrix'}))
            with np.load(dest / 'contact_features.npz', allow_pickle=False) as z:
                p10 = z['probs10_axes']
            for j, axis in enumerate('xyz'):
                p4 = np.stack([p10[:, j, idx].sum(1) for idx in api.MAP], axis=1)
                df = md.copy()
                for k in range(4):
                    df[f'p4_{k}'] = p4[:, k]
                bag = df[df.label >= 0].groupby(['file', 'label'], as_index=False)[[f'p4_{k}' for k in range(4)]].mean()
                bag['pred4'] = bag[[f'p4_{k}' for k in range(4)]].to_numpy().argmax(1)
                axes.append(dict(model=kind, variant=name, axis=axis, **{k: v for k, v in api.metrics(bag).items() if k != 'confusion_matrix'}))
            print(kind, name, allresults[-1], flush=True)
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(allresults).to_csv(a.output / 'frozen_variant_metrics.csv', index=False)
    pd.DataFrame(axes).to_csv(a.output / 'axis_metrics.csv', index=False)
    api.savej(a.output / 'runtime.json', dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__, device=a.device, source_sha256=api.sha(__file__), reference_loader_sha256=api.sha(BASE), variants=len(VARIANTS), models=len(protocol['audit_models']), windows=len(md), raw_mean_preserved_in_export=True, max_demean_recompute_difference=float(np.max(abs(current - q0)))))
if __name__ == '__main__':
    main()
