"""Check exported contracts and draw audit figures without selecting a winner."""
from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import dct
HERE = Path(_c2r_resolve_path(__file__)).resolve().parent
spec = importlib.util.spec_from_file_location('variants', HERE / 'export_variants.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def main():
    old = Path(_c2r_resolve_path('/home/huangyating/anomaly_detection/four_class_chain_20260922/contact/retrained_fcn_fixed_external'))
    md = pd.read_csv(old / 'metadata.csv')
    checks = []
    for v in m.VARIANTS:
        for model in ['retrained_fcn', 'retrained_final', 'official_final']:
            path = HERE / 'variants' / v / model
            newmd = pd.read_csv(path / 'metadata.csv')
            assert md.equals(newmd)
            with np.load(path / 'contact_features.npz', allow_pickle=False) as z:
                assert z['hidden_mean'].shape == (88, 128) and z['hidden_axes'].shape == (88, 3, 128)
                assert np.isfinite(z['hidden_mean']).all()
                assert np.allclose(z['probs10_mean'].sum(-1), 1, atol=1e-06)
                assert np.allclose(z['probs4_mean'].sum(-1), 1, atol=1e-06)
            prov = json.loads((path / 'provenance.json').read_text())
            for p, h in prov['weights'].items():
                assert m.api.sha(p) == h
            checks.append(dict(variant=v, model=model, rows=88, metadata_order_identical=True, weights_hashes_unchanged=True))
    for r in json.loads((HERE / 'input_manifest.json').read_text()):
        assert m.api.sha(r['file']) == r['sha256']
    oldf = np.load(old / 'contact_features.npz', allow_pickle=False)
    newf = np.load(HERE / 'variants/v0/retrained_fcn/contact_features.npz', allow_pickle=False)
    delta = float(np.max(abs(oldf['hidden_mean'] - newf['hidden_mean'])))
    pdelta = float(np.max(abs(oldf['probs4_mean'] - newf['probs4_mean'])))
    assert delta < 0.001 and pdelta < 1e-05
    assert np.array_equal(oldf['probs4_mean'].argmax(1), newf['probs4_mean'].argmax(1))
    physical = []
    for fs in [4000, 12000, 48000]:
        n = np.arange(fs)
        x = np.cos(2 * np.pi * 137 * (n + 0.5) / fs)
        c = dct(x, type=2, norm='backward')
        peak = int(np.argmax(abs(c)))
        assert peak == 274
        physical.append(dict(fs_hz=fs, duration_s=1, n_samples=fs, dct_peak_index=peak, peak_physical_hz=peak / 2))
    full_packet_hz = 274 * 4000 / (2 * 4096)
    opposite = dct(-np.cos(2 * np.pi * 137 * (np.arange(4000) + 0.5) / 4000), type=2, norm='backward')
    assert opposite[274] < 0
    m.api.savej(HERE / 'verification.json', dict(exports=checks, input_hashes_unchanged=True, v0_hidden_max_difference_from_previous=delta, v0_p4_max_difference_from_previous=pdelta, v0_window_predictions_identical=True, signed_dct_and_time_contract_verified=True, synthetic_fixed_duration=physical, synthetic_4096_full_packet_peak_physical_hz=full_packet_hz, note='Synthetic tests establish coordinate/implementation facts, not diagnosis or learned invariance.'))
    figdir = HERE / 'figures'
    figdir.mkdir(exist_ok=True)
    data = pd.read_csv(HERE / 'frozen_variant_metrics.csv')
    names = list(m.VARIANTS)
    short = ['Current AC', 'Ref <2 kHz', 'Raw DC', 'Raw DC + ref <2k', 'Both AC', 'Both AC + ref <2k']
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    colors = ['#236DAB', '#E89C32', '#7676BD']
    for j, model in enumerate(['retrained_fcn', 'retrained_final', 'official_final']):
        s = data[data.model == model].set_index('variant').loc[names]
        for k, metric in enumerate(['file_accuracy', 'file_macro_f1']):
            ax[k].bar(np.arange(len(names)) + (j - 1) * 0.24, s[metric], width=0.23, color=colors[j], label=model)
    for k, title in enumerate(['Original frozen head: file accuracy', 'Original frozen head: macro-F1']):
        ax[k].set(ylim=(0, 1), title=title)
        ax[k].set_xticks(range(len(names)), short, rotation=30, ha='right')
        ax[k].grid(axis='y', alpha=0.22)
        ax[k].set_axisbelow(True)
    ax[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Predeclared preprocessing audit | 8 known-class recordings | no model training', fontsize=13)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(figdir / f'01_frozen_preprocessing.{ext}', dpi=180)
    plt.close(fig)
    diag = pd.read_csv(HERE / 'query_dc_diagnostics.csv')
    diag = diag[diag.label >= 0]
    axisdata = pd.read_csv(HERE / 'axis_metrics.csv')
    axisdata = axisdata[axisdata.model == 'retrained_fcn']
    mat = axisdata.pivot(index='variant', columns='axis', values='accuracy').loc[names, ['x', 'y', 'z']].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11.8, 5.0), constrained_layout=True)
    for j, axis in enumerate('xyz'):
        col = f'no_demean_dc_energy_fraction_{axis}'
        ax[0].scatter(np.full(len(diag), j) + (np.arange(len(diag)) % 9 - 4) * 0.022, diag[col], alpha=0.5, color=colors[j], s=22)
        ax[0].plot([j - 0.18, j + 0.18], [diag[col].median()] * 2, color='black', lw=2)
    ax[0].set(xticks=[0, 1, 2], xticklabels=list('XYZ'), ylabel='DCT DC coefficient energy / total coefficient energy', ylim=(0, 1.02), title='Measured raw mean contribution')
    ax[0].grid(axis='y', alpha=0.22)
    im = ax[1].imshow(mat, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    for i in range(6):
        for j in range(3):
            ax[1].text(j, i, f'{100 * mat[i, j]:.1f}%', ha='center', va='center', color='white' if mat[i, j] > 0.5 else 'black')
    ax[1].set(xticks=[0, 1, 2], xticklabels=list('XYZ'), yticks=range(6), yticklabels=short, title='Axis sensitivity: file accuracy (exploratory)')
    fig.colorbar(im, ax=ax[1], shrink=0.8)
    fig.suptitle('Axis ablations are diagnostic; no axis is selected using these 8 test recordings', fontsize=12)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(figdir / f'02_dc_axis_diagnostics.{ext}', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(exports_verified=len(checks), original_inputs_and_weights_unchanged=True, v0_hidden_maxdiff=delta)))
if __name__ == '__main__':
    main()
