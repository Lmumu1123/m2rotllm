"""Check metadata and synthetic operator equality; no model training/inference."""
from pathlib import Path
import hashlib
import json
import csv
import numpy as np

OUT = Path(__file__).parent
BASE = Path('/home/huangyating/anomaly_detection')
ENV = BASE/'environment_validation_20260919/results'
V2 = BASE/'encoder_validation_20260923_v2/geometry_corrected'
sources = [BASE/'environment_validation_20260919/scripts/radar_environment.py',
           V2/'provenance/preprocess.py', V2/'provenance/extract_features.py']
schema = json.loads((ENV/'feature_schema.json').read_text())
edges = np.asarray(schema['band_edges_hz'])
env = np.load(ENV/'features.npz', allow_pickle=False)
v2 = np.load(V2/'radar/features.npz', allow_pickle=False)
with (ENV/'windows.csv').open() as f:
    rows = list(csv.DictReader(f))

def feat(iq, nfft, normalized):
    sp = np.fft.fft(iq*np.hanning(192)[None,:,None], n=nfft, axis=1)
    f = np.fft.fftfreq(nfft, 1/2000.)
    k = np.flatnonzero((f >= 5)&(f <= 800))
    power = np.median(np.mean(abs(sp[:, k])**2+abs(sp[:, (-k)%nfft])**2, axis=0),axis=-1)
    if normalized:
        power = power/(2000*np.sum(np.hanning(192)**2))
        # v2 export persists float32 log power before re-expansion.
        power = 10.**np.log10(power+1e-14).astype(np.float32).astype(np.float64)
    b = np.array([power[(f[k] >= lo)&(f[k] < hi)].mean() for lo,hi in zip(edges[:-1],edges[1:])])
    log = np.log10(np.maximum(b,1e-14)) if normalized else np.log10(b+1e-14)
    return log-log.mean()

rng=np.random.default_rng(17)
t=np.arange(192)/2000.
examples=[]
for name, freq in [('near_5_6_hz_band',5.5), ('shaft_tone_33hz',33.)]:
    iq=np.broadcast_to(np.exp(2j*np.pi*freq*t)[None,:,None],(20,192,12)).copy()
    iq-=iq.mean(axis=1,keepdims=True)
    a,b=feat(iq,2048,False),feat(iq,4000,True)
    examples.append(dict(fixture=name, max_abs_feature_difference=float(abs(a-b).max()),
                         mean_abs_feature_difference=float(abs(a-b).mean())))
iq=(rng.normal(size=(20,192,12))+1j*rng.normal(size=(20,192,12))).astype(np.complex64)
iq-=iq.mean(axis=1,keepdims=True)
a,b=feat(iq,2048,False),feat(iq,4000,True)
examples.append(dict(fixture='synthetic_complex_broadband',max_abs_feature_difference=float(abs(a-b).max()),
                     mean_abs_feature_difference=float(abs(a-b).mean())))
report=dict(date='2026-09-24', raw_environment_path_exists=(BASE/'data9.18').exists(),
            exact_band_edges_equal=bool(np.array_equal(edges,v2['band_edges_hz'])),
            all_environment_windows_have_full_scheduled_chirp_coverage=all(float(r['valid_fraction'])==.96 for r in rows),
            environment_feature_keys_and_shapes={k:list(env[k].shape) for k in env.files},
            environment_frame_binned_power_min=float(env['frame_power_binned'].min()),
            old_frame_fft_size=2048,new_frame_fft_size=4000,
            synthetic_operator_checks=examples,
            compatible_for_unqualified_frozen_model_test=False,
            inference_run=False,
            status='Stopped: frequency-grid band averaging operators differ; no environment ROI IQ retained in feature archive.',
            source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'compatibility.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
