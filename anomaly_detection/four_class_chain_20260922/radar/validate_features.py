"""Artifact integrity and a physical-frequency sanity check, without training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from extract_features import ROOT, SOURCE, digest, phase_frame_psd, write_json

z=np.load(ROOT/'features.npz',allow_pickle=False)
m=pd.read_csv(ROOT/'metadata.csv')
assert len(m)==614 and int((m.label>=0).sum())==412
assert np.array_equal(z['labels'],m.label.to_numpy())
assert np.array_equal(z['bag_id'],m.bag_id.to_numpy(str))
assert not m.duplicated(['bag_id','recording_row']).any()
for k in ['frame_shape','phase_shape','coherent_shape']:
    assert z[k].shape==(614,128)
    assert np.max(abs(z[k].mean(1)))<1e-6
assert z['shape_phase'].shape==(614,256)
for rec,part in m.groupby('source_npz'):
    with np.load(SOURCE/rec,allow_pickle=False) as src:
        assert len(part)==len(src['labels'])
        assert np.array_equal(part.recording_row.to_numpy(),np.arange(len(part)))
for item in json.loads((ROOT/'input_manifest.json').read_text()):
    assert digest(Path(item['path']))==item['sha256']
    assert digest(Path(item['path']).with_suffix('.json'))==item['sidecar_sha256']

# A known 100 Hz phase-rate tone is resolved at its true frequency; arbitrary
# frame signs do not affect this per-frame POWER feature (no coherence assumed).
f=np.arange(5.,800.01,.5);t=np.arange(191)/2000
base=np.sin(2*np.pi*100*t)
x=np.broadcast_to(base[None,None,:,None],(1,20,191,12)).copy()
x[:,1::2]*=-1
mask=np.ones((1,20,191),bool)
p=phase_frame_psd(x,mask,f)
peak=float(f[p[0].argmax()])
assert abs(peak-100)<=1
checks=dict(status='passed',rows=614,four_class_rows=412,unique_bags=12,
    input_npz_and_sidecar_sha256_unchanged=True,finite_shape_features=True,
    feature_row_to_bag_mapping_valid=True,within_frame_known_tone_hz=100,
    recovered_peak_hz=peak,artificial_frame_signs='alternating; power remains valid',
    limitations=['not a hardware parser or packet-loss validation',
                 'does not certify correct physical ROI or synchronized clocks'])
write_json(ROOT/'verification.json',checks)
print(json.dumps(checks,ensure_ascii=False,indent=2))
