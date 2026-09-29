#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt,hilbert
from sklearn.metrics import accuracy_score,f1_score
from run_order_diagnostics import OUT,FS,GRID,axis_spectrum,to_order,sha

def main():
    rpm=np.array([1000,2000,3000]);t=np.arange(FS)/FS
    raw=np.array([np.tile(np.sin(2*np.pi*(r/60)*5*t)[:,None],(1,3)) for r in rpm])
    f,p,_=axis_spectrum(raw)
    x=to_order(f,p,rpm,[0,1,2])
    order_peaks=GRID[x.argmax(-1)]
    assert np.max(abs(order_peaks-5))<.2
    f2,p2,_=axis_spectrum(raw*np.array([.2,3.,10.])[None,None,:])
    amp=to_order(f2,p2,rpm,[0,1,2])
    assert np.max(abs(amp-x))<1e-3
    am=np.array([np.tile(((1+.5*np.cos(2*np.pi*(r/60)*5*t))*np.cos(2*np.pi*1000*t))[:,None],(1,3)) for r in rpm])
    sos=butter(4,[500,1800],fs=FS,btype='bandpass',output='sos')
    envelope=np.abs(hilbert(sosfiltfilt(sos,am,axis=1),axis=1))[:,400:-400]
    ef,ep,_=axis_spectrum(envelope);ex=to_order(ef,ep,rpm,[0,1,2])
    envelope_peaks=GRID[ex.argmax(-1)]
    assert np.max(abs(envelope_peaks-5))<.2
    packet=pd.read_csv(OUT/'packet_predictions.csv');session=pd.read_csv(OUT/'session_predictions.csv')
    metrics=pd.read_csv(OUT/'metrics.csv')
    for row in metrics.itertuples():
        d=packet if row.level=='packet' else session
        d=d[d.method==row.method]
        if row.holdout_rpm!='pooled_oof':d=d[d.holdout_rpm==int(row.holdout_rpm)]
        assert len(d)==row.n
        assert abs(accuracy_score(d.label,d.prediction)-row.accuracy)<1e-12
        assert abs(f1_score(d.label,d.prediction,labels=range(4),average='macro',zero_division=0)-row.macro_f1)<1e-12
    verify=json.loads((OUT/'verification.json').read_text())
    assert len(verify['model_checks'])==36
    for ck in verify['model_checks']:
        assert len(ck['train_sessions'])==8 and len(ck['test_sessions'])==4
        assert set(ck['train_sessions']).isdisjoint(ck['test_sessions'])
    protocol=json.loads((OUT/'protocol_before_results.json').read_text())
    assert all(sha(path)==checksum for path,checksum in protocol['input_hashes'].items())
    result=dict(ok=True,sine_true_order=5,raw_peak_orders=order_peaks.tolist(),
        envelope_peak_orders=envelope_peaks.tolist(),amplitude_invariance_max_error=float(np.max(abs(amp-x))),
        all_metrics_recomputed=True,all36_fits_holdout_sessions_disjoint=True,inputs_still_unchanged=True,
        verification_script_sha256=sha(__file__))
    hz=np.load(OUT/'hz_control_features.npz',allow_pickle=False)
    orders=np.load(OUT/'order_features.npz',allow_pickle=False)
    at_reference=orders['rpm']==2000
    assert np.array_equal(hz['xyz_concat_fixed_hz'][at_reference],orders['xyz_concat_order'][at_reference])
    assert np.array_equal(hz['xy_concat_fixed_hz'][at_reference],orders['xy_concat_order'][at_reference])
    hp=pd.read_csv(OUT/'hz_control_packet_predictions.csv');hs=pd.read_csv(OUT/'hz_control_session_predictions.csv')
    hm=pd.read_csv(OUT/'hz_control_metrics.csv')
    for row in hm.itertuples():
        d=hp if row.level=='packet' else hs
        d=d[d.method==row.method]
        if row.holdout_rpm!='pooled_oof':d=d[d.holdout_rpm==int(row.holdout_rpm)]
        assert len(d)==row.n
        assert abs(accuracy_score(d.label,d.prediction)-row.accuracy)<1e-12
        assert abs(f1_score(d.label,d.prediction,labels=range(4),average='macro',zero_division=0)-row.macro_f1)<1e-12
    result.update(hz_control_metrics_recomputed=True,hz_and_order_identical_at_reference_2000rpm=True)
    (OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
