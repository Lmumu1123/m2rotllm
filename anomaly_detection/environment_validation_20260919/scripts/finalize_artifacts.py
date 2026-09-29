"""Verify cached results and exported models; write reproducibility metadata."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
from radar_environment import ROOT, snapshot, self_check
from predict_features import predict


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    out=ROOT/'results'
    assert '/envs/m2vllm/' in sys.executable, sys.executable
    self_check()
    expected=json.loads((out/'input_manifest.json').read_text())
    assert snapshot()==expected, 'Inputs changed since extraction'
    hashes=json.loads((out/'input_sha256.json').read_text())
    assert {r['file']:r['size'] for r in expected}=={r['file']:r['bytes'] for r in hashes}
    assert len({r['sha256'] for r in hashes})==len(hashes)
    meta=pd.read_csv(out/'windows.csv',dtype={'label':str})
    primary=pd.read_csv(out/'file_predictions.csv',dtype={'label':str,'prediction':str})
    metrics=pd.read_csv(out/'cross_environment_metrics.csv')
    assert len(expected)==42 and len(meta)==652 and len(metrics)==30
    arrays=np.load(out/'features.npz',allow_pickle=False)
    assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
    assert all(len(arrays[k])==len(meta) for k in arrays.files if k!='frequencies')
    assert set(meta.file)=={r['file'] for r in expected}
    checks=[]
    for item in json.loads((ROOT/'models/manifest.json').read_text()):
        result=predict(ROOT/'models'/item['file'],out/'features.npz',out/'windows.csv',
                       item['target'],item['task']=='P0_three_speed').set_index('file')
        direction=item['source']+'->'+item['target']
        prev=primary[(primary.task==item['task'])&(primary.direction==direction)&
                     (primary.method=='iq_multi_shape')].set_index('file').loc[result.index]
        assert result.prediction.equals(prev.prediction), item['file']
        probability_cols=[c for c in result if c.startswith('p_')]
        max_error=float(np.max(np.abs(result[probability_cols].to_numpy()-prev[probability_cols].to_numpy())))
        assert max_error<1e-10, (item['file'],max_error)
        assert set(result.index).isdisjoint(item['training_files'])
        assert all(name.startswith(item['source']+'-') for name in item['training_files'])
        checks.append(dict(model=item['file'],files=len(result),max_probability_error=max_error))
    edges=np.unique(np.r_[5,np.arange(6,102,2),np.arange(105,405,5),np.arange(420,801,20)])
    assert len(edges)-1==arrays['iq_multi_shape'].shape[1]==128
    descriptions={
        'raw_amplitude':'12 log10 mean range-cell powers; no mechanical absolute unit',
        'phase_single':'128 log10 binned powers of within-frame phase increments',
        'iq_multi_abs':'128 log10 binned masked-LS median IQ powers after per-cell amplitude scaling',
        'iq_multi_shape':'iq_multi_abs minus its per-window mean across 128 bands',
        'iq_frame_shape':'128 log10 complete-frame-average power bands, minus per-window mean',
        'power_binned':'128 relative linear powers before iq_multi_abs log transform',
        'frame_power_binned':'128 relative linear complete-frame powers; scaling differs from power_binned',
        'spectra':'1591 masked-LS frequencies, median of 12 range/RX cell powers',
        'frequencies':'5 to 800 Hz inclusive, 0.5 Hz grid; frequency axis only'}
    save(out/'feature_schema.json',dict(
        rows='Same order as windows.csv; one nominal 2-second window per row except frequencies',
        band_edges_hz=edges.tolist(),bands='Left-closed, right-open; last edge 800 Hz',
        dimensions={k:list(arrays[k].shape) for k in arrays.files},description=descriptions,
        units='Relative signal features, not calibrated displacement or acceleration; not teacher embeddings',
        off_calibration='Computed during evaluate_environment.py, not stored as a primary array',
        validity='Fixed original radar configuration; 96% scheduled chirp coverage'))
    source_hashes={}
    for path in sorted((ROOT/'scripts').glob('*.py')):
        ast.parse(path.read_text())
        source_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    now=datetime.now(timezone.utc).isoformat()
    runtime=json.loads((out/'runtime_environment.json').read_text())
    runtime['final_verification_time_utc']=now
    runtime['final_source_hashes']=source_hashes
    runtime['final_python']=sys.executable
    runtime['execution']='CPU, OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2'
    runtime['final_packages']={k:v.__version__ for k,v in [('numpy',np),('scipy',scipy),
                              ('pandas',pd),('sklearn',sklearn),('matplotlib',matplotlib)]}
    save(out/'runtime_environment.json',runtime)
    proposal=ROOT.parents[1]/'research_proposals/contact_radar_alignment_20260919'
    (ROOT/'radar_original_assumed.cfg').write_text((proposal/'雷达原始配置.cfg').read_text())
    figure_count=len(list((ROOT/'figures').glob('*.png')))
    assert figure_count==6
    verification=dict(timestamp_utc=now,python=sys.executable,status='passed',
        original_files=len(expected),windows=len(meta),primary_metric_rows=len(metrics),
        supplementary_metric_rows=len(pd.read_csv(out/'supplementary_unseen_distance_metrics.csv')),
        input_size_mtime_manifest_unchanged=True,recorded_sha256_count=len(hashes),
        identical_file_hashes=False,all_feature_values_finite=True,all_scripts_parse=True,
        parser_synthetic_and_masked_ls_self_check='passed',reloaded_models=checks,
        figure_groups=figure_count,
        note='File hashes were computed after extraction; finalization compares sizes/mtime and the saved hash inventory. Neither proves UDP completeness or independent recording.')
    save(out/'final_verification.json',verification)
    save(out/'watch_status.json',dict(timestamp_utc=now,status='complete',
        execution_mode='foreground',conda_environment='m2vllm',active_background_monitor=False,
        files=len(expected),GB=sum(r['size'] for r in expected)/1e9,windows=len(meta),
        completion_signal_received=True,previous_report_stale=False,
        report=str(ROOT/'实验结论.md')))
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
