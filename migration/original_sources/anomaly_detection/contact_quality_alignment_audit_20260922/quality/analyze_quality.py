#!/usr/bin/env python3
"""Read-only, label-independent signal diagnostics plus exploratory error joins."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import kurtosis
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
INPUT=Path('/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results')
META=Path('/home/huangyating/anomaly_detection/retention_alignment_20260922/preprocessing/variants/v0/retrained_fcn/metadata.csv')
PRED=Path('/home/huangyating/anomaly_detection/acceptance_contact98_radar95/window_predictions.csv')
FS=4000
BANDS=[('0_5',0,5),('5_100',5,100),('100_800',100,800),('800_1800',800,1800),('1800_2000',1800,2000.01)]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savej(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def longest(mask):
    a=np.r_[False,np.asarray(mask,dtype=bool),False].astype(int)
    d=np.diff(a);start=np.flatnonzero(d==1);end=np.flatnonzero(d==-1)
    return int(np.max(end-start)) if len(start) else 0
def safe(v):return float(v) if np.isfinite(v) else None

def main():
    if (OUT/'verification.json').exists():raise FileExistsError('Preserve completed audit; use a separate directory for rerun')
    protocol=dict(created_before_signal_error_comparison=True,fs_hz=4000,window_samples=4000,
        observed_boundary_proxy='Largest absolute value over all available raw_xyz, selected without outcomes; not confirmed hardware full scale',
        review_flags=dict(endpoint_fraction_at_least=.001,endpoint_run_samples_at_least=4,
            constant_run_samples_at_least=20),
        flags_are_not_rejection_rules=True,never_drop_windows=True,never_tune_thresholds_to_accuracy=True,
        objective_quality='finite/shape/complete-packet provenance/constant runs/exact duplicates; RMS, kurtosis, crest, DC and spectrum are descriptive',
        intrinsic_fault_warning='Large crest/kurtosis/amplitude may be fault signal, not bad data',
        comparison='v0 retained_head_v1 heldout only; within-recording wrong/correct medians, not pooled causal inference or model confidence',
        unknowns=['hardware ADC range and units','actual timestamp and clock drift','true inter-packet gaps','dropped unseen packet starts','raw DAT currently unavailable'],
        no_training=True,no_label_quality_inferred_from_teacher_confidence=True,
        periodogram='Hann, nfft=4000, 1 Hz grid; quarter-window nfft=1000, 4 Hz grid; no synthetic missing samples')
    savej(OUT/'protocol.json',protocol)
    md=pd.read_csv(META);md.insert(0,'global_row',np.arange(len(md)))
    caches={};manifest=[];files=[];issues=[]
    for fn in md.file.unique():
        path=INPUT/fn
        with np.load(path,allow_pickle=False) as z:caches[fn]=z['raw_xyz'].copy()
        m=json.loads(path.with_suffix('.json').read_text());q=m['quality'];w=m['windows']
        manifest.append(dict(path=str(path),sha256=sha(path),metadata_sha256=sha(path.with_suffix('.json'))))
        reason_counts=Counter(r['reason'] for r in q['issues'])
        known=[int(r['packet_ordinal_observed']) for r in w]
        ord_gaps=int(sum(max(b-a-1,0) for a,b in zip(known[:-1],known[1:])))
        rec=dict(file=fn,bag_id=md[md.file==fn].bag_id.iloc[0],state=m['metadata']['state'],
            label=m['metadata']['label'],baud=m['metadata']['baud_candidate'],
            complete_packets=q['complete_packets'],observed_packet_starts=q['packet_starts_observed'],
            complete_per_observed_start=q['complete_packets']/q['packet_starts_observed'],
            retained_windows=len(w),dcn_rejected_complete=len(m['rejected_complete_packets']),
            issue_count=q['issue_count'],observed_ordinal_gaps_among_retained=ord_gaps,
            timestamped_retained_windows=sum(r['absolute_window_start'] is not None for r in w),
            true_missing_samples_unknown=True,original_DAT_path_available=False,
            **{f'issues_{k}':int(reason_counts[k]) for k in ['invalid_record','unterminated_tail','counter_discontinuity','unexpected_packet_marker','packet_restart_before_4096','counter_or_nonfinite']})
        files.append(rec)
        for item in q['issues']:
            issues.append(dict(file=fn,byte=item['byte'],reason=item['reason'],expected=item.get('expected'),
                observed=item.get('observed'),retained_rows=item.get('retained_rows')))
    # This proxy is chosen without reading model outcomes. True ADC rails are unknown.
    boundary=float(max(np.max(np.abs(x)) for x in caches.values()))
    rows=[];axes=[];spectra=[];time_vectors=[];hashes=[]
    for r in md.itertuples():
        x=caches[r.file][r.window_row].astype(np.float64)
        meta=json.loads((INPUT/r.file).with_suffix('.json').read_text())['windows'][r.window_row]
        finite=np.isfinite(x).all();ac=x-x.mean(0,keepdims=True);rms=np.sqrt(np.mean(ac**2,axis=0))
        f,p=periodogram(ac,fs=FS,window='hann',nfft=4000,axis=0,detrend=False,scaling='density')
        total=p.sum(0);band={name:p[(f>=lo)&(f<hi)].sum(0)/total for name,lo,hi in BANDS}
        low=(f>=10)&(f<=100);peaks=f[low][np.argmax(p[low],axis=0)]
        pp=p[(f>=5)&(f<=2000)].sum(1);pp=pp/pp.sum();spectra.append(pp)
        quarters=ac.reshape(4,1000,3);qrms=np.sqrt(np.mean(quarters**2,axis=1))
        fq,pq=periodogram(quarters,fs=FS,window='hann',nfft=1000,axis=1,detrend='constant',scaling='density')
        lq=(fq>=10)&(fq<=100);qpeaks=fq[lq][np.argmax(pq[:,lq,:],axis=1)]
        same=np.all(x[1:]==x[:-1],axis=1)
        row=dict(global_row=r.global_row,file=r.file,window_row=r.window_row,bag_id=r.bag_id,
            state=r.state,label=r.label,baud=r.baud,packet_ordinal_observed=r.packet_ordinal_observed,
            sample_rate_hz=meta['sample_rate_hz'],duration_s=meta['duration_s'],shape_ok=x.shape==(4000,3),
            finite=bool(finite),complete_packet_provenance=True,known_internal_counter_break=False,
            timestamp_available=meta['absolute_window_start'] is not None,
            ac_vector_rms=float(np.linalg.norm(rms)),raw_abs_max=float(abs(x).max()),
            dc_vector_norm=float(np.linalg.norm(x.mean(0))),
            identical_xyz_step_fraction=float(same.mean()),max_constant_xyz_run=longest(same)+1,
            observed_abs_boundary=boundary,endpoint_fraction_any_axis=float(np.mean(np.any(abs(x)==boundary,axis=1))),
            endpoint_run_max=max(longest(abs(x[:,a])==boundary) for a in range(3)),
            dominant_peak_hz=float(f[low][np.argmax(p[low].sum(1))]),
            subwindow_rms_cv_mean=float(np.mean(qrms.std(0)/qrms.mean(0))),
            quarter_peak_span_hz=float(np.max(np.ptp(qpeaks,axis=0))),
            spectral_entropy=float(-np.sum(pp*np.log(np.maximum(pp,1e-30)))/np.log(len(pp))),
            median_axis_kurtosis=float(np.median(kurtosis(ac,axis=0,fisher=False,bias=False))),
            max_axis_crest=float(np.max(abs(ac).max(0)/rms)))
        for name,lo,hi in BANDS:row[f'energy_{name}_fraction']=float(p[(f>=lo)&(f<hi)].sum()/p.sum())
        for a,name in enumerate('xyz'):
            endpoint=abs(x[:,a])==boundary;same_a=x[1:,a]==x[:-1,a]
            ps=p[low,a];
            axis=dict(global_row=r.global_row,axis=name,file=r.file,window_row=r.window_row,bag_id=r.bag_id,
                state=r.state,label=r.label,baud=r.baud,dc=float(x[:,a].mean()),ac_rms=float(rms[a]),
                peak_to_peak=float(np.ptp(x[:,a])),minimum=float(x[:,a].min()),maximum=float(x[:,a].max()),
                pearson_kurtosis=float(kurtosis(ac[:,a],fisher=False,bias=False)),crest_factor=float(abs(ac[:,a]).max()/rms[a]),
                zero_step_fraction=float(same_a.mean()),longest_constant_run=longest(same_a)+1,
                observed_boundary_fraction=float(endpoint.mean()),observed_boundary_run=longest(endpoint),
                min_value_repeat_fraction=float(np.mean(x[:,a]==x[:,a].min())),max_value_repeat_fraction=float(np.mean(x[:,a]==x[:,a].max())),
                dominant_10_100_peak_hz=float(peaks[a]),quarter_peak_span_hz=float(np.ptp(qpeaks[:,a])),
                quarter_rms_cv=float(qrms[:,a].std()/qrms[:,a].mean()),
                **{f'energy_{k}_fraction':float(v[a]) for k,v in band.items()})
            axes.append(axis)
        # Only engineering symptoms trigger review. No biological/fault labels enter.
        row['hard_invalid']=not finite or not row['shape_ok'] or bool(np.any(rms==0))
        row['proxy_review_required']=row['endpoint_fraction_any_axis']>=.001 or row['endpoint_run_max']>=4 or row['max_constant_xyz_run']>=20
        row['quality_status']='hard_invalid' if row['hard_invalid'] else 'numerically_valid_with_review_proxy' if row['proxy_review_required'] else 'numerically_valid_timing_unverified'
        hashes.append(hashlib.sha256(x.astype(np.float32).tobytes()).hexdigest());rows.append(row);time_vectors.append(x)
    df=pd.DataFrame(rows);counts=Counter(hashes);df['exact_duplicate_window']=np.array([counts[h]>1 for h in hashes])
    power=np.stack(spectra)
    for bag,ids in df.groupby('bag_id').groups.items():
        ix=np.array(list(ids))
        for i in ix:
            others=ix[ix!=i];ref=power[others].mean(0);ref/=ref.sum()
            df.loc[i,'within_recording_js_distance']=jensenshannon(power[i],ref,base=2)
    # Outcomes read only now, after all signal-only metrics and flags were computed.
    pred=pd.read_csv(PRED)
    pred=pred[(pred.modality=='contact')&(pred.version=='retained_head_v1')&(pred.scope=='heldout_recordings')]
    assert len(pred)==58 and pred.row.nunique()==58
    pred=pred[['row','fold','bag_id','truth','prediction']].rename(columns={'row':'global_row','bag_id':'prediction_bag_id'})
    merged=df.merge(pred,on='global_row',how='left',validate='1:1')
    known=merged[merged.truth.notna()].copy();assert np.array_equal(known.bag_id,known.prediction_bag_id)
    assert np.array_equal(known.label.to_numpy(),known.truth.to_numpy())
    known['correct']=known.truth==known.prediction
    assert known.correct.sum()==46
    merged['heldout_correct']=np.where(merged.truth.notna(),merged.truth==merged.prediction,np.nan)
    df=merged
    df.to_csv(OUT/'window_quality.csv',index=False);pd.DataFrame(axes).to_csv(OUT/'axis_quality.csv',index=False)
    files=pd.DataFrame(files);files.to_csv(OUT/'recording_packet_quality.csv',index=False)
    pd.DataFrame(issues).to_csv(OUT/'parser_issue_locations.csv',index=False)
    known.to_csv(OUT/'heldout_window_quality_and_error.csv',index=False)
    metrics=['ac_vector_rms','dc_vector_norm','raw_abs_max','max_axis_crest','median_axis_kurtosis',
        'endpoint_fraction_any_axis','endpoint_run_max','subwindow_rms_cv_mean','dominant_peak_hz',
        'quarter_peak_span_hz','energy_100_800_fraction','energy_800_1800_fraction',
        'energy_1800_2000_fraction','spectral_entropy','within_recording_js_distance']
    comp=[]
    for bag,g in known.groupby('bag_id'):
        if g.correct.nunique()!=2:continue
        good=g[g.correct];bad=g[~g.correct]
        for k in metrics:
            gm=float(good[k].median());bm=float(bad[k].median())
            comp.append(dict(bag_id=bag,state=g.state.iloc[0],baud=int(g.baud.iloc[0]),metric=k,
                correct_n=len(good),wrong_n=len(bad),correct_median=gm,wrong_median=bm,
                within_recording_wrong_minus_correct=bm-gm))
    comparison=pd.DataFrame(comp);comparison.to_csv(OUT/'within_recording_error_comparison.csv',index=False)
    pooled=[]
    for k in metrics:
        r=comparison[comparison.metric==k]
        pooled.append(dict(metric=k,pooled_correct_median=float(known[known.correct][k].median()),
            pooled_wrong_median=float(known[~known.correct][k].median()),mixed_recordings=len(r),
            median_within_recording_difference=float(r.within_recording_wrong_minus_correct.median()),
            positive_difference_recordings=int((r.within_recording_wrong_minus_correct>0).sum()),
            negative_difference_recordings=int((r.within_recording_wrong_minus_correct<0).sum())))
    pd.DataFrame(pooled).to_csv(OUT/'exploratory_error_summary.csv',index=False)
    outfiles=known.groupby(['bag_id','state','baud'],as_index=False).agg(windows=('global_row','size'),correct=('correct','sum'),
        proxy_review_windows=('proxy_review_required','sum'),endpoint_fraction_median=('endpoint_fraction_any_axis','median'),
        rms_median=('ac_vector_rms','median'),dominant_peak_median=('dominant_peak_hz','median'))
    outfiles['wrong']=outfiles.windows-outfiles.correct
    outfiles=outfiles.merge(files[['bag_id','complete_per_observed_start','issue_count']],on='bag_id')
    outfiles.to_csv(OUT/'heldout_recording_summary.csv',index=False)
    # Figures retain every observation; failed predictions marked, not removed.
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    display=[r.state+' '+str(r.baud) for r in files.itertuples()]
    ax[0].bar(range(len(files)),files.complete_per_observed_start,color=['#2878A8' if b==115200 else '#D98134' for b in files.baud])
    ax[0].set(ylim=(0,1),ylabel='Complete packets / observed packet starts',title='Packet transport/export completeness')
    ax[0].set_xticks(range(len(files)),display,rotation=65,ha='right',fontsize=8)
    for i,r in files.iterrows():ax[0].text(i,r.complete_per_observed_start+.015,f'{r.complete_packets}/{r.observed_packet_starts}',ha='center',fontsize=8)
    ax[1].bar(range(len(outfiles)),outfiles.correct,color='#348B79',label='Correct windows')
    ax[1].bar(range(len(outfiles)),outfiles.wrong,bottom=outfiles.correct,color='#CA5F59',label='Wrong windows')
    ax[1].set_xticks(range(len(outfiles)),[r.state+' '+str(r.baud) for r in outfiles.itertuples()],rotation=65,ha='right',fontsize=8)
    ax[1].set(ylabel='Retained 1 s windows',title='Contact heldout errors: all retained windows');ax[1].legend(frameon=False)
    fig.suptitle('File-level packet failures do not establish corruption inside retained complete windows',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(figdir/f'01_packet_quality_and_errors.{ext}',dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
    bags=list(outfiles.bag_id)
    for a,(metric,title) in zip(ax.flat,[('ac_vector_rms','AC vector RMS'),('endpoint_fraction_any_axis','Observed boundary occupancy'),('within_recording_js_distance','Spectrum distance to other windows in same recording'),('subwindow_rms_cv_mean','Quarter-window AC RMS coefficient of variation')]):
        for i,bag in enumerate(bags):
            g=known[known.bag_id==bag]
            for good,color,marker in [(True,'#2878A8','o'),(False,'#D95E56','x')]:
                sel=g[g.correct==good];jitter=(np.arange(len(sel))-(len(sel)-1)/2)*.035
                a.scatter(i+jitter,sel[metric],color=color,marker=marker,s=36,alpha=.8,label='correct' if i==0 and good else 'wrong' if i==0 else None)
        a.set(title=title);a.set_xticks(range(len(bags)),[r.state+' '+str(r.baud) for r in outfiles.itertuples()],rotation=65,ha='right',fontsize=7)
        a.grid(axis='y',alpha=.2)
    ax[0,0].legend(frameon=False);fig.suptitle('Exploratory comparisons stratified by recording; no quality-based data removal',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(figdir/f'02_quality_by_recording.{ext}',dpi=170)
    plt.close(fig)
    # Representative trace chosen solely by maximum endpoint occupancy, not prediction.
    main=df[df.label>=0];pick=int(main.sort_values(['endpoint_fraction_any_axis','global_row'],ascending=[False,True]).iloc[0].global_row)
    xx=time_vectors[pick];axis=int(np.argmax(np.mean(abs(xx)==boundary,axis=0)))
    endpoint=np.flatnonzero(abs(xx[:,axis])==boundary);center=int(endpoint[0]) if len(endpoint) else 2000
    lo=max(0,center-120);hi=min(4000,center+280)
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    ax[0].plot(np.arange(lo,hi)/FS,xx[lo:hi,axis],color='#2878A8',lw=1)
    for val in [-boundary,boundary]:ax[0].axhline(val,color='#C34A4A',ls='--',alpha=.65)
    ax[0].set(xlabel='Time within retained window (s)',ylabel='Original sensor output; units unconfirmed',title=f'Boundary proxy example: global row {pick}, axis {"XYZ"[axis]}')
    yy=pd.DataFrame(axes);knownaxes=yy[yy.label>=0]
    for i,axisname in enumerate('xyz'):
        q=knownaxes[knownaxes.axis==axisname]
        ax[1].scatter(np.full(len(q),i)+(np.arange(len(q))%9-4)*.025,q.observed_boundary_fraction,s=20,alpha=.5,color=['#2878A8','#348B79','#D98134'][i])
    ax[1].set(xticks=range(3),xticklabels=list('XYZ'),ylabel='Exact observed-boundary sample fraction',title='Proxy distribution over all 58 known windows')
    fig.suptitle('Repeated endpoints suggest review; hardware range is unknown, so clipping is not confirmed',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(figdir/f'03_boundary_proxy.{ext}',dpi=170)
    plt.close(fig)
    savej(OUT/'input_manifest.json',manifest)
    assert all(sha(r['path'])==r['sha256'] for r in manifest)
    savej(OUT/'verification.json',dict(status='complete',windows=len(df),axis_rows=len(axes),recordings=len(files),
        heldout_windows=len(known),correct_windows=int(known.correct.sum()),wrong_windows=int((~known.correct).sum()),
        finite_windows=int(df.finite.sum()),shape_ok_windows=int(df.shape_ok.sum()),hard_invalid_windows=int(df.hard_invalid.sum()),
        exact_duplicate_windows=int(df.exact_duplicate_window.sum()),review_proxy_windows=int(df.proxy_review_required.sum()),
        known_review_proxy_windows=int(known.proxy_review_required.sum()),observed_absolute_boundary=boundary,
        representative_trace_global_row=pick,representative_selection_uses_prediction=False,
        all_input_hashes_unchanged=True,metadata_sha256=sha(META),predictions_sha256=sha(PRED),
        script_sha256=sha(__file__),python=sys.executable,quality_flags_use_prediction=False,
        no_windows_removed=True,no_training_or_threshold_optimization=True))
    print((OUT/'verification.json').read_text())

if __name__=='__main__':main()
