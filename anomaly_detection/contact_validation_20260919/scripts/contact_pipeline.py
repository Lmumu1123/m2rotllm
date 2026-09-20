"""Conservative parsing and 1-second BearLLM inputs for packetized XYZ data.

Raw files and existing BearLLM weights are read-only. Packet boundaries are never
joined, host receive timestamps are never treated as sample-clock timestamps.
"""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import welch
from scipy.stats import kurtosis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent/'接触式'
FS=4000
PACKET_SIZE=4096
N_FREQ=24000
HEADER=re.compile(r'\r?\n\[(\d{2}:\d{2}:\d{2}\.\d{3})\]收←◆')
NUMBER=r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
RECORD=re.compile(r'(!?)(\d+),('+NUMBER+r'),('+NUMBER+r'),('+NUMBER+r')')


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def dcn_1s(x,fs=FS,target=N_FREQ):
    """Mean removal + upstream DCT-II/pad/energy normalization, exactly 1 s.

    No time padding, resampling, Hann window, magnitude or log transform.
    A 4-kHz signal supplies only 4,000 DCT coefficients; zeros above are an
    unavailable-band convention, not measured absence of high-frequency energy.
    """
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=1 or len(x)!=fs:
        raise ValueError(f'Exactly {fs} contiguous samples required for one second; got {x.shape}')
    if not np.isfinite(x).all():raise ValueError('Nonfinite samples')
    x=x-x.mean()
    if np.mean(x*x)<=1e-16:raise ValueError('Degenerate AC energy; reject axis/window')
    c=dct(x,type=2,norm='backward')
    out=np.zeros(target,dtype=np.float64)
    n=min(len(c),target);out[:n]=c[:n]
    energy=float(out@out)
    if energy<=0:raise ValueError('No retained frequency energy')
    out*=.01*np.sqrt(target/energy)
    return out.astype(np.float32)


def prepare_pair(query,reference,query_run_id,reference_run_id,fs=FS):
    """Return [3 axes, query/reference, 24000], preserving axis identity.

    Caller must match condition, sensor orientation, mounting and load. Run IDs
    must identify acquisition sessions, not different windows of the same run.
    """
    if not query_run_id or not reference_run_id or query_run_id==reference_run_id:
        raise ValueError('An independently recorded healthy reference is required')
    query=np.asarray(query);reference=np.asarray(reference)
    if query.shape!=(fs,3) or reference.shape!=(fs,3):
        raise ValueError(f'Expected query/reference shape ({fs},3)')
    return np.stack([np.stack([dcn_1s(query[:,a],fs),dcn_1s(reference[:,a],fs)]) for a in range(3)])


def read_recording(path):
    payload=path.read_bytes()
    raw=payload.decode('ascii')
    txt=path.with_name('SaveWindows'+path.stem[4:]+'.TXT')
    annotated=txt.read_bytes().decode('gb18030')
    boundaries=[];removed=0
    for m in HEADER.finditer(annotated):
        pos=m.start()-removed
        boundaries.append(dict(payload_offset=pos,receive_time=m.group(1)))
        removed+=m.end()-m.start()
    cleaned=HEADER.sub('',annotated)
    if cleaned!=raw:raise ValueError(f'DAT/TXT disagree after header removal: {path.name}')
    # Receiver boundaries are not always corrupt. Given the observed dropped
    # counter character at one such boundary, conservatively exclude all records
    # spanning these markers. Do not guess missing characters or repair values.
    rows=[];excluded=[];segments=[];current=[];packet=0;offset=0;previous=None
    def flush():
        nonlocal current
        if current:segments.append(current);current=[]
    parts=raw.split(';')
    for token_index,token in enumerate(parts):
        start=offset;end=start+len(token);offset=end+1
        match=RECORD.fullmatch(token)
        transport=[r for r in boundaries if start<r['payload_offset']<end]
        reason=None
        if not match:reason='incomplete_or_invalid_record'
        elif token_index==len(parts)-1:reason='unterminated_last_record'
        elif transport:reason='record_spans_receive_marker'
        if reason:
            flush();previous=None
            excluded.append(dict(token_index=token_index,payload_start=start,payload_end=end,
                reason=reason,raw_token=token,receive_markers=transport))
            continue
        flag,counter,*xyz=match.groups();counter=int(counter);xyz=list(map(float,xyz))
        if flag or (previous is not None and counter<previous):
            flush();packet+=1;previous=None
        if previous is not None and counter!=previous+1:
            flush()
        if counter<1 or counter>PACKET_SIZE:
            flush();previous=None
            excluded.append(dict(token_index=token_index,reason='counter_out_of_range',raw_token=token))
            continue
        row=dict(counter=counter,x=xyz[0],y=xyz[1],z=xyz[2],packet_index=packet,
                 token_index=token_index,payload_start=start)
        current.append(row);rows.append(row);previous=counter
    flush()
    return dict(file=path.name,rpm=int(re.search(r'-(\d+)r',path.name).group(1)),
        dat_bytes=len(payload),txt_bytes=txt.stat().st_size,txt_payload_identical=True,
        dat_sha256=hashlib.sha256(payload).hexdigest(),txt_sha256=hashlib.sha256(txt.read_bytes()).hexdigest(),
        receive_markers=boundaries,exclusions=excluded,rows=rows,segments=segments)


def run():
    out=ROOT/'results';fig=ROOT/'figures';out.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    quality=[];tables=[];stats=[];segments_info=[];segments_npz={};strict=[];strictmeta=[]
    plotdata=[]
    for path in sorted(DATA.glob('*.DAT')):
        rec=read_recording(path);rpm=rec['rpm'];table=pd.DataFrame(rec.pop('rows'))
        table['file']=path.name;table['rpm']=rpm;tables.append(table)
        segments=rec.pop('segments');complete_packets=[]
        for packet,g in table.groupby('packet_index'):
            if len(g)==PACKET_SIZE and np.array_equal(g.counter,np.arange(1,PACKET_SIZE+1)):
                complete_packets.append(int(packet))
        rec['complete_4096_packets']=complete_packets
        rec['trusted_rows']=len(table)
        rec['max_contiguous_samples']=max(len(s) for s in segments)
        rec['strict_one_second_windows']=sum(len(s)//FS for s in segments)
        rec['unit']='unconfirmed; values retained in original units'
        rec['sampling_rate_hz']=FS
        quality.append(rec)
        psds=[];weights=[]
        for si,segment in enumerate(segments):
            st=pd.DataFrame(segment);x=st[['x','y','z']].to_numpy();key=f'{path.stem}_segment{si:02d}'
            segments_npz[key]=x.astype(np.float64)
            segments_info.append(dict(key=key,file=path.name,rpm=rpm,packet_index=int(st.packet_index.iloc[0]),
                first_counter=int(st.counter.iloc[0]),last_counter=int(st.counter.iloc[-1]),
                samples=len(st),nominal_duration_s=len(st)/FS,strict_1s_windows=len(st)//FS))
            if len(x)>=512:
                f,p=welch(x-x.mean(0),fs=FS,window='hann',nperseg=512,noverlap=256,nfft=4096,axis=0)
                psds.append(p);weights.append(len(x))
            starts=([48] if len(x)==PACKET_SIZE and st.counter.iloc[0]==1
                    else list(range(0,len(x)-FS+1,FS)))
            for start in starts:
                block=x[start:start+FS]
                strict.append(np.stack([dcn_1s(block[:,axis]) for axis in range(3)]))
                rawdir=out/'raw_windows';rawdir.mkdir(exist_ok=True)
                rawname=f'{path.stem}_segment{si:02d}_start{start}.npy'
                np.save(rawdir/rawname,block)
                strictmeta.append(dict(file=path.name,rpm=rpm,segment=si,start_in_segment=start,
                                       raw_window=str((rawdir/rawname).relative_to(ROOT))))
        values=table[['x','y','z']].to_numpy();ac=values-values.mean(0)
        for a,axis in enumerate('xyz'):
            stats.append(dict(file=path.name,rpm=rpm,axis=axis,samples=len(values),
                mean=float(values[:,a].mean()),std=float(ac[:,a].std()),
                minimum=float(values[:,a].min()),maximum=float(values[:,a].max()),
                rms_ac=float(np.sqrt(np.mean(ac[:,a]**2))),
                crest_factor=float(np.max(abs(ac[:,a]))/np.sqrt(np.mean(ac[:,a]**2))),
                excess_kurtosis=float(kurtosis(ac[:,a]))))
        plotdata.append((rpm,pd.DataFrame(segments[0])[['x','y','z']].to_numpy(),np.average(np.stack(psds),axis=0,weights=weights)))
    pd.concat(tables).to_csv(out/'parsed_samples.csv',index=False)
    pd.DataFrame(stats).to_csv(out/'axis_statistics.csv',index=False)
    pd.DataFrame(segments_info).to_csv(out/'segments.csv',index=False)
    np.savez_compressed(out/'parsed_segments.npz',**segments_npz)
    np.savez_compressed(out/'strict_1s_dcn.npz',features=np.stack(strict) if strict else np.empty((0,3,N_FREQ),np.float32),
        frequency_hz=np.arange(N_FREQ)*.5,available_band_mask=np.arange(N_FREQ)<FS)
    write_json(out/'strict_windows.json',strictmeta)
    write_json(out/'quality.json',quality)
    write_json(out/'preflight.json',dict(status='ready_for_pairing' if strict else 'insufficient_contiguous_data',
        recordings=len(quality),full_packets=sum(len(q['complete_4096_packets']) for q in quality),
        usable_1s_windows=len(strict),cross_packet_stitching=False,character_repair=False,
        healthy_reference_requirement='Independent same-condition recording, not another axis or self',
        inference_on_current_recordings='not_run: no strict one-second windows' if not strict else 'not_run: references must be assigned'))
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    shown=plotdata[:6]
    fig1,axs=plt.subplots(len(shown),2,figsize=(12,3*len(shown)),squeeze=False)
    for i,(rpm,first,psd) in enumerate(shown):
        # Display a contiguous segment, never the stitched table.
        n=min(len(first),800)
        for a,axis in enumerate('xyz'):
            axs[i,0].plot(np.arange(n)/FS,first[:n,a]-first[:n,a].mean(),label=axis,lw=.8)
            axs[i,1].plot(f,10*np.log10(psd[:,a]+1e-20),label=axis,lw=1)
        axs[i,0].set(title=f'{rpm} rpm: one contiguous fragment',xlabel='Local time (s)',ylabel='AC acceleration (unit unconfirmed)')
        axs[i,1].set(title=f'{rpm} rpm: segment-wise Welch PSD',xlabel='Frequency (Hz)',ylabel='PSD (relative dB)',xlim=(0,2000))
        axs[i,0].legend();axs[i,1].legend()
    fig1.suptitle('Current contact recordings: fragment diagnostics, not fault classification')
    fig1.tight_layout()
    for ext in ['png','svg','pdf']:fig1.savefig(fig/f'01_contact_quality.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig1)
    print(pd.DataFrame(quality)[['rpm','trusted_rows','max_contiguous_samples','strict_one_second_windows']].to_string(index=False))
    print(json.dumps(json.loads((out/'preflight.json').read_text()),ensure_ascii=False))


if __name__=='__main__':run()
