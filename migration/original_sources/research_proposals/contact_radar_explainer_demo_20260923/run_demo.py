"""Replay first window of every recording through the actual shared model.

Input is exported, normalized range-cell IQ and exported contact raw_xyz, not ADC bin.
No training, no scaler fitting, no label input into either forward function.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/huangyating/anomaly_detection')
RET=BASE/'retention_alignment_20260922'
OLD=BASE/'four_class_chain_20260922'
DATA=BASE/'four_class_preprocessing_20260921/results'
sys.path[:0]=[str(RET/'scripts'),str(RET/'radar')]
from shared_contact_model import SharedFourClassModel
from train_fixed_head import Encoder
NAMES=['正常','内圈故障','外圈故障','滚动体故障']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jwrite(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def small(a):return np.round(np.asarray(a,dtype=float),6).tolist()

def radar_features(iq,mask,freq,edges):
    z=iq[...,0].astype(np.float64)+1j*iq[...,1].astype(np.float64)
    taper=np.hanning(192);valid=mask.all(1)
    if not valid.any():raise ValueError('No complete frame')
    spec=np.fft.fft(z[valid]*taper[None,:,None],n=4000,axis=1)
    k=np.rint(freq*2).astype(int)
    power=np.median(np.mean(abs(spec[:,k])**2+abs(spec[:,-k])**2,axis=0),axis=1)
    power/=2000*np.sum(taper*taper)
    # Preserve the saved preprocessing contract, including intermediate float32.
    logpower=np.log10(power.astype(np.float32)+1e-14)
    p=10.**logpower.astype(np.float64)
    pooled=np.array([p[(freq>=a)&(freq<b)].mean() for a,b in zip(edges[:-1],edges[1:])])
    x=np.log10(np.maximum(pooled,1e-14));x-=x.mean()
    return x.astype(np.float32)

def contact_dcn(raw):
    x=raw.astype(np.float64);x-=x.mean(0)
    c=dct(x,type=2,axis=0,norm=None).T
    padded=np.pad(c,((0,0),(0,24000-c.shape[1])))
    return (0.01*np.sqrt(24000)*padded/np.linalg.norm(padded,axis=1,keepdims=True)).astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'replay')
    ap.add_argument('--bag',help='Optional exact recording bag_id; default: all 12 first windows')
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise RuntimeError('Use a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    cm=pd.read_csv(RET/'preprocessing/variants/v0/retrained_fcn/metadata.csv')
    rm=pd.read_csv(OLD/'radar/metadata.csv')
    with np.load(RET/'preprocessing/queries_and_references.npz') as q:
        refs=q['references_original'].copy();cached_q=q['query_demean'].copy()
    cached_h=np.load(RET/'preprocessing/variants/v0/retrained_fcn/contact_features.npz')['hidden_mean']
    rz=np.load(OLD/'radar/features.npz');edges=rz['band_edges_hz']
    bags=sorted(rm.bag_id.unique())
    if args.bag:
        if args.bag not in bags:raise ValueError('Unknown --bag recording')
        bags=[args.bag]
    models={};items=[];verify=[];inputs={}
    started=time.time()
    for bag in bags:
        rix=int(np.flatnonzero(rm.bag_id.eq(bag))[0]);cix=int(np.flatnonzero(cm.bag_id.eq(bag))[0])
        r=rm.iloc[rix];c=cm.iloc[cix]
        external=str(r.class_role)!='four_class'
        fold='all_known' if external else ('460800_to_115200' if int(r.baud_candidate)==115200 else '115200_to_460800')
        head_root='retained_head_conservative' if external else 'retained_head_v1'
        radar_root='stage_b_conservative_v0' if external else 'stage_b_v0'
        hp=RET/'results'/head_root/'models/v0'/fold/'retained_head.npz'
        rp=RET/'radar'/radar_root/'models/four_class_provisional_roi'/fold/'ce_feat_1_kd.pt'
        if fold not in models:
            ck=torch.load(rp,map_location='cpu',weights_only=False)
            shared=SharedFourClassModel(hp).eval();enc=Encoder(ck['input_dim'],ck['contact_mean'],ck['contact_scale']).eval()
            enc.load_state_dict(ck['encoder'])
            assert torch.equal(shared.weight10,ck['head_weight']) and torch.equal(shared.bias10,ck['head_bias'])
            assert sha(hp)==ck['shared_head_sha256']
            models[fold]=(shared,enc,ck)
        shared,enc,ck=models[fold]
        with np.load(DATA/r.source_npz) as z:
            iq=z['iq_frames'][int(r.recording_row)].copy()
            mask=z['valid_chirp_mask'][int(r.recording_row)].copy()
            freq=z['frequency_hz'].copy()
        with np.load(DATA/c.file) as z:raw=z['raw_xyz'][int(c.window_row)].copy()
        for path in [DATA/r.source_npz,DATA/c.file,hp,rp]:inputs[str(path)]=sha(path)
        x=radar_features(iq,mask,freq,edges)
        dx=float(np.max(abs(x-rz['frame_shape'][rix])))
        assert dx<2e-5,dx
        q=contact_dcn(raw);dq=float(np.max(abs(q-cached_q[cix])))
        assert dq<2e-6,dq
        std=((x-ck['radar_mean'])/ck['radar_scale']).astype(np.float32)
        # Only signal arrays and fixed healthy references enter contact inference.
        hviews=[]
        with torch.inference_mode():
            for ref in refs:
                pair=np.stack([q,np.broadcast_to(ref,q.shape)],axis=1)
                hviews.append(shared.contact_hidden(torch.tensor(pair)).numpy())
            hc=np.mean(np.stack(hviews).astype(np.float64),axis=0).astype(np.float32).mean(0)
            hr=enc(torch.tensor(std[None]))
            cp=shared.classify_embedding(torch.tensor(hc[None]))
            rpred=shared.classify_embedding(hr)
            cached_cp=shared.classify_embedding(torch.tensor(cached_h[cix:cix+1]))
            cache_std=((rz['frame_shape'][rix]-ck['radar_mean'])/ck['radar_scale']).astype(np.float32)
            cached_rp=shared.classify_embedding(enc(torch.tensor(cache_std[None])))
        pc=cp['probabilities4'][0].numpy();pr=rpred['probabilities4'][0].numpy()
        assert pc.argmax()==cached_cp['probabilities4'][0].argmax().item()
        assert pr.argmax()==cached_rp['probabilities4'][0].argmax().item()
        hiderr=float(np.max(abs(hc-cached_h[cix])))
        valid=not bool(r.geometry_roi_mismatch)
        truth=0 if str(r.state)=='bigNormal' else int(r.label)
        truth_text=NAMES[truth] if truth>=0 else '底座/保持座未知故障（不在四类内）'
        item=dict(bag_id=bag,state=str(r.state),fold=fold,model_role='外部案例：保守 all_known 模型' if external else '录制留出：对应折模型（开发数据）',
            radar_index=rix,contact_index=cix,radar_window=int(r.recording_row),contact_window=int(c.window_row),
            truth=truth,truth_name=truth_text,geometry_valid=valid,range_center_m=float(r.range_center_m),
            radar_probability=pr.tolist(),contact_probability=pc.tolist(),radar_prediction=NAMES[int(pr.argmax())],contact_prediction=NAMES[int(pc.argmax())],
            contact_correct=bool(pc.argmax()==truth) if truth>=0 else None,radar_correct=bool(pr.argmax()==truth) if truth>=0 else None,
            iq_shape=list(iq.shape),contact_shape=list(raw.shape),radar_feature_shape=list(x.shape),dcn_shape=list(q.shape),
            radar_feature=small(x),radar_standardized=small(std),radar_scaler_mean=small(ck['radar_mean']),radar_scaler_scale=small(ck['radar_scale']),
            frequency_centers=small((edges[:-1]+edges[1:])/2),
            radar_hidden=small(hr[0].numpy()),contact_hidden=small(hc),
            iq_first_frame=small(iq[0,:,0,:]),iq_time_ms=small(np.arange(192)/2),
            xyz_display=small(raw[::8]),xyz_time_s=small(np.arange(0,4000,8)/4000),
            model_path=str(rp),shared_head_path=str(hp),radar_source=str(DATA/r.source_npz),contact_source=str(DATA/c.file))
        items.append(item)
        np.savez_compressed(args.output/(bag+'.npz'),radar_feature=x,radar_standardized=std,radar_hidden=hr.numpy(),contact_dcn=q,contact_hidden=hc,
                            radar_probability4=pr,contact_probability4=pc)
        verify.append(dict(bag_id=bag,radar_feature_max_error=dx,contact_dcn_max_error=dq,contact_hidden_max_error=hiderr,
                            radar_probability_max_error=float(np.max(abs(pr-cached_rp['probabilities4'][0].numpy()))),
                            contact_probability_max_error=float(np.max(abs(pc-cached_cp['probabilities4'][0].numpy()))),same_head=True))
        print(bag, 'contact:',item['contact_prediction'],'radar:',item['radar_prediction'],'ROI:',valid,flush=True)
    payload=dict(title='真实保存模型：双模态输入输出演示',classes=NAMES,items=items,
       boundary='每份录制按顺序取第一个有效窗口；同一编号不是同一时刻。HTML 回放本次 Python 实际推理，不在浏览器重新运行网络，也没有运行 Qwen。',
       input_boundary='雷达起点是已经归一化、去帧均值的距离门 IQ，不是原始 ADC bin。标签在推理完成后用于对照，不是模型输入。')
    jwrite(args.output/'demo_data.json',payload)
    template=(ROOT/'demo_template.html').read_text()
    (args.output/'模型输入输出演示.html').write_text(template.replace('__DEMO_JSON__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<'+chr(92)+'/')))
    report=dict(status='passed',no_training=True,no_scaler_refit=True,labels_used_in_forward=False,selection='first valid exported window per recording; independent of predictions',
       predictions_match_archived_input_path=True,scope='12 first-window examples, not a new dataset accuracy estimate',
       elapsed_seconds=time.time()-started,checks=verify,input_sha256=inputs,
       unchanged_inputs=all(sha(p)==v for p,v in inputs.items()),python=sys.executable)
    assert report['unchanged_inputs']
    jwrite(args.output/'verification.json',report)
    pd.DataFrame([{k:r[k] for k in ['bag_id','truth_name','contact_prediction','radar_prediction','geometry_valid','model_role']} for r in items]).to_csv(args.output/'sample_predictions.csv',index=False)
    print('Wrote',args.output/'模型输入输出演示.html',flush=True)

if __name__=='__main__':main()
