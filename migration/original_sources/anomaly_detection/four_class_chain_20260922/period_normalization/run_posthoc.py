"""Fixed posthoc dominant-period normalization diagnostic, never tune on cases."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
CHAIN=ROOT.parent
SOURCE=CHAIN.parent/'four_class_preprocessing_20260921/results'
Q=np.linspace(.5,20.,128)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savej(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def markdown(df):
    out=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        out.append('| '+' | '.join(f'{v:.5g}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(out)

def f0_and_normalize(power,frequency):
    use=(frequency>=10)&(frequency<=100)
    recording_power=np.median(power,axis=0)
    f0=float(frequency[use][np.argmax(recording_power[use])])
    eval_hz=Q*f0
    valid=(eval_hz>=frequency.min())&(eval_hz<=frequency.max())
    out=np.full((len(power),len(Q)),np.nan,np.float64)
    log=np.log10(np.maximum(power,1e-20))
    for i,row in enumerate(log):
        out[i,valid]=np.interp(eval_hz[valid],frequency,row)
        out[i,valid]-=out[i,valid].mean()
    assert np.isfinite(out[:,valid]).all()
    return f0,out,valid,eval_hz

def extract():
    cm=pd.read_csv(CHAIN/'contact/retrained_fcn_fixed_external/metadata.csv')
    rm=pd.read_csv(CHAIN/'radar/metadata.csv')
    with np.load(CHAIN/'radar/features.npz') as z:
        edges=z['band_edges_hz'];radar_raw=z['frame_shape'];assert np.array_equal(z['bag_id'],rm.bag_id.to_numpy())
    with np.load(CHAIN/'contact/retrained_fcn_fixed_external/contact_features.npz') as z:
        hidden=z['hidden_mean'];assert np.array_equal(z['file_names'],cm.file.to_numpy())
    contact_raw=np.zeros((len(cm),3*(len(edges)-1)))
    contact_period=np.full((len(cm),128),np.nan)
    radar_period=np.full((len(rm),128),np.nan)
    frequencies=[];manifest=[]
    for modality,meta,col in [('contact',cm,'file'),('radar',rm,'source_npz')]:
        for name,inds in meta.groupby(col,sort=False).indices.items():
            p=SOURCE/name
            with np.load(p,allow_pickle=False) as z:
                local=meta.iloc[inds].window_row.to_numpy() if modality=='contact' else meta.iloc[inds].recording_row.to_numpy()
                if modality=='contact':
                    f=z['common_frequency_hz'];full=z['common_power'][local].astype(np.float64)
                    power=np.median(full,axis=1)
                    binned=np.stack([full[:,:,(f>=a)&(f<b)].mean(-1) for a,b in zip(edges[:-1],edges[1:])],axis=-1)
                    raw=np.log10(binned+1e-20);raw-=raw.mean(-1,keepdims=True)
                    contact_raw[inds]=raw.reshape(len(inds),-1)
                else:
                    f=z['frequency_hz'];power=10.**z['frame_log_power'][local].astype(np.float64)
                f0,period,valid,eval_hz=f0_and_normalize(power,f)
                if modality=='contact':contact_period[inds]=period
                else:radar_period[inds]=period
            one=meta.iloc[inds[0]]
            frequencies.append(dict(modality=modality,file=name,bag_id=one.bag_id,state=one.state,
                label=int(one.label),baud=int(one.baud if modality=='contact' else one.baud_candidate),
                f0_hz=f0,peak_search_boundary=f0 in [10.,100.],n_windows=len(inds),
                observed_relative_bins=int(valid.sum()),max_observed_relative_frequency=float(Q[valid][-1]),
                geometry_roi_mismatch=bool(one.geometry_roi_mismatch) if modality=='radar' else False,
                source_frequency_min_hz=float(f.min()),source_frequency_max_hz=float(f.max()),
                absolute_grid_max_hz=float(eval_hz.max())))
            manifest.append(dict(file=str(p),sha256=sha(p)))
    arrays=dict(contact_spectral=contact_raw,contact_period_normalized=contact_period,
        contact_hidden=hidden,contact_hidden_plus_period_normalized=np.concatenate([hidden,contact_period],axis=1),
        radar_frame_shape=radar_raw,radar_period_normalized=radar_period)
    np.savez_compressed(ROOT/'features.npz',**arrays,relative_frequency=Q,
        contact_period_valid_mask=np.isfinite(contact_period),radar_period_valid_mask=np.isfinite(radar_period),
        contact_bag_id=cm.bag_id.to_numpy(str),radar_bag_id=rm.bag_id.to_numpy(str))
    cm.to_csv(ROOT/'contact_rows.csv',index=False);rm.to_csv(ROOT/'radar_rows.csv',index=False)
    fm=pd.DataFrame(frequencies);fm.to_csv(ROOT/'recording_frequencies.csv',index=False)
    savej('input_manifest.json',manifest)
    return arrays,cm,rm,fm

def weights(meta):
    n=meta.groupby('bag_id').size()
    return np.array([1/n[b] for b in meta.bag_id])*len(meta)/len(n)

def fit_transform(x,meta,train):
    """Only source observed entries fit column statistics; absent columns ignored."""
    a=x[train];w=weights(meta[train]);valid=np.isfinite(a)
    mass=(valid*w[:,None]).sum(0);observed=mass>0
    mean=np.sum(np.where(valid,a,0)*w[:,None],axis=0)/np.maximum(mass,1e-12)
    var=np.sum(np.where(valid,(a-mean)**2,0)*w[:,None],axis=0)/np.maximum(mass,1e-12)
    scale=np.sqrt(var);scale[scale<1e-12]=1.
    transformed=np.where(np.isfinite(x),(x-mean)/scale,0.)
    transformed[:,~observed]=0.
    assert np.isfinite(transformed).all()
    clf=LogisticRegression(C=1.,max_iter=3000,solver='lbfgs',random_state=42)
    clf.fit(transformed[train],meta.loc[train,'label'].to_numpy(),sample_weight=w)
    return clf.predict_proba(transformed),dict(mean=mean,scale=scale,observed=observed,
        coef=clf.coef_,intercept=clf.intercept_,classes=clf.classes_)

def aggregate(p,meta):
    rows=[]
    for bag,idx in meta.groupby('bag_id',sort=True).indices.items():
        r=meta.iloc[idx[0]];prob=p[idx].mean(0)
        rows.append(dict(bag_id=bag,state=r.state,label=int(r.label),prediction=int(prob.argmax()),
            n_windows=len(idx),healthy_probability=float(prob[0]),
            **{f'p{i}':float(prob[i]) for i in range(4)}))
    return pd.DataFrame(rows)

def evaluate(arrays,cm,rm,fm):
    results=[];predictions=[];external=[];prototype=[]
    for source,target in [(115200,460800),(460800,115200),(None,None)]:
        fold=f'{source}_to_{target}' if source else 'all_known_external_only'
        for method,x in arrays.items():
            modality='contact' if method.startswith('contact') else 'radar'
            meta=cm if modality=='contact' else rm
            baud=meta.baud.to_numpy() if modality=='contact' else meta.baud_candidate.to_numpy()
            train=(meta.label>=0).to_numpy()
            if source:train&=baud==source
            test=(meta.label>=0).to_numpy()&(baud==target) if source else (meta.label<0).to_numpy()
            assert not np.any(train&test)
            prob,state=fit_transform(x,meta,train)
            assert np.array_equal(state['classes'],[0,1,2,3])
            out=aggregate(prob[test],meta[test])
            out['method']=method;out['fold']=fold;out['posthoc']=True
            out['geometry_scope']='outer/keep wrong ROI remains; no correction from normalized spectra'
            if source:
                results.append(dict(method=method,fold=fold,n_train_bags=meta[train].bag_id.nunique(),
                    n_test_bags=len(out),file_accuracy=accuracy_score(out.label,out.prediction),
                    file_macro_f1=f1_score(out.label,out.prediction,labels=range(4),average='macro',zero_division=0),
                    file_balanced_accuracy=balanced_accuracy_score(out.label,out.prediction),
                    window_accuracy_secondary=float(np.mean(prob[test].argmax(1)==meta[test].label.to_numpy())),
                    ignored_source_absent_features=int(np.sum(~state['observed']))))
                predictions.append(out)
            else:external.append(out)
            md=ROOT/'models'/fold;md.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(md/f'{method}.npz',**state)
        for modality in ['contact','radar']:
            fs=fm[(fm.modality==modality)&(fm.label>=0)]
            if source:fs=fs[fs.baud==source]
            for lab,group in fs.groupby('label'):
                prototype.append(dict(fold=fold,modality=modality,label=int(lab),state=group.state.iloc[0],
                    n_training_files=len(group),f0_mean_hz=float(group.f0_hz.mean()),
                    f0_min_hz=float(group.f0_hz.min()),f0_max_hz=float(group.f0_hz.max())))
    metrics=pd.DataFrame(results);metrics.to_csv(ROOT/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(ROOT/'file_predictions.csv',index=False)
    ex=pd.concat(external,ignore_index=True);ex.to_csv(ROOT/'external_predictions.csv',index=False)
    pd.DataFrame(prototype).to_csv(ROOT/'training_frequency_prototypes.csv',index=False)
    summaries=[]
    for method,group in ex.groupby('method',sort=False):
        healthy=group[group.state=='bigNormal'];other=group[group.state=='keep']
        summaries.append(dict(method=method,external_healthy_files=len(healthy),healthy_accepted=int((healthy.prediction==0).sum()),
            healthy_acceptance=float(np.mean(healthy.prediction==0)),mean_healthy_probability=float(healthy.healthy_probability.mean()),
            unknown_base_fault_files=len(other),unknown_rejection_metric='not calibrated, not measured'))
    es=pd.DataFrame(summaries);es.to_csv(ROOT/'external_summary.csv',index=False)
    return metrics,ex,es

def report(fm,metrics,ex,es):
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),sharey=True)
    for ax,modality in zip(axes,['contact','radar']):
        sub=fm[fm.modality==modality].reset_index(drop=True)
        colors=['#3274a1' if v>=0 else '#e1812c' for v in sub.label]
        ax.bar(np.arange(len(sub)),sub.f0_hz,color=colors)
        ax.set_xticks(np.arange(len(sub)),[f'{r.state}\n{r.baud}' for r in sub.itertuples()],rotation=65,ha='right',fontsize=8)
        ax.set_title(modality+' dominant peak, not confirmed shaft speed');ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,110)
    axes[0].set_ylabel('Dominant 10–100 Hz peak (Hz)');fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(ROOT/f'dominant_periods.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    lines=['# 主周期归一化：事后诊断检查','',
        '**这是受到另一台正常电机预测失败启发的 posthoc 检查。所有新增设置在本检查计算结果前写入 protocol.json；原主实验协议及结果未修改。不能将这里的新外部结果继续称为独立封存测试。**','',
        '**结果：六个固定方法对 bigNormal 均为 0/2 正常接受。雷达主周期归一化的两方向文件 macro-F1 从原谱的 1.000 / 1.000 降至 0.667 / 0.333，没有得到稳定补救收益；接触式四种方法两方向均为 1.000，但外部仍全部失败。**','',
        '主台四类接触主峰全部为 33 Hz，keep 接触主峰为 66 Hz。bigNormal 两份接触主峰落在搜索上界 100 Hz，雷达落在下界 10 Hz：边界命中说明本搜索不能提供可信的周期基频估计，不能将它们乘 60 当作已知 rpm。本检查保留这些失败，不扩大搜索范围追求更好的外部分类结果。','',
        '## 固定算法','',
        '逐文件在 10–100 Hz 内选窗口中位功率谱最大峰 f0。接触式先对 XYZ 功率取中位数，雷达使用原 frame_log_power 还原的逐帧谱。估计不用故障标签，也不针对外部健康记录挑峰。f0 只是主周期候选，不是已知轴转频；机械谐波、共振和雷达帧滤波都可能影响它。','',
        '把每个窗口 log10 功率插值到固定相对频率 q=0.5…20 的 128 点线性网格，实际查询频率为 q×f0。在原始 5–800 Hz 之外记缺测，不外推。仅在有效点减去本窗口均值；分类器按训练文件等权拟合列均值/方差，缺测映射到标准化后的零，训练全缺列忽略。mask 单独保存但不作为分类输入。','',
        'f0 使用该文件所有无标签窗口，因此属于离线录制级自归一化；不能据此声称在线首窗部署已验证。毫米波帧谱真实分辨能力约 10.42 Hz，0.5 Hz 网格并不提高真实分辨率。','',
        '原谱与相对频率谱的网格、支持范围也随之改变，当前差异不能全部归因于“除以主周期”这一个因素。该检查回答固定的简单补救方案是否有效，不是孤立所有频谱处理因素的最终消融。','',
        '模型固定 L2 多项逻辑回归 C=1；只做两个串口波特率整录制互换方向，以及全部八个已知类录制拟合后的外部案例。keep/bigNormal 全部排除拟合、scaler 与阈值选择。冻结接触式 hidden 来自主实验固定的 BearLLM FCN 及外部参考方案。','',
        '## 全部主峰估计','',markdown(fm[['modality','state','baud','f0_hz','observed_relative_bins','max_observed_relative_frequency','geometry_roi_mismatch']]),'',
        '![主周期候选](dominant_periods.png)','',
        '## 两个方向全部结果','',markdown(metrics),'',
        '四类中外圈已有距离门错误，当前表格仅诊断算法与混淆风险。谱归一化不能恢复未保存的目标距离单元。两个方向仅各四个测试文件，不是新的独立轴承试验。','',
        '## 全部外部结果','',markdown(es),'',markdown(ex[['method','state','bag_id','prediction','healthy_probability']]),'',
        'bigNormal 只有另一台电机的正常类，不能证明外部故障分类；keep 是未知底座故障且距离门错误，未拟合开放集拒识阈值，所以不报告未知故障拒识准确率。表中保留所有方法和失败。若某个特征组合改善，仅说明值得按预先冻结方案补采独立数据复验，不能据此选择“最佳模型”后宣布泛化成功。','',
        '## 可复现文件','',
        '- `protocol.json`：计算前固定的新增协议。','- `run_posthoc.py`：完整计算脚本。','- `features.npz`：原谱/主周期归一化特征、相对频率轴及缺测 mask。','- `recording_frequencies.csv`：逐文件 f0 及有效频率覆盖。','- `training_frequency_prototypes.csv`：各训练折的类别主频均值/范围，仅作诊断。','- `metrics.csv`、`file_predictions.csv`、`external_predictions.csv`：全部结果。','- `models/`：训练来源统计和线性模型参数。','',
        '下一步应先确认各录制真实转速，再判断主峰对应基频还是谐波；随后用同转速、同距离和同负载的额外故障/正常电机数据检验是否仍需要此归一化。']
    (ROOT/'主周期归一化事后检查.md').write_text('\n'.join(lines)+'\n')

def main():
    if '/envs/m2vllm/' not in sys.executable:raise ValueError('Use m2vllm')
    started=time.monotonic();before=sha(ROOT/'protocol.json')
    arrays,cm,rm,fm=extract();metrics,ex,es=evaluate(arrays,cm,rm,fm);report(fm,metrics,ex,es)
    assert before==sha(ROOT/'protocol.json')
    savej('runtime.json',dict(python=sys.executable,numpy=np.__version__,elapsed_seconds=time.monotonic()-started,
        protocol_sha256=before,script_sha256=sha(__file__),posthoc=True,original_main_protocol_modified=False,
        no_external_label_used_in_fit=True,models_per_fold=6,folds=3,raw_inputs_modified=False))
    print(metrics.to_string(index=False));print(es.to_string(index=False));print(fm[['modality','state','baud','f0_hz']].to_string(index=False))

if __name__=='__main__':main()
