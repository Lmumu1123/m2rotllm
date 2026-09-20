"""Source-only model selection, whole-file cross-environment evaluation."""
from pathlib import Path
import json, hashlib, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)
METHODS=['raw_amplitude','phase_single','iq_multi_abs','iq_multi_shape','iq_frame_shape']


def weights(meta):
    # Each file has equal weight within a class; each class has equal total weight.
    counts=meta.groupby('file').size()
    files=meta.drop_duplicates('file')
    classes=files.groupby('label').size()
    w=np.array([1/(counts[r.file]*classes[r.label]) for r in meta.itertuples()])
    return w*len(w)/w.sum()


def fit(X,meta,C):
    w=weights(meta)
    scaler=StandardScaler().fit(X,sample_weight=w)
    model=LogisticRegression(C=C,max_iter=3000,solver='lbfgs',random_state=42)
    model.fit(scaler.transform(X),meta.label.values,sample_weight=w)
    return scaler,model


def file_predictions(model,scaler,X,meta):
    prob=model.predict_proba(scaler.transform(X))
    temp=pd.DataFrame(prob,columns=model.classes_)
    temp['file']=meta.file.values
    p=temp.groupby('file',sort=True)[list(model.classes_)].mean()
    info=meta.drop_duplicates('file').set_index('file').loc[p.index]
    pred=np.asarray(model.classes_)[np.argmax(p.values,axis=1)]
    out=info[['environment','distance_cm','label']].reset_index()
    out['prediction']=pred
    for c in model.classes_:out['p_'+str(c)]=p[c].values
    return out


def score(pred):
    return dict(macro_f1=float(f1_score(pred.label,pred.prediction,average='macro',zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(pred.label,pred.prediction)),
                n_files=len(pred),n_classes=pred.label.nunique())


def choose_C(X,meta):
    scores=[]
    for C in [.1,1.,10.]:
        fold=[]
        for d in sorted(meta.distance_cm.unique()):
            tr=(meta.distance_cm!=d).values;va=~tr
            if meta[tr].label.nunique()!=meta.label.nunique():continue
            sc,mo=fit(X[tr],meta[tr],C)
            fold.append(score(file_predictions(mo,sc,X[va],meta[va]))['macro_f1'])
        scores.append((float(np.mean(fold)),C))
    # Ties favour the stronger regularization, independent of target outcomes.
    best=sorted(scores,key=lambda v:(-v[0],v[1]))[0]
    return best[1],best[0],scores


def calibrated_features(meta,powers,quality):
    # Range consistency check is geometry-only, with a fixed 10-cm tolerance.
    # Estimate an environment's range offset only from its allowed off records.
    # No target rotating records enter calibration or model selection.
    q=pd.DataFrame(quality)
    off=q[q.rpm==0].copy()
    off['offset']=off.roi_peak_m-off.distance_cm/100
    offset=off.groupby('environment')['offset'].median().to_dict()
    references={}; rejected=[]
    for row in q[q.rpm==0].itertuples():
        residual=abs(row.roi_peak_m-row.distance_cm/100-offset[row.environment])
        if residual>.10:
            rejected.append(dict(file=row.file,range_residual_m=residual,
                                 reason='off target range inconsistent; pending acquisition metadata'))
            continue
        idx=(meta.file==row.file).values
        noise=np.median(powers[idx],axis=0)
        noise=median_filter(noise,size=5,mode='nearest')
        references[(row.environment,row.distance_cm)]=noise
    transformed=np.full_like(powers,np.nan)
    for i,row in enumerate(meta.itertuples()):
        key=(row.environment,row.distance_cm)
        if key not in references:continue
        noise=references[key]
        floor=max(float(np.median(noise))*.01,1e-16)
        x=np.log1p(powers[i]/np.maximum(noise,floor))
        transformed[i]=x-x.mean()
    common=set.intersection(*[set(d for e,d in references if e==env) for env in ['room','narrow']])
    return transformed,sorted(common),rejected,offset


def savefig(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def run():
    meta=pd.read_csv(OUT/'windows.csv',dtype={'label':str})
    arrays=np.load(OUT/'features.npz')
    quality=json.loads((OUT/'quality.json').read_text())
    feats={m:arrays[m] for m in METHODS}
    p1,common,rejected,offset=calibrated_features(meta,arrays['power_binned'],quality)
    p1f,_,_,_=calibrated_features(meta,arrays['frame_power_binned'],quality)
    feats['iq_multi_off']=p1;feats['iq_frame_off']=p1f
    metrics=[];all_predictions=[];selections=[]
    tasks=[('P0_four_state',np.ones(len(meta),bool),METHODS),
           ('P0_three_speed',(meta.label!='off').values,METHODS)]
    if common:
        matched=((meta.label!='off')&meta.distance_cm.isin(common)).values
        tasks.append(('P1_matched_geometry_three_speed',matched,
                      ['phase_single','iq_multi_shape','iq_frame_shape','iq_multi_off','iq_frame_off']))
    for task,eligible,methods in tasks:
        for source,target in [('room','narrow'),('narrow','room')]:
            tr=eligible&(meta.environment==source).values
            te=eligible&(meta.environment==target).values
            for method in methods:
                x=feats[method]
                assert np.isfinite(x[tr|te]).all(),(task,method)
                C,cv,grid=choose_C(x[tr],meta[tr])
                sc,mo=fit(x[tr],meta[tr],C)
                pred=file_predictions(mo,sc,x[te],meta[te])
                info=dict(task=task,direction=source+'->'+target,method=method)
                metrics.append(dict(**info,C=C,source_cv_macro_f1=cv,**score(pred)))
                selections.append(dict(**info,source_only_C_grid=grid))
                for col,value in info.items():pred[col]=value
                all_predictions.append(pred)
                for distance,g in pred.groupby('distance_cm'):
                    # Store per-distance details in the predictions; main table remains file-level.
                    pass
                print(info,score(pred),flush=True)
    metrics=pd.DataFrame(metrics)
    preds=pd.concat(all_predictions,ignore_index=True)
    metrics.to_csv(OUT/'cross_environment_metrics.csv',index=False)
    preds.to_csv(OUT/'file_predictions.csv',index=False)
    (OUT/'source_model_selection.json').write_text(json.dumps(selections,indent=2))
    calibr=dict(range_offset_by_environment_from_off_m=offset,P1_common_distances_cm=common,
               rejected_off_references=rejected,reference_tolerance_m=.10,
               note='P1 excludes off from diagnosis tests; matched P0 comparators use same distances')
    (OUT/'calibration_audit.json').write_text(json.dumps(calibr,indent=2))
    # By-distance results are file-level, with all task/direction/method groups retained.
    by=[]
    for keys,g in preds.groupby(['task','direction','method','distance_cm']):
        by.append(dict(zip(['task','direction','method','distance_cm'],keys),**score(g)))
    pd.DataFrame(by).to_csv(OUT/'per_distance_metrics.csv',index=False)
    plots(meta,arrays,quality,metrics,preds)
    report(meta,quality,metrics,calibr)


def plots(meta,arrays,quality,metrics,preds):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'room':'#2875B8','narrow':'#D78428'}
    q=pd.DataFrame(quality)
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    r=np.arange(128)*3430000/512*299792458/(2*49.99e12)
    for ie,env in enumerate(['room','narrow']):
        for j,d in enumerate([20,40,80]):
            ax=axs[ie,j]
            for state,ls in [('rotating','-'),('off','--')]:
                rows=q[(q.environment==env)&(q.distance_cm==d)&((q.rpm!=0) if state=='rotating' else (q.rpm==0))]
                p=np.mean(np.stack(rows.median_range_profile),axis=0)
                ax.plot(r,10*np.log10(p+1),ls,label=state,color=colors[env])
            ax.axvline(d/100,color='gray',lw=1,alpha=.6)
            ax.set(title=f'{env}, {d} cm',xlim=(0,1.4),xlabel='Range (m)',ylabel='ADC spectrum power (dB)')
            ax.legend(fontsize=8)
    fig.suptitle('Measured range profiles: geometry and off-reference audit (not fault results)')
    fig.tight_layout();savefig(fig,'01_range_audit')
    fig,axs=plt.subplots(1,2,figsize=(13,4),sharey=True)
    for ax,task in zip(axs,['P0_four_state','P0_three_speed']):
        tab=metrics[metrics.task==task].pivot(index='method',columns='direction',values='macro_f1').reindex(METHODS)
        tab.plot.bar(ax=ax,color=['#2875B8','#D78428'],rot=25)
        ax.set(title=task,ylabel='File-level macro-F1',ylim=(0,1.05),xlabel='')
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('Cross-environment evaluation; source-only hyperparameter selection')
    fig.tight_layout();savefig(fig,'02_cross_environment')
    fig,axs=plt.subplots(3,3,figsize=(13,9),sharex=True)
    f=arrays['frequencies'];spectra=arrays['spectra']
    for i,rpm in enumerate(['1000','2000','3000']):
        for j,d in enumerate([20,40,80]):
            ax=axs[i,j]
            for env in ['room','narrow']:
                ids=((meta.label==rpm)&(meta.distance_cm==d)&(meta.environment==env)).values
                p=np.median(spectra[ids],axis=0);p/=max(p.sum(),1e-30)
                ax.plot(f,10*np.log10(p+1e-12),c=colors[env],lw=1,label=env)
            ax.axvline(int(rpm)/60,c='gray',ls='--',lw=.8)
            ax.set(title=f'{rpm} rpm, {d} cm',xlim=(5,160),xlabel='Slow-time frequency (Hz)',ylabel='Normalized power (dB)')
            ax.legend(fontsize=8)
    fig.suptitle('Masked harmonic-LS IQ spectra; frame coherence is an assumption; dashed: nominal shaft rate')
    fig.tight_layout();savefig(fig,'03_matched_condition_spectra')
    chosen='iq_frame_shape'
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,direction in zip(axs,['room->narrow','narrow->room']):
        p=preds[(preds.task=='P0_three_speed')&(preds.direction==direction)&(preds.method==chosen)]
        labels=['1000','2000','3000'];cm=confusion_matrix(p.label,p.prediction,labels=labels)
        ax.imshow(cm,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=labels,yticklabels=labels,
               xlabel='Predicted rpm',ylabel='Filename rpm label',title=direction)
    fig.suptitle('Preselected frame-power baseline, whole-file predictions')
    fig.tight_layout();savefig(fig,'04_speed_confusion')


def markdown_table(df):
    # Avoid depending on tabulate; values are short and do not include Markdown pipes.
    out=['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    out+=['| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(out)


def report(meta,quality,metrics,calibration):
    q=pd.DataFrame(quality)
    inventory=q.groupby(['environment','label','distance_cm']).size().rename('files').reset_index()
    table=metrics.copy()
    for col in ['macro_f1','balanced_accuracy','source_cv_macro_f1']:table[col]=table[col].map(lambda x:f'{x:.3f}')
    report='''# room/narrow 毫米波环境验证：实际数据结果

已在 conda `m2vllm` 中完成。建议先读 [实验结论与算法细节](实验结论.md)，并结合 [未见距离与主频对照](补充验证.md)：简单主频对照同样达到 100%，而未见距离测试出现明显下降，不能仅凭下方主表认定环境影响已被消除。

本报告基于目录的一次稳定输入快照，处理前后 manifest 不变；是否与上传源文件完整一致，应结合上传完成确认和源端校验和。数据包含正常轴承旋转与停止状态，没有故障类或同步加速度，因此这里的分类结果是运动状态/转速结果，不是故障诊断或跨模态迁移结果。

## 解析与协议

假设使用用户前述 cfg：256 ADC、4RX、192 chirp/100 ms、chirp 周期 500 μs；按 TI 两 lane 的 2I/2Q 布局解析，使用固定共轭方向使目标距离峰位于正频率。仅有 bin 时，不能严格证明帧起点与 UDP 缺包重排正确；具体配置、采集版本及独立 run 信息仍需采集记录佐证。

各窗口 2 秒，不重叠；每次训练只用一个环境。所有标准化和分类器参数由源环境训练产生，C=0.1/1/10 只在源环境按距离留出验证选择。先平均同一文件的窗口概率，再计算文件级 macro-F1 与 balanced accuracy。训练权重使同类文件等权、类别总权重相同。文件名转速没有输入模型；距离只用于所有方法一致的几何 ROI。

同一次采集拆分文件必须合并为同一 run；当前结果按文件计数。在未确认独立录制前，不把它称为独立 run 泛化。两个环境不足以估计任意新环境的总体表现。

## 数据清点

'''
    report+=f'共 {len(q)} 个文件，约 {q["size"].sum()/1e9:.3f} GB，{len(meta)} 个完整 2 秒窗口。末尾非整帧余量逐文件记录，不作为零填充测量。\n\n'
    report+=markdown_table(inventory)+'\n\n'
    report+='## 处理方法\n\n'
    report+='''- `raw_amplitude`：ROI 三 bin×4RX 的 log 幅度功率，检查对静态几何/环境的依赖。
- `phase_single`：最强单元的包内相位增量，真实 mask 上作谐波回归；不跨帧差分。
- `iq_multi_abs`：按复幅度尺度归一化的多单元 IQ、帧内去均值、mask 谐波回归、稳健功率汇聚。
- `iq_multi_shape`：上一方法的 log 频带谱减频带均值，检验幅值归一化是否保留任务信息。
- `iq_frame_shape`：完整帧内估谱，跨帧平均功率；不要求绝对帧间相干，但实际分辨率受 96 ms 片段限制。
- `iq_multi_off` / `iq_frame_off`：分别在上述前端上使用同环境同距离 off 的噪声谱归一化。仅评价旋转记录，off 参考不进入测试状态分类。

mask 谐波回归包含常数、cos、sin 项，已与直接最小二乘对照核查。它没有把缺口当成观测零值，但也不保证消除全部谱泄漏；跨帧相干性未经独立标定时，相应结果属于条件性结果。原始 ADC 削顶无法通过这些算法恢复。

## 实际结果

'''
    report+=markdown_table(table[['task','direction','method','C','source_cv_macro_f1','macro_f1','balanced_accuracy','n_files']])+'\n\n'
    report+='### 预先指定比较的解读\n\n'
    diffs=[]
    for direction in ['room->narrow','narrow->room']:
        part=metrics[(metrics.task=='P0_three_speed')&(metrics.direction==direction)].set_index('method')
        base=float(part.loc['phase_single','macro_f1'])
        value=float(part.loc['iq_frame_shape','macro_f1'])
        diffs.append(value-base)
        report+=f'- {direction}：完整帧多单元谱形状为 {value:.3f}，单单元相位基线为 {base:.3f}，差值 {(value-base)*100:+.1f} 个百分点。\n'
    if all(x>0 for x in diffs):
        report+='\n在这两个固定方向上，该预先指定前端均提高了三转速识别的文件级分数；样本量、独立性和故障任务外推仍受下述限制。不能仅凭这两个点宣称统计显著或普适改进。\n\n'
    else:
        report+='\n该预先指定前端没有在两个方向都优于相位基线；本数据尚不支持“环境处理稳定改善泛化”的结论。应结合质量异常与逐距离结果定位，不仅挑选有利方向。\n\n'
    report+='P1 与 P0 的公平比较仅使用相同有效校准距离子集；不要把该子集上的分数直接与全距离主表比较。\n\n'
    report+=f'P1 两环境共同有效距离：{calibration["P1_common_distances_cm"]} cm；几何一致性采用统一 10 cm 残差容许值。待核查 off 文件：`{json.dumps(calibration["rejected_off_references"],ensure_ascii=False)}`。\n\n'
    clipped=q[q.clip_fraction>.001][['file','clip_fraction','roi_peak_m']]
    report+='## 数据质量与解释边界\n\n'
    if len(clipped):
        report+='以下文件至少 0.1% 的已处理 int16 数值接近上下限（阈值 |x|≥32760），需要核查增益/削顶；主表保留它们，没有依据性能删除：\n\n'+markdown_table(clipped)+'\n\n'
    report+='''域差异变小本身不是成功：应同时保持不同转速的可分性。更高的正常旋转/停止识别分数不证明保留了轴承故障信息。若复杂前端没有超过简单前端，应按实测结果缩减设计；不能把候选算法写成已经有效。

## 图与复现产物

![距离与参考检查](figures/01_range_audit.png)

![跨环境分类](figures/02_cross_environment.png)

![相同工况跨环境频谱](figures/03_matched_condition_spectra.png)

![预先指定的帧内功率前端混淆矩阵](figures/04_speed_confusion.png)

结果明细在 `results/cross_environment_metrics.csv`、`file_predictions.csv`、`per_distance_metrics.csv`；数据与配置在 `input_manifest.json`、`quality.json`、`processing_config.json`。保留输入快照、源环境参数选择日志与全部逐文件预测。

解析依据：[TI RAW Capture](https://www.ti.com/lit/an/swra581b/swra581b.pdf)、[TI 对 AWR1843 格式的确认](https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/967227/dca1000evm-raw-data-format)。
'''
    (ROOT/'环境泛化验证报告.md').write_text(report)


if __name__=='__main__':run()
