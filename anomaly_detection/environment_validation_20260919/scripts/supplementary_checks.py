"""Declared diagnostic follow-ups, separate from the frozen primary table.

These checks were added after primary results to examine distance dependence
and whether simple shaft-frequency evidence explains the speed task.
No primary preprocessing or source-model-selection rules are retuned.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_environment import ROOT,OUT,FIG,choose_C,fit,file_predictions,score,markdown_table,savefig


def main():
    meta=pd.read_csv(OUT/'windows.csv',dtype={'label':str})
    arrays=np.load(OUT/'features.npz')
    q=pd.DataFrame(json.loads((OUT/'quality.json').read_text()))
    main_metrics=pd.read_csv(OUT/'cross_environment_metrics.csv')
    # Cross environment AND an unseen distance, no target-based model selection.
    result=[];predictions=[]
    methods=['phase_single','iq_multi_shape','iq_frame_shape']
    for source,target in [('room','narrow'),('narrow','room')]:
        for held in [20,40,80]:
            tr=((meta.environment==source)&(meta.label!='off')&(meta.distance_cm!=held)).values
            te=((meta.environment==target)&(meta.label!='off')&(meta.distance_cm==held)).values
            for method in methods:
                x=arrays[method]
                C,cv,grid=choose_C(x[tr],meta[tr])
                scaler,model=fit(x[tr],meta[tr],C)
                pred=file_predictions(model,scaler,x[te],meta[te])
                info=dict(direction=source+'->'+target,held_distance_cm=held,method=method)
                result.append(dict(**info,C=C,source_cv_macro_f1=cv,**score(pred)))
                for key,value in info.items():pred[key]=value
                predictions.append(pred)
    result=pd.DataFrame(result)
    result.to_csv(OUT/'supplementary_unseen_distance_metrics.csv',index=False)
    pd.concat(predictions).to_csv(OUT/'supplementary_unseen_distance_predictions.csv',index=False)

    # Physical control: strongest frequency in a fixed 10..60 Hz operating band.
    # Filename RPM labels are only used for source centroids and evaluation.
    f=arrays['frequencies'];band=(f>=10)&(f<=60)
    power=arrays['spectra'][:,band]
    peaks=f[band][power.argmax(axis=1)]
    per_window=meta[['file','environment','distance_cm','label','window_start_s']].copy()
    per_window['peak_hz']=peaks
    per_window.to_csv(OUT/'dominant_peak_windows.csv',index=False)
    records=[]
    for name,g in per_window.groupby('file'):
        first=g.iloc[0]
        records.append(dict(file=name,environment=first.environment,distance_cm=first.distance_cm,
                            label=first.label,median_peak_hz=float(g.peak_hz.median()),
                            min_peak_hz=float(g.peak_hz.min()),max_peak_hz=float(g.peak_hz.max())))
    records=pd.DataFrame(records)
    records.to_csv(OUT/'dominant_peak_files.csv',index=False)
    physical=[]
    for source,target in [('room','narrow'),('narrow','room')]:
        src=records[(records.environment==source)&(records.label!='off')]
        dst=records[(records.environment==target)&(records.label!='off')].copy()
        centers=src.groupby('label').median_peak_hz.median()
        dst['prediction']=centers.index.to_numpy()[np.argmin(abs(dst.median_peak_hz.to_numpy()[:,None]-centers.to_numpy()[None,:]),axis=1)]
        physical.append(dict(direction=source+'->'+target,**score(dst),source_centers_hz=centers.to_dict()))
    (OUT/'physical_peak_baseline.json').write_text(json.dumps(physical,indent=2))

    # Export the two primary P0 spectral-shape models per task without selecting
    # on the target results. These are speed/state classifiers, not fault models.
    model_dir=ROOT/'models';model_dir.mkdir(exist_ok=True)
    previous=pd.read_csv(OUT/'file_predictions.csv',dtype={'label':str,'prediction':str})
    exported=[]
    for task in ['P0_three_speed','P0_four_state']:
        for source,target in [('room','narrow'),('narrow','room')]:
            method='iq_multi_shape';direction=source+'->'+target
            mask=np.ones(len(meta),bool) if task=='P0_four_state' else (meta.label!='off').values
            tr=mask&(meta.environment==source).values;te=mask&(meta.environment==target).values
            selected=main_metrics[(main_metrics.task==task)&(main_metrics.direction==direction)&(main_metrics.method==method)].iloc[0]
            sc,mo=fit(arrays[method][tr],meta[tr],float(selected.C))
            pred=file_predictions(mo,sc,arrays[method][te],meta[te]).set_index('file')
            expected=previous[(previous.task==task)&(previous.direction==direction)&(previous.method==method)].set_index('file')
            assert pred.prediction.equals(expected.loc[pred.index].prediction)
            for label in mo.classes_:
                assert np.allclose(pred['p_'+label],expected.loc[pred.index,'p_'+label],atol=1e-8)
            name=f'{task}__{source}_to_{target}__{method}.npz'
            np.savez_compressed(model_dir/name,mean=sc.mean_,scale=sc.scale_,coef=mo.coef_,
                intercept=mo.intercept_,classes=mo.classes_.astype(str),feature_name=method,C=float(selected.C))
            exported.append(dict(file=name,task=task,source=source,target=target,
                                 training_files=sorted(meta[tr].file.unique().tolist())))
    (model_dir/'manifest.json').write_text(json.dumps(exported,indent=2))

    fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,direction in zip(axs,['room->narrow','narrow->room']):
        table=result[result.direction==direction].pivot(index='held_distance_cm',columns='method',values='macro_f1')
        table[methods].plot.bar(ax=ax,rot=0,color=['#647589','#12877D','#805BAA'])
        ax.set(title=direction,xlabel='Distance absent from source training (cm)',ylabel='File-level macro-F1',ylim=(0,1.05))
    fig.suptitle('Supplementary diagnostic: environment + distance shift, three-speed task')
    fig.tight_layout();savefig(fig,'05_unseen_distance')
    fig,ax=plt.subplots(figsize=(9,4))
    for env,offset,color in [('room',-.1,'#2875B8'),('narrow',.1,'#D78428')]:
        rows=records[(records.environment==env)&(records.label!='off')]
        for j,label in enumerate(['1000','2000','3000']):
            points=rows[rows.label==label]
            jitter=np.linspace(-.045,.045,len(points))
            ax.scatter(j+offset+jitter,points.median_peak_hz,c=color,label=env if j==0 else None,s=35)
            ax.hlines(int(label)/60,j-.35,j+.35,color='#647589',ls='--',lw=1)
    ax.set(xticks=[0,1,2],xticklabels=['1000','2000','3000'],xlabel='Filename nominal rpm (evaluation only)',ylabel='Measured strongest 10..60 Hz peak (Hz)',title='Physical sanity check; dashed lines are nominal shaft rates')
    ax.legend();ax.grid(axis='y',alpha=.15);fig.tight_layout();savefig(fig,'06_dominant_peak')

    disp=result.copy()
    for c in ['macro_f1','balanced_accuracy','source_cv_macro_f1']:disp[c]=disp[c].map(lambda v:f'{v:.3f}')
    text='''# 补充诊断检查：距离外推与转频基线

这部分在查看主实验后增加，用于检查“高分是否只是转频容易识别”和“距离变化是否仍然稳健”。它是事后诊断性补充；主实验处理规则、参数网格及结果没有依据这些目标域结果修改。

## 环境与距离同时留出

训练只用源环境的两个距离，测试仅用目标环境的第三个距离。C 仍只在源环境剩余两距离交叉验证选择。目标距离作为已知部署几何用于相同 ROI 规则，未给分类器输入距离/转速标签。每个测试单元只有 5 或 7 个文件，结论应克制。

'''+markdown_table(disp)+'\n\n![距离共同留出](figures/05_unseen_distance.png)\n\n'
    text+='## 无复杂分类器的物理对照\n\n固定在 10–60 Hz 运行频带内选最大峰，每文件取窗口峰值中位数；用源环境每类的峰值中位数作为原型，对目标文件做最近原型分类。目标文件名转速只用于事后评价，没有用于寻峰。该范围利用了本任务已知运行范围，不是任意转速算法。\n\n主频对照共用已处理的多单元 IQ 频谱；它检验的是复杂谱特征分类是否必要，不是完全独立的原始信号前端对照。\n\n'
    for r in physical:text+=f'- {r["direction"]}：macro-F1={r["macro_f1"]:.3f}，balanced accuracy={r["balanced_accuracy"]:.3f}，n={r["n_files"]}；源原型 Hz：{r["source_centers_hz"]}。\n'
    text+='\n若这个简单基线已足够准确，说明本数据最稳健的信息是主转频；不能用转速分类高分证明复杂环境算法或故障诊断创新。后续需要故障状态×相同转速的独立采集。\n\n![主频核查](figures/06_dominant_peak.png)\n\n'
    text+='## 可复用处理产物\n\n`models/` 保存四个主实验谱形状模型的 scaler、线性系数、类别与训练文件列表。导出时逐文件验证预测及概率与主结果一致。它们识别转速/停止状态，不是 BearLLM 教师、跨模态学生或故障分类模型。`results/features.npz` 保存全部处理特征，行顺序对应 `results/windows.csv`。\n'
    (ROOT/'补充验证.md').write_text(text)
    print(result.to_string(index=False))
    print('Physical baseline:',json.dumps(physical))
    print('Four primary models exported; predictions match recorded outputs.')


if __name__=='__main__':main()
