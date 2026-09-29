#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
B=Path(__file__).resolve().parent/'radar'
rm=pd.read_csv(B/'radar_metadata.csv');files=pd.read_csv(B/'file_predictions.csv')
rows=[];equality=[]
for head in ['native15','new_contact4']:
    for method in ['ce_only','embedding_only','full']:
        for seed in [17,42,73]:
            pieces=[]
            for source,target in [(115200,460800),(460800,115200)]:
                fold=f'{source}_to_{target}';z=np.load(B/'models'/f'{fold}__{head}__{method}__seed{seed}'/'predictions.npz',allow_pickle=False)
                mask=(rm.label>=0)&(rm.baud_candidate==target)
                d=rm[mask].copy();d['prediction']=z['probs4'][mask].argmax(1)
                d['gear_mass']=z['probs15'][mask,10:].sum(1);d['pred15']=z['probs15'][mask].argmax(1)
                pieces.append(d)
            windows=pd.concat(pieces);assert len(windows)==412 and windows.row.nunique()==412
            fs=files[(files['head']==head)&(files.method==method)&(files.seed==seed)&(files.role=='test')]
            assert len(fs)==8 and fs.bag_id.nunique()==8
            for unit,all_data in [('window',windows),('file',fs)]:
                for subset in ['all_four_provisional_roi','geometry_valid_subset']:
                    dd=all_data.copy();labs=[0,1,2,3]
                    if subset=='geometry_valid_subset':
                        dd=dd[~dd.geometry_roi_mismatch] if unit=='window' else dd[dd.geometry_valid]
                        labs=[0,1,3]
                    rows.append(dict(head=head,method=method,seed=seed,unit=unit,subset=subset,n=len(dd),correct=int((dd.label==dd.prediction).sum()),
                        accuracy=float(accuracy_score(dd.label,dd.prediction)),macro_f1=float(f1_score(dd.label,dd.prediction,labels=labs,average='macro',zero_division=0)),
                        macro_f1_labels=','.join(map(str,labs)),gear_mass=float(dd.gear_mass.mean()) if unit=='window' else float(dd.native15_gear_mass.mean()),
                        pred15_other_rate=float((dd.pred15>=10).mean()) if unit=='window' else float((dd.native15_pred>=10).mean())))
for fold in ['115200_to_460800','460800_to_115200']:
    for seed in [17,42,73]:
        a=np.load(B/'models'/f'{fold}__native15__embedding_only__seed{seed}'/'predictions.npz',allow_pickle=False)
        b=np.load(B/'models'/f'{fold}__new_contact4__embedding_only__seed{seed}'/'predictions.npz',allow_pickle=False)
        error=float(np.abs(a['hidden']-b['hidden']).max());assert error==0
        equality.append(dict(fold=fold,seed=seed,hidden_max_abs_error=error,exact_equal=True))
(B/'embedding_only_head_swap_check.json').write_text(json.dumps(equality,indent=2)+'\n')
d=pd.DataFrame(rows);d.to_csv(B/'pooled_per_seed_metrics.csv',index=False)
summary=d.groupby(['head','method','unit','subset'],as_index=False).agg(accuracy_mean=('accuracy','mean'),accuracy_min=('accuracy','min'),accuracy_max=('accuracy','max'),macro_f1_mean=('macro_f1','mean'),gear_mass_mean=('gear_mass','mean'),pred15_other_rate=('pred15_other_rate','mean'),n_per_seed=('n','first'))
summary.to_csv(B/'pooled_summary.csv',index=False)
table=[]
for head in ['native15','new_contact4']:
    for method in ['ce_only','embedding_only','full']:
        win=summary[(summary['head']==head)&(summary.method==method)&(summary.unit=='window')&(summary.subset=='all_four_provisional_roi')].iloc[0]
        fs=summary[(summary['head']==head)&(summary.method==method)&(summary.unit=='file')&(summary.subset=='all_four_provisional_roi')].iloc[0]
        sub=files[(files['head']==head)&(files.method==method)&(files.role=='test')]
        label='官方原生15类头（四组条件判断）' if head=='native15' else '另训接触四类头'
        names={'ce_only':'只要求类别正确','embedding_only':'只匹配接触平均特征','full':'类别＋特征＋对比＋KD'}
        table.append(f"| {label} | {names[method]} | {win.accuracy_mean:.2%}（{win.accuracy_min:.2%}—{win.accuracy_max:.2%}） | {fs.accuracy_mean:.2%} | {sub.mse_standardized.mean():.4f} | {sub.cosine_standardized.mean():.4f} |")
geo=summary[(summary.subset=='geometry_valid_subset')&(summary.unit=='window')]
gtable=[]
for r in geo.itertuples():gtable.append(f'| {r.head} | {r.method} | {r.accuracy_mean:.2%} | {r.macro_f1_mean:.4f} |')
readme=f'''# 雷达接入 RotLLM：实际训练验证

本轮新增训练36个雷达网络：两个冻结判断模块 × 两个互补录制划分 × 三种训练要求 × 三个随机种子。每个网络固定训练400步，最后一步直接评价，未根据测试选择checkpoint、随机种子、预处理或超参数。只使用前向前指定的 `raw_rms001` 接触分支。

## 最重要的观察

**直接接入原生RotLLM时，雷达只学类别能得到很高的开发准确率，但其特征反而远离接触教师。只学接触特征时，它确实更像教师，却跟着教师给出低准确率。另加仅用接触训练数据学得的四类判断模块后，相同的雷达特征可以正确分类。**

这不是已经证明迁移方法完全通用，而是得到一个可检查的失败原因：**必须分别检查接触特征和接触判断模块是否适用于新测量条件，不能以“雷达最后答对”替代“雷达学会了接触知识”的证据。**

## 主结果

每个随机种子合并两个方向，412个雷达测试窗口各计一次，8条录制各计一次。下表为三个种子的平均，括号为种子范围；种子没有增加物理样本量。

| 判断模块 | 雷达如何训练 | 2秒窗口正确率 | 整条录制正确率 | 平均特征差异MSE↓ | 平均特征相似度cosine↑ |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

MSE和cosine比较的是按接触训练数据尺度标准化后的整条录制平均特征。它们没有百分比含义，也不是“知识继承比例”。

这里还完成了一个特别直接的对照：**只匹配特征的六个网络，在原生头、新四类头两种运行中的614个窗口hidden逐元素完全相同，最大差为0；仅换读取这些特征的判断模块，录制准确率就从25%变为100%。** 文件 `embedding_only_head_swap_check.json` 保存全部六组检查。这说明该组结果的差别确实来自读取模块，而不是重新训练出了另一种雷达特征。

## 为什么原生头的CE-only高分不能当作知识迁移成功

训练CE-only时，真实故障标签直接要求雷达输出正确类别。雷达网络可以学会一组能使冻结分类器答对的数字，并不需要逼近真实接触特征。实测其标准化平均特征MSE约6.82、cosine约-0.008；只做特征匹配时MSE约0.20、cosine约0.635。

原生接触教师自己在这批接触输入上也只有25%的整条录制正确率。特征匹配与KD要求学生接近该教师，而CE要求答对真实标签，两种要求存在冲突。当前固定权重下，完整组合没有解决这种冲突。我们没有用测试集继续搜索损失权重来掩盖这个结果。

加新四类头后，接触本身的留出片段准确率是84.48%、录制8/8；雷达接入后多种方法都很高。它仍不能证明更细粒度知识迁移：每折每类只有一条训练录制，类别代表特征与录制平均特征无法区分；完整方法相比简单特征匹配也没有明显收益。

## 15类和四类口径

原生模块是15类，正常0、内圈1—3、外圈7—9、滚动体4—6构成四个组，10—14为齿轮故障。雷达主指标是“已知任务属于四种轴承状态”下的条件分类。所有模型均完整保存原15类概率及齿轮概率，主测试原15类top-1落到齿轮的比例均为0，不存在靠隐藏大量齿轮预测制造高分的问题。

新四类头的主要输出本来就是四类，保存的原15类概率仅是用同一hidden经过旧头的诊断，不把它称为新头的输出。

## 预处理、训练和数据隔离

- 雷达输入是历史IQ预处理后保存的128维 `frame_shape`，不是直接把原始IQ送神经网络。本轮没有重算原始bin。
- 输入StandardScaler只用本折雷达训练录制拟合，每条录制权重相等。接触特征均值、尺度也只用本折接触训练录制、等录制权重拟合。
- 学生：128→128→LayerNorm→GELU→Dropout0.1→64→GELU→128，最后输出 `ReLU(contact_mean + contact_scale * output)`，保持接触隐藏特征非负。
- 每步每训练录制随机取16个窗口；AdamW，lr=0.001，weight_decay=0.01，梯度裁剪5；400步，seed17/42/73。
- CE-only仅用类别交叉熵；embedding-only仅用标准化整段平均特征MSE；full=CE+MSE+0.1×同类对比+0.5×KD。KD在T=2下比较教师与学生各录制的平均条件四类概率。
- 接触encoder、原15类头、新四类头全程冻结。新四类头先用接触训练录制建立，雷达训练不更新它；没有将接触测试窗口用于训练这个头。
- 115200/460800是串口波特率和两次录制标记，不是转速。按整条录制留出，没有混合随机窗口。
- 36个模型全部使用 `weights_only=True` 重新载入，重新从保存的scaler和权重计算概率并核验；输入、原encoder、原头和新接触头文件SHA256全部保持不变。

## 外圈距离单元错误与有效子集

主表包括此前已知距离单元错误的外圈雷达，故必须标记为“已有开发数据、外圈ROI暂不可信”。原始ADC bin在本环境不可用，本轮没有假装完成修复。以下仅排除已知错误ROI的外圈样本，保留同一个四类预测器，不重新训练、不把外圈概率删掉。仅在正常/内圈/滚动体三个真实标签上计算macro-F1。

| 判断模块 | 方法 | 有效子集窗口正确率 | 三个真实标签macro-F1 |
|---|---|---:|---:|
{chr(10).join(gtable)}

这是敏感性检查，不能替代干净的四类物理实验。

## 输出文件

- `pooled_per_seed_metrics.csv`：每种方法每个seed合并两方向的412窗口/8录制结果；`pooled_summary.csv`为三个seed汇总。应优先引用这两个文件。
- `metrics.csv`：每折每seed的训练、测试与有效ROI子集指标。`summary_mean_folds_seeds.csv`是折均值，片段数不等时略不同于合并窗口口径。
- `file_predictions.csv`：逐录制四类概率、预测、特征误差、几何标记、原15类齿轮概率等。
- `models/*/model.pt`：36个模型全部保存，包含冻结头、训练scaler和输入来源哈希。
- `models/*/predictions.npz`：614个窗口对应的128维hidden、标准化hidden、四类概率、原15类概率、bag和标签。文件顺序与 `radar_metadata.csv` 相同。
- `contact_teacher_file_predictions.csv`：各折教师在接触输入上的结果（先平均XYZ hidden再分类，用于本训练的教师目标）。
- `training_trace.csv`：每100步损失；`verification.json`：重载与只读检查；`protocol.json`：运行前协议。

重跑脚本为上级目录 `run_radar_transfer.py`。完成目录有保护，不能静默覆盖已验收的训练结果。汇总脚本为上级目录 `summarize_radar.py`。
'''
(B/'README_雷达迁移实测.md').write_text(readme)
print(summary.to_string(index=False))
