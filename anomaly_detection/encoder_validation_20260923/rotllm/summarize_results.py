from c2r_paths import resolve_path as _c2r_resolve_path
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
B = Path(_c2r_resolve_path(__file__)).resolve().parent
VARIANTS = ['raw_rms001', 'demean_rms001', 'raw_rms1', 'demean_rms1']
rows = []
for variant in VARIANTS:
    for unit in ['window', 'file']:
        native = pd.read_csv(B / 'variants' / variant / f'native_{unit}_predictions.csv')
        native = native[native.label >= 0]
        for method, pred in [('native_bearing_conditional', 'pred4'), ('native_gear_as_errors', 'pred4_gear_errors')]:
            rows.append(dict(variant=variant, method=method, unit=unit, n=len(native), correct=int((native.label == native[pred]).sum()), accuracy=accuracy_score(native.label, native[pred]), macro_f1=f1_score(native.label, native[pred], labels=range(4), average='macro', zero_division=0)))
        pieces = []
        for fold in ['115200_to_460800', '460800_to_115200']:
            df = pd.read_csv(B / 'variants' / variant / 'heads' / fold / f'{unit}_predictions.csv')
            df = df[df.split == 'test'].copy()
            df['fold'] = fold
            pieces.append(df)
        joined = pd.concat(pieces, ignore_index=True)
        keys = ['file', 'window_row'] if unit == 'window' else ['file']
        assert not joined.duplicated(keys).any()
        joined.to_csv(B / 'variants' / variant / f'new_head_pooled_test_{unit}_predictions.csv', index=False)
        rows.append(dict(variant=variant, method='new_contact_only_linear_head', unit=unit, n=len(joined), correct=int((joined.label == joined.pred4).sum()), accuracy=accuracy_score(joined.label, joined.pred4), macro_f1=f1_score(joined.label, joined.pred4, labels=range(4), average='macro', zero_division=0)))
        (B / 'variants' / variant / f'new_head_pooled_test_{unit}_confusion.json').write_text(json.dumps({'labels': ['normal', 'inner', 'outer', 'ball'], 'matrix': confusion_matrix(joined.label, joined.pred4, labels=range(4)).tolist()}, indent=2) + '\n')
result = pd.DataFrame(rows)
result.to_csv(B / 'pooled_metrics.csv', index=False)
summary = []
for v in VARIANTS:
    g = result[result.variant == v]
    vals = {}
    for method, prefix in [('native_bearing_conditional', 'native'), ('new_contact_only_linear_head', 'new')]:
        for unit in ['window', 'file']:
            r = g[(g.method == method) & (g.unit == unit)].iloc[0]
            vals[prefix + '_' + unit] = f'{100 * r.accuracy:.2f}%（{r.correct}/{r.n}）'
    summary.append(f"| {v} | {vals['native_window']} | {vals['native_file']} | {vals['new_window']} | {vals['new_file']} |")
prov = json.loads((B / 'provenance.json').read_text())
verify = json.loads((B / 'verification.json').read_text())
native = pd.read_csv(B / 'variants/raw_rms001/native_window_predictions.csv')
main = native[native.label >= 0]
readme = f"# RotLLM 接触教师实测结果\n\n本轮实际下载并运行官方 SFN 预训练权重，完成接触输入、原生分类器和新四分类头验证。没有运行 Qwen 或用语言模型生成答案，也没有训练雷达网络。\n\n## 最重要的结果\n\n**直接使用官方 SFN 的原生分类部分，目前在我们的主数据上只得到 22.41% 的一秒片段正确率、25% 的整段录制正确率。冻结 SFN，只新训练一个四分类判断模块后，两个方向留出的 58 个片段中答对 49 个（84.48%），8 条录制全部答对。**\n\n整条录制的结果是平均各片段的预测概率后再判断，所以 8/8 不代表每秒都能判断正确。没有达到接触片段 98% 的目标。\n\n## 输入和模型来自哪里\n\n- 官方仓库：https://github.com/SIA-IDE/RotLLM\n- 固定 commit：`{prov['commit']}`。\n- `weights/encoder_weights.pth` 加载 SFN 卷积部分；`weights/proj_weights.pth` 的 `proj.0.*` 和 `proj.2.*` 分别对应官方 `SpecFoldNet.fc.0.*` 和 `fc.2.*`。这两个层输出 128 维特征与 15 类 logits。\n- 分类路径全部 {prov['strict_state_keys']} 个 state-dict 项 strict=True 加载，没有随机缺失参数。剩余 projection 层属于到 Qwen 词向量的映射，本轮没有执行，不将未执行部分称为已验收。\n- 权重使用 `torch.load(weights_only=True)`；模型 `eval()`，所有卷积、BN、原 15 类头均冻结。\n- 12 个接触文件共 88 个一秒窗口，每窗口原始 `raw_xyz` 为 4000×3。主四类是 8 条录制、58 个窗口；另外 `keep` 18 个窗口、`bigNormal` 12 个窗口仅作外部观察。\n\n## 预处理差异必须明确\n\n官方 `code/dataset_constructor/dct_process.py` 的 `process_single_data` 是 DCT→补到24000→RMS归一到1；同文件当前入口却调用 `multipy_data()`，把已保存数据再除100。官方仓库没有提供产生这些权重的输入尺度运行记录，因此不能假装这个歧义已解决。\n\n在第一次模型前向之前，`protocol_before_results.json` 已指定主分支 **raw_rms001：不去均值，最终RMS=0.01**。同时固定报告另外三个敏感性分支，不根据测试表现挑主分支。`raw_rms1` 对应构造函数RMS=1；`demean` 表示先对各轴一秒窗口去均值。\n\n输入处理为每轴 DCT-II、补零到24000、指定RMS，折成 3×8000 送 SFN。三轴各自前向，没有挑“最好轴”。原生分类平均三个轴的 softmax 概率；128维教师特征则平均三个轴的 hidden。二者不应混为同一操作，`probs15_from_mean_hidden` 另存了先平均特征再分类的结果。\n\n4 kHz 原信号只有前4000个 DCT 系数有观测数据，补零后后两条折叠频谱通道全部为零。补零统一输入形状，无法补回缺失的高频信息。它是适用性限制，尚不能单独判定为低分的原因。\n\n## 原生15类如何对应四类\n\n官方 SQLite 的 `label_note` 与 `code/fine_tune/qwen.py` 的 `label_ids` 相互核验：\n\n| 我们的标签 | 官方类别 |\n|---|---|\n| 0 正常 | 0 |\n| 1 内圈 | 1、2、3（不同严重度） |\n| 2 外圈 | 7、8、9（不同严重度） |\n| 3 滚动体 | 4、5、6（不同严重度） |\n| 任务范围外 | 10—14，各种齿轮故障 |\n\n主表使用“已知属于这四种轴承状态”的条件分类，将上述四组概率求和并归一化。另完整保留15类概率、15类最高概率类别和齿轮概率质量；也报告允许齿轮预测计错的口径，不静默丢掉齿轮错误。主数据15类最高概率落在齿轮的比例为 {(main.pred15 >= 10).mean():.2%}，齿轮概率总和的均值为 {main.gear_mass.mean():.8f}，两种四类评价正确率相同。外部 bigNormal 的齿轮概率更高，完整见逐文件表。\n\n## 四个预先声明的分支结果\n\n| 预处理 | 原生头：片段 | 原生头：录制 | 新四类头：留出片段 | 新四类头：留出录制 |\n|---|---:|---:|---:|---:|\n{chr(10).join(summary)}\n\n上表只代表已有开发录制上的跨录制检查，不能称为新采集盲测。新四类头的结果合并两个互补方向，每条测试录制只计一次。\n\n## 新四分类模块如何训练\n\n这不是修改原15类头，也没有微调encoder。做法是另加一个 `StandardScaler + multinomial LogisticRegression`，输入冻结SFN输出的128维特征。\n\n- 115200那次录制训练→460800那次测试：32个训练窗口、26个测试窗口；测试23/26=88.46%，录制4/4。\n- 反方向：26个训练窗口、32个测试窗口；测试26/32=81.25%，录制4/4。\n- `115200/460800` 是采集串口波特率与录制标记，不是转速。相同物理轴承分别两次录制，不能称为“未见轴承”验证。\n- 标准化只拟合本折接触训练窗口。固定参数 `C=1`、`class_weight=balanced`、`max_iter=2000`、`solver=lbfgs`，不使用测试标签选择参数、迭代次数或预处理。\n- 所有窗口先按录制划分，没有随机混合一条录制的片段进入两边。\n- 导出的 `weight4`、`bias4` 已把标准化合并回原128维特征，推理直接 `softmax(hidden @ weight4.T + bias4)`。无需再次 StandardScaler。\n- 两个新头对于另外一台正常电机的 `bigNormal` 均为0/2条判正常。`keep` 为训练类别外的故障，只报告预测，不将其硬塞进四类准确率。\n\n## 可以得到和不能得到的结论\n\n1. **已经验证第二个真实预训练模型能加载、提取特征，并可以接一个仅用接触训练数据建立的判断模块。** 未从头训练SFN，没有把随机初始化网络伪装成预训练教师。\n2. 原生头低分、冻结特征配新头变好，说明这些特征中存在本地四类可利用的信息；不等于原生模型已经无需适配，也不等于学生已获得丰富诊断知识。\n3. 录制只有8条，每折每类只有1条训练录制。依然无法检验“每次录制的细节知识”和“类别代表特征”的区别；这些训练/测试数据也已经被多轮开发查看。\n4. 新接触四分类模块与原15类路径分别保存。原模型所有参数、BN缓冲和权重文件逐项/哈希不变。本轮没有原LMR测试集，**没有测量原15类测试准确率的变化，不能据此写成通过了0.5个百分点统计回归验收。**\n5. 后续雷达应固定使用预先指定主分支和本折训练头。外圈雷达旧ROI错误与这次接触测试无关，但继续做雷达对接时仍应保留“暂定ROI”的限制。\n\n## 给后续雷达实验的接口\n\n- `teacher_features.npz`：主分支；`hidden_mean`=(88,128)、`hidden_axes`=(88,3,128)、`probs15_mean`=(88,15)、`probs4_mean`=(88,4)、`labels/label`、`bag_id`、`baud`、`state/states`、`file_names`、`window_rows`。字符串为固定Unicode数组，读取不需要pickle。\n- `metadata.csv`：与NPZ行严格对应。主标签0/1/2/3分别正常/内圈/外圈/滚动体，外部为-1。\n- `heads/115200_to_460800/head.npz` 和反向目录：`weight4`=(4,128)，`bias4`=(4,)，标准化已折入权重。\n- `native_head15.npz`：`weight15`=(15,128)，`bias15`=(15,)，并保存四类分组索引。\n- `variants/*`：四个预声明分支所有特征、原生预测、新头、逐窗与逐文件预测；所有敏感性结果保留。\n- `metrics.csv`：各折原始指标；`pooled_metrics.csv`：两个方向合并、每个测试录制只计一次的指标。\n- `provenance.json`、`verification.json`：commit、SHA256、权重加载范围、无修改检查、数值折回误差。\n\n运行环境：`/home/huangyating/miniconda3/envs/m2vllm/bin/python`，GPU0，CPU线程2。本轮前向和线性头训练脚本耗时约 {verify['elapsed_seconds']:.2f} 秒（不含仓库下载）。\n\n重跑：`OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/huangyating/miniconda3/envs/m2vllm/bin/python {B}/run_teacher.py`，然后运行同目录 `summarize_results.py`。\n"
(B / 'README_接触教师实测.md').write_text(readme)
print(result.to_string(index=False))
