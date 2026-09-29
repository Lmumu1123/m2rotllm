# 接触式教师模型与开源基线核验笔记

核验日期：2026-09-23。仅核验原论文、作者仓库、官方模型卡、源码及文件清单；未启动训练、未下载大型权重。下文“已公开权重”不等于已在本机完成 checkpoint 加载与准确率复现。

## 1. 建议优先级与研究问题

建议核心异构教师集合为 **BearLLM FCN、RotLLM SFN、原始时域 InceptionTime、TS2Vec、UniFault Tiny（取决于加载验收）**。预算紧时先做前三个加 TS2Vec。MOMENT 为通用时序预训练的额外压力测试；TF-C、TSLANet 可作为补充。不要只做 BearLLM 与 RotLLM：两者均来自同一团队，均依赖统一 DCT 频域输入，无法充分支持跨表示形式的结论。

必须区分三类模型：

1. 作者发布了接触振动预训练权重，可直接检验“已有模型再利用”：BearLLM、RotLLM；UniFault 有 Tiny 下载链接，尚待加载验收。
2. 作者发布算法和训练实现，需要我们用接触式训练数据建立教师：InceptionTime、TS2Vec、TF-C、TSLANet。这类验证方法兼容性，而不是现成轴承基础模型的零样本迁移。
3. 通用时序基础模型有预训练权重，但未自带我们的轴承四分类头：MOMENT。应先用接触训练集训练头、确认教师可诊断，再冻结用于毫米波迁移。

| 教师 | 表示及可提取接口 | 权重状态 | 代码许可证 | 推荐角色 |
|---|---|---|---|---|
| BearLLM FCN | DCT 查询+健康参考；128 维倒数层 | Hugging Face 有 vibration_adapter.pth，约 4.27 MB | GitHub 未见独立 LICENSE；HF 模型卡标 MIT，需区分代码与权重许可 | 当前主基线，固定既有实现 |
| RotLLM SFN | 单条 DCT 折叠输入；128 维倒数层 | 官方 GitHub 有 encoder_weights.pth，2,731,545 bytes | MIT | 必做，同领域异网络、无需健康参考 |
| InceptionTime | 原始时序，多尺度卷积+全局池化 | 未核验到现成通用轴承权重；需训练 | GPL-3.0 | 必做，原始时域与纯监督学习对照 |
| TS2Vec | 原始时序；逐时刻或整段 embedding，默认 320 维 | 官方提供自监督训练流程；未核验到轴承通用权重 | MIT | 必做或优先补充，非分类监督表示教师 |
| UniFault Tiny | 时序 patch Transformer；token 均值接分类头 | 官方 Tiny 链接可跳转至 HTTP 200 Dropbox；未逐个 checkpoint 验收 | MIT | 领域预训练 Transformer 的强补充 |
| MOMENT-Small | 时序 Transformer；embedding API，512 维隐藏表示 | HF 明确提供预训练 safetensors；37.9M 参数 | 代码和模型卡均 MIT | 通用基础模型压力测试 |
| TF-C | 时域/频域双分支、各 128 维投影 | 官方完整预训练/迁移代码与数据；未核验轴承通用权重 | 主仓库 MIT；vendored baseline 需单查 | 方法相关基线，防止“时频一致性”被当新概念 |
| TSLANet | 自适应频谱块+交互卷积 | 官方训练实现；未核验通用轴承权重 | MIT | 轻量频域归纳偏置补充 |

## 2. 各模型核验及接入要点

### 2.1 BearLLM

AAAI 2025；官方要求一秒待测振动和一秒健康参考；仓库 FCN 为查询、参考、残差三路卷积，再分类。`linear1→ReLU` 后可取 128 维特征，末层为十分类。权重页有 `vibration_adapter.pth` 和 LoRA 文件。[论文](https://arxiv.org/abs/2408.11281)、[作者仓库](https://github.com/SIA-IDE/BearLLM)、[FCN 源码](https://github.com/SIA-IDE/BearLLM/blob/main/models/FCN.py)、[官方权重](https://huggingface.co/SIA-IDE/BearLLM/tree/main)。

接入建议：继续以当前已验收的本地版本为基准，并记录 checkpoint、输入预处理、参考信号政策。对比教师时不能给 BearLLM 更多目标测试条件的健康数据。已有四分类头若经过目标接触数据微调，应称“经接触域适配的教师”，与原始开源权重分列。

### 2.2 RotLLM

论文发表于 Engineering Applications of Artificial Intelligence，DOI `10.1016/j.engappai.2025.112544`。官方 SFN 默认输入 `(1,24000)`，折为 3 个子谱，128 维隐藏层后输出 15 类；仓库含真实 encoder 权重文件。作者明确说明代码仍是研究草稿，完整 LMR 信号尚未在该仓库发布。[论文](https://doi.org/10.1016/j.engappai.2025.112544)、[仓库](https://github.com/SIA-IDE/RotLLM)、[SFN](https://github.com/SIA-IDE/RotLLM/blob/main/code/models/SFN.py)、[权重](https://github.com/SIA-IDE/RotLLM/blob/main/weights/encoder_weights.pth)。

接入建议：截取 `fc[0]→fc[1]` 后的 128 维，将后续 `fc[2]` 作为冻结头；先核对 15 类标签中轴承四类的映射，不能假定前四项就是我们的四类。需要注意脚本先进行能量归一化，另有整体 `/100` 后处理；直接沿用 BearLLM 的幅值流程前必须检查 checkpoint 训练时的版本。[DCT 处理源码](https://github.com/SIA-IDE/RotLLM/blob/main/code/dataset_constructor/dct_process.py)。

对我们最有价值的失败定位：若 BearLLM 可迁移而 SFN 不行，先排查健康参考、幅值标定和分类词表，再讨论架构差异；若这些都一致后仍存在差异，才构成教师表示与毫米波可观测信息之间的适配问题。

### 2.3 InceptionTime / WDCNN

InceptionTime 官方仓库提供多尺度一维卷积分类实现；源码有全局平均池化，可提取分类前表示，模型可按输入长度构建。原方法包含多模型集成，因此要明确报告使用单个 Inception 网络还是完整 ensemble。[作者仓库](https://github.com/hfawaz/InceptionTime)、[模型源码](https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py)、[论文](https://arxiv.org/abs/1909.04939)。

接入建议：在 4 kHz 的 4000 点原始时域上训练接触教师；这是不依赖 DCT 或健康参考的对照。限定相同接触训练录制、相同四类标签、相近优化预算。若论文预算不足，使用一个单网络并清楚命名；不必为这个验证重新训练一个更大的基础模型。

WDCNN 可作更便宜替代，但必须准确写实现来源：RotLLM 仓库内有作者用于对比的 WDCNN，**不是已经核实的 WDCNN 原作者实现**；该文件展示 12000 点输入、固定展平维度和 15 类输出。不能直接输入 4000 点且声称原生兼容；需改为自适应池化/匹配维度并从头训练，报告为适配实现。[对比实现](https://github.com/SIA-IDE/RotLLM/blob/main/code/models/WDCNN.py)。

### 2.4 TS2Vec

AAAI 2022 官方实现；输入为 `[样本,时间点,通道]`，提供逐时刻表示和 `encoding_window='full_series'` 的整段表示，默认 320 维。仓库给出从头自监督训练的代码，不能把它称为已发布通用轴承预训练权重。[论文](https://arxiv.org/abs/2106.10466)、[仓库](https://github.com/zhihanyue/ts2vec)。

接入建议：只用接触训练录制预训练 TS2Vec，再用相同接触训练标签建立一个线性四分类头。冻结 encoder+head，训练雷达适配器输出 320 维或通过教师专属投影映射。这样可检验方法是否依赖 BearLLM 的分类监督和频谱归纳偏置。预训练期间使用测试录制，即使不用标签，也应明确列为 transductive 协议，不能混入主结果。

### 2.5 UniFault

截至核验日，作者仓库有 Tiny 下载链接、Transformer 和微调代码。Tiny 配置为维度 128、4 层、4 头，patch 默认 64；分类头由目标标签数构造，encoder 返回 token，预测前对 token 求均值。[作者仓库](https://github.com/emadeldeen24/UniFault)、[模型](https://github.com/emadeldeen24/UniFault/blob/main/model/model.py)、[微调入口](https://github.com/emadeldeen24/UniFault/blob/main/fine_tune.py)。

论文 v2 写 0.1 秒窗口并统一至 1024 点；代码的预处理 README 给通用 1024 点流程。**“0.1 秒”和“采样率”不是同一概念，不能据此认定所有输入真实有效采样率为 10.24 kHz。** 我们 4 kHz 的 0.1 秒只有 400 个原始样本，插值到 1024 点不会补充带宽。论文不同段落对训练/评估数据集名有不一致，不能未经审计就照搬其零样本表述。[论文 v2](https://arxiv.org/html/2504.01373v2)、[预处理说明](https://github.com/emadeldeen24/UniFault/blob/main/data_preprocessing/Preprocessing_README.md)。

验收重点：Tiny 下载链接跳转至 Dropbox 文件夹并返回 HTTP 200，只证明链接存活；仍需列出文件、校验 hash、实际加载。官方采用按名称和形状过滤后 `strict=False` 加载，改变序列长度时可能丢弃位置编码而未报错。必须报告参数加载覆盖率、遗漏参数、输入时长、有效带宽，不能将部分随机初始化模型冒充完整预训练教师。论文 Lite/Base 与仓库 Tiny/Small/Base 命名也应据 checkpoint 实际配置对应。

### 2.6 MOMENT

ICML 2024 通用时序基础模型，官方支持 embedding 与 classification 模式。Small 权重已发布，约 37.9M 参数；配置输入长度 512、patch 8、隐藏维度 512，模型卡标 MIT。[论文](https://arxiv.org/abs/2402.03885)、[仓库](https://github.com/moment-timeseries-foundation-model/moment)、[Small 模型卡](https://huggingface.co/AutonLab/MOMENT-1-small)、[配置](https://huggingface.co/AutonLab/MOMENT-1-small/blob/main/config.json)。

接入建议：512 点可表示 4 kHz 下 128 ms 的片段，再在一秒 bag 中聚合多个片段。另一方案是把一秒抗混叠降采样到 512 点，但这会降低有效带宽，必须单列处理条件；禁止把两种方案混称输入标准化。先做接触端线性探针，若教师本身不能识别目标故障，它不适合作为迁移成功率比较的等价教师，但可以作为“教师适用性”边界实验。

### 2.7 TF-C、TSLANet 与不建议当前押注的项目

TF-C（NeurIPS 2022）官方代码提供时间与频率双分支一致性预训练，包含 Paderborn FD-A/FD-B 跨工况数据。它是**同一传感器信号两种表示的一致性学习**，不能直接等同于加速度与雷达跨传感器迁移。[论文入口与官方仓库](https://github.com/mims-harvard/TFC-pretraining)。我们应将其改为明确命名的跨模态 baseline，不能把改写版本当原论文原生方法。仓库 FD 配置固定 5120 长度，移植必须确认张量轴与 batch 独立性。

TSLANet（ICML 2024）将自适应频谱块与卷积结合，提供官方代码和自监督流程，但本次未核验到可直接用于轴承四类的通用 checkpoint。[论文](https://proceedings.mlr.press/v235/eldele24a.html)、[仓库](https://github.com/emadeldeen24/TSLANet)。适合作为轻量时频网络补充，优先级低于核心四教师。

BearingFM 论文提供转速归一化与机理数据增强的思路，但本次在作者公开 PDF 与检索中未核验到对应官方代码+预训练权重。只能作为方法相关文献，不能承诺可立即运行；也不能写成确定未开源。[作者网页 PDF](https://cyang3993.github.io/files/2024_bearing_fault_diag.pdf)。仅有 ImageNet 预训练 CNN 的轴承论文也不能算“接触式故障知识预训练模型”。

## 3. 怎样公平比较不同教师

下面是针对本项目的建议协议，不是上述论文已验证的结论。

### 3.1 两个互补实验轨道

**轨道 A：已有模型再利用。** 保留 BearLLM、RotLLM、UniFault 的发布权重与原生预处理，各自先进行接触端验收；必要适配只允许使用接触训练录制。每个教师保留独立原生分类头，雷达推理必须经过这个被冻结的头。该轨道最贴近“接触模型如何让雷达用”的目标，但教师预训练数据量不同，不可把差异全部归因于骨干。

**轨道 B：受控方法通用性。** 在相同公开接触训练集、相同 4 kHz 带宽、相同源任务和标签下训练时域 CNN、TS2Vec 等教师。随后采用相同雷达训练录制、相同适配器参数预算、相同验证搜索次数。该轨道分离教师数据量和预处理的影响。

教师至少记录：接触测试 macro-F1、balanced accuracy、准确率、校准误差；是否经过目标接触训练适配；采样率、窗口时长、带宽；embedding 维度；checkpoint 加载覆盖率；标签映射；原生头是否冻结。

### 3.2 不同维度不是通用性的核心困难

为每个教师定义 `z_t=E_t(x_c)∈R^{d_t}` 和冻结的 `h_t`。雷达共享骨干输出 `r=E_r(x_r)`，通过轻量 `P_t` 输出 `d_t` 维。分别训练 `E_r,P_t` 能证明“同一训练方法适用于多种教师”；**不能证明同一个已经训练好的雷达 encoder 无需训练即可接入任何模型**。

更强协议：先训练单个共享雷达 encoder 并冻结；对未参与雷达训练的新教师，只用接触数据拟合旧教师空间到新教师空间的线性/ridge 变换，不使用新雷达标签；比较新教师冻结头能否直接消费雷达输出。这可测试表示的可复用性，但成功仅对测试过的模型族、任务与数据条件成立。

若每换教师都重新训练整个雷达 encoder，应命名为“跨教师适用的迁移框架”；若只需轻量投影校准，可称“共享雷达表示的轻量教师适配”；若未见教师完全不需目标雷达训练，则单独报告“未见教师迁移”。三者不可混称“通用 encoder”。

### 3.3 从失败定位有意义的算法修改

| 观察到的失败 | 优先检验 | 有意义的修改 |
|---|---|---|
| 接触教师自己低分 | 预处理、带宽、参考、标签映射、接触信号质量 | 接触端最小适配；不要归因于雷达 encoder |
| 接触强、雷达 CE 强，特征 MSE 迁移差 | 特征缩放、维度、不可观测成分、类内几何 | 教师空间标准化；关系蒸馏；可预测子空间；避免一味增大 MSE |
| 训练集对齐好，新录制差 | 录制 ID、转速、环境等捷径 | 留录制测试、同类不同录制检索、环境扰动，不必扩展到跨机械结构 |
| 新分类头可用，原冻结头不可用 | 只学到类别可分性，未获得原坐标接口 | 保留冻结原生头约束，校准轻量投影，报告兼容性差距 |
| 只有重训雷达 encoder 才能换教师 | 学到教师特定坐标 | 共享骨干+教师专属投影；关系目标；未见教师轻量适配实验 |

## 4. 最小可执行教师矩阵

第一轮只做：BearLLM、RotLLM、时域 InceptionTime、TS2Vec；每种教师跑同一组 CE-only、logit-KD、feature-MSE、关系蒸馏、当前 bag 方法。先用一套固定超参数完成可行性矩阵，再对所有方法给予相同验证搜索预算。

每行至少包含：接触教师分数、雷达直接监督分数、雷达经冻结教师头分数、与 CE-only 的差值、是否用了雷达标签、训练数据规模、均值与以独立录制为单位的区间。当前 8 段主录制只能做开发与初步检验，不能因为窗口数多就把跨教师结论写成已充分证实。

若需第五个优先加 UniFault Tiny；如权重验收或输入定义阻塞，MOMENT-Small 可立即替换成“通用时序预训练”探索项，但务必先确认接触端能力。
