# MBHM 全量验证与 BearLLM 开源代码复现

本次已按用户确认的**当前开源仓库配置**完成训练和全量验证：Qwen2.5-1.5B-Instruct、FCN 最多 50 轮（第 19 轮触发源码学习率早停）、LoRA 完整 50 轮。论文原始模型与划分存在差别，详见 `outputs/protocol_audit.md`。

全部完成性门槛已通过，见 [实测结果](outputs/full/RESULTS_zh.md) 和 [完成检查](outputs/full/completion_check.log)。逐条预测核验确认官方和新模型各覆盖全部 135,516 条振动数据、12,279 条测试查询生成及全部 600 条公开语料，见 `outputs/full/prediction_coverage.json`。训练曲线提供 [PNG](outputs/full/training_curves.png) 与 [PDF](outputs/full/training_curves.pdf)。

**重训未达到官方权重的测试生成分数**：官方 97.7034%；重训并保存训练末期 BatchNorm 状态为 93.7780%；同一次重训沿用原仓库保存方式（微调前 BatchNorm）为 95.7570%。这些是重建测试查询集上的实测值，不是论文指标复现声明。

## 环境和资源

- 独立 Conda 环境：`m2vllm`，路径 `/home/huangyating/miniconda3/envs/m2vllm`。
- 环境配置：`environment-m2vllm.yml`，不使用系统 site-packages 或其他项目的 Python 包。
- 已安装依赖锁文件：`requirements-m2vllm.lock.txt`；Conda 基础包精确清单：`outputs/full/conda-explicit.txt`。环境自检已通过包来源隔离、`pip check`、两卡 BF16 运算及真实 FCN 前向/反向/优化步骤。
- 当前 Blackwell GPU 使用独立安装的官方 PyTorch 2.11.0 + CUDA 12.8，取代上游 torch 2.6.0 的版本约束；这是硬件兼容差异，实际版本与检查记录保存在 `outputs/full/environment_ready.json`。
- 完整数据：`/media/nas_users/huangyating/bearllm-assets/mbhm_dataset`。
- 官方权重：`/media/nas_users/huangyating/bearllm-assets/bearllm_weights`。
- 重新训练和评测产物：`/media/nas_users/huangyating/bearllm-runs/released-code-seed42`。
- 日志：仓库 `outputs/full`。GPU 0 用于重新训练，GPU 1 用于官方权重评测。

```bash
source /home/huangyating/miniconda3/etc/profile.d/conda.sh
conda activate m2vllm
cd /home/huangyating/BearLLM
python run_demo.py
```

也可不激活环境，使用 `./scripts/run_demo.sh` 或 `./scripts/run_m2vllm.sh run_demo.py`。启动器清理外部 Python 路径，启用模型离线加载。

在 `m2vllm` 中已重新运行完整官方 Demo：退出码 0，耗时 12.350 秒（含模型加载），生成类别为 Fault-Free。日志与响应见 `outputs/full/demo_m2vllm.log`、`outputs/full/demo_response_m2vllm.txt`；该示例没有独立真实标签。批量评测与原始入口的两条不同信号测试中，适配器 logits、完整提示嵌入最大差值均为 0，贪心生成 token 完全一致，见 `outputs/full/evaluation_equivalence.json`。

新训练权重的完整 Demo 也已通过：退出码 0，耗时 10.486 秒，生成 Fault-Free；见 `outputs/full/demo_trained.log` 和 `outputs/full/demo_response_trained.txt`。

在其他机器重建环境可执行 `conda env create -f environment-m2vllm.yml`；需要本次安装的精确 Python 依赖时，再用环境内的 `python -m pip install -r requirements-m2vllm.lock.txt`。首次运行仍需下载下述数据及模型，并配置 `.env`。

首次配置可将仓库 `.env.example` 复制为 `.env`，把 `DATA_DIR` 设为资源根目录，例如本机的 `/media/nas_users/huangyating/bearllm-assets`。保留模板中的 `DESCRIPTION_LEN=5`、`LLM_HIDDEN_SIZE=1536` 和 `SIGNAL_TOKEN_ID=151925`。

## 数据与评测范围

固定数据版本 `78cd9b8b6b65cd43eebf4879b060d783d5f7dcfe`，共有 135,516 条振动信号、1,043 个 condition、9 个数据源。HDF5 形状 `(135516, 24000)`，float32；已经做过 DCN，训练和评测不重复预处理。公开语料只有 600 条（4 个任务各 150 条），不能当作论文所述 542,064 对完整语料。

源码要求同 condition 的正常参考信号，满足要求的查询为 122,792 条，12,724 条无法按原方法配对。完整数据检查覆盖全部 135,516 条；模型评测分为主实验与补充实验：

| 数据源 | 全部信号 | 原方法可配对 | 缺少同工况正常参考 |
|---|---:|---:|---:|
| CWRU | 3,932 | 70 | 3,862 |
| DIRG | 7,140 | 1,020 | 6,120 |
| HIT | 9,648 | 9,648 | 0 |
| IMS | 46,480 | 46,480 | 0 |
| JUST | 43,986 | 43,986 | 0 |
| MFPT | 78 | 36 | 42 |
| NCEPU | 3,600 | 900 | 2,700 |
| PU | 7,316 | 7,316 | 0 |
| XJTU | 13,336 | 13,336 | 0 |
| 总计 | 135,516 | 122,792 | 12,724 |

主实验沿用源码的三次参考抽样，合计 368,376 对。额外 12,724 条查询从**同一数据源**的正常样本中抽取三次参考，合计 38,172 对；这组参考未保证工况相同，单独报告，不计作原方法/论文可比结果。两部分合起来覆盖所有公开振动样本。

尤其要注意，原方法可配对的 CWRU、DIRG、NCEPU 子集仅含正常样本，不能用这些子集的高准确率证明其故障识别能力。MFPT 可配对子集包含 18 条正常样本及 18 条类别 8 故障样本。

## 训练与数据划分

固定 seed=42，按源码 SQL 顺序，对有正常参考的查询按序号模 10 分配 7:2:1，再对每个查询抽取三条参考。唯一查询 train/val/test = 85,955 / 24,558 / 12,279；配对后 = 257,865 / 73,674 / 36,837。

正常参考池包括所有划分，因而存在跨划分参考重用。这是源码协议的特性，不是严格无泄漏基准。所有 600 条已发布语料查询都落在该训练查询集合中。对这 600 条生成文本的评测是**训练语料回放**；另外对完整测试查询集合生成分类回答，分别统计。

FCN 训练保留 batch=1024、AdamW 学习率 1e-4 和每 batch 更新 ReduceLROnPlateau 的源码设置。主导出 `fcn` 沿用源码“验证 loss 或 accuracy 任一改善就保存”的规则，验证 loss 也按源码取 batch 平均；后续语言适配器由该导出初始化。另保存最后一轮和验证准确率最佳 checkpoint，在完整测试配对集上分别报告，方便解释检查点选择的影响。

正式预训练的 train/val/test DataLoader 都保留源码的 `shuffle=True`，不启用 persistent workers。开发阶段的调试训练独立归档，正式结果从零开始，不混用调试检查点。

LoRA 保留 r=4、alpha=32、dropout=0.1、all-linear、batch=1、梯度累积 4、学习率 1e-4、cosine、50 轮。冻结参数的特征编码器仍按源码 model.train() 更新 BatchNorm 缓冲；导出时额外保存这些实际缓冲，使重新加载后的推理和训练完成时模型一致。初始化适配器与最终导出权重分开保存，不覆盖官方权重。

微调直接使用原生 `transformers.Trainer`，保留其梯度累积损失计算。50 轮共 7,500 次参数更新，每轮 600 个语料 ID 各使用一次；最终导出的 398 个 LoRA 张量与最后检查点一致，72 个冻结基础参数张量未变，BatchNorm 的计数与统计正确保存。22 项导出校验详见 `outputs/full/training_export_verification.json`。

为核对原仓库只保存 PEFT、保留初始振动适配器文件的行为，另导出 `source_export_weights` 并在全部 12,279 条测试查询上做对照。它与主权重的 LoRA、配置和非 BatchNorm 参数一致，仅恢复初始 BatchNorm 状态；不重新训练或按测试分数选择结果。详见 [保存方式对照](outputs/full/source_export_diagnostic.md)。

## 命令

```bash
# Qwen、官方 BearLLM 权重及示例（固定版本并校验摘要）
python scripts/download_demo_assets.py \
  --data-dir /media/nas_users/huangyating/bearllm-assets \
  --endpoint https://hf-mirror.com --connections 16 --no-proxy

# 全数据下载及校验（包含 HDF5，公开文件固定 revision）
python scripts/download_mbhm.py \
  --data-dir /media/nas_users/huangyating/bearllm-assets/mbhm_dataset \
  --endpoint https://hf-mirror.com --no-proxy --connections 32

# FCN 预训练
CUDA_VISIBLE_DEVICES=0 ./scripts/run_m2vllm.sh scripts/train_bearllm.py pretrain \
  --output-dir /media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain

# 使用新训练的 FCN 初始化并微调语言模型
CUDA_VISIBLE_DEVICES=0 ./scripts/run_m2vllm.sh scripts/train_bearllm.py finetune \
  --fcn-dir /media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn \
  --output-dir /media/nas_users/huangyating/bearllm-runs/released-code-seed42/finetune

# 官方权重：原协议所有配对及全部 600 条语料
CUDA_VISIBLE_DEVICES=1 ./scripts/run_m2vllm.sh scripts/evaluate_mbhm.py \
  --stage all --corpus-batch-size 16 \
  --output /media/nas_users/huangyating/bearllm-runs/released-code-seed42/official_evaluation

# 官方权重：原协议排除样本的补充实验
CUDA_VISIBLE_DEVICES=1 ./scripts/run_m2vllm.sh scripts/evaluate_mbhm.py \
  --stage vibration --reference-policy excluded-same-source-fallback \
  --output /media/nas_users/huangyating/bearllm-runs/released-code-seed42/official_fallback

# 完整测试集查询的自然语言分类，单次固定参考，贪心生成
CUDA_VISIBLE_DEVICES=1 ./scripts/run_m2vllm.sh scripts/evaluate_mbhm.py \
  --stage heldout-generation --split test --corpus-batch-size 32 --max-new-tokens 64 \
  --output /media/nas_users/huangyating/bearllm-runs/released-code-seed42/official_heldout
```

重新训练模型评测使用同一脚本，增加 `--checkpoint /media/nas_users/huangyating/bearllm-runs/released-code-seed42/finetune/weights`，并使用新的输出目录。训练中断后，在原命令末尾加 `--resume`；评测使用相同参数、同一输出目录会自动接续 JSONL。已有 checkpoint 禁止无意覆盖，变更配置须换目录。

重新训练的权重也可直接运行官方 Demo：

```bash
./scripts/run_demo.sh \
  --checkpoint /media/nas_users/huangyating/bearllm-runs/released-code-seed42/finetune/weights \
  --seed 42 --output outputs/full/demo_response_trained.txt
```

原仓库保存方式的对照权重位于 `/media/nas_users/huangyating/bearllm-runs/released-code-seed42/source_export_weights`，可以通过相同的 `--checkpoint` 参数加载。该目录对应单独的保存方式对照；主实验目录 `trained_*` 均使用保存训练末期 BatchNorm 的 `finetune/weights`。

所有阶段结束后核验逐条预测覆盖并生成汇总和训练曲线：

```bash
./scripts/run_m2vllm.sh scripts/verify_prediction_coverage.py
./scripts/run_m2vllm.sh scripts/summarize_reproduction.py --require-complete
./scripts/run_m2vllm.sh scripts/plot_training.py
```

`--require-complete` 在训练、数量、配置、指标或权重初始化链核验未完成时会返回失败，避免把中间结果视为完整复现。覆盖校验另逐行检查查询、参考、标签和预测值，确认主实验与补充实验的查询并集等于全数据集。

## 指标解释

振动指标来自适配器内部十分类 logits，分别报告同一导出检查点不叠加 LoRA 增量的基础适配器，以及叠加 LoRA 后的适配器。新训练模型导出的基础适配器已包含微调时更新的 BatchNorm 缓冲，因此前者不等同于微调前的初始模型。它们不同于语言模型生成文本的分类准确率。语言任务使用真实贪心生成，无法解析的回答计作错误；参考回答 NLL/困惑度单独报告，不能当作生成正确率。

文本比较包含规范化精确匹配、unigram F1、词级 ROUGE-L；它们仅度量文字重合，不证明维护建议或风险分析的事实正确性。此次开源仓库复现不将这些分数冒充论文 BERTScore 或全部消融实验的结果。

公开语料 ID 394（维护）与 ID 505（风险）的参考回答本身未提供实际故障诊断。官方模型可以精确复现这些无实质回答，说明高文字重合不能替代内容审查。样本仍保留在全量评测中，不依据结果事后删除。
