# 保持原能力的接触式—毫米波对齐实验

主报告：[原能力保持与跨模态对齐实测报告.md](/home/huangyating/anomaly_detection/retention_alignment_20260922/原能力保持与跨模态对齐实测报告.md)。运行环境为 `conda m2vllm`；原权重、原数据和此前实验未覆盖。

最终保守 v0 的原测试十类/四类 Acc 为 98.8598%/99.0472%，相对原模型均下降 0.0081 个百分点；全部原训练十类/四类为 99.9930%/99.9942%。总体及九个数据源通过 ≤0.5 pp 回归；个别细分工况未通过。当前八个主录制不足以证明泛化，外圈错误 ROI 和 bigNormal 0/2 仍是未解决项。

## 产物导航

| 路径 | 用途 |
|---|---|
| `preprocessing/` | 六种预处理×三套冻结权重、逐轴及本地正常参考排查 |
| `source/` | 原 MBHM 同协议全量验证/测试、1174 训练 query 回放缓存及带宽实验 |
| `results/retained_head_v1/` | 原头、本地训练负对照、共享头两方向留出；不以 all_known 冒充留出 |
| `results/retained_head_conservative/` | 增加逐来源约束的两个保守 all_known 头 |
| `results/full_old_train_regression/` | 模型冻结后全量 85955 原训练 query 回归 |
| `results/retention_verification/` | 11 头重载、逐条件退化、四类统一模型与原 FCN 接口核验 |
| `radar/` | 198 次学生训练及消融，86 保存模型核验，共享空间图 |
| `radar/radar_only_conservative_v0/` | 实际仅雷达推理的 embedding、p10、p4 与几何有效性 |
| `results/llm_native_p10_combined/` | 152 行原生 p10 的真实 LLM 生成汇总、13 行冲突 |
| `inputs/radar_only/` | 从前轮复制并校验的无标签、无接触输入；包含训练录制，不作为独立测试集 |
| `figures/` | 主报告彩图及可导出的 PDF/SVG |
| `references/` | 官方论文/源码核查、版本与引用 |
| `artifact_manifest.json` | 本目录交付文件 SHA-256；不含自身与 __pycache__ |

## 直接运行仅雷达推理

先进入实验目录。下面命令会新建输出目录，不覆盖归档结果；`radar_repeat` 必须尚不存在或为空。

```bash
cd /home/huangyating/anomaly_detection/retention_alignment_20260922
/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python radar/infer_fixed_head.py \
  --model radar/stage_b_conservative_v0/models/four_class_provisional_roi/all_known/ce_feat_1_kd.pt \
  --features inputs/radar_only/features.npz \
  --metadata inputs/radar_only/metadata.csv \
  --output results/radar_repeat
```

输入特征第一维与 metadata 行数必须一致，`frame_shape` 为 128 维帧内谱形状。特征必须来自同一处理契约，不能把任意 FFT 或原始 IQ 直接传入此接口。`bag_id` 仅用于汇总；类别、状态、接触特征不进入推理。四类顺序固定为 normal/inner/outer/ball。

输出 `window_outputs.npz` 的 `embedding[N,128]`、`probability10[N,10]`、`probability4[N,4]`；文件汇总在 `file_probabilities.csv`。`diagnosis_status` 为无效或未验证几何时，概率只作排查，不作诊断。所有头、encoder、训练集尺度都在可信本地 `.pt` 检查点中。

当前保守模型由全部已知四类录制训练，因此上述十二文件重跑用于接口一致性验证；不能从中报告训练集外准确率。

## 接触式与雷达共用模型

```python
from pathlib import Path
import sys
import torch

root = Path('/home/huangyating/anomaly_detection/retention_alignment_20260922')
sys.path.insert(0, str(root / 'scripts'))
from shared_contact_model import SharedFourClassModel

model = SharedFourClassModel(
    root / 'results/retained_head_conservative/models/v0/all_known/retained_head.npz'
).eval()

# dcn_pairs: 已按训练契约处理的 FloatTensor [N,2,24000]，含查询和参考。
# h_radar: 对应上述共享头训练的雷达 encoder 输出 FloatTensor [N,128]。
with torch.inference_mode():
    # contact_logits4 = model(dcn_pairs)
    # radar_outputs = model.classify_embedding(h_radar)
    pass
```

公开 forward 返回四类 logits，内部保留十类概率用于原任务回归和原 linear3。不能将其他 teacher、其他预处理版本或任意 128 维特征混接到当前头。

多参考/多轴聚合须明确：原 MBHM 基线逐参考 softmax 后平均概率，调用 `contact_views(..., pool='probability')`；本地 teacher 先平均参考/轴 hidden，再分类，使用 `pool='hidden'`。这两种运算不可交换，不能不说明便混用指标。

## 重跑主要训练与语言推理

以下均在本目录、`m2vllm` 下运行，使用新的输出目录。复现实验仍依赖当前机器上的 BearLLM 仓库、原权重及前轮数据路径；压缩包不包含完整 MBHM、原始 IQ 或基础 LLM。

```bash
# 36 次共享头训练，含原域遗忘负对照；保持现有结果不动。
conda run --no-capture-output -n m2vllm python scripts/train_retained_head.py \
  --output results/retained_head_repeat

# 使用已归档的 v1 逐折共享头，重跑主预处理三种雷达损失、三个种子。
conda run --no-capture-output -n m2vllm python radar/train_shared_head.py \
  --contact-dir preprocessing/variants/v0/retrained_fcn \
  --head-root results/retained_head_v1/models/v0 \
  --output radar/stage_b_repeat

# 保守 all_known 对应的完整雷达训练；不是新本地留出实验。
conda run --no-capture-output -n m2vllm python radar/train_shared_head.py \
  --contact-dir preprocessing/variants/v0/retrained_fcn \
  --head-root results/retained_head_conservative/models/v0 \
  --only-all-known --seeds 42 --methods ce_feat_1_kd \
  --output radar/conservative_repeat

# 真正运行原 p10 → 已训练 linear3/LoRA → Qwen，不做人为严重度展开。
conda run --no-capture-output -n m2vllm python scripts/run_p10_llm.py \
  --inputs radar/llm_original_p10_inputs_plus_conservative.csv \
  --no-controls --device cuda:0 --output results/llm_repeat
```

保守头训练 `scripts/refine_retention.py` 使用固定输出路径且拒绝覆盖；重跑应复制本实验的脚本、缓存和预处理到新实验目录，保留归档结果，勿直接删除结果后原地重跑。其 `ROOT` 随脚本所在目录计算；仍须可访问前轮实验依赖。全旧训练集评估、验证及绘图脚本也采用本目录固定产物路径，应在新副本中运行。

输入/环境依赖：

- `/home/huangyating/BearLLM`：原 FCN 定义。
- `/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn`：本次原模型。
- `/media/nas_users/huangyating/bearllm-runs/released-code-seed42/finetune/weights`：已训练语言适配资产。
- `/media/nas_users/huangyating/bearllm-assets`：MBHM、官方权重及基础语言模型。
- `/home/huangyating/anomaly_detection/four_class_chain_20260922`：前轮特征、提取脚本与对照。
- `/home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results`：本次原处理 NPZ。

阶段目录保留生成时的脚本快照、协议、逐文件预测、概率和验证文件；记录里的绝对路径是本次实际执行路径。`environment_versions.json`、`final_integrity.json`、manifest 可用于核对，不代表把完整依赖和原始数据一并发布。
