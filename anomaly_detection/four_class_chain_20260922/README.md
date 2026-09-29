# 四类接触式—毫米波全链条验证

2026-09-22；conda `m2vllm`。已实际完成接触教师验证与适配、雷达特征、学生训练、仅雷达推理和真实 Qwen/BearLLM 生成。**这是实验版本：外圈/keep 雷达 ROI 错误尚需原 bin 修复，跨电机主链失败，不能直接部署为可靠诊断系统。**

- [全链条实测报告](全链条实测报告.md)：具体算法、损失、基线、消融、全部关键结果及限制。
- [下一轮算法与补采实验方案](下一轮算法与补采实验方案.md)：48 对核心采集、可靠对应、环境/结构泛化、14 天安排。
- [一条命令仅雷达推理](scripts/雷达独立推理.md)：已实跑；不需要接触式数据、标签或 state。
- [最终独立审查](audit/final_result_review.md)：指标重算、数据隔离、外部例外及语言检查。

## 保存的模型与输出

主学生：[四分类 all_known/distill_seed42.pt](results/chain_v1/models/four_class_provisional_roi/all_known/distill_seed42.pt)。这个模型用 8 个已知类录制拟合，适合接口演示与独立外部检查，不能用其训练文件计测试成绩。对应四类头在同目录 `contact_head.npz`。

真正留出的两个完整方向模型在 `0000/`、`1111/`；三类几何相容版本在 `three_class_geometry_compatible/{000,111,all_known}/`。保存了这六个条件下全部 8 种学生消融的 seed42 检查点，共 48 个。每个 checkpoint 自带 encoder、输入 scaler、固定/自有分类头和训练 bag 列表。

| 路径 | 内容 |
|---|---|
| `protocol.json` | 正式训练前固定的设置与用户已确认信息 |
| `contact/` | 原接触式 6 组推理、冻结特征、真实语言生成与一致性检查 |
| `radar/features.npz`、`radar/metadata.csv` | 614 行特征与行映射；主输入 `frame_shape[614,128]` |
| `results/chain_v1/metrics.csv` | 888 行文件指标；正式 200 步结果 |
| `results/chain_v1/method_summary.csv` | 四类/三类、全部 21 方法汇总 |
| `results/chain_v1/file_predictions.csv` | 每折/方法/seed 的逐文件概率 |
| `results/chain_v1/external_summary.csv` | bigNormal 正常接受数；四类全部 0/2，三类接触频谱有 2/2 例外 |
| `results/chain_v1/splits.json` | 26 个划分，含两个外部专用 all_known |
| `results/chain_v1/verification.json` | 48 个模型仅雷达重载检查、集合隔离 |
| `results/radar_only_chain_demo/predictions.csv` | 无标签独立命令的真实 12 文件输出；不是新性能测试 |
| `results/latency_benchmark.md` | CPU 学生微基准，明确不含采集、特征和 LLM |
| `period_normalization/` | 外部失败后的固定事后检查，保留归一化失败 |
| `figures/` | 四幅新增实测/系统彩图，PNG、SVG、PDF |
| `audit/` | 输入哈希、预处理、协议及训练结果独立核查 |

`results/smoke_5steps` 只用于开发连通性，不是正式结果；资料包不包含该临时目录。读取 CSV 时给 `fold` 指定字符串类型，避免把 `0000` 读成整数 0。

## 复现正式训练

当前训练读取本目录已冻结的接触与雷达特征，另外从原预处理目录读取接触频谱/统计基线。`--output` 只改变输出目录，不会切换数据集。新 ROI/新采集数据必须建立新版输入与协议，再训练，不能仅改变输出名便声称已使用新数据。

```bash
cd /home/huangyating/anomaly_detection/four_class_chain_20260922
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python scripts/run_chain.py --output results/chain_repeat --device cuda:1
```

拒绝非空输出目录；固定 200 步、3 个种子、四类和三类全部划分。脚本输入依赖及权重哈希记录在各级 manifest/runtime 中。`--quick` 只跑两个方向及 all_known 的 seed42，不能将其输出冒充完整正式协议。

接触特征重新导出时用新目录：

```bash
/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python contact/evaluate_contact.py --output contact_repeat --device cuda:0
```

雷达特征重新导出时同样用新目录：

```bash
/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python radar/extract_features.py \
  --source /home/huangyating/anomaly_detection/four_class_preprocessing_20260921/results \
  --output radar_repeat
```

以上命令示范复算；原输入只保留旧目标门，复算不会自动修复几何。

## 仅雷达分类，不加载语言模型

```bash
/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python scripts/infer_radar.py \
  --model results/chain_v1/models/four_class_provisional_roi/all_known/distill_seed42.pt \
  --features results/radar_only_demo_inputs/features.npz \
  --metadata results/radar_only_demo_inputs/metadata.csv \
  --output results/radar_only_repeat.csv
```

模型是本地生成的可信 checkpoint。雷达输入只需特征，元数据用于文件汇聚及几何状态；几何未核验时不默认放行。

## 仅雷达分类加实际 BearLLM/Qwen 生成

```bash
/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python scripts/predict_chain.py \
  --model results/chain_v1/models/four_class_provisional_roi/all_known/distill_seed42.pt \
  --features results/radar_only_demo_inputs/features.npz \
  --metadata results/radar_only_demo_inputs/metadata.csv \
  --output results/radar_only_chain_repeat --device cuda:0
```

这是从**预先提取的雷达特征**到分类/真实文本的独立命令，不是直接解析新 ADC bin 的命令。最终包含原始生成、结构化候选、冲突回退和 ROI 状态；并明确没有已校准未知故障检测。

重跑 110 条正式语言接口评估时：

```bash
/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python contact/run_llm_bridge.py --no-controls \
  --inputs results/chain_v1/llm_bridge_inputs.csv \
  --output contact/llm_bridge_repeat --device cuda:0
/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python contact/guard_llm_output.py --directory contact/llm_bridge_repeat
```

原 LLM 脚本本身允许写指定目录，务必使用上述新输出名；一条命令的 `predict_chain.py` 则主动拒绝非空输出。

## 依赖与资料包范围

本机已用 Python 3.12.14、PyTorch 2.11.0+cu128、NumPy 1.26.0、SciPy 1.15.2、scikit-learn 1.6.1、pandas 2.2.3、matplotlib 3.10.1 完成计算。完整本轮包版本见 [environment_versions.json](environment_versions.json)。没有改动 conda 环境安装。

外部依赖仍在原位置：

- `/home/huangyating/anomaly_detection/four_class_preprocessing_20260921`：原预处理导出及解析脚本。
- `/home/huangyating/BearLLM`：原仓库代码。
- `/media/nas_users/huangyating/bearllm-runs/released-code-seed42`：已训练 FCN/最终 LoRA 权重。
- `/media/nas_users/huangyating/bearllm-assets`：官方适配器、Qwen 基座及相关资产。

资料包包含本轮新增代码、冻结特征、学生/线性头、协议、结果和图，不复制原始 ADC、原始接触数据、完整上游仓库或大模型基座。普通雷达分类可以使用包内已提取特征与学生；重新提取教师或生成语言需要上述本机依赖。脚本含明确本机路径，不宣称无需配置即可在任意机器运行。

[交付校验](artifact_verification.json) 记录文件链接与来源哈希检查；[文件清单](artifact_manifest.json) 记录资料包文件 SHA256。更早的研究方案与环境实验仍在原目录，最新四类结果以上述报告为准。
