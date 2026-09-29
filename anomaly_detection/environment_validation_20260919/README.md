# room/narrow 毫米波环境处理与实际验证

**已完成，使用 conda `m2vllm`，日期 2026-09-19。** 输入为 `/home/huangyating/anomaly_detection/data9.18` 的 42 个 bin（10.556 GB）；得到 652 个不重叠的名义 2 秒窗口。原始文件未修改。上传监测已结束，目前没有本实验的后台监测或训练任务。

先读 **[实验结论与详细处理方案](实验结论.md)**，再读 [全部主实验](环境泛化验证报告.md) 和 [距离外推与主频对照](补充验证.md)。当前数据只有正常轴承的旋转/停止记录，结论限于状态和转频信息，不能代表故障诊断、跨模态迁移或跨电机结构泛化。

多单元 IQ 谱形状的三转速文件级 macro-F1 为 room→narrow 100.0%、narrow→room 93.3%；相位基线为 92.2%、87.5%。**但同一处理频谱上的简单主频分类两个方向都是 100%，且环境与未见距离同时变化时谱形状模型有明显失败。** 本实验没有证明已经消除环境影响。off 噪声校准也没有带来稳定额外收益。

## 可复现运行

本轮所有实际数据处理、评估和补充实验均在 `m2vllm` 中运行。仅使用 CPU；下列线程限制避免占用过多计算资源。

```bash
cd /home/huangyating/anomaly_detection/environment_validation_20260919
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
conda run --no-capture-output -n m2vllm python scripts/radar_environment.py self-check
conda run --no-capture-output -n m2vllm python scripts/radar_environment.py extract --conjugate
conda run --no-capture-output -n m2vllm python scripts/evaluate_environment.py
conda run --no-capture-output -n m2vllm python scripts/supplementary_checks.py
```

如果 shell 尚未初始化 conda，可将 `conda` 替换为 `/home/huangyating/miniconda3/bin/conda`。已有处理特征时，可以直接执行后两条命令。解析器的数据路径在 `scripts/radar_environment.py` 的 `DATA` 常量中。

本批次的完整性与模型重载复核可运行 `conda run --no-capture-output -n m2vllm python scripts/finalize_artifacts.py`。该脚本特意核对这批 42 个文件的已保存快照，不是任意新数据集的通用验收程序。`requirements.txt` 记录已有环境的实际依赖版本，本次没有安装或升级依赖。

用已导出的模型复现目标环境预测，无须重新拟合：

```bash
conda run --no-capture-output -n m2vllm python scripts/predict_features.py \
  --model models/P0_three_speed__room_to_narrow__iq_multi_shape.npz \
  --environment narrow --exclude-off \
  --output results/reloaded_model_predictions.csv
```

这些模型是本实验的三转速/四状态线性分类器，不是故障模型，也不是 BearLLM/RotLLM。

## 输出说明

| 路径 | 内容 |
|---|---|
| `results/features.npz` | 5 种前端特征、两种线性功率谱、5–800 Hz 的多单元谱；行顺序对应 windows.csv |
| `results/windows.csv` | 文件、环境、距离、评价标签、窗口起点、有效采样比例 |
| `results/feature_schema.json` | 数组维度、频带边界、数值含义 |
| `results/cross_environment_metrics.csv` | 30 行主实验结果，包含所有基线和 off 消融 |
| `results/file_predictions.csv` | 全部主实验逐文件预测及概率 |
| `results/source_model_selection.json` | 仅使用源环境的选参记录 |
| `results/supplementary_unseen_distance_*` | 环境与距离同时留出的 18 项结果和逐文件预测 |
| `results/dominant_peak_*`、`physical_peak_baseline.json` | 主频的逐窗口/逐文件值与简单分类对照 |
| `results/quality.json`、`calibration_audit.json` | 帧尾、距离峰、近满量程、off 参考质量 |
| `results/input_manifest.json`、`input_sha256.json` | 本次输入快照及 42 个原始文件的 SHA256 |
| `results/runtime_environment.json`、`final_verification.json` | 解释器、依赖版本、脚本哈希与最终验证 |
| `models/` | 四个 IQ 谱形状模型及其源环境训练文件清单 |
| `figures/` | 六组实测图，每组 PNG、SVG、PDF |

`scripts/watch_upload.py`、`results/watch.log`、`parser_prefix_audit.csv` 是上传阶段的历史工具/记录；不是当前运行状态或完整采集的额外证据。`watch_status.json` 已更新为完成。

## 适用配置和解释边界

`radar_original_assumed.cfg` 归档用户提供的配置；本轮没有向硬件发送配置。解析假设为 AWR1843/DCA1000、2I/2Q 数据、4RX、256 ADC、192 chirp/100 ms、500 μs/chirp。新配置不能混用当前时间轴。帧间缺少的 8 个采样时隙用 mask 表达，没有插值为实测数据。相邻三个补零 range-bin 互相关，不是三个独立距离分辨单元。

文件时间戳/大小在处理前后不变，并已计算 SHA256；这不替代源端哈希、DCA1000 丢包日志和采集配置核对。同次采集拆分的文件应归为同一 run；独立录制关系尚未确认，因此按文件报告，未把窗口数量当独立样本量。

主实验参数没有依据补充测试结果重新优化。主频对照使用已知 1000–3000 rpm 运行范围对应的固定 10–60 Hz 搜索带，并共用多单元 IQ 估谱前端；它检验复杂特征分类是否必要，不能单独证明前端每一步都必要或无效。

完整研究路线见 [SenSys 投稿与 14 天全链条验证](../../research_proposals/contact_radar_alignment_20260919/SenSys投稿与14天全链条验证.md)。
