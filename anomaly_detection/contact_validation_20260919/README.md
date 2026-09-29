# 4 kHz 接触式数据与 BearLLM 接入验证

本轮已使用 conda `m2vllm` 检查用户的三份 XYZ 导出、核对 BearLLM 已完成的复现结果，实现严格预处理和三轴 embedding 接口，并实际运行冻结教师的 MBHM 带宽消融。

**先读 [接触式预处理与 embedding 验证报告](接触式预处理与embedding验证报告.md)。** 当前数据没有完整、可信的 1 秒连续窗口，`strict_1s_dcn.npz` 的样本维为 0；没有将碎片拼成有效教师样本，也没有对这三份记录宣称诊断准确率。MBHM 带宽消融和合成接口测试均明确单列。

## 运行

```bash
cd /home/huangyating/BearLLM
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
conda run --no-capture-output -n m2vllm python /home/huangyating/anomaly_detection/contact_validation_20260919/scripts/contact_pipeline.py
conda run --no-capture-output -n m2vllm python /home/huangyating/anomaly_detection/contact_validation_20260919/scripts/bandwidth_audit.py
conda run --no-capture-output -n m2vllm python /home/huangyating/anomaly_detection/contact_validation_20260919/scripts/verify_pipeline.py
```

`conda` 不在 PATH 时使用 `/home/huangyating/miniconda3/bin/conda`。没有安装或升级依赖。原始接触式文件、BearLLM 源码和既有权重没有修改。带宽消融只作前向推理，无参数训练；BatchNorm 缓冲在运行前后核验未变。

## 已实现的入口

- `scripts/contact_pipeline.py`：TXT 的 GB18030 接收标记解析、DAT 一致性校验、包内序号检查、保守剔除接收边界记录、严格 1 秒取段、逐轴 DCN、片段质量统计和图。
- `scripts/teacher_bridge.py`：读取重训 FCN、官方最终适配器、重训最终适配器；后两者正确加载其 LoRA 分类层，不加载整个 LLM。
- `scripts/encode_contact_pair.py`：给定独立同工况 query/reference，保存 `[3,128,47]` 特征图、`[3,128]` 隐藏向量、`[3,10]` 分类输出，三行依次为 XYZ。
- `scripts/bandwidth_audit.py`：固定 MBHM 测试查询子集上比较原带宽、低于 2 kHz、低于 800 Hz；选择种子与所有查询 ID 已保存。
- `scripts/verify_pipeline.py`：频率坐标、与数据集 DCN 一致性、无效输入拒绝、合成接口和冻结 BN 检查。

当前 `contact_pipeline.py` 的 DATA 常量指向这批原始目录。下一批应使用独立输出目录保存审计；不要将不同记录/工况混为同一 run。完整 4096 点包默认取中间 `[48:4048]` 的 4000 点，有效窗口保存为 `results/raw_windows/*.npy`；本批没有满足条件的窗口，因此该目录可能不存在。

补采合格后，以下命令展示接口用法。示例路径是待采集数据的占位路径，不是本批已经生成的信号：

```bash
conda run --no-capture-output -n m2vllm python /home/huangyating/anomaly_detection/contact_validation_20260919/scripts/encode_contact_pair.py \
  --query /path/to/query_xyz_4000.npy \
  --reference /path/to/healthy_reference_xyz_4000.npy \
  --query-run-id query_run_01 --reference-run-id reference_run_01 \
  --condition-id motorA_2000rpm_loadA_mountA \
  --teacher retrained_fcn --device cpu \
  --output /path/to/contact_embedding.npz
```

调用者须确认输入已通过连续性检查、参考确为正常、两次录制独立且轴向/转速/负载/安装匹配。不同字符串 run ID 本身不能证明这些事实。接口会拒绝样本数错误、非有限值、零交流能量和同名 query/reference run。

## 结果位置

| 文件 | 内容 |
|---|---|
| `results/quality.json` | 原始 SHA256、接收标记、排除记录与最长连续片段 |
| `results/parsed_samples.csv`、`parsed_segments.npz`、`segments.csv` | 保留 XYZ、包编号、序号和连续片段；片段之间不可视作连续 |
| `results/axis_statistics.csv` | 原单位均值、交流 RMS、峰值与峭度 |
| `results/preflight.json`、`strict_windows.json` | 本批严格教师窗口数量为 0 |
| `results/strict_1s_dcn.npz` | 空样本数组 `(0,3,24000)`，另含 0.5 Hz 频率轴和可观测频带 mask |
| `results/bandwidth_metrics.csv`、`bandwidth_predictions.csv` | 3 个教师 × 3 个带宽，共 9 项 MBHM 消融及逐条预测 |
| `results/bandwidth_subset.csv`、`bandwidth_protocol.json` | 426 条查询、参考、划分、各类数量、权重哈希及评测限制 |
| `results/mbhm_hidden_embeddings.npz` | MBHM 子集的实际 128 维 embedding；不是用户三份记录的 embedding |
| `results/synthetic_interface_test.npz` | 合成正弦接口检查；不是实测诊断 |
| `results/verification.json`、`uncentered_dc_audit.json` | 验证记录与未去均值的 DC 能量诊断 |
| `figures/` | 实测接触式质量图和 MBHM 带宽消融图，PNG/SVG/PDF |

没有用三个正常转速文件训练故障分类器，也没有把同一文件的窗口随机划分后报告泛化分数。
