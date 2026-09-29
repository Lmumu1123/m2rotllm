# 雷达学生 CPU 推理成本

测量对象是已训练的 `four_class_provisional_roi/all_known/distill_seed42.pt`，采用 128 维已提取雷达特征、学生 encoder 和冻结的**适配后接触式分类头**。只测工程成本，不读取或使用标签。

| 项目 | 实测值 |
|---|---:|
| Encoder 参数量 | 33,344 |
| 冻结适配分类头参数量 | 516 |
| 总模型参数量 | 33,860 |
| 标准化缓冲值数量 | 256 |
| 保存 checkpoint 文件大小 | 150,191 字节（146.67 KiB） |

| 测量范围 | p50（ms） | p95（ms） | 均值（ms） |
|---|---:|---:|---:|
| 单窗口预载 Tensor：标准化→encoder→冻结头→softmax | 0.0282 | 0.0325 | 0.0285 |
| 单窗口内存特征→结果字典：包含上述模型过程及输入/结果转换 | 0.0319 | 0.0363 | 0.0323 |
| 32 窗口批次：预载 Tensor→softmax | 0.0453 | 0.0502 | 0.0458 |

每项分别预热 200 次，正式测量 500 次。单窗口每次仅输入 `[1,128]`，没有把全部 614 窗口的批吞吐冒充单条延迟。Batch 32 的平均吞吐为 698,012 窗口/秒，摊销值 0.00143 ms/窗口仅用于吞吐解释。

CPU：`AMD EPYC`。PyTorch `2.11.0+cu128`，NumPy `1.26.0`，Python `3.12.14`；conda `m2vllm`。Torch intra-op 固定 2 线程，inter-op 1 线程，模型为 float32、eval/inference_mode。GPU 被隐藏且未初始化，本测量不争用训练 GPU。CPU 未绑核，其他主机任务未隔离，因此这些数值是当时服务器实测微基准，不能当作确定性的硬实时保证。

**不包含** ADC 解析、距离 FFT/选门、特征提取、2 秒采集、文件 I/O、权重加载、接触式 encoder 或 LLM 推理。两秒窗口的采集等待也不能从这些耗时中省略，因此不能称为完整链路端到端实时延迟。模型当前仍带有 provisional ROI 限制；成本低不证明诊断有效。

输入标准化的 float32 工程实现与原 source scaler 算术结果的概率最大绝对差为 `1.49e-08`。完整逐次耗时、软件信息、模型与脚本 SHA256 见 [latency_benchmark.json](latency_benchmark.json)。

复现时使用新输出名，避免覆盖本次证据：

```bash
/home/huangyating/miniconda3/bin/conda run --no-capture-output -n m2vllm \
  python /home/huangyating/anomaly_detection/four_class_chain_20260922/scripts/benchmark_student.py \
  --output-prefix /home/huangyating/anomaly_detection/four_class_chain_20260922/results/latency_benchmark_repeat
```
