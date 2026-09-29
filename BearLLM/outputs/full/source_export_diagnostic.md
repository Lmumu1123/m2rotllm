# 原仓库保存方式对照（恢复初始BN）

使用同一次50轮训练结束后的LoRA与配置，字节哈希完全一致；仅将振动适配器恢复为微调前保存的版本。72个非BN基础参数张量逐值一致，唯一差异为6层BN的18个状态张量。未重新训练、未调参、未筛选样本，主权重与主评测结果保持原样。

全部12,279条测试查询使用与主评测相同的查询顺序、健康参考、提示、seed42、greedy生成、batch32和max_new_tokens64。进程退出码0，耗时167.34秒；10项完整性核验全部通过，无截断或无法解析的生成结果。

| 指标 | 主模型：微调后BN | 原仓库保存方式：微调前BN | 差值（百分点） |
|---|---:|---:|---:|
| 自然文本生成分类准确率 | 93.777995% | 95.756983% | +1.978989 |
| 同一首个参考下内部分类头准确率 | 94.030459% | 96.156039% | +2.125580 |

自然生成的配对变化：主结果错误、对照正确320条；主结果正确、对照错误77条；预测类别改变452条。

| 数据集 | 查询数 | 主模型生成准确率 | 原保存方式生成准确率 |
|---|---:|---:|---:|
| CWRU | 7 | 100.000000% | 100.000000% |
| DIRG | 102 | 98.039216% | 98.039216% |
| HIT | 964 | 96.991701% | 96.680498% |
| IMS | 4648 | 98.192771% | 98.687608% |
| JUST | 4399 | 87.906342% | 93.271198% |
| MFPT | 4 | 75.000000% | 75.000000% |
| NCEPU | 90 | 98.888889% | 98.888889% |
| PU | 731 | 94.528044% | 92.065663% |
| XJTU | 1334 | 94.377811% | 94.752624% |

该对照量化了持久化BN运行统计这一导出行为的影响。保留微调后BN能恢复训练结束时的实际模型状态；恢复初始BN则复现原仓库只保存PEFT、保留原振动适配器文件的加载行为。本报告同时保留两种结果，不按测试分数选择模型。官方权重同协议生成准确率为97.703396%，仅供参照，其原始训练成员未公开。

主要产物：

- 权重与差异清单：`/media/nas_users/huangyating/bearllm-runs/released-code-seed42/source_export_weights/diagnostic_manifest.json`
- 全量逐条预测：`/media/nas_users/huangyating/bearllm-runs/released-code-seed42/source_export_heldout/heldout_predictions.jsonl`
- 全量指标：`/media/nas_users/huangyating/bearllm-runs/released-code-seed42/source_export_heldout/heldout_metrics.json`
- 完整校验与配对分析：`/home/huangyating/BearLLM/outputs/full/source_export_diagnostic.json`
- 运行命令与退出码：`/home/huangyating/BearLLM/outputs/full/source_export_run.json`
- 日志：`/home/huangyating/BearLLM/outputs/full/source_export_heldout.log`
