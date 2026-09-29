# 第二轮：修复原始雷达后验证故障知识和多教师复用

从 [第二轮验证汇报_故障知识与多教师复用.md](/home/huangyating/anomaly_detection/encoder_validation_20260923_v2/第二轮验证汇报_故障知识与多教师复用.md) 开始阅读。

已完成：24个原始文件校验、12份雷达重新导出、第三个官方预训练教师UniFault接入、204次雷达学生训练、3教师×6物理统计读取、独立重算。

| 目录 | 内容 |
|---|---|
| `raw/` | 原始SHA、DAT独立解析、距离修复对比、采集质量审计 |
| `geometry_corrected/` | 固定0.29–0.61米ROI的新信号导出与614个雷达窗口 |
| `controls/` | Bear固定头6方法×2方向×3种子，36模型 |
| `rotllm_radar/` | Rot两种头×3方法×2方向×3种子，36模型 |
| `teachers/` | 三教师接口说明、UniFault官方权重来源和接触端实验 |
| `unifault_radar/` | 有符号／ReLU输出×3方法×2方向×3种子，36模型 |
| `heldout_radar_class/` | 4个留出类别×4方法×2方向×3种子，96模型 |
| `results/contact_only_stitch/` | 冻结Bear目标雷达，仅用接触训练数据建立转换后接Rot |
| `results/physical_readouts/` | 冻结表示的额外连续量读取与真类别均值对照 |
| `results/independent_audit/` | 描述量、读取器、36个Uni学生及理论下界独立复算 |
| `results/summary/` | 合并方向后的计分、全部结果CSV、论文用PNG/PDF/SVG |
| `literature/` | 原论文核验、证据边界和下一轮实验详案 |

本轮仍使用8条已有开发录制；不是新的物理泛化数据。原始文件和旧模型未覆盖。所有实际计算使用 `/home/huangyating/miniconda3/envs/m2vllm/bin/python`。
