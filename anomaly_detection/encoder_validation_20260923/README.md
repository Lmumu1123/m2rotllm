# 2026-09-23 Encoder 迭代验证

优先阅读 [第一轮迭代验证结果_通俗汇报.md](/home/huangyating/anomaly_detection/encoder_validation_20260923/第一轮迭代验证结果_通俗汇报.md)。已完成72个雷达学生训练、第二个官方预训练接触模型推理，以及不重训雷达的模型间转换检查。

这是一批已有开发录制上的受控实验。原始外圈距离单元问题尚未修复；用户 Mac 原始文件需同步到服务器，见 [接收说明](/home/huangyating/anomaly_detection/encoder_validation_20260923/data_audit/README.md)。不能据此宣布达到接触98%／雷达95%的最终目标。

| 目录 | 内容 |
|---|---|
| `controls/` | 六种BearLLM固定头对照，36个雷达模型 |
| `rotllm/` | 官方SFN权重与四个预处理分支、接触四类读取头、36个雷达模型 |
| `results/contact_only_stitch/` | 接触训练数据拟合的模型间转换、冻结雷达评估 |
| `results/iteration_summary/` | 两个方向合并后计分的CSV、图、来源哈希 |
| `scripts/` | 转换与统一汇总代码 |
| `data_audit/` | 原始数据可用性、ROI、接收和重处理入口 |

所有实际计算均使用 `/home/huangyating/miniconda3/envs/m2vllm/bin/python`。训练脚本拒绝覆盖已有实验输出；汇总和审计脚本可对已保存结果重跑。
