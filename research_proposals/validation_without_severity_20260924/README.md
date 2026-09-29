# 无严重度数据的验证方案与七个问题答复

2026-09-24。本目录记录本轮文献调研、既有结果解释、数据审计和环境前端兼容性检查，没有新增模型训练成绩。

- **先读：[七个问题答复与无严重度验证方案](/home/huangyating/research_proposals/validation_without_severity_20260924/七个问题答复与无严重度验证方案.md)**。含 UniFault、分类参考基线、物理读出误差、转换模块、错误接口，以及 36／48 条采集方案。
- [详细研究依据与候选实验](/home/huangyating/research_proposals/validation_without_severity_20260924/literature_and_protocols.md)。具体划分和能量基线的信息来源以主文的补充说明为准。
- [真实数据清单](/home/huangyating/research_proposals/validation_without_severity_20260924/data_inventory.md)。42 条环境雷达、12 对原始双模态和单独接触转速文件的可用性。
- [UniFault 与接口核验](/home/huangyating/research_proposals/validation_without_severity_20260924/unifault_and_interfaces.md)。明确哪些雷达 encoder 重训、哪些冻结。
- [环境前端兼容性检查](/home/huangyating/research_proposals/validation_without_severity_20260924/env_compatibility/README.md)。旧环境 128 维特征不等价于当前输入，因此没有直接推理或生成环境准确率；检查脚本已在 m2vllm 环境运行。

当前最值得验证的是：雷达表示是否保留了类别和转速之外、可被接触端新读取器使用的实测故障信号信息，并能在距离／环境变化后保持。多教师能分类、向量相似和新故障迁移是不同结论，需分别验收。
