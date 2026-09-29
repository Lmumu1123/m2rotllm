# 对齐链路、冻结复用、baseline与补采计划

2026-09-24。

**目标澄清：本项目主目标是让后续雷达数据通过统一跨模态接口使用接触模型诊断；减少重采／升级成本只是可能的附带收益。** 本轮关于物理读取器、冻结backbone、转换实验力度及配置选择的最新说明见 [目标澄清与cfg小试](/home/huangyating/research_proposals/alignment_and_collection_20260924/目标澄清_读取器与冻结验证_CFG小试.md)。此前转换成绩保留，定位为辅助兼容性证据。

## 先看这三份

1. [链路说明与下一轮验证](/home/huangyating/research_proposals/alignment_and_collection_20260924/链路说明与下一轮验证.md)：从原始数据到128维对齐、分类头／读取器／转换器区别、雷达预处理细节，以及本次实际完成的Bear→Uni冻结补测。
2. [现场物理采集方案](/home/huangyating/research_proposals/alignment_and_collection_20260924/physical_collection.md)：先小试，再48条核心、32条扩展；质量、时序、分组、命名、存储和操作步骤。
3. [baseline与创新边界](/home/huangyating/research_proposals/alignment_and_collection_20260924/baselines_and_novelty.md)：精选分类算法、迁移方法、机制对照和额外教师，附原论文／作者实现。

## 现场用表

优先使用 [80条简版操作表](/home/huangyating/research_proposals/alignment_and_collection_20260924/collection_operator_main80.csv)。计划条件已经填好，实际日期、文件、实体、距离和转速证据等待采集者填写。没有实际转速反馈时填 `setpoint_only`，不要把计划值抄成实测值。

完整表用于后续机器整理，不要求人工逐条填写一百多项：相同硬件／cfg／采集软件的信息每个配置版本记录一次，分析时按ID关联；哈希、数据量、完整包数、有效窗口等由程序计算。现场必须保留真实改动、失败重采与原始日志。

- [核心48条完整表](/home/huangyating/research_proposals/alignment_and_collection_20260924/collection_manifest_core48.csv)
- [扩展32条完整表](/home/huangyating/research_proposals/alignment_and_collection_20260924/collection_manifest_extension32.csv)
- [合并80条完整表](/home/huangyating/research_proposals/alignment_and_collection_20260924/collection_manifest_main80.csv)
- [背景24条](/home/huangyating/research_proposals/alignment_and_collection_20260924/collection_manifest_background24.csv)
- [可选组合条件8条](/home/huangyating/research_proposals/alignment_and_collection_20260924/collection_manifest_optional_combined8.csv)
- [可选keep 12条](/home/huangyating/research_proposals/alignment_and_collection_20260924/collection_manifest_optional_keep12.csv)

首批先完成off及两条正常重复的小试，检查至少10个完整接触包和雷达日志，再按表批量采。正式每条以≥20个完整接触包且≥30秒稳定段为目标；90秒仅为当前460800文本链路的预算，真实时长由小试决定。

## 本次实际运行的补测

[冻结Bear雷达→UniFault实验](/home/huangyating/anomaly_detection/frozen_bear_to_unifault_20260924/README.md)：仅拟合两折的接触转换，36份旧雷达权重和Uni分类头不变。完整方法99.92%窗口／100%录制；人为类别向量95.06%窗口／100%录制，全部方法均公开。仍是八个已有开发录制，不能视为新增盲测。

采集CSV全部为待执行计划，不是新增采集结果。baseline调研没有冒充新跑出的模型成绩；雷达配置没有自动下发。
