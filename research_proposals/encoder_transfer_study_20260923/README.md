# 阅读入口

本次调研聚焦：毫米波 encoder 是否学到可复用的接触诊断知识，以及跨接触模型的适用性；不再把跨机械结构泛化列为主任务。

**建议先读：[通俗讲解：结果表是什么意思、其他模型怎样验证](/home/huangyating/research_proposals/encoder_transfer_study_20260923/通俗讲解_现有结果与其他模型验证.md)。** 原报告第 2 节也已改用通俗解释；不需要先看懂公式再讨论实验。

1. [主报告：文献、算法与完整实验方案](/home/huangyating/research_proposals/encoder_transfer_study_20260923/Encoder知识迁移验证_文献调研与SenSys实验方案.md)
2. [开源接触式教师与权重核验](/home/huangyating/research_proposals/encoder_transfer_study_20260923/model_baselines_notes.md)
3. [encoder 指标、反证实验、跨教师协议](/home/huangyating/research_proposals/encoder_transfer_study_20260923/representation_validation_notes.md)
4. [传感领域近邻与论文定位](/home/huangyating/research_proposals/encoder_transfer_study_20260923/sensing_prior_art_notes.md)

## 建议先讨论的三个实验

- **真实教师 vs 类别原型 vs 同类错配**：保留类别信息，破坏细粒度教师信息，判断蒸馏是否超出四分类。
- **接触式新任务头直接用于冻结雷达 encoder**：任务相对雷达蒸馏阶段未监督，且不是原四类的简单合并。
- **留出教师的 contact-only 接口校准**：雷达只向 A 学，冻结后仅用接触数据拟合 A→B，比较 B 原生、接触拼接和雷达拼接。

## 当前证据

归档结果中 CE-only、袋均值匹配和完整蒸馏的录制级分类都满分，但特征保真度不同。每折每类只有一个训练袋，使录制目标与类别原型完全重合。当前数据不足以区分两种知识解释。

[既有结果的只读复核](/home/huangyating/research_proposals/encoder_transfer_study_20260923/evidence/existing_summary.json)与[汇总表](/home/huangyating/research_proposals/encoder_transfer_study_20260923/evidence/existing_method_comparison.csv)来自8个开发录制，不是新盲测。错误ROI等既有问题保留说明。

## 本轮完成边界

2026-09-23 核验原论文、会议记录、官方代码／模型资产；形成完整研究方案；在 `m2vllm` 只读复核已有结果并生成彩图。未进行新模型训练、实机采集或雷达配置修改。“权重公开”不等于已在本机复现。
