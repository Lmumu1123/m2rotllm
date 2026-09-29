# 冻结 BearLLM / Qwen 的真实生成验证

110 行输入全部完成真实生成，使用 `conda m2vllm`、GPU 0、冻结的 Qwen2.5-1.5B-Instruct 和已有 BearLLM LoRA。没有人工替换输出，没有把真值写进提示词，也没有根据目标结果修改提示词。

这是**新增的四类语义桥**：先取接触教师或雷达学生的四类概率，再展开到 BearLLM 的十类概率，经过当前最终适配器的 LoRA `linear3` 生成 5 × 1536 信号 token，注入 Qwen 输入嵌入。它证明当前分类器可以连接已有语言生成模型；不能称为原振动 encoder 对雷达的直接零样本诊断。

```text
p4 = [正常, 内圈, 外圈, 滚动体]
q10[0]   = p4[0]
q10[1:4] = p4[1] / 3
q10[7:10]= p4[2] / 3
q10[4:7] = p4[3] / 3
signal_tokens = LoRA_linear3(q10).reshape(5,1536)
```

三个严重度位置均分仅用于兼容原接口，**没有推断严重度**。固定英文提示要求只返回 Normal、Inner race fault、Outer race fault、Rolling element fault 中的一个部位类别，不要求模型生成维护建议。精确提示词、模型 SHA256、输入 SHA256 和推理参数见 [provenance.json](provenance.json)。合并权重代数实现与实际 LoRA 层输出亦已核验。

## 结果按任务与测试范围分别统计

主实验使用训练脚本导出的两个波特率录制留出方向，五个方法均保留：`contact_hidden`、`radar_frame_shape`、`supervised`、`distill`、`no_radar_CE`。

| 任务及测试范围 | 每方法、每方向的文件数 | 生成准确率 / macro-F1 | 对输入概率 argmax 的忠实度 |
|---|---:|---:|---:|
| 四类，0000 → 另一组录制 | 4 | 所有方法 100% / 100% | 所有方法 100% |
| 四类，1111 → 另一组录制 | 4 | 所有方法 100% / 100% | 所有方法 100% |
| 几何相容三类，000 → 另一组录制 | 3 | 所有方法 100% / 100% | 所有方法 100% |
| 几何相容三类，111 → 另一组录制 | 3 | 所有方法 100% / 100% | 所有方法 100% |
| 另一正常电机 bigNormal | 2 个真实文件，每任务/方法复用 | 所有方法正常误报率 100%；不计算多类 F1 | 除下述一例外均忠实 |
| 未知底座故障 keep | 2 个真实文件，每任务/方法复用 | 不计算正确故障部位或多类 F1 | 均忠实于输入的已知类预测 |

三类任务 macro-F1 的类别集合严格为 normal/inner/ball，即 `[0,1,3]`；没有把不存在的外圈类放入分母。外部正常电机的分数单独保存为正常误报率；keep 的真值保留 -1，现有闭集输出不能证明已识别未知故障。

四类任务名称中的 `provisional_roi` 有实际限制：雷达质量表显示两条外圈录制提取的距离单元与已确认几何不符，因此这些雷达预测只保留作计算流程诊断，不构成有效的外圈物理诊断证据。下述输出状态检查将对应雷达结果标为 `invalid_target_roi`；没有用故障标签代替几何判断。

110 条是**算法/任务/文件组合的预测条目，不是 110 次独立物理实验**。其中主实验有 70 条重复任务/方法预测，涉及原 8 个主四类文件；另外 40 条来自 4 个外部文件。当前成绩不能替代独立轴承、重新装配、跨环境和跨电机的新增测试。

全体 110 条生成无解析失败、无截断。对概率 argmax 的总体忠实度为 109/110，即 99.09%；这是接口忠实度，不能当作诊断准确率。

## 保留的一例接口失真

三类任务、`all_known`、`no_radar_CE`、`20260920-202115-bigNormal-460800`：

| 输入正常概率 | 输入内圈概率 | 输入外圈概率 | 输入滚动体概率 | LLM 实际输出 |
|---:|---:|---:|---:|---|
| 0.04834 | **0.57361** | 0 | 0.37806 | Rolling element fault |

输入的最大概率是内圈，LLM 却生成滚动体。这个失真保留在主结果中，不能按真值或概率 argmax 重写生成文本。它提示软类别混合生成的信号嵌入仍可能改变决策；产品展示应把可审计分类器结果与 LLM 文本分开记录，未来若加入确定性标签约束应作为新接口单独评测。

此外，外部正常电机在上游已经被误判，LLM 也没有修复这种跨结构失败。不能从主实验全对推导出语言模型带来了额外故障知识。

## 独立的一致性后处理

针对上述实际发现的一例不一致，新增了 [guard_llm_output.py](../guard_llm_output.py)。脚本只读取原输出并写新文件：当 `parsed_coarse_label != argmax(p0..p3)`，结构化诊断保留分类器原始概率和 argmax，语言文本回退为该类别的标准短语。其他 109 条文本保持原样。

结果见 [guarded_predictions.csv](guarded_predictions.csv) 与 [guard_audit.json](guard_audit.json)：原始 LLM 忠实度仍为 109/110，1 条回退；后处理的一致性为 110/110，**这是规则保证，不能算成 LLM 学得性能或准确率提升**。原 CSV、JSONL 与主生成指标未修改。实际扰动真值和 state 后，所有回退决策完全不变。

同一文件还根据真实 [雷达质量表](../../radar/recording_quality.csv) 以 `bag_id` 关联 `geometry_roi_mismatch`。24 条重复任务/方法的雷达结果被标为 `invalid_target_roi`，涉及外圈和 keep 的异常距离；这个状态不用于重写概率或原生成结果。接触式预测不受雷达 ROI 状态影响。

所有结果均保留 `known_class_candidate_only=True`、`unknown_detection_supported=False`。闭集最大概率高低都不能证明未知故障已被识别；尤其 keep 不得因碰巧生成某个故障类别就记作未知故障识别成功。

## 可复现文件

- [generation_predictions.csv](generation_predictions.csv)：全部输入概率、生成文本、解析标签、忠实度、真值和任务元数据。
- [generation_predictions.jsonl](generation_predictions.jsonl)：同样内容的逐条 JSON。
- [generation_metrics.csv](generation_metrics.csv)：按 task / fold / direction / method / seed / evaluation_scope 分组；外部样本没有混入主 F1。
- [generation_metrics_by_state.csv](generation_metrics_by_state.csv)：额外按 state 分组，便于检查各故障部位及外部正常/未知数据。
- [inputs_used.csv](inputs_used.csv)：本次实际输入快照。

```bash
conda run --no-capture-output -n m2vllm python \
  /home/huangyating/anomaly_detection/four_class_chain_20260922/contact/run_llm_bridge.py \
  --inputs /home/huangyating/anomaly_detection/four_class_chain_20260922/results/chain_v1/llm_bridge_inputs.csv \
  --no-controls \
  --output /home/huangyating/anomaly_detection/four_class_chain_20260922/contact/llm_bridge_chain_v1
```

四个 oracle 一热控制及原接触 FCN 错误概率控制保存在相邻的 `llm_bridge_controls/`，不与以上主实验混合。oracle 控制仅验证接口对明确四类语义的表达能力。
