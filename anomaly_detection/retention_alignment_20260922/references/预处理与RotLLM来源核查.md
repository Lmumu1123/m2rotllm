# 预处理与 RotLLM 来源核查

核查日期：2026-09-22。本轮只依赖作者论文、作者仓库、已下载官方资产与制造商技术资料，不使用自动生成的论文解读作为证据。

| 来源 | 本次实际访问 | 可支持的内容与边界 |
|---|---|---|
| [BearLLM arXiv 原文 v2](https://arxiv.org/html/2408.11281v2) | 正文 HTML 可读 | 固定时长、DCN、同工况健康参考、概率语义投影；不等于未知台架泛化保证 |
| [BearLLM 官方 DCN](https://github.com/SIA-IDE/BearLLM/blob/main/functions/dcn.py) | 远端 raw 与本地代码核查 | demo 的 DCN 不减均值 |
| [BearLLM 官方数据接口](https://github.com/SIA-IDE/BearLLM/blob/main/functions/mbhm.py) | 本地官方仓库核查 | 条件 ID 匹配健康参考；原缓存划分规则 |
| [MBHM 构造代码](https://huggingface.co/datasets/SIA-IDE/MBHM/blob/main/src/dcn.py) | 本次网页抓取失败；已有官方资产文件可读 | 本地版本减均值；不能声称本次在线成功读取该文件 |
| 已发布 MBHM data.hdf5 | 前轮全量 DC 项审计；本轮固定 9 行参考重读 | 发布资产与代码契约需区分；未篡改数据来“符合论文” |
| [RotLLM 出版社页](https://doi.org/10.1016/j.engappai.2025.112544) | 搜索索引可读官方摘要/方法片段；完整正文打开被 403 拒绝 | SFN、15 类、语义投影和可选 RAG 的概述；不支持声称完整方法已逐项复现 |
| [RotLLM 官方 SFN](https://github.com/SIA-IDE/RotLLM/blob/main/code/models/SFN.py) | 远端 raw 与本地代码核查 | 3 段折叠，宽卷积、多尺度、通道注意力和残差；没有把 SFN误写成平滑模块 |
| [RotLLM 官方投影](https://github.com/SIA-IDE/RotLLM/blob/main/code/fine_tune/convert_weights.py) | 本地官方仓库核查 | 15 类概率语义分支与 720 维连续特征残差、零初始化、11×2048 token 维度 |
| [RotLLM 模型代码](https://github.com/SIA-IDE/RotLLM/blob/main/code/fine_tune/rotllm.py) | 本地官方仓库核查 | 发布 forward 对 encoder 使用 no_grad，与 README 高层描述不能混同 |
| [SKF 轴承振动说明](https://evolution.skf.com/us/vibrations-help-find-faulty-components-3/) | 官方网页可读 | 速度与几何影响故障频率，谐波和包络的重要性；本项目未通过此文献获得设备实际转速 |

本地 BearLLM 提交：`4bdd29bf013971909e81cd79732c4b3813101ca9`。RotLLM 副本提交：`2fda9859da3537acf9909b62010995d7366b17ae`。运行脚本和权重具体 SHA-256 见 `preprocessing/protocol.json`、各变体 `provenance.json` 和 `verification.json`。

本轮为独立诊断实验；没有调用 SFN 发布权重替换 BearLLM，也没有把论文中的总体性能当作本地数据结果。新增图中的数据均来自本次实际冻结推理；DCT 坐标验证是标明的解析/合成测试。
