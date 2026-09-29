# C2R 模型与恢复依赖清单（2026-09-29）

本清单只读取项目，没有修改或删除原文件。逐文件真实路径、大小、SHA-256、内容重复组、参数数组名称以及相邻配置见 `model_inventory.json`。这里的“模型文件”也包括优化器/随机数恢复状态、分类头和含参数的 NPZ，因此不能把文件数量等同于独立模型数量。

共清点 **784 个模型与参数文件，3,995,720,677 字节**；按 SHA-256 去重为 **3,822,354,195 字节**。其中公共 Qwen 基座单文件 3,087,467,144 字节；其他模型与参数合计约 908 MB。除 Qwen 基座外，最大的单个权重是 RotLLM 投影（66,792,572 字节），均低于 100 MB。

## 必须一并迁移的模型和配置

| 模型/用途 | 服务器原路径 | 保留要求 |
|---|---|---|
| 当前主接触教师 Bear FCN | `/media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn/` | `feature_encoder.pth`、`classifier.pth` 均保留；同时保存 FCN 定义和预处理 |
| 已做源域回归的历史分类头 | `/home/huangyating/anomaly_detection/retention_alignment_20260922/results/retained_head_v1/models/v0/all_known/retained_head.npz` | 独有训练产物，保存原参数及 `source_scaler_mean/scale`、标签分组；不要被 R2 诊断头替换 |
| Bear 原生/再训练语言接口 | `/media/nas_users/huangyating/bearllm-assets/bearllm_weights/`；`.../bearllm-runs/released-code-seed42/finetune/weights/` | `adapter_model.safetensors`、`vibration_adapter.pth`、`adapter_config.json` 是配套文件，必须同版本使用 |
| Bear 复现各阶段与可恢复训练状态 | `.../bearllm-runs/released-code-seed42/{pretrain,finetune,source_export_weights,diagnostics}/` | 保留全部独有权重；继续原训练还需 optimizer、scheduler、rng_state、trainer_state、training_args 以及数据划分 |
| 历史雷达编码器、分类头、模型间转换 | `/home/huangyating/anomaly_detection/` 下各实验的 `models/`、`heads/`、`mappings/` | 不能只挑最高分 checkpoint；各折/种子对应训练 scaler、教师统计量、划分与配套代码一起保留 |
| 最新 R2 编码器 | `anomaly_detection/r2_validation_20260928/experiments/` | 完整保留各消融方法与重复种子。当前跨转速结果未达到用户的 95% 雷达目标，不应改名为通过验收的部署模型 |
| R2 接触诊断头 | `anomaly_detection/r2_validation_20260928/contact/diagnostic_heads/` | 三个留转速头仅用于诊断；`accepted_for_source_retention=False`，不满足已验证的“原准确率下降≤0.5百分点”交付条件 |
| RotLLM 上游权重 | `anomaly_detection/encoder_validation_20260923/rotllm/vendor/RotLLM/weights/` | 原生 encoder 与 projection 一并保留；原生分类头 15 类，不等同于 Bear 十类头 |
| UniFault 上游权重 | `anomaly_detection/encoder_validation_20260923_v2/teachers/vendor/weights/pretrain-epoch=1.ckpt` | 9,834,974 字节；对应官方代码与适配代码一并保留，不能用新四类头分数宣称官方原生头分数 |
| UniFault 本地四类头及转换 | `.../teachers/unifault/heads/`；`anomaly_detection/frozen_bear_to_unifault_20260924/results/mappings/` | 四类头、按训练折拟合的缩放、映射参数及训练袋身份必须配套 |

`BearLLM/outputs/training_smoke/` 是训练通路冒烟验证，权重不作为主模型。用户要求完整迁移，所以应一并归档并明确标签；服务器精简保留时，只有在远端归档已校验后才可去掉这些重复实验副本。

## 不下载完整源数据也能跑接触推理

当前 `four_class_chain_20260922/contact/evaluate_contact.py` 的 `external_references()` 从完整 MBHM 数据库和 HDF5 选择 9 条公开健康参考。相同参考已保存为：

- `anomaly_detection/four_class_chain_20260922/contact/external_references.npz`：键为 `dcn`、`file_id`，仅约 757 KB；
- 同目录 `external_reference_protocol.json`：选择协议和来源身份。

迁移版应优先使用这两份固定参考，保留原逻辑作为回退。这样运行新接触数据时不必为 9 条参考下载 13 GB 源数据。原始预处理、9 条参考、均值方式和十类合并为四类的映射都属于模型接口，不能只复制 `.pth`。

完整源域重新训练/从原始样本重新评测仍需要 MBHM。现有 `retention_alignment_20260922/source/` 的冻结特征缓存可以复查相关分类头回归，但不等同于重新运行完整原始波形端到端评测。

## 可固定版本重下载的公共资源

| 资源 | 固定仓库与 revision | 大文件及校验 |
|---|---|---|
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct`，`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `model.safetensors`，3,087,467,144 字节；SHA-256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 官方 Bear 适配器 | `SIA-IDE/BearLLM`，`fd2859d9ea8fbebe7f815ca8e02ca48bb1025191` | 两个权重和 `adapter_config.json` 已存在本机，合计约 23 MB，建议直接保全 |
| MBHM | 数据集 `SIA-IDE/MBHM`，`78cd9b8b6b65cd43eebf4879b060d783d5f7dcfe` | `data.hdf5`，13,009,537,400 字节；SHA-256 `ba5e5c9da0538e8d0dfd22d683b95098cb1abee14c6c6e2bbc478ee6339040b8` |

现成下载与校验源清单：`/media/nas_users/huangyating/bearllm-assets/asset_manifest.json` 和 `.../mbhm_dataset/mbhm_manifest.json`；它们也嵌入本次 JSON 清单。**Qwen 基座可固定 revision 恢复，项目训练所得的适配器、分类头和雷达编码器不能依靠重新下载公共基座代替。**

Qwen 的 `config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`、`merges.txt` 和 `LICENSE` 也要保存。Bear 训练的 `l3.npy` 是语义向量初始化缓存，应随训练记录保留；它与固定 Qwen/tokenizer/description token 设置关联。

## 上游版本和许可证

- BearLLM：源码 revision 见 JSON；本地 checkout 未找到 `LICENSE`，本清单不擅自给上游源码添加新许可证，保留上游署名和来源。
- RotLLM：`https://github.com/SIA-IDE/RotLLM`，commit `2fda9859da3537acf9909b62010995d7366b17ae`，源码包含 MIT 许可证。
- UniFault：`https://github.com/emadeldeen24/UniFault`，commit `4b20ce54f31507bfb9e486e5126cb025bf46e194`，源码包含 MIT 许可证；权重 SHA-256 `a00b8aa0220f7ce4bdfa5e4f91d1a923a880ffa6f5fa0f7369d3b46e87335ed7`。官方权重下载入口在既有 provenance 中。
- Qwen：本地基座附 Apache-2.0 LICENSE，应同 token/config 文件保留。

上述源码许可证信息来自本地文件，并不额外断言第三方独立权重具有未明确标注的许可。

## 服务器清理的安全边界

建议先建立独立“模型保留包”，包含本清单中的权重、同目录配置/训练状态、参考库、模型定义、标签映射/预处理配置、环境锁定和 SHA 清单。模型定义与少量配置占用很小，删除它们只留裸权重会使后续微调困难。**清理前必须确认远端及另一台主机可恢复并通过校验，本次清点未执行删除。**

- **保留包核验后可清理旧副本**：历史重复模型目录、冒烟模型、图表/日志/文档的服务器工作副本、可重算的中间特征与临时缓存；前提是已迁移到 C2R 或另一主机。
- **不能仅因“不上传 GitHub”就删除**：R2 的 `data/upload/`、`data/wide_cache_local/`、唯一原始 DAT/bin、旧实验尚未备份的数据。先在新主机完成单独复制或确认 Mac 原始数据与处理清单足以重建。
- **可重新下载但并非已备份**：MBHM 13 GB、Qwen 公共基座；如果服务器后续还要运行完整 Bear LLM，仍需本地 Qwen。只保留分类教师/雷达编码器用于数值分类或其微调时通常不必加载语言基座。
- **不要批量删除**：`/media/nas_users/huangyating` 全目录、`/home/huangyating/.cache`、Miniconda、无关项目或任何符号链接目标；它们可能被其他项目共享。

另外，`bearllm-runs/released-code-seed42/{official_evaluation,trained_evaluation}/vibration_predictions.jsonl` 各约 266 MB，是结果而非原始数据。上传时应无损压缩或分块保存，不应因超出单文件限制直接遗漏。
