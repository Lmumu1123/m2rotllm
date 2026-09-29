# Third-party sources and notices

本文件记录本次迁移实际使用的第三方来源、固定版本和本地保存的许可证，不给整个 C2R 仓库统一赋予 MIT 或其他新许可证。各上游文件、模型及数据仍保留其原有权利与声明。本文件基于本机已保存源码、模型资产清单和 LICENSE 核对，不表示本次重新核验了远端主分支的最新版本。

## BearLLM

- 代码来源：[SIA-IDE/BearLLM](https://github.com/SIA-IDE/BearLLM)，本地基准 commit `4bdd29bf013971909e81cd79732c4b3813101ca9`。
- 迁移位置：`BearLLM/`；本项目对复现、预处理、路径和训练验证的本地修改也在交接范围，不能把整个目录声称为未经修改的上游快照。
- 官方权重：[SIA-IDE/BearLLM](https://huggingface.co/SIA-IDE/BearLLM)，固定 revision `fd2859d9ea8fbebe7f815ca8e02ca48bb1025191`；迁移位置 `external/bearllm-assets/bearllm_weights/`。
- **本地 BearLLM 代码 checkout 未找到 LICENSE 文件。** 本迁移保留作者、README 与来源，不将它推定为 MIT，也不通过本仓库根目录给它重新授权。不能将另一个项目或 MBHM 数据卡的许可直接套用到 BearLLM 代码和权重。
- 本地复现/微调输出在 `external/bearllm-runs/` 和项目实验目录中；这些输出与官方权重分别登记，不混为同一个模型。

## Qwen2.5-1.5B-Instruct

- 来源：[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，固定 revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- 迁移位置：`external/bearllm-assets/qwen_weights/`。大权重在 `migration/model_parts/qwen_model/` 分片分发，经恢复脚本重组为 `model.safetensors`；分片不改变原权重内容或许可。
- 本地实际保存的 [LICENSE](../../external/bearllm-assets/qwen_weights/LICENSE) 为 Apache License 2.0，版权声明为 `Copyright 2024 Alibaba Cloud`。原许可证全文保留。
- 原始 `model.safetensors` 为 3,087,467,144 字节，SHA256 为 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。恢复后的文件须匹配这个哈希。
- 配置、tokenizer 和词表随权重保留；本文件不把量化版本、其他尺寸或更新 revision 视为等价替代。

## RotLLM

- 来源：[SIA-IDE/RotLLM](https://github.com/SIA-IDE/RotLLM)，两个本地副本核对为 commit `2fda9859da3537acf9909b62010995d7366b17ae`。
- 位置：`anomaly_detection/encoder_validation_20260923/rotllm/vendor/RotLLM/`，以及早期调研副本 `research_proposals/contact_radar_alignment_20260919/references/RotLLM/`。
- 原 [LICENSE](../../anomaly_detection/encoder_validation_20260923/rotllm/vendor/RotLLM/LICENSE) 为 MIT，版权声明为 `Copyright (c) 2025 SIA-IDE`；两份源码中的许可证均保留。
- 保留官方 encoder/projection 权重、词表映射与本项目输入适配、头和转换实验。目录名 `references` 下含实际代码及权重，不只是文献链接。

## UniFault

- 来源：[emadeldeen24/UniFault](https://github.com/emadeldeen24/UniFault)，固定 commit `4b20ce54f31507bfb9e486e5126cb025bf46e194`。
- 原源码位置：`anomaly_detection/encoder_validation_20260923_v2/teachers/vendor/UniFault/`。
- 原 [LICENSE](../../anomaly_detection/encoder_validation_20260923_v2/teachers/vendor/UniFault/LICENSE) 为 MIT，版权声明为 `Copyright (c) 2025 Emadeldeen Eldele`。
- 本项目使用作者 README 链接的 Tiny `pretrain-epoch=1.ckpt`，SHA256 为 `a00b8aa0220f7ce4bdfa5e4f91d1a923a880ffa6f5fa0f7369d3b46e87335ed7`。
- 项目另有去除无关 Lightning 依赖、保留前向结构的轻量推理实现；原 vendor 副本与修改实现各自保存，详情见[教师接口记录](../../anomaly_detection/encoder_validation_20260923_v2/teachers/教师输入接口与第三模型验证.md)。新本地四分类头不等同于官方 16 类头。

## MBHM 公共源数据

- 来源：[SIA-IDE/MBHM](https://huggingface.co/datasets/SIA-IDE/MBHM)，固定 revision `78cd9b8b6b65cd43eebf4879b060d783d5f7dcfe`。
- 本地数据卡开头标记 `license: mit`；这是该数据卡的声明，不扩展为本仓库其他组件的许可证。保留数据卡、元信息和记录的来源。
- 大 `data.hdf5` 是独立恢复的数据资产，不放入普通代码目录；原记录大小 13,009,537,400 字节，SHA256 为 `ba5e5c9da0538e8d0dfd22d683b95098cb1abee14c6c6e2bbc478ee6339040b8`。恢复与排除范围按迁移数据清单执行。
- 原公开数据组成部分的说明仍按上游资料解释；本交接不新增对这些数据的授权承诺。

## 其余依赖与项目图表

NumPy、SciPy、PyTorch、Transformers、PEFT、h5py、pandas、scikit-learn、Matplotlib 等作为环境依赖安装。版本记录不意味着重新分发它们的全部安装目录；原服务器 `.venv`、conda 目录、浏览器 `node_modules` 和临时报告渲染包不作为项目源码上传。

`research_proposals/` 中已核对到的 PDF 为本项目生成的科学图表，未发现下载的论文全文 PDF。调研原文采用出处链接和记录；上游仓库自带 README、LICENSE、代码和模型保留原归属。

本项目的分片、路径适配、复现实验及本地模型清单用于完整交接，不表示上游作者认可了当前模型性能、论文结论或部署用途。
