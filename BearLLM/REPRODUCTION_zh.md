# BearLLM 官方 Demo 复现

本文保留第一阶段官方 Demo 的运行记录。后续任务已扩展为全量 MBHM 验证和重新训练，当前环境统一使用独立 Conda `m2vllm`，完整说明见 [FULL_REPRODUCTION_zh.md](FULL_REPRODUCTION_zh.md)。以下 `.venv` 版本和耗时属于历史记录；`scripts/run_demo.sh` 现已切换到 `m2vllm`。

第一阶段目标：使用官方 Qwen2.5-1.5B-Instruct、BearLLM 权重和 MBHM 示例信号运行推理。

## 实际结果（2026-09-19）

已完成官方示例的 GPU 端到端推理，进程退出码 0。完整输出保存在 `outputs/official_demo.txt`，模型生成的类别为 **Fault-Free（无故障）**，随后生成风险分析和维护建议。

另从 `/tmp` 通过启动脚本指定 `--data`、`--seed 42`、`--output` 复跑，退出码同样为 0，保存结果与默认入口一致。该次运行耗时 **17.726 秒**（含模型加载，文件已在本地缓存）。保存的响应见 `outputs/demo_response.txt`；机器可读验证记录见 `outputs/verification.json`，命令和耗时见 `outputs/demo_runtime.json`。

这证明当前环境下的模型加载、信号编码、LoRA 加载与文本生成流程可以运行，不代表已经验证论文分类准确率。官方示例没有独立真实标签；类别是此次模型的预测。

## 本机路径

- 代码：`/home/huangyating/BearLLM`
- 当前虚拟环境：`/home/huangyating/miniconda3/envs/m2vllm`；历史 Demo 环境：`/home/huangyating/BearLLM/.venv`
- 模型和示例数据：`/media/nas_users/huangyating/bearllm-assets`
- 环境配置：仓库根目录 `.env`
- 环境快照：`outputs/environment.json`

## 第一阶段历史环境说明

代码基于上游提交 `4bdd29bf013971909e81cd79732c4b3813101ca9`。
本机 Python 3.12.3，GPU 为 NVIDIA RTX PRO 5000 72GB Blackwell（sm_120）。
保留容器自带的 `torch 2.10.0a0+b558c986e8.nv25.11 / CUDA 13.0`，并通过 `--system-site-packages` 供独立虚拟环境使用。该环境已执行 GPU 矩阵运算和振动编码器前向检查。

上游固定的 torch 2.6.0 不是本机使用的版本，因此这是针对当前 GPU 的兼容复现；不直接运行上游 `pip install -r requirements.txt`，以免替换现有支持 Blackwell 的 PyTorch。上游 Transformers/PEFT 从 Git 主分支安装，没有锁定版本；本次使用固定版本，见 `requirements-demo.lock.txt`。

```bash
cd /home/huangyating/BearLLM
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-demo.lock.txt
```

以上命令适用于当前容器或已安装兼容 PyTorch 的环境；锁文件只记录独立环境安装的依赖，不含继承的系统 PyTorch。完整关键版本见环境快照。

`pip check` 另报容器已有的 `nvidia-resiliency-ext` 缺少 `pynvml`，与本 Demo 的导入和推理链路无关，没有修改该系统组件。

## 下载权重和示例

下载脚本固定各仓库 revision，只下载推理所需的 12 个文件。LFS 文件校验 SHA-256，普通文件校验 Git blob SHA-1；成功后保存 `asset_manifest.json`。

```bash
cd /home/huangyating/BearLLM
.venv/bin/python scripts/download_demo_assets.py \
  --data-dir /media/nas_users/huangyating/bearllm-assets \
  --endpoint https://hf-mirror.com --connections 16 --no-proxy
```

本机原 Hugging Face 代理不可用，镜像直连可用，所以此处使用 `--no-proxy`；其他环境可省略该参数，并按需使用默认官方 endpoint。脚本支持重跑，已存在且通过校验的文件会跳过。

资源来源：

- [上游代码](https://github.com/SIA-IDE/BearLLM)
- [基础模型 Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- [BearLLM 适配器](https://huggingface.co/SIA-IDE/BearLLM)，revision `fd2859d9ea8fbebe7f815ca8e02ca48bb1025191`
- [MBHM 示例](https://huggingface.co/datasets/SIA-IDE/MBHM)，revision `78cd9b8b6b65cd43eebf4879b060d783d5f7dcfe`

## 运行官方 Demo

```bash
cd /home/huangyating/BearLLM
./scripts/run_demo.sh
# 保存响应；默认 seed=42，max_new_tokens=2048
./scripts/run_demo.sh --seed 42 --output outputs/demo_response.txt
```

启动脚本自动切换到仓库目录、使用独立 Python 环境并设置 Hugging Face 离线模式。默认使用可见的第一张 GPU，指定其他 GPU 可用 `CUDA_VISIBLE_DEVICES=1 ./scripts/run_demo.sh`。

模型采用官方生成配置（采样、temperature=0.7、top_p=0.8、top_k=20），启动入口默认固定随机种子 42。不同随机种子、硬件和库版本仍可能产生不同的自然语言表述。

## 当前兼容修复

`src/fine_tuning.py` 将 `Trainer` 和 `TrainingArguments` 改为在 `fine_tuning()` 内导入。原 Demo 导入模型定义时会加载 Trainer，从而触发当前容器 Apex 不提供 `amp` 的错误。修改仅调整导入时机，推理仍使用原始 DCN、振动适配器、LoRA 和提示词。

PEFT 固定为 0.15.2，已确认读取的官方 LoRA 参数为 `r=4`、`lora_alpha=32`、10 类目标模块。更早的 0.14.0 无法完整读取当前官方适配器配置，不用于最终环境。

`run_demo.py` 和 `src/demo.py` 补充输入文件、随机种子、生成长度和输出文件参数，校验占位符及信号数组。每次推理使用独立临时信号缓存，避免原始固定 `./cache.npy` 被其他进程覆盖。模型架构、官方权重、DCN 算法和提示词不变。

运行日志中的两个上游警告已核对：Transformers 4.49.0 的滑动窗口提示由 `config.sliding_window` 触发，但实际配置 `use_sliding_window=false`；零元素张量来自 `ConvWide` 中已定义但未在其前向调用的通道注意力层。此次推理未出现异常退出或权重缺失警告。日志保存在 `outputs/official_demo.stderr.log`。

## 自有数据格式

JSON 包含 `instruction`、`vib_data`、`ref_data`。两路信号均应是 1 秒数据，`ref_data` 是相应工况的无故障参考。两个数组可以有不同采样率；官方示例长度分别为 12000 和 48000。程序通过 DCN 变换到长度 24000。

`instruction` 中必须保留且仅保留一个 `#state_place_holder#`，用于插入振动语义表示。示例 JSON 没有独立的真实类别字段，模型生成的类别不能单独用于证明分类准确率。

```bash
cd /home/huangyating/BearLLM
./scripts/run_demo.sh \
  --data /绝对路径/your_data.json \
  --seed 42 --max-new-tokens 2048 \
  --output outputs/your_response.txt
```

零能量、非有限值和非一维信号会被拒绝；不会自动推断采样率或信号时长，因此需要自行保证输入数据各对应 1 秒。
