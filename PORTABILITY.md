# 新主机运行与路径迁移

本仓库保留了实验归档和可重新运行的代码。推荐通过 `scripts/c2r.py` 创建新的运行目录；归档结果用于对照，不直接覆盖。新运行仍使用原来的数据，不能算新增物理实验。

## 1. 环境

完整复现环境使用 Python 3.12。当前服务器实际为 Python 3.12.14、PyTorch 2.11.0+cu128、NumPy 1.26.0、SciPy 1.15.2、scikit-learn 1.6.1。先进入仓库根目录：

```bash
conda create -n m2vllm python=3.12 pip -y
conda activate m2vllm
# Linux NVIDIA 主机：
python -m pip install -r environment/core-cuda128.txt
# 只使用 CPU 时改为：
# python -m pip install -r environment/core-cpu.txt
python scripts/c2r.py doctor
python scripts/c2r.py verify --output runs/migration_smoke.json
```

CPU 与 CUDA 两份核心依赖择一安装。NVIDIA 驱动不包含在 Python 环境中。Mac 本地 BIN/DAT 预处理只需对应工具目录下的 `requirements.txt`；不必安装上述 Linux CUDA 环境。

2026-09-29 已对两套「核心依赖＋LLM 额外依赖」分别执行 `pip --isolated install --dry-run --ignore-installed`：CPU 与 CUDA 12.8 均解析成功，包括 PyTorch 2.11.0、h5py 3.16.0 和传递依赖。现有环境中的已安装包未参与这次解析；完整报告见 `migration/verification/dependency_resolution.json`。这仅验证索引可取得匹配的包元数据且依赖约束可满足，**没有创建全新环境完成安装，也未验收另一台主机的驱动或运行结果**。`environment/resolved-*-with-llm.txt` 保存本次解析得到的完整版本清单。

CUDA 依赖文件用 `--find-links https://download.pytorch.org/whl/cu128/torch/` 只从 PyTorch 官方页面选择 Torch，其他包仍走 PyPI。这样避免宽泛额外索引把 cuDNN 等包转到缺少独立元数据的镜像；所有原来固定的包版本均未改变。

`doctor` 检查固定接触参考、接触 FCN 权重、保留准确率的历史分类头以及两种模态的特征缓存。Qwen 大模型和原始数据在报告中单独列为可选依赖；普通接触/雷达分类不需要加载 Qwen。要运行文本诊断或原 BearLLM 语言模型训练，再安装 `environment/llm-extra.txt`，并按仓库的大文件恢复说明恢复 Qwen 权重。

## 2. 无原始大数据也可以完成的复现

仓库中的 `contact/contact_features.npz` 和 `radar/features.npz` 已包含本轮模型训练所需输入。固定参考是从公开数据训练部分选出的原来那九条记录，缓存于 `four_class_chain_20260922/contact/external_references.npz`。迁移时已与原 MBHM 读取结果逐元素对比相等；没有重新选择参考。

下面每次使用一个**尚不存在**的工作目录：

```bash
# 接触式：重新拟合按转速留出的诊断头和接触频谱基线。
python scripts/c2r.py run contact-heads --workspace runs/contact_heads_01

# 雷达单模态：SVM、KNN、随机森林等。
python scripts/c2r.py run baselines --workspace runs/radar_baselines_01

# 雷达→接触特征：六种训练方法、三个留出转速、三个种子。
python scripts/c2r.py run alignment --workspace runs/alignment_01 --device cuda:0

# 先跑较小的计算复现：一个种子、完整损失；仍保留三个留出转速。
python scripts/c2r.py run alignment --workspace runs/alignment_small_01 \
  --device cuda:0 --methods full --seeds 17

# 依次执行接触诊断头、雷达传统基线、主要跨模态对齐实验。
python scripts/c2r.py run full --workspace runs/full_cached_01 --device cuda:0
```

主对齐默认使用 `--head archived --feature complex_shape`。其他现存实验接口是 `--head diagnostic`、`--feature phase_shape`、`--feature geometry_complex_shape`。`diagnostic` 会在该新目录先拟合训练折接触头，它没有通过原公开数据准确率下降不超过 0.5 个百分点的验收，不能替换已认证历史头。主对齐命令不会把诊断头当成原模型。

六种方法名称为 `head_CE contact_session_MSE full class_prototype_MSE class_code_MSE same_class_wrong_speed_MSE`；还会按原代码生成雷达直接分类 MLP 对照。损失权重、训练步数、划分、种子和输入顺序均保持归档代码的设置。`full` 是主链路计算复现，不自动执行每一个历史探索实验；历史的 RotLLM、UniFault、物理指标读取器和转换器代码及结果仍在原日期目录中。

运行目录保留原相对布局，例如 `runs/alignment_01/anomaly_detection/r2_validation_20260928/experiments/...`。模型、其他日期的源代码和辅助资源通过指向本仓库的链接引用；在该目录继续实验时应保持本仓库存在。归档的实验输出目录不会被复制进新实验目录，避免把旧输出误认为新结果。

## 3. 从已导出的 HDF5 重新跑信号处理

把省略的大数据从 Mac 或数据备份复制到新主机。设置 `C2R_DATA_ROOT` 为包含 `upload/` 和 `wide_cache_local/` 的目录，而不是某一个 HDF5 文件：

```bash
export C2R_DATA_ROOT="/你的数据盘/R2_processed"
python scripts/c2r.py run radar --from-exports --workspace runs/radar_frontend_01
python scripts/c2r.py run contact --from-exports --workspace runs/contact_frontend_01 --device cuda:0
python scripts/c2r.py run full --from-exports --workspace runs/full_from_exports_01 --device cuda:0
```

这些命令复现当前已审计的 144 段雷达、12 个接触会话、343 个接触窗口协议；源脚本会检查固定数量。未来新增数据应在新协议下修改数量检查及划分，不能把新数据无记录地塞进本轮归档。

若只剩原始 BIN/DAT，先按 `anomaly_detection/local_preprocessing_r2_20260928/README_本地预处理.md` 运行本地工具重新生成 HDF5。R2 帧内为 5 kHz，存在帧间缺口；不能把全部 chirp 当连续 4800 Hz 信号处理。接触式保存完整 4096 点、4000 Hz 的 XYZ 包。内圈 Z 轴截顶、20 cm ROI 需要重新导出的已知问题没有因迁移而消失。

## 4. BearLLM 公开数据回归与语言模型

安装可选依赖、恢复 Qwen 后，设置公开数据目录并生成仅含本机路径及公开常数的配置：

```bash
python -m pip install -r environment/llm-extra.txt
export C2R_MBHM_ROOT="/你的数据盘/mbhm_dataset"
python scripts/c2r.py configure
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
cd BearLLM
python scripts/train_bearllm.py --help
python scripts/evaluate_mbhm.py --help
```

`configure` 只会新建 `BearLLM/.env`，已有文件时拒绝覆盖。它使用 `DESCRIPTION_LEN=5`、`LLM_HIDDEN_SIZE=1536`、`SIGNAL_TOKEN_ID=151925`，这些是原公开配置中的常数。完整训练/评估参数和原结果见 `BearLLM/FULL_REPRODUCTION_zh.md` 与 `external/bearllm-runs/released-code-seed42` 下的运行清单；若更换参数，应保存为新的实验。

旧 `BearLLM/scripts/run_m2vllm.sh` 默认查找 `$HOME/miniconda3/envs/m2vllm/bin/python`，还会清除 `PYTHONPATH`；迁移后推荐如上直接使用激活环境中的 Python。不要照抄旧文档中的服务器绝对路径。

## 5. 路径适配与验证的边界

`c2r_paths.py` 显式把原 `/home/huangyating/...` 映射为仓库或本次运行目录，把 NAS 的模型目录映射为 `external/`，把大数据映射为 `C2R_DATA_ROOT`、`C2R_MBHM_ROOT`。可选 `C2R_ASSETS_ROOT` 和 `C2R_BEAR_RUNS_ROOT` 可以指定模型的其他位置。没有修改系统 HOME，也不要求新主机创建旧用户名。

迁移仅修改含旧路径、旧环境断言或需要解释历史路径的 Python 源码。修改清单及原/新 SHA 在 `migration_source_adaptations.json`。原始源码保存在 `migration/original_sources/`，包括原算法注释；AST 渲染后的执行副本可能改变排版和注释。历史 JSON、CSV、实验划分和预测数组保持原值，其中的旧路径是实验来源记录。

`verify` 运行 16 个本地解析测试、7 个雷达特征测试，重载一个实际雷达 checkpoint 并重放 32 个测试窗口，再以实际接触 FCN 对一个缓存接触包和九条参考执行 27 次视图推理。模型重放的主进程通过审计钩子禁止读取原服务器工程和 NAS 数据/模型路径；子进程的 23 个数值测试不继承这个钩子，但只使用测试生成的合成输入。

迁移验收还在新工作目录实际跑过接触诊断头，1,715 行窗口预测、60 行会话预测及各项指标与原结果逐值相同；雷达 SVM/KNN/随机森林及质量控制对照的三个转速折也均完成。CPU 上另跑完 `full + direct_MLP`、seed 17、三个转速留出折，共生成六个新 checkpoint，验证重新训练的入口和文件依赖可用。CPU 重训的随机训练轨迹和结果与原 GPU 归档不同，不能保证更换计算平台后重训逐位复现，也没有用这次小测替换原科研结果。记录见 `migration/verification/portable_alignment_training.json`。

接触端同时比较两件事：同 CPU、同缓存输入下，迁移前后的输出应一致；相对原先 float64 HDF5/GPU 归档，另行报告数值误差。不能把后者误差解释为算法修改，或用较宽阈值代替前者检查。这里的烟测证明迁移后的核心前端和模型接口可运行，不等同于在新主机重训完全部模型、重新测出新的物理泛化结果。
