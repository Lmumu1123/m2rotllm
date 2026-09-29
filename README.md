# C2R：接触振动模型与毫米波雷达的跨模态诊断

这是截至 **2026-09-29** 的完整科研交接快照。`C2R` 从独立根提交建立，不以原仓库 `main` 为父提交，也没有复制 `main` 的工作树。本分支用于迁移、复核和继续实验。

**最新 R2 跨转速验证尚未达到接触式 >98%、雷达 >95% 的目标。** 历史八录制实验、同工况重复实验和最新跨转速实验的协议不同，不能直接比较准确率。请先读 [最新实验汇报](anomaly_detection/r2_validation_20260928/本轮实验汇报.md) 和 [项目导览](migration/docs/README_项目导览.md)。

## 包含的内容

| 目录 | 内容 |
|---|---|
| `anomaly_detection/` | 全部接触/雷达预处理、基线、蒸馏、对照实验、权重、特征缓存、逐样本预测、汇总结果及失败实验记录 |
| `BearLLM/` | 实际使用的模型定义、训练/推理程序、历史 smoke 结果；不是只有一个分类头 |
| `research_proposals/` | 研究方案、导师汇报、原理说明、采集方案、演示及参考实现 |
| `external/bearllm-runs/` | NAS 上的 BearLLM 复现训练、优化器状态、导出模型和完整评测结果 |
| `external/bearllm-assets/` | 官方适配器、Qwen 配置/词表、公开数据元信息；大文件按下文还原 |
| `migration/model_parts/` | Qwen 基座的完整无损分片；不是下载地址占位文件 |
| `migration/` | 模型清单、文件校验、环境版本、原始源码、路径修改记录、迁移验证及清理说明 |
| `scripts/`、`c2r_paths.py` | 新机器检查、还原、复核和隔离实验入口 |

原始 `.bin`/`.DAT`、本批 NAS 的 HDF5 与宽 ROI 缓存、13 GB 的 MBHM 信号集没有加入 Git。已有研究目录中的中间 NPZ、训练/测试划分和所有模型保留。精确清单见 [excluded_files.csv](migration/manifests/excluded_files.csv)。环境目录、缓存、嵌套 `.git`、含本机配置的 `.env` 不迁移。

## 在另一台机器启动

建议先准备 **30 GB 可用空间**（不含额外数据集）；Git 对象、工作树、完整模型还原和新实验会各占空间。

```bash
git clone --branch C2R --single-branch https://github.com/Lmumu1123/m2rotllm.git C2R
cd C2R
conda create -n m2vllm python=3.12 pip -y
conda activate m2vllm

# Linux CPU：用于验证与小规模重跑
python -m pip install -r environment/core-cpu.txt
python scripts/c2r.py doctor
python scripts/verify_repository.py
python scripts/c2r.py verify --output ../c2r_verify.json
```

Linux NVIDIA 机器改装 `environment/core-cuda128.txt`，执行时用 `--device cuda:0`。CPU 和 CUDA 两组依赖均已忽略现有安装进行完整解析并通过，见 [解析记录](migration/verification/dependency_resolution.json)；这不等同新主机驱动验收。原机器是 Python 3.12、PyTorch 2.11.0+cu128；[原环境清单](migration/environment/runtime_original.json) 和 [完整包版本](migration/environment/pip-freeze-versions.txt) 用于对照。不能直接把服务器的 `miniconda3/` 拷到 Mac 使用。Mac 的原始数据导出使用 [本地预处理工具](anomaly_detection/local_preprocessing_r2_20260928/README_本地预处理.md) 及其独立依赖；本迁移验收针对 Linux，未宣称在所有平台重新安装测试过。

`verify` 会执行原始数据解析与雷达特征数值测试、加载真实接触权重和固定健康参考、重放保存的雷达模型。模型重放会拒绝读取旧服务器的项目及 NAS 资源。更完整的 171 个 checkpoint 审计：

```bash
python scripts/audit_all_checkpoints.py --output ../c2r_all_checkpoint_audit
```

不运行语言生成时，诊断分类、缓存特征训练和上述检查都不需要完整 Qwen。运行原 BearLLM 语言生成/LoRA 时：

```bash
python -m pip install -r environment/llm-extra.txt
python scripts/restore_artifacts.py --qwen
python scripts/c2r.py configure
# 如需读取两份完整历史逐样本评测 JSONL，再执行：
python scripts/restore_artifacts.py --results
```

还原脚本会校验每个分片和重组文件的 SHA-256。完整基座固定为 `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。**所有实际训练的适配器、雷达 encoder、分类头、映射参数和训练恢复状态均在分支内。** 公开 MBHM 信号只有进行完整源域训练/回归时才需要另行下载；九条健康参考已缓存。

## 继续实验，不覆盖历史结果

所有 `run` 命令要求一个**尚不存在**的工作目录。下列命令使用已存特征重跑，属于计算复现，不增加独立物理测试样本：

```bash
# 雷达传统分类基线
python scripts/c2r.py run baselines --workspace ../c2r_baselines_01

# 主配置：冻结旧接触头，六种训练目标加直接分类对照，三个种子
python scripts/c2r.py run alignment --workspace ../c2r_alignment_01 --device cuda:0

# 接触诊断头 + 基线 + 主输入下的对齐（不是全部历史实验的一键重跑）
python scripts/c2r.py run full --workspace ../c2r_main_chain_01 --device cuda:0
```

`--head diagnostic` 是只使用训练折接触数据的新四分类头，尚未满足源域准确率下降 ≤0.5 个百分点的验收，不能据此替换已验收的旧头。相位/几何特征对照可用 `--feature phase_shape` 或 `--feature geometry_complex_shape`，每次换一个新 workspace。完整方法与负对照、物理读取器以及训练/测试划分见 [实验源码](anomaly_detection/r2_validation_20260928/experiments/run_alignment.py)。其他诊断和历史实验入口见 [安全运行说明](migration/docs/归档保护与复现命令审计.md)。

若已将本地导出的 `upload/` 和 `wide_cache_local/` 复制到新机器：

```bash
export C2R_DATA_ROOT=/你的磁盘/R2数据
python scripts/c2r.py run full --from-exports --workspace ../c2r_from_exports_01 --device cuda:0
```

`C2R_DATA_ROOT` 指向**包含** `upload/` 的目录。接触 DAT 持续覆盖同故障同转速的多次雷达测量；仍按接触会话分组划分，不能把共享同一 DAT 的雷达录制随机分到训练和测试两边。采集、数据重新导出、20 cm ROI 补导、2 g 截顶问题见 [数据恢复与后续实验](migration/docs/数据恢复与后续实验.md)。

## 迁移范围与模型保留

- [模型与参数清单](migration/model_inventory.json)：各文件角色、原始位置及 SHA-256；包括历史模型，不能把所有文件都视作可部署最佳模型。
- [源文件快照](migration/manifests/source_snapshot.csv)：原始文件与新仓库路径对应。结果 JSON/CSV 中的旧路径作为历史记录保留；程序通过 `c2r_paths.py` 显式解析。
- [路径适配说明](PORTABILITY.md)：修改仅发生在迁移副本，未重训或改动模型权重。原始源码另保存在 `migration/original_sources.tar.gz` 和适配记录内。
- [服务器清理建议](migration/docs/服务器清理建议.md)：本次只整理，未删除原数据/代码。先在新机器复核，再按清单决定清理。
- [第三方来源与许可](migration/docs/THIRD_PARTY_NOTICES.md)：保留各上游许可；本交接快照没有给第三方代码重新赋予许可证。

后续实验优先解决输入质量与跨转速接触教师，再检验真实接触监督是否优于类别原型。历史高分、未通过的诊断头和无效中间实验均保留并注明，不能挑选少数高分作为最终论文结论。
