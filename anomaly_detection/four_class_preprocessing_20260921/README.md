# 四分类预处理与服务器输入包

先读 [预处理方案与实测审计](预处理方案与实测审计.md)。当前结果是 **接触式已生成 DCN；雷达为距离门未确认的候选导出；未运行模型**。

## 已有结果

`results/` 约 323 MB，24 个 NPZ 及其 JSON、汇总与校验文件。可将整个本目录复制到服务器，不需要为了运行 encoder 上传约 9.8 GB 的 ADC 原始数据。重新选择雷达距离门则需要原始 bin。

- 四类标签：normal=0、inBroken=1、outBroken=2、roll=3。keep/bigNormal=-1，训练前必须过滤。
- 四类合计 58 个接触式窗口、412 个雷达窗口；所有文件合计 88 / 614。
- 每份 NPZ 的行号对应同名 JSON 的 `windows[row]`。必须保留 JSON 和汇总，不能只传特征数组而丢失录制身份。
- `recording_bags.json` 仅记录候选同录制关联，**不是同步窗口配对表**；文件名时间差不能用作逐采样包时间偏移。
- `input_manifest.json` 为原始文件 SHA256，`verification.json` 为全部 NPZ 校验与输出 SHA256，`runtime.json` 记录配置/脚本哈希与运行环境。
- `assignments.template.json` 给出八个四类接触式文件的实际文件名。工况、独立 run 和划分尚未确认，因此字段为 null；填好另存为 `assignments.verified.json`。不能把故障类别编码进 condition_id。

## 接触式输入

每份接触式 NPZ 包含：

- `raw_xyz[N,4000,3]`：每包中心一秒原值，轴序 x/y/z。
- `dcn[N,3,24000]`：按轴去均值的已处理 DCN，**不要再次 DCN**。
- `available_dcn_mask[24000]`：只有前 4000 维有测量带宽依据。
- `common_power[N,3,1591]` 和 `common_frequency_hz[1591]`：独立的 5–800 Hz 功率谱诊断分支，不能代替上述 DCN 输入原 FCN。
- `labels[N]`：本项目四类标签，和原模型十类编号不是同一编号体系。

用独立正常 reference 配对后，单个窗口输入 FCN 是 `[3,2,24000]`。三轴在 batch 维，第二维是 query/reference。不能把雷达放入 reference 通道。

服务器已有独立 FCN 权重时：

```bash
python server_encode_contact.py \
  --data results \
  --assignments assignments.verified.json \
  --bearllm-repo /home/huangyating/BearLLM \
  --weights /media/nas_users/huangyating/bearllm-runs/released-code-seed42/pretrain/fcn \
  --device cuda:0 \
  --output contact_embeddings_v1
```

上面仓库/权重路径来自旧实验记录，请按服务器实际位置调整；本机未连接服务器。脚本需要服务器现有 PyTorch，仅支持 `models/FCN.py` 的独立十类 FCN，严格加载 encoder/classifier，不支持把最终 LoRA 适配器直接当作独立 FCN。

`assignments.verified.json` 每个文件条目必须有非空 `condition_id`、`independent_run_id`、`split`。split 为 train/val/test/reference；reference 表示部署前独立固定校准库，不能把测试记录改名为 reference。相同真实 run 不能跨 split；正常参考须同工况、不同 run，来自 train 或固定 reference。雷达 counterpart 也必须放入相同划分，此关联需在训练代码中使用 bag 清单保持一致。

输出 `contact_embeddings.npz` 包含 `[N,3,128,47]` 特征图、`[N,3,128]` hidden 与 hidden_l2、`[N,3,10]` logits10。原始 hidden 给原头，L2 版本用于对比学习；不能混用。`pairing_coverage.json` 记录参考来源和缺参考样本，拒绝静默遗漏。

## 雷达输入

优先使用谱分支：

```python
from server_radar_io import spectral_inputs, frame_inputs

x, labels, frequency_hz = spectral_inputs(
    'results/20260920-192216-inBroken-115200.npz'
)
# x: [N,1591]，送入明确以1591维谱形状为输入的雷达encoder。
# 这不是BearLLM的24000维DCN；雷达encoder及128维投影需要训练/适配。

iq, valid, times, labels = frame_inputs(
    'results/20260920-192216-inBroken-115200.npz'
)
# iq: [N,20,12,2,192]；valid: [N,20,192]。
# 逐帧卷积 -> masked pooling -> 汇聚20帧；不能跨帧拼成连续信号。
```

原始 NPZ 中 IQ 维度为 `[N,20,192,12,2]`。空间单元顺序为 RX0的三个bin、RX1的三个bin，依此类推。最后一维是 I/Q，不是两种模态。

`phase_rate_frames[N,20,191,12]` 有对应 `valid_phase_mask`；单位 rad/s。`observed_time_s[20,192]` 保留每帧末尾的 4.5 ms 首点间隔。`raw_cell_power[N,12]` 保留原始幅值对照，`frame_log_power` 是没有跨频率去均值的谱；但它仍经过逐单元 IQ 幅值归一化，不能称为完全原始的绝对振动能量。

`coherent_log_shape` / `coherent_power` 为旧 mask 最小二乘思路的两秒谱，仅在跨帧相干性有依据时使用。主 `frame_log_shape` 的 0.5 Hz 网格是补零结果，实际分辨能力约 10.42 Hz。

当前距离自动候选：normal/inBroken 0.462m、roll 0.482m、outBroken 0.884m，必须先核对实际几何。未提供位置前不要将此版本当作故障泛化最终输入。

## 重跑与验证

预处理只用 NumPy/SciPy。本机实际使用 Python 3.9、NumPy 2.0.2、SciPy 1.13.1，独立环境位于 `/tmp/c2rllm-preprocess-venv`，未修改原项目环境。服务器可在独立环境安装 `requirements.txt`；不要为了这些脚本覆盖正在使用的模型环境。

```bash
python -m unittest -v test_preprocess.py
python verify_exports.py results
```

距离确认后，例如实际目标为 0.80m（这里只是命令示例，不是本批已确认距离）：

```bash
python preprocess.py \
  --data /path/to/data接触式和非接触式同时采集 \
  --config 雷达原始配置.cfg \
  --distance-m 0.80 \
  --output results_confirmed_geometry
python verify_exports.py results_confirmed_geometry
```

本目录归档一份原始 cfg 供服务器重跑；默认脚本配置路径指向项目原位置，独立上传后请显式使用 `--config 雷达原始配置.cfg`。若不同文件实际距离不同，应按真实记录分组，分别指定各自距离和新输出目录，后续合并清单；不能因故障类别而选不同距离门。

脚本拒绝非空输出目录，避免覆盖候选版或混入旧结果。仅处理单一模态可用 `--modality contact` 或 `--modality radar`。没有指定 `--distance-m` 时默认输出明确标注的宽距离候选版。

`preprocess.py` 严格检查当前 R0 配置，不能拿 R1/R2/R3 新配置生成的文件直接沿用旧时间轴。

## 交付状态

已验证数值处理与数据接口；尚未执行服务器 encoder、训练雷达学生或报告准确率。正常参考与工况元数据、雷达距离门是下一步实验必须补齐的信息。先保留各自的一秒/两秒窗口，再按稳态录制做集合级迁移，不需要在预处理阶段统一采集总时长或建立虚假的逐窗同步。
