# 跨模态时序对齐 — 复现指南

本文档提供在另一台服务器上完整复现本项目的操作步骤。

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 20.04+） |
| Python | 3.10+ |
| Conda | Miniconda 或 Anaconda |
| 磁盘空间 | ≥ 5 GB（不含原始数据；雷达 bin 约 1 GB） |
| 内存 | ≥ 8 GB |

---

## 2. 克隆仓库

```bash
git clone https://github.com/Lmumu1123/m2rotllm.git
cd m2rotllm
```

---

## 3. 创建 Conda 环境

```bash
# 创建环境
conda create -n rotllm python=3.12 -y
conda activate rotllm

# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 核心依赖：

```
torch>=2.2.0
lightning>=2.2.0
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.0.0
h5py>=3.10.0
transformers>=4.45.0
...
```

---

## 4. 准备数据

### 4.1 目录结构

建议将原始数据放在项目外的独立目录（避免误提交大文件）：

```bash
mkdir -p ~/data/anomaly_detection
```

目录结构：

```
~/data/anomaly_detection/
├── 20260807-153437-1786088077053250.bin    # 毫米波 IQ 数据 (~1 GB)
└── 15-34-23-384/
    ├── data_0.csv                           # 触觉传感器 CSV
    ├── data_0.bin
    ├── data.wplay
    └── Matlab/
        └── readMatData.m
```

### 4.2 数据获取

原始数据需从实验采集设备导出：

- **雷达 bin**：TI mmWave SDK 录制的 LVDS/ADC 原始 IQ 数据
- **触觉 CSV**：接触式传感器上位机导出的 `data_0.csv`

> 注意：原始数据文件不包含在 Git 仓库中（体积过大），需单独传输。

### 4.3 传输大文件（示例）

```bash
# 从源服务器 scp 到目标服务器
scp user@source:/home/huangyating/anomaly_detection/20260807-153437-1786088077053250.bin \
    ~/data/anomaly_detection/

scp -r user@source:/home/huangyating/anomaly_detection/15-34-23-384 \
    ~/data/anomaly_detection/
```

---

## 5. 运行跨模态对齐

### 5.1 快速测试（前缀数据，约 2 秒）

```bash
conda activate rotllm
cd m2rotllm

python cross_modal_alignment/align_cross_modal.py \
  --radar_bin ~/data/anomaly_detection/20260807-153437-1786088077053250.bin \
  --tactile_csv ~/data/anomaly_detection/15-34-23-384/data_0.csv \
  --max_loops 2000 \
  --dtw_band 60 \
  --radar_max_len 600 \
  --tactile_max_len 600 \
  --out_dir ./cross_modal_alignment/_align_out
```

### 5.2 完整对齐（更多 loop）

```bash
python cross_modal_alignment/align_cross_modal.py \
  --radar_bin ~/data/anomaly_detection/20260807-153437-1786088077053250.bin \
  --tactile_csv ~/data/anomaly_detection/15-34-23-384/data_0.csv \
  --max_loops 20000 \
  --dtw_band 80 \
  --out_dir ./cross_modal_alignment/_align_out_full
```

### 5.3 强制指定触觉特征列

若希望固定使用 X 轴加速度而非自动选择：

```bash
python cross_modal_alignment/align_cross_modal.py \
  --radar_bin ~/data/anomaly_detection/20260807-153437-1786088077053250.bin \
  --tactile_csv ~/data/anomaly_detection/15-34-23-384/data_0.csv \
  --feature_mode x_acc_x \
  --out_dir ./cross_modal_alignment/_align_out_xacc
```

或指定任意列名：

```bash
python cross_modal_alignment/align_cross_modal.py \
  --forced_feature_col "X位移幅值(um)" \
  ...
```

### 5.4 主要命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--radar_bin` | （必填） | 雷达 IQ bin 文件路径 |
| `--tactile_csv` | （必填） | 触觉 CSV 文件路径 |
| `--num_adcsamples` | 256 | 每 chirp ADC 采样点数 |
| `--num_rx` | 4 | 接收通道数 |
| `--chirps_per_loop` | 3 | 每 loop 的 chirp 数 |
| `--loop_count` | 64 | 每帧 loop 数 |
| `--max_loops` | 20000 | 最多处理的 loop 数 |
| `--dtw_band` | 80 | DTW Sakoe-Chiba 带宽度 |
| `--dtw_alpha` | 0.7 | DTW 值/导数代价权重 |
| `--max_corr_lag` | 400 | 粗对齐最大滞后搜索范围 |
| `--tactile_max_len` | 2000 | 触觉序列下采样目标长度 |
| `--radar_max_len` | 2000 | 雷达序列下采样目标长度 |
| `--out_dir` | `.` | 输出目录 |

---

## 6. 运行测试

### 6.1 合成数据单元测试（无需原始数据）

```bash
conda activate rotllm
cd m2rotllm
python cross_modal_alignment/tests/run_tests.py
```

预期输出：

```
[OK] test_best_lag_correlation
[OK] test_constrained_dtw_path_recovers_warp
[SKIP] test_end_to_end_on_real_data (set RADAR_BIN and TACTILE_CSV to enable)
[OK] test_end_to_end_on_real_data
ALL TESTS PASSED
```

### 6.2 含真实数据的端到端测试

```bash
export RADAR_BIN=~/data/anomaly_detection/20260807-153437-1786088077053250.bin
export TACTILE_CSV=~/data/anomaly_detection/15-34-23-384/data_0.csv
python cross_modal_alignment/tests/run_tests.py
```

---

## 7. RotLLM 主项目复现（可选）

若需复现 RotLLM 论文主流程：

### 7.1 准备权重与数据集

RotLLM 依赖外部大文件（通过软链接引用，不包含在仓库中）：

```bash
# 示例：创建数据目录软链接
ln -s /path/to/RotLLM/weights ./weights
ln -s /path/to/RotLLM/qwen_weights ./qwen_weights
ln -s /path/to/RotLLM/mbhm_dataset ./mbhm_dataset
```

### 7.2 配置环境变量

```bash
cp .env.example .env   # 若提供
# 编辑 .env 设置数据路径、模型路径等
```

### 7.3 运行 smoke test

```bash
python run_smoke.py
python run_inference_smoke.py
```

### 7.4 MBHM 预训练

```bash
python run_mbhm_pretrain.py
```

---

## 8. 常见问题

### Q1: `ModuleNotFoundError: No module named 'numpy'`

确保已激活 conda 环境：

```bash
conda activate rotllm
```

### Q2: `DTW failed (path not found within band)`

增大 `--dtw_band` 参数，例如 `--dtw_band 120`。

### Q3: 雷达 bin 解析失败（size not divisible）

检查 `--num_adcsamples` 和 `--num_rx` 是否与雷达配置一致。当前默认 `256` 采样点 × `4` 通道。

### Q4: 触觉特征列全为零

CSV 中部分统计量列可能为 0（设备未计算）。使用 `--feature_mode auto` 会自动选择方差最大的 X 轴相关列。

### Q5: 内存不足

减小 `--max_loops` 或增大下采样（减小 `--radar_max_len` / `--tactile_max_len`）。

---

## 9. 验证对齐效果

检查输出目录中的文件：

```bash
# 查看对齐元信息
cat cross_modal_alignment/_align_out/alignment_info.json

# 查看对齐数据前几行
head -5 cross_modal_alignment/_align_out/aligned_pair.csv
```

`alignment_info.json` 示例：

```json
{
  "selected_tactile_feature_col": "X位移幅值(um)",
  "coarse_lag_on_ds": 12,
  "dtw_band": 60,
  "used_len_for_dtw": 588
}
```
