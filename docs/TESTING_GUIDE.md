# 跨模态时序对齐 — 测试文档

## 1. 测试概述

本项目包含三层测试：

| 层级 | 测试内容 | 是否需要原始数据 |
|------|----------|-----------------|
| L1 单元测试 | 互相关滞后估计 | 否（合成数据） |
| L2 单元测试 | 受限 DTW 路径恢复 | 否（合成数据） |
| L3 集成测试 | 端到端对齐流水线 | 是（雷达 bin + 触觉 CSV） |

测试脚本：`cross_modal_alignment/tests/run_tests.py`

---

## 2. 运行方式

### 2.1 仅合成数据测试

```bash
conda activate rotllm
cd m2rotllm
python cross_modal_alignment/tests/run_tests.py
```

### 2.2 含真实数据测试

```bash
export RADAR_BIN=/path/to/radar.bin
export TACTILE_CSV=/path/to/data_0.csv
python cross_modal_alignment/tests/run_tests.py
```

---

## 3. 测试用例详情

### 3.1 test_best_lag_correlation

**目的**：验证互相关能正确估计已知延迟。

**方法**：

1. 生成 500 点随机序列 `x`
2. 构造延迟 37 点的 `y`：`y[37:] = x[:-37]` + 小噪声
3. 调用 `best_lag_correlation(x, y, max_lag=100)`
4. 断言估计延迟与真实延迟误差 ≤ 2

**通过标准**：`|est - 37| ≤ 2`

### 3.2 test_constrained_dtw_path_recovers_warp

**目的**：验证受限 DTW 能恢复非线性时间扭曲。

**方法**：

1. 生成 400 点正弦复合信号 `x`
2. 构造 320 点单调扭曲映射 `idx`（带噪声）
3. `y = x[idx] + 噪声`
4. 运行 `constrained_dtw_path(x, y, w=120)`
5. 按路径重建 `aligned`，计算 MSE

**通过标准**：`MSE(aligned, y) < 0.08`

### 3.3 test_end_to_end_on_real_data

**目的**：验证完整流水线能在真实数据上运行并产出预期文件。

**前置条件**：设置环境变量 `RADAR_BIN` 和 `TACTILE_CSV`。

**方法**：

1. 调用 `align_cross_modal.py`，参数：
   - `--max_loops 2000`
   - `--dtw_band 60`
   - `--radar_max_len 600`
   - `--tactile_max_len 600`
2. 检查输出文件存在：
   - `alignment_info.json`
   - `aligned_pair.csv`

**通过标准**：脚本退出码 0，输出文件存在。

---

## 4. 预期输出

### 4.1 全部通过

```
[OK] test_best_lag_correlation
[OK] test_constrained_dtw_path_recovers_warp
[OK] test_end_to_end_on_real_data
ALL TESTS PASSED
```

### 4.2 跳过端到端（无原始数据）

```
[OK] test_best_lag_correlation
[OK] test_constrained_dtw_path_recovers_warp
[SKIP] test_end_to_end_on_real_data (set RADAR_BIN and TACTILE_CSV to enable)
[OK] test_end_to_end_on_real_data
ALL TESTS PASSED
```

---

## 5. 手动验证对齐质量

测试通过后，建议手动检查对齐效果：

### 5.1 检查元信息

```bash
python -c "
import json
with open('cross_modal_alignment/_test_align_out/alignment_info.json') as f:
    info = json.load(f)
for k,v in info.items():
    print(f'{k}: {v}')
"
```

关注字段：

- `selected_tactile_feature_col`：自动选择的特征列
- `coarse_lag_on_ds`：粗对齐滞后（应在合理范围）
- `used_len_for_dtw`：参与 DTW 的长度

### 5.2 检查对齐数据

```bash
head -10 cross_modal_alignment/_test_align_out/aligned_pair.csv
```

CSV 列：

| 列名 | 说明 |
|------|------|
| `radar_step_idx_in_dtw` | 雷达时间步索引 |
| `radar_proxy_y` | 雷达慢时间代理值 |
| `aligned_tactile_x` | 对齐后的触觉特征值 |
| `mapped_tactile_idx_in_dtw` | DTW 映射的触觉索引 |

### 5.3 可视化（可选，需 matplotlib）

```python
import numpy as np
import pandas as pd

df = pd.read_csv("cross_modal_alignment/_test_align_out/aligned_pair.csv")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax[0].plot(df["radar_proxy_y"], label="radar")
ax[0].legend()
ax[1].plot(df["aligned_tactile_x"], label="tactile (aligned)", color="orange")
ax[1].legend()
plt.tight_layout()
plt.savefig("alignment_check.png", dpi=150)
print("Saved alignment_check.png")
```

---

## 6. 故障排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `DTW failed (path not found within band)` | 带宽度不足 | 增大 `dtw_band` 至 120+ |
| `bin size not divisible` | 雷达参数不匹配 | 检查 `num_adcsamples`/`num_rx` |
| `forced_feature_col not found` | CSV 列名不匹配 | 用 `head -1 data_0.csv` 查看列名 |
| `ModuleNotFoundError: pandas` | 依赖未安装 | `pip install pandas` |
| 端到端 SKIP | 未设置环境变量 | `export RADAR_BIN=... TACTILE_CSV=...` |

---

## 7. 持续集成建议

若后续接入 CI（GitHub Actions 等），建议：

```yaml
# .github/workflows/test.yml（示例）
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: conda-incubator/setup-miniconda@v3
        with:
          activate-environment: rotllm
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python cross_modal_alignment/tests/run_tests.py
```

端到端测试因依赖大文件，建议在 CI 中仅运行 L1/L2 合成数据测试。

---

## 8. 测试覆盖矩阵

| 模块 | 函数/入口 | L1 | L2 | L3 |
|------|-----------|----|----|-----|
| `dtw_alignment.py` | `best_lag_correlation` | ✅ | | |
| `dtw_alignment.py` | `constrained_dtw_path` | | ✅ | |
| `align_cross_modal.py` | `load_tactile_feature` | | | ✅ |
| `align_cross_modal.py` | `extract_radar_slow_proxy_from_bin` | | | ✅ |
| `align_cross_modal.py` | `main()` (CLI) | | | ✅ |
