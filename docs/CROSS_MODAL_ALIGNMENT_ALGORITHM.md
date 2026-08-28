# 跨模态时序对齐算法说明

## 1. 问题定义

给定两条不同模态、不同采样率、不同安装位置的时间序列：

- **触觉序列** \( x(t) \)：来自接触式加速度传感器 CSV
- **雷达序列** \( y(t) \)：从毫米波 IQ 原始数据提取的慢时间代理

目标：找到时间映射 \( \pi \)，使得对齐后的序列在物理事件上同步，用于后续跨模态融合。

---

## 2. 数据预处理

### 2.1 触觉特征提取

**输入**：`data_0.csv`，包含时间戳、三轴加速度/速度/位移及谱能量等 100+ 列。

**坐标系**（用户确认）：

| 轴 | 方向 |
|----|------|
| X | 朝向雷达（电机正前方） |
| Y | 朝右 |
| Z | 朝上 |

**特征选择策略**：

1. 筛选 X 轴相关列（列名含 "X" 且含 "加速度"/"速度"/"位移"/"谱能量"）
2. `auto` 模式：选择方差最大（信息量最丰富）的列
3. `x_acc_x` 模式：强制使用 `加速度X(g)` 列
4. 支持 `--forced_feature_col` 手动指定

**时间轴**：优先使用 `片上时间()` 列，计算相对秒数；否则使用行索引。

### 2.2 雷达慢时间代理提取

**输入**：TI mmWave ADC 原始 IQ `.bin` 文件（int16 交织 I/Q）。

**解析假设**：

```
每个 complex 样本 = 2 个 int16 (I, Q)
每个 chirp = num_rx × num_adcsamples 个 complex
文件按 chirp 顺序串联
```

**默认参数**（来自用户雷达配置）：

| 参数 | 值 |
|------|-----|
| num_adcsamples | 256 |
| num_rx | 4 |
| chirps_per_loop | 3 |
| loop_count | 64 |

**提取流程**：

```
bin (int16 I/Q)
  │
  ├─ 按 chirp 分块读取
  ├─ 每 chirp: mag = sqrt(I² + Q²), 对 rx×adc 求均值
  ├─ 每 loop (3 chirps): 对 chirp 幅值求均值
  └─ 输出: y[loop_idx] = 该 loop 的平均幅值
```

这产生一个 1D 慢时间序列，每个点对应一个 radar loop（时间分辨率约 frame_period / loop_count ≈ 20ms / 64 ≈ 0.31ms，但实际受 chirp 配置影响）。

### 2.3 下采样

为加速 DTW 计算，对两条序列做块平均下采样：

```python
def downsample_average(x, target_len):
    k = ceil(len(x) / target_len)
    return [mean(x[i*k:(i+1)*k]) for i in range(ceil(len(x)/k))]
```

默认目标长度：2000 点。

---

## 3. 阶段一：粗对齐（互相关滞后估计）

### 3.1 算法

对下采样后的序列 \( x_{ds} \)、\( y_{ds} \)：

1. Z-score 归一化
2. 在 \( L \in [-L_{max}, L_{max}] \) 范围内搜索：

\[
\tau^* = \arg\max_L \sum_i x_{ds}[i] \cdot y_{ds}[i + L]
\]

3. 按 \( \tau^* \) 平移序列，使重叠段对齐

### 3.2 物理意义

\( \tau^* \) 近似估计了：

- 采集启动时间差
- 结构波传播延迟（常数近似）
- 雷达处理群延迟（常数近似）

### 3.3 默认参数

- `max_corr_lag = 400`（在下采样后的索引空间中搜索）

---

## 4. 阶段二：精对齐（受限 DTW）

### 4.1 动态时间规整 (DTW)

DTW 找到最优单调路径 \( \pi = \{(i_k, j_k)\}_{k=1}^{K} \)，使得累积代价最小：

\[
\min_{\pi} \sum_{(i,j) \in \pi} C(i, j)
\]

### 4.2 代价函数

联合值与导数：

\[
C(i,j) = \alpha \cdot (x[i] - y[j])^2 + (1-\alpha) \cdot (\Delta x[i] - \Delta y[j])^2
\]

- \( \alpha = 0.7 \)（默认）：更重视幅值匹配
- 导数项抑制"形状相似但动力学错位"

### 4.3 Sakoe-Chiba 带约束

限制路径满足：

\[
|i - j| \leq w
\]

- `w = 80`（默认）：最大允许 80 个索引点的扭曲
- 防止非物理的时间倒流或过度拉伸

### 4.4 动态规划

```
D[0,0] = 0
D[i,j] = C(i,j) + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
```

回溯得到最优路径 \( (path\_x, path\_y) \)。

### 4.5 对齐映射

对雷达时间轴上的每个点 \( j \)：

```
aligned[j] = x[path_x 中映射到 j 的最后一个 i]
```

输出 `aligned_tactile_x.npy`：触觉特征映射到雷达慢时间轴。

---

## 5. 算法流程图

```
┌─────────────────┐     ┌─────────────────┐
│  触觉 CSV        │     │  雷达 IQ .bin    │
│  data_0.csv     │     │  int16 I/Q      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
  选择 X 轴特征列          chirp 幅值 → loop 均值
  x_raw[N]                 y_raw[M]
         │                       │
         ▼                       ▼
  downsample(N→2000)      downsample(M→2000)
  x_ds                       y_ds
         │                       │
         └───────────┬───────────┘
                     ▼
           互相关粗对齐 (τ*)
                     │
                     ▼
           平移后重叠段 x', y'
                     │
                     ▼
           受限 DTW (band=w, α=0.7)
                     │
                     ▼
           路径映射 → aligned_tactile_x
                     │
                     ▼
           输出 JSON/CSV/NPY
```

---

## 6. 复杂度分析

| 步骤 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 雷达 bin 读取 | O(chirps × adc × rx) | O(1)（memmap 流式） |
| 下采样 | O(N + M) | O(target_len) |
| 互相关 | O(L_max × min(N,M)) | O(1) |
| DTW | O(N × w)（带约束） | O(N × M) |
| 总计（默认参数） | ~秒级（2000点 DTW） | ~16 MB |

---

## 7. 参数调优建议

| 场景 | 建议 |
|------|------|
| 对齐效果差 | 增大 `dtw_band`（80→120→200） |
| 运行太慢 | 减小 `max_loops` 或 `radar_max_len` |
| 触觉信号弱 | 换用 `--forced_feature_col "X速度幅值(mm/s)"` |
| 需要相位对齐 | 后续增加 range bin 相位提取 + 残差补偿 |
| 多轴不确定 | 分别对 X/Y/Z 跑一遍，比较 `coarse_lag` 和 DTW cost |

---

## 8. 局限性与改进方向

### 当前局限

1. 雷达代理使用全通道幅值均值，未做 range-Doppler 处理
2. 时间轴为 loop 索引，未严格映射到秒
3. 未实现残差相位补偿步骤

### 改进方向

1. **Range FFT + 目标 bin 相位**：提取特定距离处的 \( \phi_r(t) \)
2. **分数延迟滤波**：DTW 后用 sinc 插值做亚采样级对齐
3. **学习式对齐**：用 CCA/神经网络学习跨模态映射
4. **在线对齐**：滑动窗口增量 DTW

---

## 9. 代码入口

| 文件 | 功能 |
|------|------|
| `cross_modal_alignment/align_cross_modal.py` | 端到端 CLI |
| `cross_modal_alignment/dtw_alignment.py` | DTW + 互相关核心算法 |
| `cross_modal_alignment/tests/run_tests.py` | 测试脚本 |
