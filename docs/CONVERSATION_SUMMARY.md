# 跨模态时序对齐项目对话整理

本文档整理自 Cursor 对话，记录了从问题提出、算法设计、代码实现到测试验证的完整过程。

---

## 1. 背景与问题

### 1.1 研究场景

在电机健康监测/异常检测场景中，需要融合两类传感器数据：

| 传感器 | 安装位置 | 数据类型 | 文件示例 |
|--------|----------|----------|----------|
| 77GHz 毫米波雷达 | 电机正前方 | IQ 原始 ADC 数据（`.bin`） | `20260807-153437-1786088077053250.bin` |
| 接触式加速度传感器 | 电机正上方 | CSV 特征数据 | `15-34-23-384/data_0.csv` |

### 1.2 核心矛盾：时空耦合

接触式与非接触式数据的安装点/测量点不在同一点，且：

- 接触式加速度计：单点测量
- 毫米波雷达：波束面域测量

二者存在本质的空间基准差异。仅依赖时间戳同步无法消除：

1. **结构波传播延迟**（振动从一点传到另一点需要时间）
2. **雷达信号处理固有群延迟**（滤波、FFT 等处理引入相位滞后）

这会导致微观相位错位与跨模态校验失效。

### 1.3 目标

设计并实现一套**超越简单时间戳**的深层时序对齐方案，使跨模态数据在时序上对齐，便于后续融合、异常检测等操作。

---

## 2. 算法方案讨论

### 2.1 总体思路：两阶段对齐

```
原始数据
   │
   ├─ 触觉 CSV ──► 构造 1D 特征序列 x(t)
   │
   └─ 雷达 IQ bin ──► 构造慢时间 1D 代理序列 y(t)
           │
           ▼
   阶段1：粗对齐（互相关估计常延迟 τ*）
           │
           ▼
   阶段2：精对齐（受限 DTW 非线性时间规整）
           │
           ▼
   （可选）残差相位/群延迟补偿
           │
           ▼
   对齐后的跨模态数据对
```

### 2.2 特征构造

**触觉侧：**

- 包络类：\( s_c(t) = \text{HPF}(\|a_{xyz}(t)\|) \)
- 导数类：\( s_c'(t) = \Delta s_c(t) \)

**雷达侧：**

- 从 IQ 数据提取慢时间序列
- 每 chirp 计算 I/Q 幅值均值，再按 loop 汇聚
- 可选：相位导数 \( \Delta\phi_r(t) \) 用于相位补偿

### 2.3 受限 DTW

代价函数（值 + 导数联合）：

\[
C(i,j) = \alpha (x[i]-y[j])^2 + (1-\alpha)(\Delta x[i]-\Delta y[j])^2
\]

约束：Sakoe-Chiba 带，限制最大时间扭曲，保证物理合理性。

### 2.4 传感器坐标系讨论

用户确认触觉传感器安装方向：

- **X 轴**：朝向雷达（电机正前方）→ 与雷达径向测量方向一致
- **Y 轴**：朝右
- **Z 轴**：朝上

因此对齐时优先使用 **X 轴相关特征**（加速度、速度、位移、谱能量等）。

---

## 3. 雷达配置参数

用户提供的 77GHz 毫米波雷达配置：

```
flushCfg
sensorStop
dfeDataOutputMode 1
channelCfg 15 7 0
adcCfg 2 1
adcbufCfg -1 0 1 1 1
profileCfg 0 77 420 5 80 0 0 49.99 1 256 3430 0 0 48
chirpCfg 0 0 0 0 0 0 0 1
chirpCfg 1 1 0 0 0 0 0 2
chirpCfg 2 2 0 0 0 0 0 4
frameCfg 0 2 64 20000 100 1 0
lowPower 0 0
lvdsStreamCfg -1 0 1 0
calibMonCfg 1 1
monCalibReportCfg 0 0 0
sensorStart
```

关键参数解读：

| 参数 | 值 | 含义 |
|------|-----|------|
| `numAdcSamples` | 256 | 每个 chirp 的 ADC 采样点数 |
| `numRx` | 4 | 接收通道数（channelCfg 15 = 4 RX） |
| `chirps_per_loop` | 3 | 每个 loop 3 个 chirp（chirpCfg 0/1/2） |
| `loop_count` | 64 | 每帧 64 个 loop（frameCfg 第3参数） |
| `frame_period` | 20000 μs | 帧周期 20ms → 50 Hz 帧率 |

---

## 4. 代码实现记录

### 4.1 初始实现位置

代码最初实现在 `/home/huangyating/anomaly_detection/`：

```
anomaly_detection/
├── __init__.py
├── dtw_alignment.py          # DTW + 互相关
├── align_cross_modal.py      # 端到端对齐流水线
└── tests/run_tests.py        # 测试脚本
```

### 4.2 迁移到 RotLLM 仓库

为便于版本管理与跨服务器复现，代码已迁移至：

```
RotLLM/cross_modal_alignment/
├── __init__.py
├── dtw_alignment.py
├── align_cross_modal.py
└── tests/run_tests.py
```

### 4.3 测试验证结果

在 `conda activate rotllm` 环境下运行测试，全部通过：

```
[OK] test_best_lag_correlation
[OK] test_constrained_dtw_path_recovers_warp
[OK] test_end_to_end_on_real_data
ALL TESTS PASSED
```

端到端测试自动选择的触觉特征列为：`X位移幅值(um)`。

---

## 5. 输出文件说明

对齐脚本运行后在 `out_dir` 生成：

| 文件 | 说明 |
|------|------|
| `alignment_info.json` | 对齐元信息（选用特征列、序列长度、粗延迟、DTW 参数等） |
| `aligned_pair.csv` | 雷达步索引 + 雷达代理值 + 对齐后触觉值 + 映射索引 |
| `radar_proxy_y_ds.npy` | 下采样后的雷达慢时间代理序列 |
| `aligned_tactile_x.npy` | 映射到雷达时间轴上的触觉特征 |
| `mapped_tactile_indices.npy` | DTW 路径映射的触觉索引 |

---

## 6. 后续可扩展方向

1. **物理时间轴映射**：将 loop 索引严格映射到秒级时间戳
2. **Range-Doppler 完整处理链**：替代当前幅值均值代理，使用目标 range bin 相位
3. **残差相位补偿**：DTW 后做分数延迟滤波或相位偏置消除
4. **多轴融合**：将 X/Y/Z 旋转到雷达径向分量后联合对齐

---

## 7. 相关文档索引

- [复现指南](REPRODUCTION_GUIDE.md)
- [算法说明](CROSS_MODAL_ALIGNMENT_ALGORITHM.md)
- [测试文档](TESTING_GUIDE.md)
