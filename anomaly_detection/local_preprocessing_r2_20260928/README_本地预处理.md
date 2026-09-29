# R2 新数据：Mac 本地预处理与上传说明

更新：2026-09-28。适用硬件：AWR1843 + DCA1000；本工具不连接雷达，也不下发配置。

## 1. 这次应该上传什么

**雷达上传保留相位的复数距离信号，接触式上传完整 XYZ 数据包。暂时不把它们压成模型 embedding，也不只上传图片、幅度谱或 128 维特征。**

已经确认的采集方式：

- 雷达：4 类 × 3 转速 × 3 距离 × 4 次，计划 144 段录制。
- 接触式：4000 Hz；同故障、同转速下的一条 DAT 连续覆盖全部距离和重复。
- 如果每个故障/转速恰好一条 DAT，则共有 12 条接触长记录。它们不是 144 条独立接触式录制。
- 同工况唯一 DAT 可以与该工况的多段雷达关联；此关联表示属于同一连续采集过程，**尚不表示每个 1 秒接触窗口已经与每个 2 秒雷达窗口精准同步**。

本地做确定性的文件解析、距离处理和质量记录；服务器再做频谱/时频/物理指标提取、训练划分、标准化和 encoder 训练。这能保留后续研究所需的选择空间。

## 2. R2 的时间不能按旧脚本处理

| 项目 | 已确认 R2 数值 | 用途 |
|---|---:|---|
| 每个 chirp 的 ADC 点数 | 256 | 快时间距离 FFT |
| ADC 快时间采样率 | 3.43 MHz | 距离坐标计算，不是振动采样率 |
| 接收通道 | 4 RX，单 TX | 分开保存四路 |
| chirp 周期 | 120 + 80 = 200 μs | 帧内振动采样率 5000 Hz |
| 每帧 chirp 数 | 192 | 按帧保留 |
| 帧周期 | 40 ms，即 25 帧/s | 从 cfg 计算时间 |
| 平均实际观测数 | 4800 chirp/s | 不能据此当作均匀的 4800 Hz 波形 |
| 按采样槽计的帧间缺口 | 1.6 ms，8 个槽，占 4% | 保留缺测关系 |
| 2 秒雷达窗口 | 50 帧，9600 个实际观测 | 旧脚本的 20 帧只有 0.8 秒 |
| 每帧纯 ADC 字节数 | 786432 | 检查完整帧及尾部 |
| 平均纯 ADC 数据率 | 19.6608 MB/s | 十进制，未计网络包头 |

第 `m` 帧第 `n` 个 chirp 的名义相对时刻：`t = m × 0.04 + n × 0.0002`，`n=0,…,191`。一个 2 秒窗口最后一个观测在 1.9982 秒；若铺到 5 kHz 网格上，共 10000 个槽，其中 400 个没有观测。H5 保存帧起点和帧内时间，不自动填这些槽。

这里的时间来自 cfg，前提是 ADC 文件正确从帧边界开始。它不是设备逐帧硬件时间戳；丢包仍应结合采集日志判断。

## 3. 在 Mac 上运行

### 3.1 安装依赖

将整个工具包解压到 Mac，进入工具包目录。Python 建议 3.10 或更新版本。只需要 NumPy 和 h5py，不需要 GPU、PyTorch 或下载模型。

```bash
cd "/Users/anthea/Desktop/local_preprocessing_r2_20260928"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -v
```

以下示例假定新原始数据放在 `/Users/anthea/Desktop/C2RLLM/new_R2_data`。**把这个路径替换成实际路径**；不要把旧 R0 数据混到新批次目录中。输入目录允许中文和空格，命令里保持双引号。

### 3.2 自动生成文件关联表

```bash
python build_manifest.py \
  --data-root "/Users/anthea/Desktop/C2RLLM/new_R2_data" \
  --output manifest.csv
```

脚本识别用户提供的命名格式，例如：

```text
20260927-183015-normal-1000r-20cm.bin
2026_9_27_20-48-50-out-2000r.DAT
```

类别名支持 `normal`、`in/inBroken/inner`、`out/outBroken/outer`、`roll/ball`，输出统一为 `normal/inBroken/outBroken/roll`。这只是名字映射，不是模型分类。

生成：

- `manifest.csv`：每段雷达一行，同工况的多行指向同一 `contact_id` 和 DAT。
- `manifest_review.csv`：未识别文件、同工况多个 DAT、重复编号等需要核对的项。
- `manifest_summary.json`：录制数、共享接触会话数、36 个工况的四次录制是否齐全。

**先核对这三个文件。** 理想情况下是 144 行雷达和 12 个接触会话；若同工况有多个 DAT，脚本不会凭“时间最近”猜测。需要在 CSV 中填写正确的 `contact_path/contact_id`。

没有显式重复编号时，脚本按同工况文件名时间排序临时编号 `rep1…rep4`。这个编号不证明四次之间停机、重装或独立重复，也不保证不同距离的 `rep1` 属于同一次实验轮次。若实际实验有统一轮次，按实验记录改 `repeat/session_id`。

`session_id`、`bearing_id`、`environment`、`contact_unit` 能补则补；不知道就留空，不能猜。环境列记录本轮实际地点；距离变化不能替代环境变化实验。

若 DCA 把同一录制分成多个 BIN，不能把每个分卷当成独立录制。将该行 `radar_path` 留空，`radar_parts` 填为按真实顺序排列的 JSON 数组，例如：

```json
["capture_part0.bin", "capture_part1.bin"]
```

脚本会跨分卷边界连续读取，不要求每个分卷恰好结束在帧边界。这只适用于同一 ADC 流的连续分卷；不能拼接不同次录制或未处理的 UDP 包。

### 3.3 使用本次实际 cfg 做预检查

随包 `R2_reference.cfg` 与用户确认的 R2 一致，但优先使用采集时真正保存的 cfg。各行如果有单独 cfg，可以填写 `cfg_path`；否则用下面的全局 `--cfg`。

```bash
python preprocess_local.py process \
  --data-root "/Users/anthea/Desktop/C2RLLM/new_R2_data" \
  --manifest manifest.csv \
  --cfg R2_reference.cfg \
  --radar-format headerless_reordered_adc \
  --output "/Users/anthea/Desktop/C2RLLM/R2_export" \
  --dry-run
```

`--radar-format headerless_reordered_adc` 的含义：输入是已经去网络包头、重排后的 ADC 字节流，使用已有采集程序的 IIQQ / RX 排列。它不是转换命令。如果保存的是带 UDP 包头的网络抓包，或采集程序改变了字节顺序，不能用该参数强行解释；需要先按实际格式解析。只有“文件大小能整除帧大小”不足以证明格式正确。

默认 `iiqq_neg` 沿用项目此前核验的 `I−jQ` 约定，另支持 `iiqq_pos`。工具会报告已知距离处的正负谱能量比较；若出现 `negative_frequency_mirror_dominates_check_IQ_convention`，先核查采集程序/距离轮廓，再决定是否修改 CSV 中的 `iq_order`。不要按哪种约定分类更准来选择。

### 3.4 先处理三个距离的小样

```bash
python preprocess_local.py process \
  --data-root "/Users/anthea/Desktop/C2RLLM/new_R2_data" \
  --manifest manifest.csv --cfg R2_reference.cfg \
  --radar-format headerless_reordered_adc \
  --recording normal_1000r_20cm_rep1 \
  --recording normal_1000r_40cm_rep1 \
  --recording normal_1000r_80cm_rep1 \
  --max-frames 50 \
  --output "/Users/anthea/Desktop/C2RLLM/R2_pilot"

python make_qc_report.py --output "/Users/anthea/Desktop/C2RLLM/R2_pilot"
open "/Users/anthea/Desktop/C2RLLM/R2_pilot/qc_report.html"
```

每段只处理前 2 秒雷达，接触长记录仍完整导出并且只导出一次。预览结果会标明 `is_pilot=true`，不能混入正式全量训练集。**正式输出使用另一个目录**。

若清单中 ID 不同，使用 CSV 中的实际 ID。小样主要检查距离轮廓、格式、选区和文件结构，不证明整个录制质量良好。可以再加上内圈、外圈和滚动体代表样本，但不看分类成绩来调选区。

重点看：所选距离是否接近实际被照射电机外壳，是否触及搜索区边界，是否出现大量全零 chirp 或接近 ADC 极限的值，以及接触完整包数是否为零。可先把 `qc_report.html/qc_index.csv/summary.json` 上传，让服务器侧复核后再处理全量。

### 3.5 全量导出

```bash
python preprocess_local.py process \
  --data-root "/Users/anthea/Desktop/C2RLLM/new_R2_data" \
  --manifest manifest.csv --cfg R2_reference.cfg \
  --radar-format headerless_reordered_adc \
  --output "/Users/anthea/Desktop/C2RLLM/R2_export"

python make_qc_report.py --output "/Users/anthea/Desktop/C2RLLM/R2_export"
python preprocess_local.py verify --output "/Users/anthea/Desktop/C2RLLM/R2_export"
```

默认按 4 帧一批流式处理，不把整个大 BIN 加载进内存。内存紧张可加 `--batch-frames 1`。同参数再次运行会跳过完成且来源/配置一致的记录。验证命令会检查文件 SHA256、整组是否缺失和共享接触引用是否完整。

如果之前中途失败，工具不会把半成品当作成功结果；保留错误日志，用新的输出目录重跑失败的 `--recording`。参数、源文件、清单内容或脚本变了也应使用新输出目录，避免不同版本混用。全量汇总读取输出目录下所有已完成雷达，而非只显示最后一个小批次。

## 4. 雷达究竟做了哪些预处理

```text
原始 BIN 的 ADC IQ
  → 按 IIQQ 和 4RX 解码，恢复 frame × chirp × RX × ADC
  → 每个 chirp 的快时间去均值、Hann 窗、512 点距离 FFT
  → 以实测距离为先验，在 ±12 cm 的范围内选目标
  → 保存全部帧的 4RX × 5 个相邻距离单元的复数值
  → 同时在本地保存 ±12 cm 宽距离区的复数备份与质量报告
```

### 4.1 距离 FFT 与目标选择

快时间 FFT 把每个 chirp 的 256 点变成距离上的复数反射信息。补到 512 点后距离坐标网格约 2.009 cm，物理距离分辨能力仍约 4.018 cm；不能把补零叫作分辨率翻倍。

20/40/80 cm 分别使用大约 `[8,32] / [28,52] / [68,92] cm` 的物理搜索区。搜索区里的离散 bin 由真实 cfg 计算。

选区用录制开头最多 1 秒的数据：先计算每一帧相对于该帧均值的复数变化能量，再在物理搜索区内选峰；选好后整段固定，不每个窗口跳来跳去。规则对四类故障完全相同，不读取标签、不根据分类结果调参。

这一步的目的，是降低“强静态背景反射被选成电机”的机会；**它还不是环境泛化已经解决的证明**。如果开头是电机尚未运行、强干扰或很弱信号，这个选区仍可能不可靠，要看报告和宽 ROI。距离条件只是定位电机的采集几何信息，不提供故障类别。

导出的复数数据本身**没有做慢时间逐帧去均值**。上面的帧均值操作只用于选区和质量统计，以便后续仍可比较不同背景抑制算法，避免提前消掉低频信息。

### 4.2 保存的主要信号

`radar.h5` 中：

| 数据项 | 形状/含义 |
|---|---|
| `iq` | `[总帧数,192,4,5]`，complex64，保留实部/虚部与相位 |
| `range_m/range_bin_indices` | 本段所选距离单元 |
| `rx_indices` | 四个接收通道身份 |
| `frame_start_s/chirp_offset_s` | 名义帧时刻/帧内时刻 |
| `valid_chirp` | 是否并非全零 ADC；并不等于完整丢包审计 |
| `range_profile_mean_power` | 每帧全正距离范围的平均功率轮廓 |
| `range_profile_dynamic_power` | 每帧全正距离范围的变化能量轮廓 |
| `window_first_frame/window_frame_count` | 以 50 帧组成的完整 2 秒窗口索引 |
| `cfg_json/qc_json` 属性 | cfg、采样率、质量标记、源文件哈希等 |

一个两秒块是 `[50,192,4,5]` 个复数。4RX × 5bin 是 20 个相关空间观测单元，不是 20 个独立天线。2 秒只是默认统计/训练窗口；全部完整帧都被保存，末尾不足 2 秒的帧也不丢掉，以后可以另取窗口。

不完整的最后一个 ADC 帧不会伪造填满：记录尾部字节数，原始尾部仍在本地 BIN。DCA 丢失整个包或错位不一定产生全零 chirp，必须保留采集日志。

### 4.3 为什么保留宽 ROI

默认本地 `wide_cache_local` 保存实测距离 ±12 cm 的全部复数 bin，通常约 12 个距离格点；紧凑上传版只保存其中 5 个。报告保存完整距离轮廓，能发现选点异常；但**只有功率轮廓不能恢复其他距离的相位**。如果紧凑版选错位置，服务器需要相应的宽 ROI 文件，或本地从原始 BIN 重新导出。

主导出不是对原始 BIN 的无损压缩：它有意舍弃电机附近之外的距离信号。因此本地原始 BIN、实际 cfg、采集日志、宽 ROI 暂时都要保留。complex64 也不是原始 ADC 字节的可逆编码。

## 5. 接触式怎么处理

每条 DAT 只导出一次，不按雷达距离重复复制。严格识别：

```text
!1,x,y,z;2,x,y,z;…;4096,x,y,z;
```

只将索引连续 `1…4096` 的包视为完整包；断裂、截断、无法识别的行记录在质量报告，不跨缺口拼成一秒。完整低能量包、常量包仍保留并打标记，不能因“不像故障”或模型答错就删掉。

保存 `contact.h5/raw_xyz` 为 `[包数,4096,3]` float64，保留 XYZ 三轴原数值和包的字节位置。没有转换单位、合并轴、归一化或计算 DCT。原始 DAT 还会以 `contact_original.DAT.gz` 一并保存，便于检查原解析器不认识的标记或以后补取时间信息。

4000 Hz 下每包有效采样跨度按采样槽计为 1.024 秒。兼容之前 BearLLM 一秒输入时，服务器可以取零基索引 `48:4048`，即中间 4000 点；**不是把信号“变成 24 kHz”**。模型专属的 DCN/SFN 等转换留到服务器做，便于核对每个模型真实输入约定。

持续记录的 DAT 仍可能由“采样包＋发送间隙”组成。不能把第 1 包末尾与第 2 包开头直接拼成连续振动，也不能用“包数 ×1.024秒”当作整条 DAT 墙钟时长。没有每包时间戳时，先按同一故障/转速会话关联，再设计集合级或可核验的时间段监督。

## 6. 上传清单与大致体积

```text
R2_export/
├── upload/                         ← 整个上传
│   ├── radar/<recording_id>/
│   │   ├── radar.h5
│   │   ├── recording.json
│   │   ├── actual_radar.cfg
│   │   ├── DONE.json
│   │   └── aux_*                  # 可选采集日志/时间戳
│   └── contacts/<contact_id>/
│       ├── contact.h5
│       ├── contact_original.DAT.gz
│       ├── contact_qc.json
│       └── DONE.json
├── export_catalog.json             ← 必须上传，用于核对整组缺失
├── qc_index.csv                    ← 上传
├── qc_report.html                  ← 上传
├── summary.json                    ← 上传
├── preflight.json / runtime.json    ← 上传
├── verification.json               ← 上传
└── wide_cache_local/               ← 暂留本地，选区异常时再补传相应文件
```

另上传 `manifest.csv`、`manifest_review.csv`、`manifest_summary.json` 和本工具版本。采集日志/每包时间戳若存在，把路径写到 CSV 的 `aux_paths` JSON 数组，例如 `["run_log.txt", "timestamps.csv"]`；工具会原样附带，不猜其时间语义。

**大小估算，按未 gzip 压缩的数值主体计算：**

| 每条录制时长 | 原始 ADC / 条 | 紧凑 IQ / 条 | 144 条紧凑 IQ 合计 |
|---|---:|---:|---:|
| 60 秒 | 1.180 GB | 46.08 MB | 6.636 GB |
| 90 秒 | 1.769 GB | 69.12 MB | 9.953 GB |
| 120 秒 | 2.359 GB | 92.16 MB | 13.271 GB |

其中紧凑 IQ 数据率为 `25 ×192 ×4 ×5 ×8 =768000 bytes/s`，是原始 ADC 的 `1/25.6`。另有约 0.05 MB/s 的两种全距离功率轮廓，以及掩码、元信息、接触文件等；HDF5 内部 gzip 可能进一步缩小，实际以导出目录为准，不保证固定压缩率。宽 ROI 本地缓存另占空间，典型约为紧凑 IQ 的 2.4 倍；不必一开始全部上传。

可以分段上传 `upload/radar/`，接触会话目录每个只上传一次。最终在服务器对同一目录运行 `verify`；校验和只证明文件传输完整，不证明算法/采样格式一定正确。

## 7. 这一步不要做什么

- 不按标签或模型是否答对来筛“好切片”。先报告结构性损坏与客观质量标记，再固定判据。
- 不对全体数据拟合 StandardScaler、PCA、类均值模板或任何学习参数。这些只能在之后的训练集拟合。
- 不把 192 个 chirp/帧按没有间隙的 5000 Hz 波形跨帧直接接起来。
- 不只保留 5–800 Hz 的旧 128 维特征；新数据应允许检查更宽频带及多种表征。
- 不把同一条 DAT 复制成不同距离/不同重复的独立教师记录，也不把它同时当作独立接触训练、接触测试。
- 不假定“改成 R2＋维度相同”就能直接复用旧雷达 encoder 的成绩。R2 的采样节奏、帧内长度和输入统计已经变化，需要重新提取特征并验证。

## 8. 上传后如何服务后续验证

这一格式保留了三种后续路线的条件：

1. **分类链路**：雷达复数 ROI → 帧内相位变化/复数谱/时频特征 → 雷达 encoder → 接触模型的冻结分类接口。
2. **知识验证**：接触包计算物理指标及教师表示，检查雷达表示能否被接触端读取器解释；共享同一 DAT 的来源必须公开，不能宣称已有 144 个独立同步教师。
3. **泛化验证**：按整个转速留出时，应同时留出该转速的接触长记录与雷达记录；跨距离、跨重复的协议另行明确共享接触会话及训练参考数据来自哪里。

四次录制如果来自同一安装、同一条接触长记录，就主要增加重复测量覆盖，不能等价宣称四次独立安装或四条独立接触会话。预处理阶段不随机切窗口划分训练/测试。尤其不能先把覆盖所有测试时间段的整条 DAT 求均值，再不加说明地将其作为训练目标；需要根据后续任务固定允许使用的接触参考来源。有可靠时间戳时再按时间段分开；没有时保留会话分组并限制结论。

本工具仅完成可审计的数据导出。新 144 段原始数据目前还在 Mac，本工具的自动测试与旧 DAT 回归对照不等于已经检验了新批次的实际信号质量。

## 9. 实现依据

TI 官方 ADC 数据捕获文档给出两路 LVDS 下的非交错接收通道与复数数据排列示例；AWR1843 的适用关系可参考 TI 技术人员说明。文件解析仍须结合实际采集软件及已知距离核验，不由型号名称单独确定所有字节约定。

- [TI：mmWave Radar Device ADC Raw Data Capture，SWRA581B](https://www.ti.com/lit/an/swra581b/swra581b.pdf)
- [TI：AWR1843 的数据格式与 xWR16xx/IWR6843 一致的说明](https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/1247658/awr1843boost-interleaving-of-the-multiple-emitter-data-in-raw-data-acquired-with-dca1000)
- [TI：DCA1000EVM User’s Guide](https://www.ti.com/lit/ug/spruij4a/spruij4a.pdf)

R2 帧率、缺口、字节数与体积估算由本次 cfg 直接计算。±12 cm 搜索区、开头 1 秒选区、5 个紧凑距离单元属于当前工程默认值，需通过本批质量报告确认适用性，不是 TI 保证的参数或已经发表验证过的环境泛化算法。
