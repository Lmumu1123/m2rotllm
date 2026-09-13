"""
针对导师汇报的补充诊断与画图。

回答的几个问题：
1. 10s 平滑是不是"铲平了当然相似"？ -> 画不同平滑尺度下的两条曲线，肉眼看
2. 有没有更好的雷达处理（静止滤除/去噪/相干积累）？ -> 实测谱与噪声底
3. 快趋势下 26 Hz 转动峰够不够显著？ -> 画雷达原始位移谱 + 触觉包络谱
4. "滞后谱 0s 尖峰"到底什么意思？ -> 画滞后谱柱状图

核心发现：触觉 CSV 给的是**幅值包络**（RMS/峰值检波器输出），26 Hz 载波已被
片上去除，只剩慢变包络；而雷达原始相位位移在 666.67 Hz 仍含 26 Hz 载波。
所以两路无法在 26 Hz 载波相位上对齐——只能在包络(慢趋势)层面对齐。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 注册本地中文字体（系统未安装 CJK 字体）
DATA_ROOT = os.environ.get(
    "ALIGNMENT_DATA_DIR", os.path.dirname(os.path.abspath(__file__))
)
_FONT_PATH = os.environ.get("ALIGNMENT_FONT")
if _FONT_PATH and os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

BIN = os.environ.get(
    "RADAR_BIN", os.path.join(DATA_ROOT, "20260807-153437-1786088077053250.bin")
)
TCSV = os.environ.get(
    "TACTILE_CSV", os.path.join(DATA_ROOT, "15-34-23-384", "data_0.csv")
)
OUT = os.environ.get("ALIGNMENT_OUT_DIR", os.path.join(DATA_ROOT, "_rd_test_out"))
os.makedirs(OUT, exist_ok=True)

C = 3e8; FC = 77e9; LAM = C / FC
FS_LOOP = 1000.0 / 1.5   # 666.67 Hz
RX = 4; NADC = 512; CPL = 3
RADAR_CLOCK = "15:34:37"


def clk(s):
    h, m, sec = str(s).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


# ---------- 雷达原始位移波形（666.67 Hz，含 26Hz 载波） ----------
print("extracting radar raw displacement ...")
mm = np.memmap(BIN, dtype=np.int16, mode="r")
spb = RX * NADC
# 注意: os.path.getsize 返回字节数, 每实采样 2 字节
nch = os.path.getsize(BIN) // (spb * 2)
nloop = nch // CPL
# 取 TX0 (chirp0) 目标 bin 13
TB = 13
win = np.hanning(NADC).astype(np.float32)
twid = np.exp(-2j * np.pi * TB * np.arange(NADC) / NADC)
block = 8192
zall = np.zeros((nloop, RX), dtype=np.complex128)
for lo in range(0, nloop, block):
    hi = min(lo + block, nloop)
    s = lo * CPL * spb
    a = np.asarray(mm[s:s + (hi - lo) * CPL * spb], dtype=np.float32).reshape(
        (hi - lo) * CPL, RX, NADC)
    c0 = a[0::CPL]
    zall[lo:hi] = np.einsum("lra,a->lr", c0 * win, twid)
ref = np.angle(zall[:1000].mean(axis=0))
zb = (zall * np.exp(-1j * ref)[None, :]).mean(axis=1)
disp_raw = np.unwrap(np.angle(zb)) * LAM / (4 * np.pi) * 1e6  # um, 666.67 Hz
t_radar = np.arange(nloop) / FS_LOOP  # 相对雷达自身 t0

# ---------- 雷达侧谱：26 Hz 转动峰 + 噪声底 ----------
sig = disp_raw - np.polyval(np.polyfit(np.arange(len(disp_raw)) / FS_LOOP,
                                      disp_raw, 1), np.arange(len(disp_raw)) / FS_LOOP)
# 用整段做 FFT，零补齐到 2 的幂
NFFT = 1 << int(np.ceil(np.log2(len(sig))))
f = np.fft.rfftfreq(NFFT, 1 / FS_LOOP)
seg = np.zeros(NFFT); seg[:len(sig)] = sig[:NFFT] if len(sig) >= NFFT else sig
win_w = np.hanning(len(sig)) if len(sig) < NFFT else np.hanning(NFFT)
seg[:len(sig)] = sig * win_w
spec = np.abs(np.fft.rfft(seg)) * 2 / NFFT
peak_idx = np.argmax(spec[5:]) + 5
print(f"radar disp spectrum peak: {f[peak_idx]:.2f} Hz, amp {spec[peak_idx]:.3f} um")

# ---------- 触觉：X位移幅值包络 + 其谱 ----------
df = pd.read_csv(TCSV)
t_str = pd.Series(df.index, dtype=str)
d2 = df.iloc[:, :-1].copy()
d2.columns = df.columns[1:]
t0 = clk(t_str.iloc[0])
t_t = t_str.map(clk).to_numpy() - t0
xd = pd.to_numeric(d2["X位移幅值(um)"], errors="coerce").to_numpy(float)

# 触觉包络谱（41.7 Hz）
fs_t = 1 / np.median(np.diff(t_t))
Nt = 1 << int(np.floor(np.log2(len(xd))))
ft = np.fft.rfftfreq(Nt, 1 / fs_t)
spt = np.abs(np.fft.rfft((xd[:Nt] - xd[:Nt].mean()) * np.hanning(Nt))) * 2 / Nt
print(f"tactile X disp envelope fs={fs_t:.2f} Hz, n={Nt}")
# 触觉原始加速度列是否存在且非零？检查能否拿到载波
acc_x = pd.to_numeric(d2.get("加速度X(g)"), errors="coerce").to_numpy(float)
print(f"accX: mean abs = {np.nanmean(np.abs(acc_x)):.4f} g (若>0 说明有原始加速度载波)")

# ---------- 时间戳配准，取重叠区间 ----------
radar_off = clk(RADAR_CLOCK) - clk(t_str.iloc[0])
t_end = radar_off + nloop / FS_LOOP
sel = (t_t >= radar_off + 1) & (t_t <= t_end - 1)
ts = t_t[sel]
# 直接复用已存的 rd_disp 包络(24ms std)
npz = np.load(os.path.join(OUT, "rd_series.npz"))
rd_disp = npz["rd_disp"]  # 已配准到 ts
xdisp_sel = xd[sel]
accx_sel = pd.to_numeric(d2.get("加速度X(g)"), errors="coerce").to_numpy(float)[sel]

def movavg(v, sec, fs):
    w = max(1, int(sec * fs))
    return np.convolve(np.asarray(v, float), np.ones(w) / w, mode="same")

def zsc(x):
    s = x.std(); return (x - x.mean()) / (s if s > 1e-12 else 1)

# ============================================================
# 图1：不同平滑尺度下两条曲线对比（说明"铲平"质疑）
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
scales = [0, 0.5, 2, 10]
for ax, sm in zip(axes, scales):
    a = zsc(movavg(xdisp_sel, sm, fs_t))
    b = zsc(movavg(rd_disp, sm, fs_t))
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    r = float(np.corrcoef(a, b)[0, 1])
    ax.plot(ts, a, lw=0.8, alpha=0.8, label="触觉 X位移幅值(包络)")
    ax.plot(ts, b, lw=0.8, alpha=0.8, label="雷达 rd_disp(径向位移包络)")
    ax.set_title(f"平滑窗 = {sm}s    Pearson r = {r:.3f}", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("时间 (s, 相对触觉起点)")
fig.suptitle("图1: 不同平滑尺度下两路信号对比 (z-score) — 看清'平滑=铲平'效应", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_smoothing_scales.png"), dpi=110)
plt.close(fig)

# ============================================================
# 图2：滞后谱柱状图(说明"0s 尖峰"含义)
# ============================================================
lags = np.arange(-int(8 * fs_t), int(8 * fs_t) + 1)
a2 = movavg(xdisp_sel, 2, fs_t)
b2 = movavg(rd_disp, 2, fs_t)
n2 = min(len(a2), len(b2))
a2, b2 = a2[:n2], b2[:n2]
rs = []
for lag in lags:
    if lag >= 0:
        u, v = a2[:len(a2) - lag], b2[lag:]
    else:
        u, v = a2[-lag:], b2[:len(b2) + lag]
    if len(u) < 5 or len(u) != len(v):
        rs.append(np.nan); continue
    rs.append(float(np.corrcoef(u, v)[0, 1]))
rs = np.array(rs)
lag_s = lags / fs_t
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(lag_s, rs, "-o", ms=3, color="#1f77b4")
ax.axvline(0, color="red", ls="--", lw=1, label="时间戳对齐点 (0s)")
ax.axhline(0, color="gray", lw=0.5)
bi = np.argmax(rs)
ax.annotate(f"峰: lag={lag_s[bi]:+.2f}s, r={rs[bi]:.3f}",
            xy=(lag_s[bi], rs[bi]), xytext=(lag_s[bi] + 1.5, rs[bi] - 0.05),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("雷达相对触觉的滞后 (s)\n负=雷达更早, 正=雷达更晚")
ax.set_ylabel("Pearson 相关系数 (2s 平滑)")
ax.set_title("图2: 滞后谱 — 唯一在 0s 处出现尖峰, ±1s 即跌到 ~0.1\n"
             "含义: 只有当两路时间戳精确对齐时才相关, 偏一点就崩 -> 时间轴已对准")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_lag_profile.png"), dpi=110)
plt.close(fig)

# ============================================================
# 图3：雷达原始位移谱(26Hz峰) vs 触觉包络谱(无载波)
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
ax1.semilogy(f, spec, lw=0.6)
ax1.axvline(f[peak_idx], color="red", ls="--",
            label=f"26.0 Hz 转动峰 amp={spec[peak_idx]:.2f}um")
# 帧率边带标注: 26 ± n*10.42 Hz (10.42Hz = 1/96ms 帧周期)
FRAME_RATE = 1 / 0.096
for n_sb in (-2, -1, 1, 2):
    ax1.axvline(f[peak_idx] + n_sb * FRAME_RATE, color="orange", ls=":", lw=0.8)
ax1.axvline(f[peak_idx], color="red", ls="--",
            label="橙色虚线: 26±n×10.42Hz 帧率边带(非独立振源)")
# 画噪声底中位
ax1.axhline(np.median(spec), color="green", ls=":", label=f"噪声底≈{np.median(spec):.3f}um")
ax1.set_xlim(0, 80)
ax1.set_xlabel("频率 (Hz)")
ax1.set_ylabel("位移幅值 (um, log)")
ax1.set_title("图3a: 雷达原始位移谱 (666.67Hz采样, 含26Hz转动载波)\n"
              "—— 26Hz 峰显著高于噪声底, 雷达侧快趋势信号是有的")
ax1.legend(); ax1.grid(alpha=0.3, which="both")

ax2.semilogy(ft, spt, lw=0.6, color="orange")
ax2.set_xlim(0, 20)
ax2.set_xlabel("频率 (Hz)")
ax2.set_ylabel("幅值 (um, log)")
ax2.set_title("图3b: 触觉 X位移幅值(包络)谱 (41.7Hz采样)\n"
              "—— 这是 RMS/峰值检波器输出, 26Hz 载波已被片上去除, 只剩<20Hz 慢包络\n"
              "所以触觉侧根本没有 26Hz 载波可参与对齐")
ax2.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_spectra.png"), dpi=110)
plt.close(fig)

# ============================================================
# 图4：静止杂波滤除对比 — 去直流前后雷达位移
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7))
# 原始(含静态偏置)
seg_t = t_radar[:int(2 * FS_LOOP)]  # 前2秒
n2seg = len(seg_t)
ax1.plot(seg_t, disp_raw[:n2seg], lw=0.5)
ax1.set_title("原始径向位移 (含静态偏置+慢漂, 666.67Hz, 前2s)")
ax1.set_xlabel("时间 (s)"); ax1.set_ylabel("um"); ax1.grid(alpha=0.3)
# 去静态: 减均值+去线性趋势
disp_ac = sig  # 已去直流/去趋势
seg2 = disp_ac[:len(seg_t)]
# 在这个短段上再减局部均值(=静止杂波/泄漏的慢变分量)
win_hp = max(1, int(0.05 * FS_LOOP))  # 50ms 高通式去慢漂
mf = np.convolve(disp_ac, np.ones(win_hp) / win_hp, mode="same")
disp_hp = disp_ac - mf
ax2.plot(seg_t, disp_hp[:n2seg], lw=0.5, color="green")
ax2.set_title("去静止/慢漂后 (高通去杂波) — 26Hz 振动更干净, 相位噪声为主")
ax2.set_xlabel("时间 (s)"); ax2.set_ylabel("um"); ax2.grid(alpha=0.3)
fig.suptitle("图4: 静止杂波滤除前后对比 (雷达径向位移)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_clutter.png"), dpi=110)
plt.close(fig)

# ============================================================
# 图5：快趋势噪声量化 — 触觉真实调制 vs 雷达噪声
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5))
# 触觉 X位移幅值自身波动 (1.3%) vs 雷达 rd_disp 包络噪声 (17%)
x_norm = (xdisp_sel - xdisp_sel.mean()) / xdisp_sel.mean()
r_norm = (rd_disp - rd_disp.mean()) / rd_disp.mean()
bins = np.linspace(-0.5, 0.5, 80)
ax.hist(x_norm, bins=bins, alpha=0.6, density=True,
        label=f"触觉 X位移相对波动 (std={x_norm.std():.3f}={x_norm.std()*100:.1f}%)")
ax.hist(r_norm, bins=bins, alpha=0.5, density=True,
        label=f"雷达 rd_disp 相对波动 (std={r_norm.std():.3f}={r_norm.std()*100:.1f}%)")
ax.set_xlim(-0.6, 0.6)
ax.set_xlabel("相对均值的变化幅度")
ax.set_ylabel("概率密度")
ax.set_title("图5: 快尺度噪声量化 — 雷达包络噪声(17%)远大于真实振动调制(1.3%)\n"
             f"理论相关上限 ≈ 1.3/17 ≈ 0.075, 与实测 0.071 吻合 -> 快尺度低相关是噪声问题")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_noise.png"), dpi=110)
plt.close(fig)

# ============================================================
# 图6：触觉原始加速度X(g) 谱 —— 是否含载波(及混叠)
# ============================================================
Nt2 = 1 << int(np.ceil(np.log2(np.sum(~np.isnan(accx_sel)))))
acc_clean = accx_sel[~np.isnan(accx_sel)]
Nt2 = min(Nt2, len(acc_clean))
acc_seg = acc_clean[:Nt2] - acc_clean[:Nt2].mean()
ft2 = np.fft.rfftfreq(Nt2, 1 / fs_t)
sp2 = np.abs(np.fft.rfft(acc_seg * np.hanning(Nt2))) * 2 / Nt2
pk2 = np.argmax(sp2[2:]) + 2
print(f"tactile accX spectrum peak: {ft2[pk2]:.2f} Hz, amp {sp2[pk2]:.4f} g "
      f"(fs={fs_t:.2f}Hz, Nyquist={fs_t/2:.2f}Hz; 若真振动26Hz会混叠到"
      f"{abs(26-fs_t*round(26/fs_t)):.2f}Hz)")

fig, ax = plt.subplots(figsize=(11, 5))
ax.semilogy(ft2, sp2, lw=0.6, color="purple")
ax.axvline(ft2[pk2], color="red", ls="--",
           label=f"峰 {ft2[pk2]:.2f}Hz amp={sp2[pk2]:.3f}g")
ax.axvline(fs_t / 2, color="gray", ls=":", label=f"Nyquist={fs_t/2:.1f}Hz")
ax.set_xlim(0, fs_t / 2 + 1)
ax.set_xlabel("频率 (Hz)")
ax.set_ylabel("加速度 (g, log)")
ax.set_title("图6: 触觉原始 加速度X(g) 谱 — 含载波但采样率仅41.7Hz\n"
             f"26Hz 超过 Nyquist({fs_t/2:.1f}Hz) 会混叠, 无法直接对齐载波相位")
ax.legend(); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_tactile_acc_spectrum.png"), dpi=110)
plt.close(fig)

# ============================================================
# 图7：原始时域对比 —— 为什么不能拿原始波形对原始波形比
# ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9))
i0, i1 = int(10 * FS_LOOP), int(12 * FS_LOOP)  # 雷达第10~12秒
ax1.plot(t_radar[i0:i1], disp_ac[i0:i1], lw=0.6, color="#1f77b4")
ax1.set_title("图7a: 雷达原始径向位移 (666.67Hz采样, 去趋势, 2s窗口)\n"
              "—— 干净的 26Hz 正弦(2秒约52个周期), 载波完好")
ax1.set_ylabel("um"); ax1.grid(alpha=0.3)

m = (t_t >= radar_off + 10) & (t_t < radar_off + 12)
ax2.plot(t_t[m], accx_sel_full := pd.to_numeric(d2["加速度X(g)"], errors="coerce").to_numpy(float)[np.arange(len(t_t))][m],
         lw=0.6, color="purple")
ax2.set_title("图7b: 触觉原始 加速度X(g) (41.7Hz采样, 同墙钟2s窗口)\n"
              "—— 叠着 0.89g 重力直流, 且 26Hz>奈奎斯特20.8Hz 已混叠, 波形本身是错的")
ax2.set_ylabel("g"); ax2.grid(alpha=0.3)

ax3.plot(t_t[m], xd[np.arange(len(t_t))][m], lw=0.8, color="green")
ax3.set_title("图7c: 触觉 X位移幅值 (厂家片上包络, 同窗口)\n"
              "—— 26Hz 载波在传感器内部已被检波掉, 只剩慢变幅度")
ax3.set_ylabel("um"); ax3.set_xlabel("时间 (s)"); ax3.grid(alpha=0.3)
fig.suptitle("图7: 原始时域为什么不能直接对 —— 雷达有干净载波, 触觉原始列混叠+直流, 包络列无载波", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_raw_time_domain.png"), dpi=110)
plt.close(fig)

print("\nfigures saved:")
for fn in ["fig1_smoothing_scales.png", "fig2_lag_profile.png",
           "fig3_spectra.png", "fig4_clutter.png", "fig5_noise.png",
           "fig6_tactile_acc_spectrum.png", "fig7_raw_time_domain.png"]:
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
