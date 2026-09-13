"""
验证三个问题：
Q1: 26 Hz 凭什么是电机频率？雷达谱里其他尖峰是什么？
Q2: fig2 滞后谱峰值/数值与文档 0.756@0s 是否一致？差在哪？
Q3: 各通道（位移幅值/加速度包络）平滑相关到底多少？accX 原始波形长什么样？
"""
import os, json
import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get(
    "ALIGNMENT_DATA_DIR", os.path.dirname(os.path.abspath(__file__))
)
OUT = os.environ.get("ALIGNMENT_OUT_DIR", os.path.join(DATA_ROOT, "_rd_test_out"))
BIN = os.environ.get(
    "RADAR_BIN", os.path.join(DATA_ROOT, "20260807-153437-1786088077053250.bin")
)
TCSV = os.environ.get(
    "TACTILE_CSV", os.path.join(DATA_ROOT, "15-34-23-384", "data_0.csv")
)
C = 3e8; LAM = C / 77e9; FS = 1000.0 / 1.5; RX = 4; NADC = 512; CPL = 3

def clk(s):
    h, m, sec = str(s).strip().split(":")
    return int(h)*3600 + int(m)*60 + float(sec)

# ---------- 触觉加载（与 rd_alignment_test.load_tactile_fixed 同口径） ----------
df = pd.read_csv(TCSV)
t_str = pd.Series(df.index, dtype=str)
d2 = df.iloc[:, :-1].copy(); d2.columns = df.columns[1:]
t_t = t_str.map(clk).to_numpy() - clk(t_str.iloc[0])
col = lambda name: pd.to_numeric(d2[name], errors="coerce").to_numpy(float)
xd, zd, yd = col("X位移幅值(um)"), col("Z位移幅值(um)"), col("Y位移幅值(um)")
xv = col("X速度幅值(mm/s)")
accx = col("加速度X(g)")
accx_amp = col("X加速度幅值(g)")
fs_t = 1 / np.median(np.diff(t_t))

npz = np.load(os.path.join(OUT, "rd_series.npz"))
ts = npz["t_tactile"]; rd_disp = npz["rd_disp"]
radar_off = 13.603; t_end = radar_off + 86528 / FS
sel = (t_t >= radar_off + 1) & (t_t <= t_end - 1)

print("=" * 70)
print("[Q3-交叉验证] 我的 CSV 加载 vs npz 里存的触觉列 (应全为 0):")
print("  max|diff| =", np.nanmax(np.abs(xd[sel] - npz["tactile_X位移幅值(um)"])))

def movavg(v, sec):
    w = max(1, int(sec * fs_t))
    return np.convolve(np.asarray(v, float), np.ones(w) / w, mode="same")

print("\n[Q3-平滑相关表] 各触觉通道 vs rd_disp (0/0.5/2/10s):")
channels = {
    "X位移幅值(厂家包络)": xd,
    "Z位移幅值(厂家包络)": zd,
    "X速度幅值(厂家包络)": xv,
    "accX滚动std包络0.5s": pd.Series(accx[sel]).rolling(21, center=True, min_periods=1).std().to_numpy(),
    "accX滚动std包络2s": pd.Series(accx[sel]).rolling(83, center=True, min_periods=1).std().to_numpy(),
}
for name, arr in channels.items():
    a = arr[sel] if len(arr) == len(t_t) else arr
    row = []
    for sm in [0, 0.5, 2, 10]:
        x, y = movavg(a, sm), movavg(rd_disp, sm)
        ok = np.isfinite(x) & np.isfinite(y)
        row.append(np.corrcoef(x[ok], y[ok])[0, 1])
    print(f"  {name:<22}: " + "  ".join(f"{v:+.3f}" for v in row))
print("  (JSON 里 rd_disp|X位移幅值: 0.071 / 0.533 / 0.756 / 0.970)")
print("  厂家 X加速度幅值(g) 列全零?", bool(np.all(accx_amp == 0)),
      f"(max={np.nanmax(np.abs(accx_amp))})")
print(f"  accX 均值={np.nanmean(accx):.3f}g (重力直流), 去直流后 std={np.nanstd(accx):.4f}g")

# ---------- Q2: 滞后谱精细复算 ----------
print("\n[Q2-滞后谱精细复算] (2s 平滑, 与 rd_alignment_test 同公式):")
a2, b2 = movavg(xd[sel], 2), movavg(rd_disp, 2)
ok = np.isfinite(a2) & np.isfinite(b2)
print(f"  finite 比例 = {ok.mean():.4f}  (若<1 说明有 NaN)")
r0 = float(np.corrcoef(a2, b2)[0, 1])
print(f"  r(lag=0) = {r0:.4f}   (文档表: 0.756)")
best = (-9, 0)
prof = {}
for lag in range(-int(1.0 * fs_t), int(1.0 * fs_t) + 1):
    if lag >= 0:
        u, v = a2[:len(a2)-lag], b2[lag:]
    else:
        u, v = a2[-lag:], b2[:len(b2)+lag]
    r = float(np.corrcoef(u, v)[0, 1])
    prof[lag / fs_t] = r
    if r > best[0]:
        best = (r, lag / fs_t)
print(f"  ±1s 内最优: lag={best[1]:+.3f}s r={best[0]:.4f}")
print("  lag=-0.10..+0.10s:", {f"{k:+.2f}": round(prof[k], 3) for k in sorted(prof) if abs(k) <= 0.11})

# ---------- Q1: 雷达谱尖峰清单 (bin 13 电机 / bin 141 多径) ----------
print("\n[Q1-雷达位移谱尖峰] 提取 bin13 & bin141 ...")
mm = np.memmap(BIN, dtype=np.int16, mode="r")
spb = RX * NADC
nch = mm.__len__() // (spb * 2)
nloop = nch // CPL
win = np.hanning(NADC).astype(np.float32)
TW = np.stack([np.exp(-2j * np.pi * b * np.arange(NADC) / NADC) for b in (13, 141)])
zall = [np.zeros((nloop, RX), dtype=np.complex128) for _ in range(2)]
for lo in range(0, nloop, 8192):
    hi = min(lo + 8192, nloop)
    s = lo * CPL * spb
    arr = np.asarray(mm[s:s + (hi - lo) * CPL * spb], dtype=np.float32).reshape(
        (hi - lo) * CPL, RX, NADC)
    z = np.einsum("lra,ba->lrb", arr[0::CPL] * win, TW)
    zall[0][lo:hi] = z[:, :, 0]; zall[1][lo:hi] = z[:, :, 1]

tt = np.arange(nloop) / FS
NF = 1 << int(np.ceil(np.log2(nloop)))
f = np.fft.rfftfreq(NF, 1 / FS)
for bi, TB in enumerate([13, 141]):
    ref = np.angle(zall[bi][:1000].mean(axis=0))
    zb = (zall[bi] * np.exp(-1j * ref)[None, :]).mean(axis=1)
    disp = np.unwrap(np.angle(zb)) * LAM / (4 * np.pi) * 1e6
    disp = disp - np.polyval(np.polyfit(tt, disp, 1), tt)
    spec = np.abs(np.fft.rfft(disp * np.hanning(nloop), NF)) * 2 / nloop
    band = (f >= 1) & (f <= 300)
    sb, fb = spec[band], f[band]
    idx = [i for i in range(1, len(sb) - 1) if sb[i] > sb[i-1] and sb[i] >= sb[i+1]]
    idx.sort(key=lambda i: -sb[i])
    print(f"  bin {TB} ({'电机' if TB==13 else '背景多径~5.7m'}) top 尖峰 (Hz, um):")
    print("   ", [(round(fb[i], 2), round(sb[i], 2)) for i in idx[:10]])
    noise_med = float(np.median(sb))
    pk26 = float(sb[np.argmin(np.abs(fb - 26))])
    print(f"    26Hz 处幅度={pk26:.2f}um, 噪声底中位={noise_med:.3f}um, "
          f"信噪比={pk26/noise_med:.0f}x")
