"""
裁决实验: std包络(12ms窗)测得粗滞后+1500ms, lock-in包络测得+8ms, 谁对?

方法:
1) 触觉自相关(0.5s平滑)在 0~2s 的值 —— 若慢趋势太宽, 两个滞后都能高相关
2) 两种包络互相关(雷达内部) —— 它们应跟踪同一物理幅度, lag~0 互相关
3) 统一框架下重算两种包络的滞后谱(同网格/同平滑), 打表对比
4) 波形叠加图: 触觉 vs lock-in包络, 分别按 lag=0 和 lag=+1.5s 校正, 肉眼裁决
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

DATA_ROOT = os.environ.get(
    "ALIGNMENT_DATA_DIR", os.path.dirname(os.path.abspath(__file__))
)
fp = os.environ.get("ALIGNMENT_FONT")
if fp and os.path.exists(fp):
    fm.fontManager.addfont(fp); plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

C, FC = 3e8, 77e9; LAM = C/FC
FS_LOOP = 1000.0/1.5; RX, NADC, CPL = 4, 512, 3
OUT = os.environ.get("ALIGNMENT_OUT_DIR", os.path.join(DATA_ROOT, "_fine_align_out"))

G = {"tag": "14-18",
     "radar": os.path.join(DATA_ROOT, "14-18-30-583", "20260907-141831-1788761911939322.bin"),
     "radar_clock": "14:18:31",
     "tactile": os.path.join(DATA_ROOT, "14-18-30-583", "data_0.csv")}

def clk(s):
    h, m, sec = str(s).strip().split(":")
    return int(h)*3600+int(m)*60+float(sec)

# ---------- 触觉 ----------
df = pd.read_csv(G["tactile"])
t_str = pd.Series(df.index, dtype=str)
d2 = df.iloc[:, :-1].copy(); d2.columns = df.columns[1:]
t_t = t_str.map(clk).to_numpy()
y_disp = pd.to_numeric(d2["Y位移幅值(um)"], errors="coerce").to_numpy(float)
fs_c = 125.0
tg = np.arange(t_t[0], t_t[-1], 1/fs_c)
yg = np.interp(tg, t_t, y_disp)
tg_rel = tg - t_t[0]

# ---------- 雷达 ----------
mm = np.memmap(G["radar"], dtype=np.int16, mode="r")
spb = RX*NADC; nch = len(mm)//spb; nloop = nch//CPL
TB = 11
win = np.hanning(NADC).astype(np.float32)
twid = np.exp(-2j*np.pi*TB*np.arange(NADC)/NADC)
zall = np.zeros((nloop, RX), dtype=np.complex128)
for lo in range(0, nloop, 8192):
    hi = min(lo+8192, nloop); s = lo*CPL*spb
    a = np.asarray(mm[s:s+(hi-lo)*CPL*spb], dtype=np.float32).reshape((hi-lo)*CPL, RX, NADC)
    zall[lo:hi] = np.einsum("lra,a->lr", a[0::CPL]*win, twid)
ref = np.angle(zall[:1000].mean(axis=0))
zb = (zall*np.exp(-1j*ref)[None, :]).mean(axis=1)
disp = np.unwrap(np.angle(zb))*LAM/(4*np.pi)*1e6
t_loop = np.arange(nloop)/FS_LOOP
radar_off = clk(G["radar_clock"]) - t_t[0]

# 振动频率
N = 1 << int(np.ceil(np.log2(len(disp))))
seg = np.zeros(N); seg[:len(disp)] = disp*np.hanning(len(disp))
f = np.fft.rfftfreq(N, 1/FS_LOOP)
sp = np.abs(np.fft.rfft(seg))*2/N
m_ = (f > 3) & (f < 60)
fv = f[np.where(m_)[0][np.argmax(sp[m_])]]
print(f"vib freq = {fv:.2f} Hz")

# 两种包络
W, HOP = 8, 4
st = np.arange(0, nloop-W+1, HOP)
env_std = np.array([disp[s:s+W].std() for s in st])
tc_std = (st+W/2)/FS_LOOP
demod = disp*np.exp(-2j*np.pi*fv*t_loop)
Wlp = int(0.4*FS_LOOP)
amp = np.abs(np.convolve(demod, np.ones(Wlp)/Wlp, mode="same"))*2

def movavg(v, sec, fs):
    w = max(1, int(round(sec*fs)))
    v = np.nan_to_num(v, nan=np.nanmean(v))
    return np.convolve(v, np.ones(w)/w, mode="same")

def lag_scan(a, b, lags):
    out = []
    for lag in lags:
        if lag >= 0:
            u, v = a[:len(a)-lag], b[lag:]
        else:
            u, v = a[-lag:], b[:len(b)+lag]
        m2 = np.isfinite(u) & np.isfinite(v)
        out.append(np.corrcoef(u[m2], v[m2])[0, 1] if m2.sum() > 30 else np.nan)
    return np.array(out)

# ---------- 1) 触觉自相关 ----------
print("\n[1] 触觉自相关 (0.5s平滑): 慢趋势有多宽?")
ac = lag_scan(movavg(yg, 0.5, fs_c), movavg(yg, 0.5, fs_c),
              [int(x*fs_c) for x in [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]])
for x, r in zip([0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0], ac):
    print(f"    lag={x:4.2f}s  r_auto={r:.3f}")

# ---------- 2) 两种雷达包络互相关 ----------
print("\n[2] 雷达内部: std包络 vs lock-in包络 互相关 (应 lag~0)")
ev_std_i = np.interp(tg_rel, tc_std+radar_off, env_std, left=np.nan, right=np.nan)
amp_i = np.interp(tg_rel, t_loop+radar_off, amp, left=np.nan, right=np.nan)
lags2 = np.arange(-int(1.0*fs_c), int(1.0*fs_c)+1)
r2 = lag_scan(movavg(ev_std_i, 0.5, fs_c), movavg(amp_i, 0.5, fs_c), lags2)
j2 = int(np.nanargmax(r2))
print(f"    峰: lag={lags2[j2]/fs_c*1000:+.0f}ms, r={np.nanmax(r2):.3f} "
      f"(若两包络真跟踪同一幅度, 应接近0且r高)")

# ---------- 3) 统一框架滞后谱对比 ----------
print("\n[3] 触觉 vs 两种包络, 同网格同平滑, 滞后谱在关键点取值:")
lags3 = np.arange(-int(2.5*fs_c), int(2.5*fs_c)+1, 1)
for name, ev in [("std包络", ev_std_i), ("lock-in", amp_i)]:
    rs = lag_scan(movavg(yg, 0.5, fs_c), movavg(ev, 0.5, fs_c), lags3)
    j3 = int(np.nanargmax(rs))
    r0 = rs[int(2.5*fs_c)]; r1500 = rs[int(2.5*fs_c)+int(1.5*fs_c)]
    print(f"    {name:9s}: 峰lag={lags3[j3]/fs_c*1000:+7.1f}ms r={np.nanmax(rs):.3f} | "
          f"r(0ms)={r0:.3f}  r(+1500ms)={r1500:.3f}")

# ---------- 4) 叠加图裁决 ----------
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
zz = lambda x: (x-np.nanmean(x))/(np.nanstd(x)+1e-12)
n_pl = int(min(30*fs_c, len(yg)))
tts = tg_rel[:n_pl]
ax = axes[0]
ax.plot(tts, zz(movavg(yg, 0.25, fs_c)[:n_pl]), lw=1, label="触觉 Y位移幅值(0.25s平滑)")
ax.plot(tts, zz(movavg(amp_i, 0.25, fs_c)[:n_pl]), lw=1, alpha=0.85,
        label="雷达 lock-in 包络(lag=0 即文件名时间戳)")
ax.set_title("假设A: 时间戳即对(lag≈0) —— 看两条线是否同起伏")
ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_ylabel("z-score")
ax = axes[1]
sh = int(1.5*fs_c)
ax.plot(tts, zz(movavg(yg, 0.25, fs_c)[:n_pl]), lw=1, label="触觉 Y位移幅值(0.25s平滑)")
ax.plot(tts, zz(movavg(amp_i, 0.25, fs_c)[sh:sh+n_pl]), lw=1, alpha=0.85,
        label="雷达 lock-in 包络(右移1.5s)")
ax.set_title("假设B: 雷达晚1.5s(lag=+1500ms) —— 看哪张图的起伏更同步")
ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_ylabel("z-score")
ax = axes[2]
ax.plot(tts, zz(movavg(yg, 0.25, fs_c)[:n_pl]), lw=1, label="触觉 Y位移幅值(0.25s平滑)")
ax.plot(tts, zz(movavg(ev_std_i, 0.25, fs_c)[:n_pl]), lw=1, alpha=0.85,
        label="雷达 std包络(12ms窗, lag=0)")
ax.plot(tts, zz(movavg(ev_std_i, 0.25, fs_c)[sh:sh+n_pl]), lw=1, alpha=0.6, ls="--",
        label="雷达 std包络(右移1.5s)")
ax.set_title("std包络在两个滞后下的形态(它的高噪声如何制造假峰)")
ax.legend(fontsize=9); ax.grid(alpha=.3)
ax.set_xlabel("时间 (s, 相对触觉t0)"); ax.set_ylabel("z-score")
fig.suptitle("组14-18: 粗滞后矛盾裁决 (+8ms vs +1500ms)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/adjudicate_lag.png", dpi=110)
print(f"\nsaved -> {OUT}/adjudicate_lag.png")
