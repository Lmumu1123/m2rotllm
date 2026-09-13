"""
补充实验:
A) 时间尺度扫描: 检验雷达 loop 率 666.67Hz 假设是否有系统偏差
   (雷达谱峰21.9Hz vs 触觉实测20.7Hz, 比值0.946 与旧数据24.6/26.0一致)
B) 锁定放大(lock-in)包络: disp 与 e^{-j2πf_v t} 复解调 + 低通 -> 干净振幅包络
   对比 12ms 窗 std 包络在细尺度(0.1/0.25s)的相关提升
"""
import os, json
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

GROUPS = [
    {"tag": "14-18", "radar": os.path.join(DATA_ROOT, "14-18-30-583", "20260907-141831-1788761911939322.bin"),
     "radar_clock": "14:18:31", "tactile": os.path.join(DATA_ROOT, "14-18-30-583", "data_0.csv")},
    {"tag": "14-19", "radar": os.path.join(DATA_ROOT, "14-19-26-478", "20260907-141928-1788761968133353.bin"),
     "radar_clock": "14:19:28", "tactile": os.path.join(DATA_ROOT, "14-19-26-478", "data_0.csv")},
]

def clk(s):
    h, m, sec = str(s).strip().split(":")
    return int(h)*3600+int(m)*60+float(sec)

def extract_radar(path):
    mm = np.memmap(path, dtype=np.int16, mode="r")
    spb = RX*NADC; nch = len(mm)//spb; nloop = nch//CPL
    n = min(20000, nloop)
    sub = np.asarray(mm[:n*CPL*spb], dtype=np.float32).reshape(n*CPL, RX, NADC)
    mw = sub[0::CPL].mean(axis=0)
    win = np.hanning(NADC).astype(np.float32)
    prof = np.abs(np.fft.fft(mw*win, axis=-1)).mean(axis=0)
    TB = int(np.arange(4, NADC//2)[np.argmax(prof[4:NADC//2])])
    twid = np.exp(-2j*np.pi*TB*np.arange(NADC)/NADC)
    zall = np.zeros((nloop, RX), dtype=np.complex128)
    for lo in range(0, nloop, 8192):
        hi = min(lo+8192, nloop); s = lo*CPL*spb
        a = np.asarray(mm[s:s+(hi-lo)*CPL*spb], dtype=np.float32).reshape((hi-lo)*CPL, RX, NADC)
        zall[lo:hi] = np.einsum("lra,a->lr", a[0::CPL]*win, twid)
    ref = np.angle(zall[:1000].mean(axis=0))
    zb = (zall*np.exp(-1j*ref)[None, :]).mean(axis=1)
    disp = np.unwrap(np.angle(zb))*LAM/(4*np.pi)*1e6
    return disp

def spec_peak(x, fs, fmin=3.0, fmax=60.0):
    x = x - np.nanmean(x)
    N = 1 << int(np.ceil(np.log2(len(x))))
    seg = np.zeros(N); seg[:len(x)] = x*np.hanning(len(x))
    f = np.fft.rfftfreq(N, 1/fs)
    sp = np.abs(np.fft.rfft(seg))*2/N
    m = (f >= fmin) & (f <= fmax)
    i = np.where(m)[0][np.argmax(sp[m])]
    return f[i], sp[i], f, sp

def movavg(v, sec, fs):
    w = max(1, int(round(sec*fs)))
    v = np.nan_to_num(v, nan=np.nanmean(v))
    return np.convolve(v, np.ones(w)/w, mode="same")

def lag_scan_r(a, b, lags):
    out = []
    for lag in lags:
        if lag >= 0:
            u, v = a[:len(a)-lag], b[lag:]
        else:
            u, v = a[-lag:], b[:len(b)+lag]
        m = np.isfinite(u) & np.isfinite(v)
        out.append(np.corrcoef(u[m], v[m])[0, 1] if m.sum() > 30 else np.nan)
    return np.array(out)

RES = {}
for G in GROUPS:
    tag = G["tag"]; print(f"\n===== {tag} =====")
    df = pd.read_csv(G["tactile"])
    t_str = pd.Series(df.index, dtype=str)
    d2 = df.iloc[:, :-1].copy(); d2.columns = df.columns[1:]
    t_t = t_str.map(clk).to_numpy()
    y_disp = pd.to_numeric(d2["Y位移幅值(um)"], errors="coerce").to_numpy(float)
    vf = pd.to_numeric(d2["Y振动频率(Hz)"], errors="coerce").to_numpy(float)
    fs_c = 125.0
    tg = np.arange(t_t[0], t_t[-1], 1/fs_c)
    yg = np.interp(tg, t_t, y_disp)

    disp = extract_radar(G["radar"])
    nloop = len(disp)
    fv, av, _, _ = spec_peak(disp, FS_LOOP)
    print(f"radar vib={fv:.2f}Hz, tactile实测={np.nanmean(vf):.2f}Hz, 比值={np.nanmean(vf)/fv:.4f}")

    # ---------- B) lock-in 解调包络 ----------
    k = np.arange(nloop)
    t_loop = k / FS_LOOP                      # 相对雷达起点
    demod = disp * np.exp(-2j*np.pi*fv*t_loop)
    # 低通: 0.4s 窗滑动平均 (通带~1.2Hz) -> 幅值包络
    Wlp = int(0.4*FS_LOOP)
    kern = np.ones(Wlp)/Wlp
    amp = np.abs(np.convolve(demod, kern, mode="same")) * 2  # um
    # 旧包络(对照): 12ms 窗 std
    W, HOP = 8, 4
    st = np.arange(0, nloop-W+1, HOP)
    env_std = np.array([disp[s:s+W].std() for s in st])
    tc_std = (st+W/2)/FS_LOOP
    print(f"lock-in 包络: mean={amp.mean():.1f}um std/mean={amp.std()/amp.mean()*100:.1f}%")
    print(f"std包络(12ms): mean={env_std.mean():.1f}um std/mean={env_std.std()/env_std.mean()*100:.1f}%")

    radar_off = clk(G["radar_clock"]) - t_t[0]

    # ---------- A) 时间尺度 + 滞后联合扫描 ----------
    # 假设真实 loop 率 = 666.67/s, s>1 表示真实时间比假设的更长(chirp 更慢)
    # 先在 lock-in 包络(更干净)上做, 0.5s 平滑
    scales = np.arange(0.94, 1.061, 0.002)
    best = (0, 0, 0)
    rows = []
    for s in scales:
        tc_s = t_loop * s + radar_off          # 重映射后的雷达绝对时间
        # 采样到触觉网格
        tg_rel = tg - t_t[0]
        amp_i = np.interp(tg_rel, tc_s, amp, left=np.nan, right=np.nan)
        a, b = movavg(yg, 0.5, fs_c), movavg(amp_i, 0.5, fs_c)
        lags = np.arange(-int(3*fs_c), int(3*fs_c)+1, 4)
        rs = lag_scan_r(a, b, lags)
        j = int(np.nanargmax(rs))
        rows.append((s, lags[j]/fs_c, np.nanmax(rs)))
        if np.nanmax(rs) > best[2]:
            best = (s, lags[j]/fs_c, np.nanmax(rs))
    rows = np.array(rows)
    print(f"尺度扫描最优: s={best[0]:.3f} (即真实loop率={FS_LOOP/best[0]:.1f}Hz), "
          f"lag={best[1]*1000:+.0f}ms, r={best[2]:.4f}")
    r_at_1 = rows[np.argmin(np.abs(rows[:,0]-1.0))]
    print(f"s=1.000 时: lag={r_at_1[1]*1000:+.0f}ms, r={r_at_1[2]:.4f}")
    RES[tag] = {"best_scale": round(float(best[0]), 3),
                "best_lag_ms": round(float(best[1]*1000), 0),
                "best_r": round(float(best[2]), 4),
                "r_at_scale1": round(float(r_at_1[2]), 4),
                "implied_loop_hz": round(float(FS_LOOP/best[0]), 1)}

    # ---------- B) 细尺度相关对比: std包络 vs lock-in 包络 ----------
    def scale_r(env_t, env_v, name):
        out = {}
        lag0 = best[1] if abs(best[0]-1) < 0.01 else r_at_1[1]
        # 重采样到触觉网格并按滞后平移
        ev = np.interp(tg - t_t[0], env_t, env_v, left=np.nan, right=np.nan)
        n = int(round(lag0*fs_c))
        if n >= 0:
            a_al, b_al = yg[:len(yg)-n], ev[n:]
        else:
            a_al, b_al = yg[-n:], ev[:len(ev)+n]
        for sm in [0.05, 0.1, 0.25, 0.5]:
            aa, bb = movavg(a_al, sm, fs_c), movavg(b_al, sm, fs_c)
            m = np.isfinite(aa) & np.isfinite(bb)
            out[str(sm)] = round(float(np.corrcoef(aa[m], bb[m])[0, 1]), 3)
        print(f"  {name}: {out}")
        return out
    RES[tag]["r_std_env"] = scale_r(tc_std, env_std, "std包络(12ms)")
    RES[tag]["r_lockin"] = scale_r(t_loop, amp, "lock-in包络")

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    for lag_col, lab in [(1, "r")]:
        pass
    sc = axes[0]
    sc.plot(rows[:, 0], rows[:, 2], "-o", ms=2)
    sc.axvline(1.0, color="gray", ls=":", label="s=1 (loop 666.67Hz)")
    sc.axvline(best[0], color="r", ls="--", label=f"峰 s={best[0]:.3f}")
    sc.set_xlabel("时间尺度因子 s (=666.67/真实loop率)")
    sc.set_ylabel("峰值 r (0.5s平滑)")
    sc.set_title(f"{tag} 时间尺度扫描"); sc.legend(fontsize=8); sc.grid(alpha=.3)
    lk = axes[1]
    nsh = min(int(20*fs_c), len(yg))
    a20 = movavg(yg, 0.1, fs_c); b20 = movavg(np.interp(tg-t_t[0], t_loop, amp, left=np.nan, right=np.nan), 0.1, fs_c)
    n_off = int(round(r_at_1[1]*fs_c))
    a20p = a20[n_off:][ :nsh] if n_off>0 else a20[:nsh]
    b20p = b20[n_off:][:nsh] if n_off>0 else b20[:nsh]
    zz = lambda x: (x-np.nanmean(x))/(np.nanstd(x)+1e-12)
    lk.plot(a20p, lw=.8, label="触觉Y位移幅值(0.1s平滑)")
    lk.plot(zz(b20p), lw=.8, alpha=.8, label="雷达lock-in包络(0.1s平滑)")
    lk.set_xlabel="样本"; lk.set_title(f"{tag} lock-in包络 vs 触觉 (z-score)")
    lk.legend(fontsize=8); lk.grid(alpha=.3)
    bx = axes[2]
    ks = ["0.05", "0.1", "0.25", "0.5"]
    bx.plot(ks, [RES[tag]["r_std_env"][k] for k in ks], "o-", label="std包络(12ms窗)")
    bx.plot(ks, [RES[tag]["r_lockin"][k] for k in ks], "s-", label="lock-in包络")
    bx.set_xlabel("平滑尺度(s)"); bx.set_ylabel("Pearson r")
    bx.set_title(f"{tag} 包络方法对比"); bx.legend(fontsize=8); bx.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/{tag}_scale_lockin.png", dpi=110)
    plt.close(fig)

with open(f"{OUT}/scale_lockin_results.json", "w", encoding="utf-8") as fj:
    json.dump(RES, fj, ensure_ascii=False, indent=2)
print("\nsaved ->", f"{OUT}/scale_lockin_results.json")
