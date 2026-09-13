"""
2026-09-07 两组新数据（14-18-30 / 14-19-26）细粒度跨模态对齐实验。

数据:
  雷达 77GHz FMCW: 4RX x 512 实采样, 3 chirps/loop(TX0/1/2), loop 666.67Hz
  触觉: 125Hz, Y轴主振(Y位移~300um), Y振动频率实测~20.7Hz

实验:
  1) 雷达 RD 相位位移提取 + 振动频率测定
  2) 触觉载波可用性检验(Goertzel @ 振动频率)
  3) 粗滞后扫描(±3s) + 细滞后扫描(±200ms @1.5ms步) + 抛物线亚毫秒插值
  4) 分段漂移估计(4段各自最优滞后 -> 线性时钟漂移)
  5) 漂移校正后各平滑尺度相关
  6) 循环移位零假设(检验细尺度相关是否真实)
  7) GCC-PHAT 对照
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
os.makedirs(OUT, exist_ok=True)

GROUPS = [
    {"tag": "14-18", "radar": os.path.join(DATA_ROOT, "14-18-30-583", "20260907-141831-1788761911939322.bin"),
     "radar_clock": "14:18:31",
     "tactile": os.path.join(DATA_ROOT, "14-18-30-583", "data_0.csv")},
    {"tag": "14-19", "radar": os.path.join(DATA_ROOT, "14-19-26-478", "20260907-141928-1788761968133353.bin"),
     "radar_clock": "14:19:28",
     "tactile": os.path.join(DATA_ROOT, "14-19-26-478", "data_0.csv")},
]


def clk(s):
    h, m, sec = str(s).strip().split(":")
    return int(h)*3600 + int(m)*60 + float(sec)


def load_tactile(path):
    df = pd.read_csv(path)
    t_str = pd.Series(df.index, dtype=str)
    d2 = df.iloc[:, :-1].copy(); d2.columns = df.columns[1:]
    t = t_str.map(clk).to_numpy()
    feats = {c: pd.to_numeric(d2[c], errors="coerce").to_numpy(float)
             for c in ["加速度X(g)", "加速度Y(g)", "加速度Z(g)",
                       "X位移幅值(um)", "Y位移幅值(um)", "Z位移幅值(um)"]}
    vf = pd.to_numeric(d2.get("Y振动频率(Hz)"), errors="coerce").to_numpy(float)
    return t, feats, vf


def resample_uniform(t, x, fs=125.0, max_gap=0.05):
    """插值到均匀网格; 间隙>max_gap 处置 NaN(不虚构数据)"""
    tg = np.arange(t[0], t[-1], 1/fs)
    xg = np.interp(tg, t, x)
    # 标记大间隙
    ok = np.ones(len(tg), bool)
    idx = np.searchsorted(t, tg)
    idx = np.clip(idx, 1, len(t)-1)
    gap = np.minimum(np.abs(tg - t[idx-1]), np.abs(tg - t[idx]))
    ok[gap > max_gap] = False
    xg[~ok] = np.nan
    return tg, xg


def extract_radar(path):
    mm = np.memmap(path, dtype=np.int16, mode="r")
    spb = RX*NADC  # 每 chirp 的 int16 样本数
    nch = len(mm)//spb; nloop = nch//CPL
    # 自动目标 bin: 前 20000 loops 平均谱
    n = min(20000, nloop)
    sub = np.asarray(mm[:n*CPL*spb], dtype=np.float32).reshape(n*CPL, RX, NADC)
    mw = sub[0::CPL].mean(axis=0)
    win = np.hanning(NADC).astype(np.float32)
    prof = np.abs(np.fft.fft(mw*win, axis=-1)).mean(axis=0)
    TB = int(np.arange(4, NADC//2)[np.argmax(prof[4:NADC//2])])
    twid = np.exp(-2j*np.pi*TB*np.arange(NADC)/NADC)
    zall = np.zeros((nloop, RX), dtype=np.complex128)
    block = 8192
    for lo in range(0, nloop, block):
        hi = min(lo+block, nloop); s = lo*CPL*spb
        a = np.asarray(mm[s:s+(hi-lo)*CPL*spb], dtype=np.float32).reshape((hi-lo)*CPL, RX, NADC)
        c0 = a[0::CPL]
        zall[lo:hi] = np.einsum("lra,a->lr", c0*win, twid)
    ref = np.angle(zall[:1000].mean(axis=0))
    zb = (zall*np.exp(-1j*ref)[None, :]).mean(axis=1)
    disp = np.unwrap(np.angle(zb))*LAM/(4*np.pi)*1e6
    print(f"  radar: {nch} chirps, {nloop} loops, {nloop/FS_LOOP:.1f}s, target bin {TB}")
    return disp, TB


def spec_peak(x, fs, fmin=3.0, fmax=60.0):
    x = x - np.nanmean(x)
    N = 1 << int(np.ceil(np.log2(len(x))))
    seg = np.zeros(N); seg[:len(x)] = x; seg[:len(x)] *= np.hanning(len(x))
    f = np.fft.rfftfreq(N, 1/fs)
    sp = np.abs(np.fft.rfft(seg))*2/N
    m = (f >= fmin) & (f <= fmax)
    i = np.where(m)[0][np.argmax(sp[m])]
    return f[i], sp[i], f, sp


def goertzel(x, t, fq):
    """非均匀采样下的定点频率幅值(最小二乘正弦拟合)"""
    ok = ~np.isnan(x)
    tt, xx = t[ok], x[ok]-np.nanmean(x[ok])
    c = np.exp(-2j*np.pi*fq*tt)
    return np.abs((xx*c).mean())*2


def movavg(v, sec, fs):
    w = max(1, int(round(sec*fs)))
    return np.convolve(np.nan_to_num(v, nan=np.nanmean(v)), np.ones(w)/w, mode="same")


def lag_scan(a, b, lags):
    """a,b 等长; lags>0 表示 b 滞后 a(即 b 向左移)。返回各 lag 的 r"""
    out = []
    for lag in lags:
        if lag >= 0:
            u, v = a[:len(a)-lag], b[lag:]
        else:
            u, v = a[-lag:], b[:len(b)+lag]
        m = np.isfinite(u) & np.isfinite(v)
        out.append(np.corrcoef(u[m], v[m])[0, 1] if m.sum() > 30 else np.nan)
    return np.array(out)


def parabolic(y, i):
    """抛物线插值峰值位置(索引小数, 偏移限幅±0.5防外推发散)"""
    if 0 < i < len(y)-1 and np.isfinite(y[i-1]) and np.isfinite(y[i+1]):
        den = y[i-1] - 2*y[i] + y[i+1]
        if abs(den) > 1e-12:
            off = 0.5*(y[i-1]-y[i+1])/den
            return i + np.clip(off, -0.5, 0.5)
    return float(i)


ALL = {}
for G in GROUPS:
    tag = G["tag"]
    print(f"\n===== 组 {tag} =====")
    res = {}
    # ---------- 触觉 ----------
    t_t, feats, vf = load_tactile(G["tactile"])
    dt = np.diff(t_t)
    fs_t = 1/np.median(dt)
    print(f"tactile: n={len(t_t)}, span={t_t[-1]-t_t[0]:.1f}s, fs={fs_t:.1f}Hz")
    tg, y_disp = resample_uniform(t_t, feats["Y位移幅值(um)"], fs=125.0)
    tg, y_acc = resample_uniform(t_t, feats["加速度Y(g)"], fs=125.0)
    print(f"Y振动频率(实测): mean={np.nanmean(vf):.2f} std={np.nanstd(vf):.3f}")

    # ---------- 雷达 ----------
    disp, TB = extract_radar(G["radar"])
    t_r0 = clk(G["radar_clock"])
    fv, av, f, sp = spec_peak(disp[:int(60*FS_LOOP)], FS_LOOP)
    print(f"radar disp 谱峰: {fv:.2f} Hz  ({av:.2f} um)  target_bin={TB}")
    res["radar_vib_hz"] = round(float(fv), 2)
    res["tactile_vib_hz"] = round(float(np.nanmean(vf)), 2)

    # ---------- 触觉载波检验 ----------
    for name, sig in [("accY", y_acc)]:
        g = goertzel(sig, tg, fv)
        g2 = goertzel(sig, tg, 2*fv)
        print(f"触觉 {name} 在雷达振动频率 {fv:.1f}Hz 处幅值: {g:.5f} g "
              f"(对比其std {np.nanstd(sig):.4f}) -> {'有载波' if g > 0.3*np.nanstd(sig) else '无载波(被检波/更新率不足)'})")
        res["tactile_carrier_amp"] = round(float(g), 5)

    # ---------- 公共时间轴（tg 为绝对秒，统一转到相对触觉 t0） ----------
    t0_abs = t_t[0]
    tg_rel = tg - t0_abs
    radar_off = t_r0 - t0_abs
    t_end = radar_off + len(disp)/FS_LOOP
    m = (tg_rel >= radar_off+0.5) & (tg_rel <= t_end-0.5) & np.isfinite(y_disp)
    ts = tg_rel[m]
    # 雷达包络: 8-loop(12ms) 窗 std, hop 4 loops(6ms)
    W, HOP = 8, 4
    st = np.arange(0, len(disp)-W+1, HOP)
    env = np.array([disp[s:s+W].std() for s in st])
    tc = (st + W/2)/FS_LOOP + radar_off
    ri = np.interp(ts, tc, env)
    yi = y_disp[m]
    fs_c = 125.0
    res["overlap_s"] = round(float(ts[-1]-ts[0]), 1)
    res["overlap_n"] = int(len(ts))
    print(f"overlap: {len(ts)} 样本, {ts[-1]-ts[0]:.1f}s")

    # ---------- 粗滞后扫描 ±3s（步长1样本=8ms，索引即样本数，无换算歧义） ----------
    lags_c = np.arange(-int(3*fs_c), int(3*fs_c)+1)
    res["coarse_multi"] = {}
    lag_c = None
    for sm in [1.0, 0.5, 0.25]:
        a_c, b_c = movavg(yi, sm, fs_c), movavg(ri, sm, fs_c)
        rs_c = lag_scan(a_c, b_c, lags_c)
        i0 = int(np.nanargmax(rs_c))
        # 索引→滞后样本: 网格起点 lags_c[0], 步长 1
        lc = (lags_c[0] + parabolic(rs_c, i0)) / fs_c
        res["coarse_multi"][str(sm)] = (round(lc*1000, 0), round(float(np.nanmax(rs_c)), 3))
        print(f"粗滞后(平滑{sm}s): best lag={lc*1000:+.1f} ms, r={np.nanmax(rs_c):.3f}")
        if sm == 0.5:
            lag_c = lc
            a5, b5, rs5 = a_c, b_c, rs_c
    a_c, b_c, rs_c = a5, b5, rs5
    print(f"粗滞后(采用0.5s平滑): lag={lag_c*1000:+.1f} ms")
    res["coarse_lag_ms"] = round(lag_c*1000, 1)
    res["coarse_r"] = round(float(np.nanmax(rs_c)), 3)

    # ---------- 细滞后扫描 ±200ms @ 8ms 步 (0.05s 平滑) ----------
    # 约定: lag>0 表示雷达事件晚于触觉 lag 秒。lag_scan 中 u=a[i], v=b[i+lag]。
    # 校正即: 取 u=yi[:n-off], v=ri[off:] 配对。
    sm2 = 0.05
    a_f, b_f = movavg(yi, sm2, fs_c), movavg(ri, sm2, fs_c)
    off = int(round(lag_c*fs_c))
    if off >= 0:
        a_f2, b_f2 = a_f[:len(a_f)-off], b_f[off:]
    else:
        a_f2, b_f2 = a_f[-off:], b_f[:len(b_f)+off]
    lags_f = np.arange(-int(0.2*fs_c), int(0.2*fs_c)+1)
    rs_f = lag_scan(a_f2, b_f2, lags_f)
    i1 = int(np.nanargmax(rs_f))
    lag_f = (lags_f[0] + parabolic(rs_f, i1))/fs_c + lag_c
    print(f"细滞后: best lag={lag_f*1000:+.1f} ms, r={np.nanmax(rs_f):.3f}")
    res["fine_lag_ms"] = round(lag_f*1000, 1)
    res["fine_r"] = round(float(np.nanmax(rs_f)), 3)

    # ---------- 分段漂移（在粗滞后校正后的残差上扫描） ----------
    n_al = min(len(a_f2), len(b_f2))
    segs = np.array_split(np.arange(n_al), 4)
    drift = []
    for si, sl in enumerate(segs):
        a_s, b_s = a_f2[sl], b_f2[sl]  # 已应用粗滞后
        lf = np.arange(-int(0.15*fs_c), int(0.15*fs_c)+1)
        rr = []
        for lag in lf:
            if lag >= 0:
                u, v = a_s[:len(a_s)-lag], b_s[lag:]
            else:
                u, v = a_s[-lag:], b_s[:len(b_s)+lag]
            m2 = np.isfinite(u) & np.isfinite(v)
            rr.append(np.corrcoef(u[m2], v[m2])[0, 1] if m2.sum() > 20 else np.nan)
        rr = np.array(rr)
        j = int(np.nanargmax(rr))
        dl = (lf[0] + parabolic(rr, j))/fs_c
        tc_seg = ts[:n_al][sl].mean()
        drift.append((tc_seg, dl*1000, np.nanmax(rr)))
        print(f"  段{si+1} t≈{tc_seg:6.1f}s: 残差lag={dl*1000:+6.1f} ms, r={np.nanmax(rr):.3f}")
    drift = np.array(drift)
    p1, p0 = np.polyfit(drift[:, 0], drift[:, 1], 1)
    res["drift_ms_per_s"] = round(float(p1), 4)
    print(f"残差漂移拟合: {p1*1000:+.2f} ms per 1000s (段间散布 "
          f"{np.std(drift[:,1]-np.polyval([p1,p0],drift[:,0])):.0f} ms 为估计噪声)")
    res["seg_drifts"] = [(round(float(a), 1), round(float(b), 1), round(float(c), 3)) for a, b, c in drift]

    # ---------- 对齐后各尺度相关（lag 约定同上: u=yi[:n-lag], v=ri[lag:]） ----------
    lag_n = int(round(lag_f*fs_c))
    if lag_n >= 0:
        a_al, b_al = yi[:len(yi)-lag_n], ri[lag_n:]
    else:
        a_al, b_al = yi[-lag_n:], ri[:len(ri)+lag_n]
    res["scale_r"] = {}
    for sm3 in [0, 0.1, 0.25, 0.5, 1, 2]:
        aa, bb = movavg(a_al, sm3, fs_c), movavg(b_al, sm3, fs_c)
        m3 = np.isfinite(aa) & np.isfinite(bb)
        r = float(np.corrcoef(aa[m3], bb[m3])[0, 1])
        res["scale_r"][str(sm3)] = round(r, 3)
    print("漂移校正后相关:", res["scale_r"])

    # ---------- 零假设: 循环移位 5s ----------
    sh = int(5*fs_c)
    b_sh = np.roll(ri, sh)
    lags_n = np.arange(-int(0.2*fs_c), int(0.2*fs_c)+1, 1)
    rs_n = lag_scan(a_f, b_sh, lags_n)
    res["null_r"] = round(float(np.nanmax(rs_n)), 3)
    print(f"零假设(循环移位5s): 细扫描 max r={np.nanmax(rs_n):.3f} "
          f"(真实对齐 {res['fine_r']:.3f} -> {'细尺度相关可信' if res['fine_r']-np.nanmax(rs_n) > 0.2 else '仍可能虚高'})")

    # ---------- GCC-PHAT（带限 0.2~5 Hz，fftshift 索引对齐） ----------
    def gcc_phat(a, b, fs, max_lag_s=0.5, f_lo=0.2, f_hi=5.0):
        n = len(a)
        A = np.fft.rfft(a*np.hanning(n)); B = np.fft.rfft(b*np.hanning(n))
        X = A*np.conj(B)
        fr = np.fft.rfftfreq(n, 1/fs)
        band = (fr >= f_lo) & (fr <= f_hi)
        Xb = np.zeros_like(X)
        Xb[band] = X[band]/(np.abs(X[band]) + 1e-12)
        cc = np.fft.fftshift(np.fft.irfft(Xb, n))
        lags = np.fft.fftshift(np.fft.fftfreq(n, 1/fs)).astype(int)
        ml = int(max_lag_s*fs)
        m4 = np.abs(lags) <= ml
        i = np.argmax(cc[m4])
        return lags[m4][i]/fs, cc[m4][i]
    okm = np.isfinite(a_f) & np.isfinite(b_f)
    gp_lag, gp_v = gcc_phat(a_f[okm], b_f[okm], fs_c)
    print(f"GCC-PHAT(带限): lag={gp_lag*1000:+.1f} ms (与互相关 {lag_f*1000:+.1f} ms 对照)")
    res["gcc_phat_ms"] = round(gp_lag*1000, 1)

    # ---------- 图 ----------
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    # (1) 粗滞后谱
    ax = axes[0, 0]
    ax.plot(lags_c/fs_c*1000, rs_c, lw=1)
    ax.axvline(lag_c*1000, color="r", ls="--", label=f"峰 {lag_c*1000:+.0f}ms")
    ax.set_xlabel("滞后 (ms)"); ax.set_ylabel("r (0.5s平滑)")
    ax.set_title(f"{tag} 粗滞后扫描 ±3s"); ax.legend(); ax.grid(alpha=.3)
    # (2) 细滞后谱
    ax = axes[0, 1]
    ax.plot(lags_f/fs_c*1000, rs_f, lw=1)
    ax.axvline(0, color="gray", ls=":", lw=.5)
    ax.set_xlabel("滞后 (ms)"); ax.set_ylabel("r (50ms平滑)")
    ax.set_title(f"{tag} 细滞后扫描 ±200ms  (峰 {lag_f*1000:+.1f}ms)")
    ax.grid(alpha=.3)
    # (3) 分段漂移
    ax = axes[1, 0]
    ax.plot(drift[:, 0], drift[:, 1], "o-", label="分段最优滞后")
    xs = np.linspace(drift[:, 0].min(), drift[:, 0].max(), 10)
    ax.plot(xs, np.polyval([p1, p0], xs), "r--",
            label=f"漂移 {p1*1000:+.2f}ms/1000s")
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("最优滞后 (ms)")
    ax.set_title(f"{tag} 时钟漂移估计"); ax.legend(); ax.grid(alpha=.3)
    # (4) 对齐后包络叠加 (0平滑, 前10s)
    ax = axes[1, 1]
    n4 = min(int(10*fs_c), len(a_al), len(b_al))
    a4 = (a_al[:n4]-np.nanmean(a_al))/np.nanstd(a_al)
    b4 = (b_al[:n4]-np.nanmean(b_al))/np.nanstd(b_al)
    tt4 = ts[:n4]
    ax.plot(tt4, a4, lw=.8, label="触觉 Y位移幅值")
    ax.plot(tt4, b4, lw=.8, alpha=.8, label="雷达 rd_disp(12ms窗)")
    ax.set_xlabel("时间 (s)")
    ax.set_title(f"{tag} 滞后校正后原始包络 (z-score, 前10s, r(0s)={res['scale_r']['0']})")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    # (5) 平滑尺度 r 曲线
    ax = axes[2, 0]
    ks = sorted(res["scale_r"].keys(), key=float)
    ax.plot([float(k) for k in ks], [res["scale_r"][k] for k in ks], "o-")
    ax.axhline(res["null_r"], color="r", ls=":", label=f"零假设r={res['null_r']}")
    ax.set_xlabel("平滑 (s)"); ax.set_ylabel("Pearson r")
    ax.set_title(f"{tag} 相关 vs 平滑尺度"); ax.legend(); ax.grid(alpha=.3)
    # (6) 雷达谱峰 + 触觉包络谱
    ax = axes[2, 1]
    ax.semilogy(f[f < 80], sp[f < 80], lw=.6)
    ax.axvline(fv, color="r", ls="--", label=f"雷达峰 {fv:.1f}Hz")
    ax.axvline(np.nanmean(vf), color="orange", ls="--",
               label=f"触觉实测频率 {np.nanmean(vf):.1f}Hz")
    ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("位移 (um)")
    ax.set_title(f"{tag} 雷达位移谱"); ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
    fig.suptitle(f"组 {tag}: 细粒度对齐实验", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}_fine_align.png", dpi=110)
    plt.close(fig)

    ALL[tag] = res

with open(f"{OUT}/fine_align_results.json", "w", encoding="utf-8") as fjs:
    json.dump(ALL, fjs, ensure_ascii=False, indent=2)
print(f"\nsaved -> {OUT}/fine_align_results.json, {OUT}/<tag>_fine_align.png")
