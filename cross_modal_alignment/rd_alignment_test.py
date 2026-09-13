"""Corrected radar/tactile cross-modal validation experiment.

Range-Doppler 级雷达特征 vs 旧幅值代理 的跨模态对齐对比实验。

相对 align_cross_modal.py 的修正：
1. 雷达原始数据实为 **实数 ADC 采样**（4 RX x 512 real samples / chirp）。
   旧代码按复数 I/Q interleaved 解析是错误的：实测 corr(I,Q)=0.92、
   "复数"频谱共轭对称 —— 均为实信号特征。
2. 真实 loop 率 = 1000/1.5 = 666.67 Hz（chirp 周期 = idle 420us + ramp 80us
   = 500us，3 chirps/loop），全文件 86528 loops ≈ 129.8 s，与触觉 153.5 s
   同量级。旧文档按 3.2 kHz 理解且只取前 2000 loops（实际仅 3 s），
   却与触觉全段 153.5 s 做对齐 —— 时间尺度失配 ~51 倍。
3. 触觉 CSV 每行比表头多 1 个尾字段，pandas 把第 1 列（真实时间）当作
   index，列名整体左移一位 —— 旧代码取的 "X位移幅值(um)" 实际是 Y 位移。

雷达特征（均在电机目标 range bin 上提取，slow-time 666.67 Hz）：
  crude   : 旧版代理 —— 全部实采样绝对值的 loop 均值
  rd_mag  : 目标 bin Range-FFT 幅值 |z(t)| 的滑动 std
  rd_disp : 目标 bin 相位位移包络 std(unwrap(angle z)) * lambda/(4pi)  [um]
  rd_dop  : 目标 bin 多普勒幅度 —— 96ms 窗 FFT 非直流最大幅值

输出 Pearson 对比：时间戳直接对齐 / 粗滞后 / 粗滞后+DTW（与旧协议同参数）。
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

try:
    from cross_modal_alignment.dtw_alignment import best_lag_correlation, constrained_dtw_path
except ImportError:  # pragma: no cover
    from dtw_alignment import best_lag_correlation, constrained_dtw_path

C_LIGHT = 3e8
FC = 77e9
LAMBDA = C_LIGHT / FC
FS_LOOP = 1000.0 / 1.5  # 666.67 Hz


def clock_to_sec(s: str) -> float:
    h, m, sec = str(s).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def block_mean_downsample(x: np.ndarray, target_len: int) -> np.ndarray:
    """与旧版 downsample_average 相同语义的分块均值降采样。"""
    n = len(x)
    if n <= target_len:
        return np.asarray(x, dtype=np.float64)
    k = int(np.ceil(n / target_len))
    out_len = int(np.ceil(n / k))
    out = np.empty(out_len)
    for i in range(out_len):
        out[i] = x[i * k : (i + 1) * k].mean()
    return out


def load_tactile_fixed(tactile_csv: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """正确解析触觉 CSV：首列为时间（被 pandas 当作 index），列名整体左移一位。"""
    df = pd.read_csv(tactile_csv)
    t_str = pd.Series(df.index, dtype=str)
    d2 = df.iloc[:, :-1].copy()  # 丢弃行尾多余空字段
    d2.columns = df.columns[1:]
    t0 = clock_to_sec(t_str.iloc[0])
    t_rel = t_str.map(clock_to_sec).to_numpy() - t0
    feats = {}
    for c in ["X位移幅值(um)", "Y位移幅值(um)", "Z位移幅值(um)", "X速度幅值(mm/s)", "X加速度幅值(g)"]:
        if c in d2.columns:
            feats[c] = pd.to_numeric(d2[c], errors="coerce").to_numpy(dtype=np.float64)
    return t_rel, t_str, feats


def extract_radar_features(
    radar_bin: str,
    *,
    num_rx: int = 4,
    num_adc_real: int = 512,
    chirps_per_loop: int = 3,
    target_bin: int | None = None,
    max_loops: int | None = None,
    loop_block: int = 4096,
    verbose: bool = True,
) -> dict:
    """
    分块扫描雷达 bin 文件，返回每个 loop 的：
      crude[k]  全部实采样 |x| 的均值（旧版代理，与旧代码同口径）
      zb[k]     chirp0(TX0) 在目标 range bin 的单 bin DFT，跨 RX 共轭对齐后平均
    target_bin: 未 fftshift 的正频率 bin（实信号单边谱）。None 则自动选取
    （排除泄漏 bin 0..3 后平均谱幅度最大的正频率 bin）。
    """
    file_bytes = os.path.getsize(radar_bin)
    samples_per_chirp = num_rx * num_adc_real  # int16 实采样个数
    total_chirps = file_bytes // (samples_per_chirp * 2)
    total_loops = total_chirps // chirps_per_loop
    if max_loops is not None:
        total_loops = min(total_loops, max_loops)
    if verbose:
        print(f"radar: {total_chirps} chirps, {total_loops} loops, "
              f"~{total_loops / FS_LOOP:.1f} s @ {FS_LOOP:.2f} Hz loop rate")

    mm = np.memmap(radar_bin, dtype=np.int16, mode="r")

    if target_bin is None:
        # 自动选目标 bin：前 20000 loops 平均实波形 -> FFT -> 正频率侧最大
        n = min(20000, total_loops)
        sub = np.asarray(
            mm[: n * chirps_per_loop * samples_per_chirp], dtype=np.float32
        ).reshape(n * chirps_per_loop, num_rx, num_adc_real)
        mean_wf = sub[0::chirps_per_loop].mean(axis=0)  # [rx, adc] 振动平均后近似静止
        win = np.hanning(num_adc_real).astype(np.float32)
        prof = np.abs(np.fft.fft(mean_wf * win, axis=-1)).mean(axis=0)
        cand = np.arange(4, num_adc_real // 2)
        target_bin = int(cand[np.argmax(prof[cand])])
        if verbose:
            print(f"auto-detected target range bin = {target_bin} (positive-freq side)")

    win = np.hanning(num_adc_real).astype(np.float32)
    twid = np.exp(-2j * np.pi * target_bin * np.arange(num_adc_real) / num_adc_real)

    crude = np.zeros(total_loops)
    zall = np.zeros((total_loops, num_rx), dtype=np.complex128)
    for lo in range(0, total_loops, loop_block):
        hi = min(lo + loop_block, total_loops)
        nch = (hi - lo) * chirps_per_loop
        s = lo * chirps_per_loop * samples_per_chirp
        arr = np.asarray(mm[s : s + nch * samples_per_chirp], dtype=np.float32)
        arr = arr.reshape(nch, num_rx, num_adc_real)
        crude[lo:hi] = np.abs(arr).mean(axis=(1, 2)).reshape(
            hi - lo, chirps_per_loop
        ).mean(axis=1)
        c0 = arr[0::chirps_per_loop]  # TX0
        zall[lo:hi] = np.einsum("lra,a->lr", c0 * win, twid)

    # 跨 RX 共轭对齐（静止目标在各 RX 有固定相差），再平均
    ref_phase = np.angle(zall[:1000].mean(axis=0))
    zb = (zall * np.exp(-1j * ref_phase)[None, :]).mean(axis=1)
    return {"crude": crude, "zb": zb, "target_bin": target_bin, "num_loops": total_loops}


def envelope(x: np.ndarray, win: int, hop: int, mode: str = "std") -> np.ndarray:
    n = len(x)
    starts = np.arange(0, n - win + 1, hop)
    out = np.empty(len(starts))
    for i, s in enumerate(starts):
        w = x[s : s + win]
        out[i] = w.std() if mode == "std" else w.mean()
    return out


def doppler_env(z: np.ndarray, win: int, hop: int) -> np.ndarray:
    starts = np.arange(0, len(z) - win + 1, hop)
    out = np.empty(len(starts))
    for i, s in enumerate(starts):
        F = np.abs(np.fft.fft(z[s : s + win]))
        out[i] = F[1:].max()
    return out


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dtw_align_score(x, y, *, max_lag, band, alpha):
    """粗对齐（互相关）+ 受限 DTW，返回 (对齐后 pearson, lag)。"""
    xz, yz = zscore(x), zscore(y)
    lag = best_lag_correlation(xz, yz, max_lag=max_lag)
    if lag >= 0:
        ys = yz[lag:]
        xs = xz[: len(ys)]
    else:
        xs = xz[-lag:]
        ys = yz[: len(xs)]
    L = min(len(xs), len(ys))
    xs, ys = xs[:L], ys[:L]
    px, py = constrained_dtw_path(xs, ys, w=band, alpha=alpha)
    aligned = np.zeros_like(ys)
    for a, b in zip(px, py):
        aligned[b] = xs[a]
    return pearson(ys, aligned), lag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_bin", required=True)
    ap.add_argument("--tactile_csv", required=True)
    ap.add_argument("--radar_start_clock", default="15:34:37",
                    help="雷达文件名时间戳 HH:MM:SS(.fff)")
    ap.add_argument("--target_bin", default=None, type=int,
                    help="固定 Range-FFT bin；默认从数据自动选择")
    ap.add_argument("--max_loops", default=None, type=int)
    ap.add_argument("--out_dir", default="_rd_test_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- 触觉 ----------
    t_rel, t_str, feats = load_tactile_fixed(args.tactile_csv)
    print(f"tactile: n={len(t_rel)}, span={t_rel[-1]:.1f}s, "
          f"fs~{1 / np.median(np.diff(t_rel)):.1f} Hz")

    # ---------- 雷达 ----------
    R = extract_radar_features(
        args.radar_bin, target_bin=args.target_bin, max_loops=args.max_loops
    )
    zb, crude = R["zb"], R["crude"]
    disp_um = np.unwrap(np.angle(zb)) * LAMBDA / (4 * np.pi) * 1e6  # 径向位移 [um]
    print(f"disp um: mean={disp_um.mean():.1f} std={disp_um.std():.1f}")

    radar_t0 = clock_to_sec(args.radar_start_clock) - clock_to_sec(t_str.iloc[0])
    print(f"radar offset vs tactile t0: {radar_t0:+.3f} s")

    # ---------- 包络（16 loops=24ms 与触觉输出周期匹配；多普勒窗 64 loops=96ms） ----------
    W, HOP, WD = 16, 16, 64
    starts = np.arange(0, len(zb) - W + 1, HOP)
    tc = (starts + W / 2) / FS_LOOP + radar_t0
    tc_d = (np.arange(0, len(zb) - WD + 1, HOP) + WD / 2) / FS_LOOP + radar_t0

    env = {
        "crude": (envelope(crude, W, HOP, "mean"), tc),
        "rd_mag": (envelope(np.abs(zb), W, HOP, "std"), tc),
        "rd_disp": (envelope(disp_um, W, HOP, "std"), tc),
        "rd_dop": (doppler_env(zb, WD, HOP), tc_d),
    }

    # ---------- 时间戳配准 ----------
    t_end = radar_t0 + R["num_loops"] / FS_LOOP
    sel = (t_rel >= radar_t0 + 1) & (t_rel <= t_end - 1)
    t_sel = t_rel[sel]
    fs_t = 1 / np.median(np.diff(t_rel))
    print(f"overlap: {sel.sum()} tactile samples, t=[{t_sel[0]:.1f}, {t_sel[-1]:.1f}]s")

    results = {"radar_t0_s": radar_t0, "overlap_n": int(sel.sum()),
               "target_bin": R["target_bin"], "direct": {}, "dtw": {}, "lag_s": {}}

    # ---------- r_direct：时间戳直接对齐，不做任何算法对齐 ----------
    print("\n===== r_direct（仅时间戳对齐） =====")
    print(f"{'radar \\ tactile':<12}" + "".join(f"{c.split('(')[0]:>10}" for c in feats))
    interp = {}
    for rname, (rv, rt) in env.items():
        ri = np.interp(t_sel, rt, rv)
        interp[rname] = ri
        row = f"{rname:<12}"
        for fname, fv in feats.items():
            r = pearson(ri, fv[sel])
            results["direct"][f"{rname}|{fname}"] = round(r, 4)
            row += f"{r:>10.3f}"
        print(row)

    # ---------- 粗滞后 + DTW（与旧协议同参数：600 点、band=60、alpha=0.7） ----------
    print("\n===== 粗滞后(±5s) + DTW (600 点, band=60, alpha=0.7) =====")
    for rname, ri in interp.items():
        r_d, lag = dtw_align_score(
            block_mean_downsample(ri, 600),
            block_mean_downsample(feats["X位移幅值(um)"][sel], 600),
            max_lag=30, band=60, alpha=0.7,
        )
        results["dtw"][rname] = round(r_d, 4)
        results["lag_s"][rname] = round(lag * len(t_sel) / 600 / fs_t, 2)
        print(f"  {rname:<10}: r_dtw={r_d:.3f}  lag={lag} (~{lag * len(t_sel) / 600 / fs_t:+.2f} s)")

    # ---------- 旧协议复现：前 2000 loops（=3s）对触觉全段 ----------
    print("\n===== 旧协议复现（radar 前 2000 loops=3s vs 触觉全段 153.5s） =====")
    R2 = extract_radar_features(args.radar_bin, max_loops=2000, verbose=False)
    d2 = np.unwrap(np.angle(R2["zb"])) * LAMBDA / (4 * np.pi) * 1e6
    old_feats = {
        "crude(旧代理)": R2["crude"],
        "rd_mag": np.abs(R2["zb"]),
        "rd_disp": d2,
    }
    for tcol in ["X位移幅值(um)", "Y位移幅值(um)"]:
        x_d = block_mean_downsample(feats[tcol], 600)
        for name, yv in old_feats.items():
            y_d = block_mean_downsample(yv, 600)
            r_d, lag = dtw_align_score(x_d, y_d, max_lag=400, band=60, alpha=0.7)
            print(f"  tactile={tcol.split('(')[0]:<6} radar={name:<12}: r_dtw={r_d:.3f}  lag={lag}")

    # ---------- 诊断 1：rd_disp vs X位移 滞后谱（2s 平滑，检验时间轴漂移） ----------
    print("\n===== 诊断：rd_disp vs X位移幅值 滞后谱（2s 平滑，±8s） =====")

    def movavg(v, sec):
        w = max(1, int(sec * fs_t))
        return np.convolve(np.asarray(v, float), np.ones(w) / w, mode="same")

    xs2, ys2 = movavg(feats["X位移幅值(um)"][sel], 2), movavg(interp["rd_disp"], 2)
    mid = len(t_sel) // 2
    for name, sl in [("全程", slice(None)), ("前半段", slice(None, mid)), ("后半段", slice(mid, None))]:
        a, b = xs2[sl], ys2[sl]
        best = (-2, 0)
        for lag in range(-int(8 * fs_t), int(8 * fs_t) + 1):
            u, v = (a[: len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: len(b) + lag])
            r = float(np.corrcoef(u, v)[0, 1])
            if r > best[0]:
                best = (r, lag)
        print(f"  {name}: best lag = {best[1] / fs_t:+.2f} s, r = {best[0]:.3f}")
    results["lag_profile"] = {}
    for lag_s in [-8, -4, -2, -1, 0, 1, 2, 4, 8]:
        lag = int(lag_s * fs_t)
        u, v = (xs2[: len(xs2) - lag], ys2[lag:]) if lag >= 0 else (xs2[-lag:], ys2[: len(ys2) + lag])
        results["lag_profile"][f"{lag_s:+d}s"] = round(float(np.corrcoef(u, v)[0, 1]), 3)
    print("  滞后谱:", {k: f"{v:+.3f}" for k, v in results["lag_profile"].items()})

    # ---------- 诊断 2：平滑尺度相关矩阵（趋势 vs 快波动结构） ----------
    print("\n===== 诊断：r_direct vs 平滑尺度 =====")
    results["smoothing"] = {}
    for sm in [0, 0.5, 2, 10]:
        print(f"  -- smoothing {sm}s --")
        print(f"  {'radar':<10}" + "".join(f"{c.split('(')[0]:>10}" for c in feats))
        for rname in interp:
            row = f"  {rname:<10}"
            for fname, fv in feats.items():
                a, b = movavg(interp[rname], sm), movavg(fv[sel], sm)
                ok = np.isfinite(a) & np.isfinite(b)
                r = float(np.corrcoef(a[ok], b[ok])[0, 1])
                results["smoothing"][f"{rname}|{fname.split('(')[0]}|{sm}s"] = round(r, 3)
                row += f"{r:>10.3f}"
            print(row)

    # ---------- 诊断 3：DTW 零假设（循环移位，检验 DTW 是否虚高） ----------
    print("\n===== 诊断：DTW 零假设（循环移位破坏对齐） =====")
    results["dtw_null"] = {}
    x600 = block_mean_downsample(feats["X位移幅值(um)"][sel], 600)
    for shift in [0, 500, 1000, 2000, 3000, 4000]:
        y600 = block_mean_downsample(np.roll(interp["rd_disp"], shift), 600)
        r, lag = dtw_align_score(y600, x600, max_lag=30, band=60, alpha=0.7)
        results["dtw_null"][shift] = round(r, 3)
        tag = "  <- 真实对齐" if shift == 0 else ""
        print(f"  shift={shift:5d}: r_dtw = {r:.3f}  lag={lag}{tag}")

    with open(os.path.join(args.out_dir, "rd_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    np.savez(
        os.path.join(args.out_dir, "rd_series.npz"),
        t_tactile=t_sel,
        rd_disp=interp["rd_disp"],
        rd_dop=interp["rd_dop"],
        rd_mag=interp["rd_mag"],
        crude=interp["crude"],
        **{f"tactile_{k}": v[sel] for k, v in feats.items()},
    )
    print(f"\nsaved -> {args.out_dir}/rd_results.json, rd_series.npz")


if __name__ == "__main__":
    main()
