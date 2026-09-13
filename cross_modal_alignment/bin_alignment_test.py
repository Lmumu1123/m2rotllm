"""
跨模态时序对齐 —— 接触式侧改用 data_0.bin 原始二进制（不再解析 CSV 特征列）。

data_0.bin 格式（已逆向验证，字段与 CSV 全量 100% 精确匹配）：
  - 78 字节 ASCII 头（"ADDR:80,CurrentMode:NormalMode,...AutoOutInfo:\r\n"）
  - 之后为 6422 条定长 219 字节记录（尾部余 155 字节）
  - 每条记录内的大端 u16 字段（记录内偏移）：
      @25-26  校验和（差分混乱、值随机碰撞 -> 非计数器）
      @27     0x50 ('P') 帧标记    @28-30 常量 03 d6 01
      @34     秒（0-59，分钟内回绕） @35 分钟(35=22:35)  @36-37 秒内毫秒(5ms 量化)
      @38-39  加速度X(g)   ×2048     @40-41 加速度Y(g) ×2048   @42-43 加速度Z(g) ×2048
      @50-51  X速度幅值(mm/s) ×100   @52-53 Y速度 ×100        @54-55 Z速度 ×100
      @62-63  温度(℃) ×100
      @64-65  X位移幅值(um)          @66-67 Y位移幅值(um)     @68-69 Z位移幅值(um)
      @70-71  X振动频率(Hz) ×10      @72-73 Y振动频率 ×10     @74-75 Z振动频率 ×10
  - 帧序与 CSV 行序 1:1（帧 i <-> CSV 行 i+2），6420 行全量字段精确相等 -> 无丢帧
  - 注意：data_0.bin 是 ~41.7Hz 的"特征帧"存储（与串口 115200 波特率一致，
    2kHz×3轴×2B=12kB/s > 11.5kB/s 物理上不可能），不含 2kHz 原始波形。

时间轴：
  - 设备侧时钟（分钟/秒/毫秒）直接来自 data_0.bin 时钟字节，绝对时间 22:35:42.145 起；
    帧特征的发射时刻 = 下一帧的时钟字段（= 本帧时钟 + 1 帧周期 ~24ms）。
  - PC 墙钟锚点：帧 0 发射 22:35:42.170 <-> PC 15:34:23.445（CSV 行 2 的接收时刻，
    仅用作一次墙钟锚定，特征全部来自 bin）。
  - 雷达文件名 20260807-153437 -> PC 15:34:37.000 起，666.67 Hz loop 率。

对齐验证：
  - 细滞后估计：互相关 + 抛物线亚帧插值，多平滑尺度（0/0.5/2/10s）
  - 滑窗滞后轨迹（20s 窗、5s 步）-> 时钟漂移上界
  - 输出接触式各轴位移统计；不要将其与雷达-触觉相关性层级混为一谈
  - 输出对齐后序列 CSV + 图 + JSON
"""

import argparse
import json
import os

import numpy as np

try:
    from cross_modal_alignment.rd_alignment_test import (
        FS_LOOP,
        LAMBDA,
        clock_to_sec,
        envelope,
        extract_radar_features,
        pearson,
        zscore,
    )
except ImportError:  # pragma: no cover
    from rd_alignment_test import (
        FS_LOOP,
        LAMBDA,
        clock_to_sec,
        envelope,
        extract_radar_features,
        pearson,
        zscore,
    )

TACTILE_BIN = os.environ.get("TACTILE_BIN", "15-34-23-384/data_0.bin")
RADAR_BIN = os.environ.get("RADAR_BIN", "20260807-153437-1786088077053250.bin")
RADAR_START = os.environ.get("RADAR_START_CLOCK", "15:34:37")
# PC 锚点：帧 0 特征发射时刻（设备钟 22:35:42.170）对应的 PC 接收时刻
ANCHOR_DEV = 22 * 3600 + 35 * 60 + 42.170
ANCHOR_PC = 15 * 3600 + 34 * 60 + 23.445
OUT_DIR = os.environ.get("ALIGN_OUT_DIR", "_bin_align_out")
REC_LEN = 219


def load_contact_bin(path: str) -> dict:
    """解析 data_0.bin -> 特征表 + 设备侧绝对时间轴（秒，天内）。"""
    raw = open(path, "rb").read()
    m = raw.find(b"AutoOutInfo:\r\n")
    if m < 0:
        raise ValueError("header not found")
    body = raw[m + len(b"AutoOutInfo:\r\n"):]
    n = len(body) // REC_LEN
    recs = np.frombuffer(body[: n * REC_LEN], dtype=np.uint8).reshape(n, REC_LEN)
    b = recs.astype(np.int64)

    def be(off, scale=1.0):
        return (b[:, off] * 256 + b[:, off + 1]).astype(float) * scale

    feats = {
        "accX(g)": be(38, 1 / 2048),
        "accY(g)": be(40, 1 / 2048),
        "accZ(g)": be(42, 1 / 2048),
        "velX(mm/s)": be(50, 1 / 100),
        "velY(mm/s)": be(52, 1 / 100),
        "velZ(mm/s)": be(54, 1 / 100),
        "temp(C)": be(62, 1 / 100),
        "dispX(um)": be(64),
        "dispY(um)": be(66),
        "dispZ(um)": be(68),
        "freqX(Hz)": be(70, 1 / 10),
        "freqY(Hz)": be(72, 1 / 10),
        "freqZ(Hz)": be(74, 1 / 10),
    }

    # ---- 设备时钟：分钟@35 / 秒@34(0-59) / 毫秒@36-37 (5ms 量化) ----
    smin, ssec, sms = b[:, 35], b[:, 34], b[:, 36] * 256 + b[:, 37]
    key = smin * 60000 + ssec * 1000 + sms          # 毫秒键（分钟内）
    d = np.diff(key)
    d = (d + 30000) % 60000 - 30000                  # 秒回绕
    d = (d + 3000000) % 6000000 - 3000000            # 分钟回绕（保险）
    glitch = (d < 10) | (d > 50)
    d_fix = np.where(glitch, 24.0, d)                # 4 个回绕毛刺按中位周期补
    dev_ms = np.concatenate([[key[0]], key[0] + np.cumsum(d_fix)])
    dev_abs = dev_ms / 1000.0 + 22 * 3600            # 当天秒（首帧 22:35:42.145）

    # 帧特征的发射时刻 = 下一帧的时钟（特征在 ~24ms 窗末产生并发出）
    frame_period_ms = float(np.median(d_fix))
    emit = np.concatenate([dev_abs[1:], [dev_abs[-1] + frame_period_ms / 1000.0]])
    n_glitch = int(glitch.sum())
    return {
        "feats": feats,
        "dev_abs": dev_abs,
        "emit_abs": emit,
        "n_frames": n,
        "n_glitch": n_glitch,
        "frame_period_ms": frame_period_ms,
        "dev_span_s": float(dev_abs[-1] - dev_abs[0]),
    }


def fine_lag(x_t, x_v, y_t, y_v, max_lag_s=5.0):
    """在实际重叠区间内做互相关细滞后（抛物线亚点插值），不外推钳位。x 为基准。"""
    t0 = max(x_t[0], y_t[0]) + max_lag_s      # 保证 ±max_lag 平移后仍在双方范围内
    t1 = min(x_t[-1], y_t[-1]) - max_lag_s
    dt = float(np.median(np.diff(y_t)))
    grid = np.arange(t0, t1, dt)
    yg = np.interp(grid, y_t, y_v)
    best_r, best_lag = 0.0, 0.0
    max_k = int(max_lag_s / dt)
    xz_grid = zscore(np.interp(grid, x_t, x_v))
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            a, c = xz_grid[k:], yg[: len(yg) - k]
        else:
            a, c = xz_grid[: len(xz_grid) + k], yg[-k:]
        if len(a) < 100:
            continue
        r = pearson(a, c)
        if np.isfinite(r) and abs(r) > abs(best_r):
            best_r, best_lag = r, k * dt
    r, lag = best_r, best_lag
    # 抛物线插值（在整数栅格附近重算三点）
    ks = lag / dt
    k0 = int(round(ks))
    rs = {}
    for kk in (k0 - 1, k0, k0 + 1):
        if kk >= 0:
            a, c = xz_grid[kk:], yg[: len(yg) - kk]
        else:
            a, c = xz_grid[: len(xz_grid) + kk], yg[-kk:]
        rs[kk] = pearson(a, c) if len(a) >= 100 else np.nan
    if np.all(np.isfinite([rs.get(k0 - 1, np.nan), rs[k0], rs.get(k0 + 1, np.nan)])):
        y0, y1, y2 = rs[k0 - 1], rs[k0], rs[k0 + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            ksub = k0 + 0.5 * (y0 - y2) / den
            lag = ksub * dt
    return float(r), float(lag), float(dt)


def smooth(x_t, x_v, win_s):
    if win_s <= 0:
        return x_t, x_v
    dt = float(np.median(np.diff(x_t)))
    w = max(1, int(round(win_s / dt)))
    if w < 2:
        return x_t, x_v
    k = np.ones(w) / w
    v = np.convolve(x_v, k, mode="valid")
    # v[j] = mean(x[j : j+w]) -> 窗中心时刻 = x_t[j] + (w-1)/2*dt
    t = x_t[: len(v)] + (w - 1) / 2 * dt
    return t, v


def main():
    global TACTILE_BIN, RADAR_BIN, RADAR_START, OUT_DIR
    ap = argparse.ArgumentParser(description="Validate timestamp alignment using tactile data_0.bin.")
    ap.add_argument("--tactile_bin", default=TACTILE_BIN)
    ap.add_argument("--radar_bin", default=RADAR_BIN)
    ap.add_argument("--radar_start_clock", default=RADAR_START)
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    TACTILE_BIN = args.tactile_bin
    RADAR_BIN = args.radar_bin
    RADAR_START = args.radar_start_clock
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    C = load_contact_bin(TACTILE_BIN)
    print(f"contact: {C['n_frames']} frames, device span {C['dev_span_s']:.3f}s, "
          f"frame period {C['frame_period_ms']:.1f}ms, clock glitches {C['n_glitch']}")

    # PC 时间轴（发射时刻 + 墙钟偏移）
    pc_off = ANCHOR_PC - ANCHOR_DEV
    t_pc = C["emit_abs"] + pc_off
    print(f"wall-clock anchor: dev 22:35:42.170 <-> pc 15:34:23.445 (offset {pc_off:+.3f}s)")

    # 设备时钟 vs 匀速栅格的稳定性（帧间抖动）——设备侧时间轴的均匀性
    dtpc = np.diff(t_pc) * 1000
    results_clock = {
        "dev_frame_dt_ms": {
            "median": round(float(np.median(dtpc)), 2),
            "p5": round(float(np.percentile(dtpc, 5)), 2),
            "p95": round(float(np.percentile(dtpc, 95)), 2),
        }
    }
    print(f"device-clock frame dt: median={np.median(dtpc):.1f}ms "
          f"[p5={np.percentile(dtpc,5):.1f}, p95={np.percentile(dtpc,95):.1f}]")

    # ---- 雷达（时间基准统一为"接触帧 0 发射时刻"的相对秒） ----
    t_rel = t_pc - t_pc[0]          # 相对时间轴，帧 0 = 0s
    R = extract_radar_features(RADAR_BIN)
    zb = R["zb"]
    disp_um = np.unwrap(np.angle(zb)) * LAMBDA / (4 * np.pi) * 1e6
    radar_t0 = clock_to_sec(RADAR_START) - clock_to_sec("15:34:23.445")
    print(f"radar: {R['num_loops']} loops @ {FS_LOOP:.2f}Hz = {R['num_loops']/FS_LOOP:.1f}s, "
          f"target bin {R['target_bin']}, t0 offset {radar_t0:+.3f}s vs contact frame0")

    # 24ms 包络（16 loops，与特征帧窗一致）
    W = 16
    starts = np.arange(0, len(zb) - W + 1, W)
    t_env = (starts + W / 2) / FS_LOOP + radar_t0
    env_rd_disp = envelope(disp_um, W, W, "std")
    env_rd_mag = envelope(np.abs(zb), W, W, "std")
    env_crude = envelope(R["crude"], W, W, "mean")

    contact = {
        "dispZ(um)": C["feats"]["dispZ(um)"],
        "dispX(um)": C["feats"]["dispX(um)"],
        "dispY(um)": C["feats"]["dispY(um)"],
        "velZ(mm/s)": C["feats"]["velZ(mm/s)"],
    }

    results = {
        "n_contact_frames": C["n_frames"],
        "contact_dev_span_s": round(C["dev_span_s"], 3),
        "frame_period_ms": round(C["frame_period_ms"], 2),
        "clock_glitches": C["n_glitch"],
        "radar_loops": R["num_loops"],
        "target_bin": R["target_bin"],
        "radar_t0_s": round(radar_t0, 3),
        "fine_lag": {},
        "smooth_scan": {},
        "lag_track": {},
        "anisotropy": {},
        "clock": results_clock,
    }

    # ---- 诚实口径相关矩阵（lag=0，valid 平滑，无 mode="same" 边缘伪影） ----
    print("\n===== r(lag=0) 矩阵（valid 平滑；旧代码 mode='same' 的边缘斜坡会虚高） =====")
    results["r_matrix"] = {}
    for sm in [0.0, 0.5, 2.0, 10.0]:
        ty_s, vy_s = smooth(t_env, env_rd_disp, sm)
        ty_m, vy_m = smooth(t_env, env_rd_mag, sm)
        ty_c, vy_c = smooth(t_env, env_crude, sm)
        rad = {"rd_disp": (ty_s, vy_s), "rd_mag": (ty_m, vy_m), "crude": (ty_c, vy_c)}
        g0 = max(max(t[0] for t, _ in rad.values()), t_rel[0] + sm / 2)
        g1 = min(min(t[-1] for t, _ in rad.values()), t_rel[-1] - sm / 2)
        dt = float(np.median(np.diff(ty_s)))
        grid = np.arange(g0, g1, dt)
        row_out = {}
        for rn, (tyy, vyy) in rad.items():
            yv = np.interp(grid, tyy, vyy)
            for fn, fv in C["feats"].items():
                tx, vx = smooth(t_rel, fv, sm)
                xv = np.interp(grid, tx, vx)
                row_out[f"{rn}|{fn}"] = round(pearson(xv, yv), 3)
        results["r_matrix"][f"{sm}s"] = row_out
        top = sorted(row_out.items(), key=lambda kv: -abs(kv[1]))[:5]
        print(f"  sm={sm:>4}s top5: " + ", ".join(f"{k}={v:+.3f}" for k, v in top))

    # ---- 滞后谱（2s 平滑，rd_disp vs dispX/dispZ） ----
    print("\n===== 滞后谱（2s 平滑，±10s） =====")
    ty2, vy2 = smooth(t_env, env_rd_disp, 2.0)
    results["lag_profile"] = {}
    for cn in ["dispX(um)", "dispZ(um)"]:
        tx2, vx2 = smooth(t_rel, C["feats"][cn], 2.0)
        dt = float(np.median(np.diff(ty2)))
        grid = np.arange(max(tx2[0], ty2[0]), min(tx2[-1], ty2[-1]), dt)
        xg = np.interp(grid, tx2, vx2)
        yg = np.interp(grid, ty2, vy2)
        xz = (xg - xg.mean()) / xg.std()
        prof = []
        for k in range(-int(10 / dt), int(10 / dt) + 1):
            a, c = (xz[k:], yg[: len(yg) - k]) if k >= 0 else (xz[: len(xz) + k], yg[-k:])
            prof.append((k * dt, pearson(a, c)))
        prof = np.array(prof)
        r0 = prof[np.argmin(np.abs(prof[:, 0])), 1]
        rng = float(prof[:, 1].max() - prof[:, 1].min())
        print(f"  {cn}: r(lag=0)={r0:+.3f}, 全程 r 波动范围={rng:.3f} "
              f"({'平谱：滞后不可定位' if rng < 0.15 else '存在结构'})")
        results["lag_profile"][cn] = {
            "r_lag0": round(float(r0), 3),
            "r_range": round(rng, 3),
            "profile": {f"{t:+.1f}s": round(float(v), 3) for t, v in prof[:: max(1, len(prof)//21)]},
        }

    # ---- 细滞后（强通道：crude/rd_mag vs velX/dispX；含 lag=0 参照与平谱判定） ----
    print("\n===== 细滞后估计（强通道 + lag=0 参照） =====")
    radar_envs = {"crude": (t_env, env_crude), "rd_mag": (t_env, env_rd_mag),
                  "rd_disp": (t_env, env_rd_disp)}
    for win_s in [0.0, 0.5, 2.0]:
        row = {}
        for rn, (rt, rv) in radar_envs.items():
            ty, vy = smooth(rt, rv, win_s)
            for cn in ["velX(mm/s)", "dispX(um)"]:
                tx, vx = smooth(t_rel, C["feats"][cn], win_s)
                dt = float(np.median(np.diff(ty)))
                grid = np.arange(max(tx[0], ty[0]), min(tx[-1], ty[-1]), dt)
                xg = np.interp(grid, tx, vx)
                yg = np.interp(grid, ty, vy)
                xz = (xg - xg.mean()) / xg.std()
                prof = []
                for k in range(-int(10 / dt), int(10 / dt) + 1):
                    a, c = (xz[k:], yg[: len(yg) - k]) if k >= 0 else (xz[: len(xz) + k], yg[-k:])
                    prof.append((k * dt, pearson(a, c)))
                prof = np.array(prof)
                r0 = float(prof[np.argmin(np.abs(prof[:, 0])), 1])
                i = int(np.argmax(np.abs(prof[:, 1])))
                rb, lb = float(prof[i, 1]), float(prof[i, 0])
                rng = float(prof[:, 1].max() - prof[:, 1].min())
                key = f"{rn}|{cn}"
                row[key] = {"r_lag0": round(r0, 3), "r_best": round(rb, 3),
                            "lag_best_s": round(lb, 3), "r_range": round(rng, 3)}
                print(f"  sm={win_s:>4}s {key:<24} r(0)={r0:+.3f} best={rb:+.3f}@{lb:+.2f}s "
                      f"range={rng:.3f} {'[平谱:滞后不可定位,与0滞后一致]' if rng < 0.15 else ''}")
        results["smooth_scan"][f"{win_s}s"] = row

    # ---- 滑窗滞后轨迹（时钟漂移上界） ----
    print("\n===== 滑窗滞后轨迹（30s 窗 / 5s 步，rd_disp vs dispX/dispZ） =====")
    win_s = 2.0
    ty_s, vy_s = smooth(t_env, env_rd_disp, win_s)
    seg, step = 30.0, 5.0
    tracks = []
    for cn in ["dispX(um)", "dispZ(um)"]:
        tx, vx = smooth(t_rel, contact[cn], win_s)
        t_start = max(tx[0], ty_s[0]) + seg / 2 + 1
        t_end = min(tx[-1], ty_s[-1]) - seg / 2 - 1
        tt = t_start
        while tt < t_end:
            m_x = (tx >= tt - seg / 2) & (tx <= tt + seg / 2)
            m_y = (ty_s >= tt - seg / 2) & (ty_s <= tt + seg / 2)
            if m_x.sum() > 50 and m_y.sum() > 50:
                r, lag, _ = fine_lag(tx[m_x], vx[m_x], ty_s[m_y], vy_s[m_y], max_lag_s=1.0)
                tracks.append((tt, lag, r, cn))
            tt += step
    tracks.sort()
    lags = np.array([x[1] for x in tracks])
    rs = np.array([x[2] for x in tracks])
    ok = np.abs(rs) > 0.3
    if ok.sum() > 1:
        lag_std = float(lags[ok].std())
        lag_rng = float(lags[ok].max() - lags[ok].min())
        print(f"  windows={len(tracks)}, |r|>0.3: {ok.sum()}, "
              f"lag std={lag_std*1000:.1f}ms, range={lag_rng*1000:.1f}ms")
        results["lag_track"] = {
            "n": len(tracks), "n_ok": int(ok.sum()),
            "lag_std_ms": round(lag_std * 1000, 1),
            "lag_range_ms": round(lag_rng * 1000, 1),
        }
    np.save(os.path.join(OUT_DIR, "lag_track.npy"),
            np.array([(x[0], x[1], x[2]) for x in tracks]))

    # ---- 各向异性 ----
    print("\n===== 接触式位移通道统计（不等同于跨模态相关性） =====")
    for cn in ["dispX(um)", "dispY(um)", "dispZ(um)"]:
        v = contact[cn]
        results["anisotropy"][cn] = round(float(np.std(v)), 2)
        print(f"  {cn}: std={np.std(v):.1f}")

    # ---- 导出对齐后序列（时间戳直接对齐，零额外平移；滞后谱平谱证实无需平移） ----
    sel = (t_rel >= radar_t0 + 1) & (t_rel <= radar_t0 + R["num_loops"] / FS_LOOP - 1)
    out = {
        "t_rel_s": t_rel[sel],
        "dispX_um": contact["dispX(um)"][sel],
        "dispY_um": contact["dispY(um)"][sel],
        "dispZ_um": contact["dispZ(um)"][sel],
        "velX_mm_s": C["feats"]["velX(mm/s)"][sel],
        "freqX_hz": C["feats"]["freqX(Hz)"][sel],
        "temp_c": C["feats"]["temp(C)"][sel],
        "radar_disp_env_um": np.interp(t_rel[sel], t_env, env_rd_disp),
        "radar_mag_env": np.interp(t_rel[sel], t_env, env_rd_mag),
        "radar_crude_env": np.interp(t_rel[sel], t_env, env_crude),
    }
    keys = list(out.keys())
    arr = np.column_stack([out[k] for k in keys])
    np.savetxt(os.path.join(OUT_DIR, "aligned_series.csv"), arr, delimiter=",",
               header=",".join(keys), comments="", fmt="%.6f")
    print(f"\naligned series exported: {OUT_DIR}/aligned_series.csv ({sel.sum()} rows)")
    print(f"r(velX vs radar_mag_env)  @lag0: {pearson(out['velX_mm_s'], out['radar_mag_env']):+.4f}")
    print(f"r(dispX vs radar_mag_env) @lag0: {pearson(out['dispX_um'], out['radar_mag_env']):+.4f}")
    print(f"r(velX vs radar_crude)    @lag0: {pearson(out['velX_mm_s'], out['radar_crude_env']):+.4f}")
    print(f"r(dispZ vs radar_disp_env)@lag0: {pearson(out['dispZ_um'], out['radar_disp_env_um']):+.4f}")

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"results: {OUT_DIR}/results.json")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        font_path = os.environ.get("ALIGNMENT_FONT")
        fp = (font_manager.FontProperties(fname=font_path)
              if font_path and os.path.exists(font_path) else None)
        fig, axes = plt.subplots(3, 1, figsize=(12, 11))
        ax = axes[0]
        ax.plot(out["t_rel_s"], zscore(out["velX_mm_s"]), label="接触式 X速度幅值(data_0.bin)", lw=0.8)
        ax.plot(out["t_rel_s"], zscore(out["radar_mag_env"]), label="雷达目标bin幅值包络", lw=0.8, alpha=0.8)
        ax.set_title(f"时间戳直接对齐（零平移）r={pearson(out['velX_mm_s'], out['radar_mag_env']):+.3f}",
                     fontproperties=fp)
        ax.set_xlabel("t (s，接触帧0发射时刻起)", fontproperties=fp)
        ax.legend(prop=fp)
        ax = axes[1]
        ax.plot(out["t_rel_s"], zscore(out["dispX_um"]), label="接触式 X位移幅值", lw=0.8)
        ax.plot(out["t_rel_s"], zscore(out["radar_crude_env"]), label="雷达 crude 包络", lw=0.8, alpha=0.8)
        ax.set_title(f"crude vs X位移 r={pearson(out['dispX_um'], out['radar_crude_env']):+.3f}（慢尺度共同趋势）",
                     fontproperties=fp)
        ax.set_xlabel("t (s)", fontproperties=fp)
        ax.legend(prop=fp)
        ax = axes[2]
        sm = 0.5
        tx2, vx2 = smooth(t_rel, C["feats"]["velX(mm/s)"], sm)
        ty2, vy2 = smooth(t_env, env_rd_mag, sm)
        ax.plot(tx2, zscore(vx2), lw=0.9, label="接触式 X速度幅值(0.5s平滑)")
        msk = (ty2 >= tx2[0]) & (ty2 <= tx2[-1])
        ax.plot(ty2[msk], zscore(vy2[msk]), lw=0.9, alpha=0.8, label="雷达 rd_mag(0.5s平滑)")
        ax.set_xlabel("t (s)", fontproperties=fp)
        ax.set_title("0.5s 平滑后对照（快尺度受雷达包络噪声限制）", fontproperties=fp)
        ax.legend(prop=fp)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "alignment.png"), dpi=150)
        print(f"plot: {OUT_DIR}/alignment.png")
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
