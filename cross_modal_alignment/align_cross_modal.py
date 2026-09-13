"""Legacy coarse-proxy + constrained-DTW alignment pipeline.

This module is retained as a baseline for reproducing the initial experiment.
For the corrected radar format and timestamp-based validation, use
``python -m cross_modal_alignment.rd_alignment_test`` instead.
"""

import argparse
import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd

try:
    # When executed as a module: python -m cross_modal_alignment.align_cross_modal
    from cross_modal_alignment.dtw_alignment import best_lag_correlation, constrained_dtw_path
except ImportError:  # pragma: no cover
    # When executed as a script: python align_cross_modal.py (from this directory)
    from dtw_alignment import best_lag_correlation, constrained_dtw_path


def load_tactile_feature(
    tactile_csv: str,
    *,
    feature_mode: str = "auto",
    forced_feature_col: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Returns:
      t_sec: relative time seconds starting from 0
      x: feature sequence (float32)
      col_name: selected column name
    """
    df = pd.read_csv(tactile_csv)
    if "片上时间()" in df.columns:
        t_abs = pd.to_datetime(df["片上时间()"], errors="coerce")
        # drop NaT
        ok = t_abs.notna().values
        df = df.loc[ok].reset_index(drop=True)
        t_abs = pd.to_datetime(df["片上时间()"], errors="coerce")
        t_sec = (t_abs - t_abs.iloc[0]).dt.total_seconds().astype(np.float32).to_numpy()
    else:
        # fallback: use row index as time proxy
        t_sec = np.arange(len(df), dtype=np.float32)

    # Ensure numeric columns
    numeric_cols = []
    for c in df.columns:
        if c in ["设备名称", "时间"]:
            continue
        # attempt convert
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() > 0.98:
            numeric_cols.append(c)

    if forced_feature_col is not None:
        if forced_feature_col not in df.columns:
            raise ValueError(f"forced_feature_col not found: {forced_feature_col}")
        x = pd.to_numeric(df[forced_feature_col], errors="coerce").astype(np.float32).to_numpy()
        x = np.nan_to_num(x)
        return t_sec, x, forced_feature_col

    # Candidate set: prefer X-axis-related quantities (you specified X points to radar).
    cand = []
    for c in numeric_cols:
        if "X" in c and ("加速度" in c or "速度" in c or "位移" in c or "谱能量" in c or "能量" in c):
            cand.append(c)
    if not cand:
        cand = numeric_cols

    if feature_mode == "auto":
        # Choose the most variable candidate to maximize event informativeness.
        best_c = cand[0]
        best_std = -1.0
        for c in cand:
            v = pd.to_numeric(df[c], errors="coerce").astype(np.float32).to_numpy()
            v = np.nan_to_num(v)
            sd = float(np.std(v))
            # Penalize all-zero columns
            if sd < 1e-8:
                continue
            if sd > best_std:
                best_std = sd
                best_c = c
        x = pd.to_numeric(df[best_c], errors="coerce").astype(np.float32).to_numpy()
        x = np.nan_to_num(x)
        return t_sec, x, best_c

    if feature_mode == "x_acc_x":
        # Prefer raw acceleration X if available.
        # Your CSV header includes "加速度X(g)" (in some datasets it might be "加速度X(g)" exactly).
        for c in ["加速度X(g)", "加速度X(g) ", "加速度X(g)"]:
            if c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce").astype(np.float32).to_numpy()
                x = np.nan_to_num(x)
                return t_sec, x, c
        raise ValueError("feature_mode=x_acc_x but column not found.")

    raise ValueError(f"Unknown feature_mode: {feature_mode}")


def extract_radar_slow_proxy_from_bin(
    radar_bin: str,
    *,
    num_adcsamples: int,
    num_rx: int,
    chirps_per_loop: int,
    loop_count: int,
    max_loops: Optional[int] = None,
) -> np.ndarray:
    """
    Extract a 1D radar proxy over slow-time loops, using only magnitudes
    (no full range/Doppler processing). This keeps memory manageable.

    Assumptions (based on TI mmWave typical raw ADC):
    - ADC output is complex I/Q interleaved as int16 pairs.
    - Per chirp: num_rx * num_adcsamples complex samples.
    - File concatenates chirps sequentially.

    Returns:
      y: shape [num_total_loops_processed], float32
    """
    file_bytes = os.path.getsize(radar_bin)
    int16_count = file_bytes // 2
    total_complex = int16_count // 2  # each complex = 2 int16 (I,Q)

    complex_per_chirp = num_rx * num_adcsamples
    if total_complex % complex_per_chirp != 0:
        raise RuntimeError(
            f"bin size not divisible by complex_per_chirp. total_complex={total_complex}, complex_per_chirp={complex_per_chirp}"
        )
    total_chirps = total_complex // complex_per_chirp
    chirps_per_frame = chirps_per_loop * loop_count

    if total_chirps % chirps_per_frame != 0:
        # still allow approximate truncation
        frames = total_chirps // chirps_per_frame
        total_chirps = frames * chirps_per_frame
        total_complex = total_chirps * complex_per_chirp

    num_frames = total_chirps // chirps_per_frame
    total_loops = num_frames * loop_count

    if max_loops is None:
        loops_to_use = total_loops
    else:
        loops_to_use = min(int(max_loops), total_loops)

    # We will process chirps in chunks and accumulate loop values.
    # loop index maps to chirp indices:
    #   frame f (0..num_frames-1), loop l (0..loop_count-1)
    #   chirp indices for that (f,l): start = f*chirps_per_frame + l*chirps_per_loop + k, k=0..chirps_per_loop-1
    y = np.zeros((loops_to_use,), dtype=np.float32)

    int16_per_complex = 2
    int16_per_chirp = num_rx * num_adcsamples * int16_per_complex

    # chunk size in chirps
    chirps_chunk = 2048
    mm = np.memmap(radar_bin, dtype=np.int16, mode="r")

    def chirp_block_magnitude(chirp_start: int, chirp_count: int) -> np.ndarray:
        """
        Return magnitude proxy per chirp (averaged across rx+adc) for a block.
        """
        s = chirp_start * int16_per_chirp
        e = s + chirp_count * int16_per_chirp
        # NumPy 2.x is stricter about copy avoidance; allow a small copy for stability.
        block = np.asarray(mm[s:e], dtype=np.int32)
        block = block.reshape(chirp_count, num_rx * num_adcsamples * 2)  # last dim is I/Q interleaved
        I = block[:, 0::2].astype(np.float32, copy=False)
        Q = block[:, 1::2].astype(np.float32, copy=False)
        mag = np.sqrt(I * I + Q * Q)  # [chirp_count, num_rx*num_adcsamples]
        # mean magnitude across rx and fast-time samples
        per_chirp = mag.mean(axis=1)
        return per_chirp.astype(np.float32)

    # Fill y loop by loop by using chirp magnitudes and averaging chirps_per_loop
    chirp_idx = 0
    loop_idx = 0
    for f in range(num_frames):
        if loop_idx >= loops_to_use:
            break
        frame_chirp_base = f * chirps_per_frame
        for l in range(loop_count):
            if loop_idx >= loops_to_use:
                break
            loop_chirp_start = frame_chirp_base + l * chirps_per_loop
            # process just chirps_per_loop chirps
            mags = chirp_block_magnitude(loop_chirp_start, chirps_per_loop)
            y[loop_idx] = float(mags.mean())
            loop_idx += 1

    return y


def downsample_average(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n <= target_len:
        return x
    k = int(np.ceil(n / target_len))
    out_len = int(np.ceil(n / k))
    out = np.zeros((out_len,), dtype=np.float32)
    for i in range(out_len):
        s = i * k
        e = min(n, (i + 1) * k)
        out[i] = float(np.mean(x[s:e]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radar_bin", required=True, type=str)
    ap.add_argument("--tactile_csv", required=True, type=str)
    ap.add_argument("--feature_mode", default="auto", type=str)
    ap.add_argument("--forced_feature_col", default=None, type=str)
    ap.add_argument("--num_adcsamples", default=256, type=int)
    ap.add_argument("--num_rx", default=4, type=int)
    ap.add_argument("--chirps_per_loop", default=3, type=int)
    ap.add_argument("--loop_count", default=64, type=int)
    ap.add_argument("--max_loops", default=20000, type=int)
    ap.add_argument("--dtw_band", default=80, type=int)
    ap.add_argument("--dtw_alpha", default=0.7, type=float)
    ap.add_argument("--max_corr_lag", default=400, type=int)
    ap.add_argument("--tactile_max_len", default=2000, type=int)
    ap.add_argument("--radar_max_len", default=2000, type=int)
    ap.add_argument("--out_dir", default=".", type=str)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t_sec, x_raw, x_col = load_tactile_feature(
        args.tactile_csv,
        feature_mode=args.feature_mode,
        forced_feature_col=args.forced_feature_col,
    )

    y_raw = extract_radar_slow_proxy_from_bin(
        args.radar_bin,
        num_adcsamples=args.num_adcsamples,
        num_rx=args.num_rx,
        chirps_per_loop=args.chirps_per_loop,
        loop_count=args.loop_count,
        max_loops=args.max_loops,
    )

    # Downsample both sequences for faster DTW.
    x_ds = downsample_average(x_raw, target_len=args.tactile_max_len)
    y_ds = downsample_average(y_raw, target_len=args.radar_max_len)

    # Coarse lag from correlation (on derivatives is more stable for alignment).
    lag = best_lag_correlation(x_ds, y_ds, max_lag=args.max_corr_lag)

    # Shift y by lag for visualization/alignment; DTW then refines with a band.
    if lag >= 0:
        # y starts later -> overlap starts at y index lag
        y_shifted = y_ds[lag:]
        x_for_dtw = x_ds[: y_shifted.shape[0]]
        y_for_dtw = y_shifted
    else:
        # y starts earlier -> x starts later
        x_shifted = x_ds[-lag:]
        y_for_dtw = y_ds[: x_shifted.shape[0]]
        x_for_dtw = x_shifted

    # Ensure same length for DTW feasibility inside the band.
    L = min(len(x_for_dtw), len(y_for_dtw))
    x_for_dtw = x_for_dtw[:L]
    y_for_dtw = y_for_dtw[:L]

    path_x, path_y = constrained_dtw_path(
        x_for_dtw,
        y_for_dtw,
        w=args.dtw_band,
        alpha=args.dtw_alpha,
    )

    # Build aligned tactile values on radar axis:
    # path_y is non-decreasing; keep last mapped x for each y.
    aligned = np.zeros((len(y_for_dtw),), dtype=np.float32)
    mapped_x_idx = np.zeros((len(y_for_dtw),), dtype=np.int32)
    for px, py in zip(path_x, path_y):
        aligned[py] = x_for_dtw[px]
        mapped_x_idx[py] = px

    out = {
        "selected_tactile_feature_col": x_col,
        "tactile_feature_mode": args.feature_mode,
        "x_raw_len": int(len(x_raw)),
        "y_raw_len": int(len(y_raw)),
        "x_ds_len": int(len(x_ds)),
        "y_ds_len": int(len(y_ds)),
        "coarse_lag_on_ds": int(lag),
        "dtw_band": int(args.dtw_band),
        "dtw_alpha": float(args.dtw_alpha),
        "dtw_path_len": int(len(path_x)),
        "used_len_for_dtw": int(L),
    }

    # Save outputs
    with open(os.path.join(args.out_dir, "alignment_info.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(args.out_dir, "radar_proxy_y_ds.npy"), y_for_dtw)
    np.save(os.path.join(args.out_dir, "aligned_tactile_x.npy"), aligned)
    np.save(os.path.join(args.out_dir, "mapped_tactile_indices.npy"), mapped_x_idx)

    # Also write a compact CSV for downstream usage.
    csv_path = os.path.join(args.out_dir, "aligned_pair.csv")
    import csv as _csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = _csv.writer(f)
        wr.writerow(["radar_step_idx_in_dtw", "radar_proxy_y", "aligned_tactile_x", "mapped_tactile_idx_in_dtw"])
        for j in range(len(y_for_dtw)):
            wr.writerow([int(j), float(y_for_dtw[j]), float(aligned[j]), int(mapped_x_idx[j])])

    print("Done.")
    print("Selected tactile feature:", x_col)
    print("Radar proxy len (processed):", len(y_raw))
    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    main()
