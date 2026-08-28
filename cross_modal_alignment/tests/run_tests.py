import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


def test_best_lag_correlation():
    from cross_modal_alignment.dtw_alignment import best_lag_correlation

    rng = np.random.default_rng(0)
    x = rng.normal(size=500).astype(np.float32)
    delay = 37
    y = np.zeros_like(x)
    y[delay:] = x[:-delay]
    y = y + 0.01 * rng.normal(size=y.shape).astype(np.float32)

    est = best_lag_correlation(x, y, max_lag=100)
    assert abs(est - delay) <= 2, f"expected lag~{delay}, got {est}"


def test_constrained_dtw_path_recovers_warp():
    from cross_modal_alignment.dtw_alignment import constrained_dtw_path

    rng = np.random.default_rng(1)
    n = 400
    m = 320

    t = np.linspace(0, 1, n, dtype=np.float32)
    x = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 11 * t)
    x = x.astype(np.float32) + 0.05 * rng.normal(size=n).astype(np.float32)

    base = np.linspace(0, n - 1, m, dtype=np.float32)
    noise = rng.normal(scale=2.0, size=m).astype(np.float32)
    idx = np.clip((base + noise).astype(np.int32), 0, n - 1)
    idx = np.maximum.accumulate(idx)

    y = x[idx] + 0.05 * rng.normal(size=m).astype(np.float32)

    path_x, path_y = constrained_dtw_path(x, y, w=120, alpha=0.7)

    aligned = np.zeros_like(y, dtype=np.float32)
    for px, py in zip(path_x, path_y):
        aligned[py] = x[px]

    mse = float(np.mean((aligned - y) ** 2))
    assert mse < 0.08, f"DTW alignment mse too high: {mse}"


def test_end_to_end_on_real_data():
    radar_bin = os.environ.get("RADAR_BIN")
    tactile_csv = os.environ.get("TACTILE_CSV")
    if not radar_bin or not tactile_csv:
        print("[SKIP] test_end_to_end_on_real_data (set RADAR_BIN and TACTILE_CSV to enable)")
        return

    out_dir = os.path.join(REPO_ROOT, "cross_modal_alignment", "_test_align_out")
    script = os.path.join(REPO_ROOT, "cross_modal_alignment", "align_cross_modal.py")
    cmd = [
        sys.executable,
        script,
        "--radar_bin",
        radar_bin,
        "--tactile_csv",
        tactile_csv,
        "--max_loops",
        "2000",
        "--dtw_band",
        "60",
        "--radar_max_len",
        "600",
        "--tactile_max_len",
        "600",
        "--out_dir",
        out_dir,
    ]
    subprocess.check_call(cmd, cwd=REPO_ROOT)

    assert os.path.exists(os.path.join(out_dir, "alignment_info.json"))
    assert os.path.exists(os.path.join(out_dir, "aligned_pair.csv"))


def main():
    test_best_lag_correlation()
    print("[OK] test_best_lag_correlation")

    test_constrained_dtw_path_recovers_warp()
    print("[OK] test_constrained_dtw_path_recovers_warp")

    test_end_to_end_on_real_data()
    print("[OK] test_end_to_end_on_real_data")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
