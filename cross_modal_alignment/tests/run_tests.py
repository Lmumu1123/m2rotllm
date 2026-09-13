import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_best_lag_correlation():
    from cross_modal_alignment.dtw_alignment import best_lag_correlation

    rng = np.random.default_rng(0)
    x = rng.normal(size=500).astype(np.float32)
    delay = 37
    # y is x delayed: y[j] = x[j-delay] for j>=delay
    y = np.zeros_like(x)
    y[delay:] = x[:-delay]
    # add small noise
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

    # Build a monotonic time warp mapping from y index -> x index.
    # Stretch/compress with noise but keep monotonicity.
    base = np.linspace(0, n - 1, m, dtype=np.float32)
    noise = rng.normal(scale=2.0, size=m).astype(np.float32)
    idx = np.clip((base + noise).astype(np.int32), 0, n - 1)
    idx = np.maximum.accumulate(idx)  # monotonic

    y = x[idx] + 0.05 * rng.normal(size=m).astype(np.float32)

    path_x, path_y = constrained_dtw_path(x, y, w=120, alpha=0.7)

    # Reconstruct x mapped onto y axis using the DTW path.
    aligned = np.zeros_like(y, dtype=np.float32)
    for px, py in zip(path_x, path_y):
        aligned[py] = x[px]

    mse = float(np.mean((aligned - y) ** 2))
    assert mse < 0.08, f"DTW alignment mse too high: {mse}"


def test_end_to_end_on_real_data():
    """Run the legacy pipeline when external raw files are supplied."""
    radar_bin = os.environ.get("RADAR_BIN")
    tactile_csv = os.environ.get("TACTILE_CSV")
    if not radar_bin or not tactile_csv:
        print("[SKIP] real-data test (set RADAR_BIN and TACTILE_CSV to enable)")
        return

    out_dir = os.environ.get(
        "ALIGN_OUT_DIR", str(REPO_ROOT / "cross_modal_alignment" / "_test_align_out")
    )
    script = REPO_ROOT / "cross_modal_alignment" / "align_cross_modal.py"
    subprocess.check_call(
        [
            sys.executable,
            str(script),
            "--radar_bin", radar_bin,
            "--tactile_csv", tactile_csv,
            "--max_loops", os.environ.get("ALIGN_MAX_LOOPS", "2000"),
            "--dtw_band", os.environ.get("ALIGN_DTW_BAND", "60"),
            "--radar_max_len", os.environ.get("ALIGN_RADAR_MAX_LEN", "600"),
            "--tactile_max_len", os.environ.get("ALIGN_TACTILE_MAX_LEN", "600"),
            "--out_dir", out_dir,
        ],
        cwd=REPO_ROOT,
    )

    assert (Path(out_dir) / "alignment_info.json").exists()
    assert (Path(out_dir) / "aligned_pair.csv").exists()


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
