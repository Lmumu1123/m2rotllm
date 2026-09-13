# Testing guide

## Code-only checks

These tests use synthetic sequences and do not require raw data, model weights, CUDA, or MBHM:

```bash
conda activate rotllm
python -m cross_modal_alignment.tests.run_tests
```

The checks cover:

- known-delay recovery by `best_lag_correlation`;
- constrained-DTW recovery on a synthetic monotonic warp;
- an optional legacy end-to-end smoke test.

The real-data check is skipped unless both variables are set:

```bash
export RADAR_BIN=/path/to/radar.bin
export TACTILE_CSV=/path/to/data_0.csv
python -m cross_modal_alignment.tests.run_tests
```

Optional test controls are `ALIGN_MAX_LOOPS`, `ALIGN_DTW_BAND`, `ALIGN_RADAR_MAX_LEN`, `ALIGN_TACTILE_MAX_LEN`, and `ALIGN_OUT_DIR`.

## Corrected experiment checks

Run the corrected parser with a limited prefix before processing the full file:

```bash
python -m cross_modal_alignment.rd_alignment_test \
  --radar_bin /path/to/radar.bin \
  --tactile_csv /path/to/data_0.csv \
  --max_loops 20000 \
  --target_bin 13 \
  --out_dir ./runs/rd_prefix
```

For the full validation, omit `--max_loops`. Inspect:

- `radar_t0_s` and the reported overlap duration;
- automatically selected or explicitly supplied `target_bin`;
- direct timestamp correlations at each smoothing scale;
- `lag_profile` for a peak near zero;
- `dtw_null` to determine whether a DTW score is distinguishable from circular shifts.

Do not report the DTW score without its null comparison.
