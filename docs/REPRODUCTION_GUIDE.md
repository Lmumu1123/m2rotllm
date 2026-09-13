# Reproduction guide

## 1. Clone and install

```bash
git clone https://github.com/Lmumu1123/m2rotllm.git
cd m2rotllm
conda create -n rotllm python=3.12 -y
conda activate rotllm
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with paths on the new server. No path in the repository requires `/home/huangyating`; the original local layout can be used through environment variables or `.env` values.

## 2. MBHM data

Place these files in one directory:

```text
mbhm_dataset/
├── metadata.parquet
└── data.hdf5
```

`data.hdf5` must contain a `vibration` dataset with shape `[N, 24000]`. The current local artifact has 135,516 samples and labels 0–9. It is intentionally not stored in Git.

Set at least:

```dotenv
MBHM_DIR=/path/to/mbhm_dataset
MBHM_METADATA_PATH=/path/to/mbhm_dataset/metadata.sqlite
MBHM_DCN_DATASET_PATH=/path/to/mbhm_dataset/data.hdf5
```

Convert the metadata and create the compatibility wrapper:

```bash
python adapt_mbhm.py --mbhm-dir /path/to/mbhm_dataset
```

The generated wrapper is only 1 KB but contains an absolute HDF5 external link. Re-run the command after moving the data directory to another server.

Run MBHM training/evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python run_mbhm_pretrain.py \
  --epochs 5 --batch-size 256 --num-workers 4 --seed 42
```

For a quick loader/model check:

```bash
python run_mbhm_pretrain.py \
  --max-samples 100 --batch-size 16 --num-workers 0 \
  --epochs 0 --no-init-encoder
```

The MBHM runner uses `code/` directly. A local `src -> code` symlink is no longer needed.

## 3. Corrected radar/tactile experiment

Keep raw files outside the repository, for example:

```text
alignment_data/
├── 20260807-153437-1786088077053250.bin
└── 15-34-23-384/data_0.csv
```

Run the corrected analysis:

```bash
python -m cross_modal_alignment.rd_alignment_test \
  --radar_bin /path/to/alignment_data/20260807-153437-1786088077053250.bin \
  --tactile_csv /path/to/alignment_data/15-34-23-384/data_0.csv \
  --radar_start_clock 15:34:37 \
  --out_dir ./runs/rd_alignment
```

Use `--target_bin 13` to reproduce the recorded target-bin choice explicitly. Omit `--max_loops` for the full file; use a smaller value for a parser smoke test.

For the raw tactile binary timestamp check:

```bash
python -m cross_modal_alignment.bin_alignment_test \
  --tactile_bin /path/to/alignment_data/15-34-23-384/data_0.bin \
  --radar_bin /path/to/alignment_data/20260807-153437-1786088077053250.bin \
  --radar_start_clock 15:34:37 \
  --out_dir ./runs/bin_alignment
```

Supplementary scripts can use the same data root:

```bash
export ALIGNMENT_DATA_DIR=/path/to/alignment_data
export ALIGNMENT_OUT_DIR=$PWD/runs/rd_diagnostics
python -m cross_modal_alignment.fine_align
python -m cross_modal_alignment.scale_lockin_test
```

## 4. Expected files

The corrected analysis writes `rd_results.json` and `rd_series.npz`. The binary timestamp analysis writes `results.json`, `aligned_series.csv`, `lag_track.npy`, and, when matplotlib is available, `alignment.png`.
