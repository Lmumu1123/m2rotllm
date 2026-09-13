# m2rotllm / RotLLM

This repository contains the RotLLM rotating-machinery health-management code and the reproducible experiments added during the MBHM and mmWave–tactile alignment work.

The repository deliberately excludes raw sensor files, MBHM HDF5 data, model checkpoints, Qwen weights, and generated logs. Put those assets on the target server and configure their paths in `.env`.

## Repository layout

```text
code/                         RotLLM models, pre-training and fine-tuning
cross_modal_alignment/        Radar/tactile alignment and validation scripts
adapt_mbhm.py                 Convert MBHM metadata and create an HDF5 wrapper
run_mbhm_pretrain.py          Train/evaluate SFN on the 10-class MBHM bearing set
run_smoke.py                  RotLLM encoder/projection smoke test
run_inference_smoke.py        Optional Qwen multimodal inference smoke test
datasets/                     Small tracked metadata/examples
.env.example                  Portable configuration template
```

## Environment

```bash
git clone https://github.com/Lmumu1123/m2rotllm.git
cd m2rotllm
conda create -n rotllm python=3.12 -y
conda activate rotllm
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` for the server's data, weights, Qwen, and log directories. Relative paths are resolved from the repository root by the supplied runners.

Run the code-only checks at any time:

```bash
python -m cross_modal_alignment.tests.run_tests
```

The real-data test is skipped unless `RADAR_BIN` and `TACTILE_CSV` are set.

## MBHM reproduction

The current MBHM artifact used locally has 135,516 samples of length 24,000, a `vibration` HDF5 dataset, and labels 0–9. The large `data.hdf5` file is not stored in Git.

After placing `metadata.parquet` and `data.hdf5` in the configured MBHM directory:

```bash
python adapt_mbhm.py --mbhm-dir /path/to/mbhm_dataset
CUDA_VISIBLE_DEVICES=0 python run_mbhm_pretrain.py \
  --epochs 5 --batch-size 256 --num-workers 4 --seed 42
```

`adapt_mbhm.py` creates `metadata.sqlite` for the existing RotLLM dataloader and a small `data_as_rotllm.hdf5` external-link wrapper. The wrapper contains an absolute link to `data.hdf5`; recreate it after moving the dataset to another server. For a quick loader check, add `--max-samples 100 --num-workers 0 --no-init-encoder`.

If `weights/encoder_weights.pth` is available, MBHM evaluation initializes the SFN encoder from it. Use `--no-init-encoder` to train from scratch.

## Corrected radar/tactile validation

The final analysis found that the recorded radar file is real ADC data, not interleaved complex I/Q: 4 RX × 512 real samples/chirp, 3 chirps/loop, and a 666.67 Hz loop rate. The corrected main program is:

```bash
python -m cross_modal_alignment.rd_alignment_test \
  --radar_bin /path/to/20260807-153437-1786088077053250.bin \
  --tactile_csv /path/to/15-34-23-384/data_0.csv \
  --radar_start_clock 15:34:37 \
  --out_dir ./runs/rd_alignment
```

It extracts a target range-bin phase displacement and compares it with the corrected tactile columns under direct timestamp alignment, smoothing-scale scans, lag scans, and a circular-shift DTW null test. The recorded conclusion is that the phase-displacement envelope is useful for slow trends (up to about `r=0.97` after 10 s smoothing), while the initial magnitude-proxy/DTW improvement must not be treated as evidence of alignment because DTW can overfit.

For the raw tactile binary timestamp check:

```bash
python -m cross_modal_alignment.bin_alignment_test \
  --tactile_bin /path/to/15-34-23-384/data_0.bin \
  --radar_bin /path/to/20260807-153437-1786088077053250.bin \
  --radar_start_clock 15:34:37 \
  --out_dir ./runs/bin_alignment
```

See [the reproduction guide](docs/REPRODUCTION_GUIDE.md), [the experiment/status briefing](docs/跨模态时序对齐实验结果与当前状态汇报.md), [the corrected algorithm notes](docs/CROSS_MODAL_ALIGNMENT_ALGORITHM.md), [the current status report](docs/ALIGNMENT_STATUS.md), and the [MBHM feasibility report](docs/轴承故障分类可行性汇报.md) for data-format details and interpretation. The remaining diagnostic scripts in `cross_modal_alignment/` are supplementary analyses; they use the same external-data convention through `ALIGNMENT_DATA_DIR`, `RADAR_BIN`, `TACTILE_CSV`, and `ALIGNMENT_OUT_DIR` where applicable.

## Original RotLLM workflow

RotLLM uses an SFN vibration encoder, a projection layer into the Qwen embedding space, and instruction fine-tuning with LoRA. The original training modules remain under `code/pre_train/` and `code/fine_tune/`. `code/` is now imported directly as a repository package; no `src -> code` symlink is required.

## Citation

```bibtex
@article{RotLLM,
  title = {A Unified Rotating Machinery Health Management Framework Leveraging Large Language Models for Diverse Components, Conditions, and Tasks},
  author = {Peng, Haotian and Gao, Jie and Liu, Jiawei and Du, Jinsong and Wang, Wei},
  year = {2025},
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {162},
  pages = {112544},
  doi = {10.1016/j.engappai.2025.112544}
}
```
