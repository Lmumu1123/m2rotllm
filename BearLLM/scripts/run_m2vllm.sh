#!/usr/bin/env bash
# All full-reproduction commands use the independent Conda environment.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${M2VLLM_PYTHON:-$HOME/miniconda3/envs/m2vllm/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "m2vllm Python not found: $python_bin; create environment-m2vllm.yml first." >&2
  exit 1
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$python_bin" -u "$@"
